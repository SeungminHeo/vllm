# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for the A.X-K3 gated residual stream.

Written with PyTorch ops, one write + read pair of ``AXK3GatedResidual`` is
about thirty small elementwise kernels, and a 32-layer forward runs 64 pairs:
two thirds of the kernels of a decode step. The kernels here keep the rounding
points of those formulas (bf16 stream, bf16 normalized stream, bf16 projection
outputs, fp32 gates), so the results differ from them only by the summation
order inside the reductions and dot products.

One pair becomes four launches:

* ``_write_norm_kernel``: the write of the previous block (optional) and the
  per-branch RMS normalization of the stream.
* ``_project_kernel``: the two projections that read the normalized stream,
  the low-rank ``read_down`` (+ SiLU) and ``write_proj`` (+ write gate), as
  FMA loops. A GEMM launch for these shapes costs more than the whole loop
  while the batch is small; above ``PROJECT_MAX_ROWS`` rows they go through
  ``F.linear`` and ``_project_epilogue_kernel`` instead.
* ``F.linear`` for ``read_up``.
* ``_contract_kernel``: the sigmoid read gate and the branch mean.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

# Rows (tokens) up to which the projections run in the FMA kernel.
PROJECT_MAX_ROWS = 48
# Projection outputs per program and stream columns per loop step.
_PROJECT_OUTPUTS = 4
_PROJECT_BLOCK = 8192
# Block-input positions per program.
_CONTRACT_TILE = 512


@triton.jit
def _write_norm_kernel(
    stream_ptr,  # [M, N * H] bf16
    block_ptr,  # [M, H] bf16, block output (HAS_WRITE)
    gate_ptr,  # [M, N] fp32, write gate (HAS_WRITE)
    gamma_ptr,  # [N * H] bf16 (HAS_NORM)
    written_ptr,  # [M, N * H] bf16, out (HAS_WRITE)
    normed_ptr,  # [M, N * H] bf16, out (HAS_NORM)
    N: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    HAS_WRITE: tl.constexpr,
    HAS_NORM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    base = row * (N * H) + branch * H

    x = tl.load(stream_ptr + base + offs, mask=mask, other=0.0)
    if HAS_WRITE:
        gate = tl.load(gate_ptr + row * N + branch)
        block = tl.load(block_ptr + row * H + offs, mask=mask, other=0.0)
        scaled = (gate * block.to(tl.float32)).to(tl.bfloat16)
        x = (x.to(tl.float32) + scaled.to(tl.float32)).to(tl.bfloat16)
        tl.store(written_ptr + base + offs, x, mask=mask)
    if HAS_NORM:
        xf = x.to(tl.float32)
        rstd = tl.rsqrt(tl.sum(xf * xf, axis=0) / H + EPS)
        gamma = tl.load(gamma_ptr + branch * H + offs, mask=mask, other=0.0)
        normed = xf * rstd * gamma.to(tl.float32)
        tl.store(normed_ptr + base + offs, normed.to(tl.bfloat16), mask=mask)


@triton.jit
def _project_kernel(
    normed_ptr,  # [M, W] bf16
    down_ptr,  # [R, W] bf16, read_down weight
    proj_ptr,  # [N, W] bf16, write_proj weight (last program of a row)
    bias_ptr,  # [N] fp32, write bias (last program of a row)
    low_ptr,  # [M, R] bf16, out: silu(read_down(normed) / N)
    gate_ptr,  # [M, N] fp32, out: 2 * sigmoid(write_proj(normed) / N + bias)
    N: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    OUTPUTS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Programs 0 .. R / OUTPUTS - 1 of a row take OUTPUTS rows of read_down
    # each; one more program, launched only for a module that writes, takes
    # the N rows of write_proj.
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    is_down = group < R // OUTPUTS
    outs = tl.arange(0, OUTPUTS)

    acc = tl.zeros([OUTPUTS], dtype=tl.float32)
    for start in range(0, W, BLOCK_W):
        cols = start + tl.arange(0, BLOCK_W)
        col_mask = cols < W
        x = tl.load(normed_ptr + row * W + cols, mask=col_mask, other=0.0)
        if is_down:
            w_ptrs = down_ptr + (group * OUTPUTS + outs)[:, None] * W + cols[None, :]
            w = tl.load(w_ptrs, mask=col_mask[None, :], other=0.0)
        else:
            w_ptrs = proj_ptr + outs[:, None] * W + cols[None, :]
            w_mask = (outs < N)[:, None] & col_mask[None, :]
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)

    # The formulas divide the bf16 projection output by N (exact in bf16).
    scaled = acc.to(tl.bfloat16).to(tl.float32) / N
    if is_down:
        low = scaled * tl.sigmoid(scaled)
        tl.store(low_ptr + row * R + group * OUTPUTS + outs, low.to(tl.bfloat16))
    else:
        out_mask = outs < N
        bias = tl.load(bias_ptr + outs, mask=out_mask, other=0.0)
        gate = 2.0 * tl.sigmoid(scaled + bias)
        tl.store(gate_ptr + row * N + outs, gate, mask=out_mask)


@triton.jit
def _project_epilogue_kernel(
    down_ptr,  # [M, R] bf16, read_down(normed)
    proj_ptr,  # [M, N] bf16, write_proj(normed) (HAS_WRITE)
    bias_ptr,  # [N] fp32, write bias (HAS_WRITE)
    low_ptr,  # [M, R] bf16, out
    gate_ptr,  # [M, N] fp32, out (HAS_WRITE)
    N: tl.constexpr,
    R: tl.constexpr,
    HAS_WRITE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ranks = tl.arange(0, BLOCK_R)
    rank_mask = ranks < R
    down = tl.load(down_ptr + row * R + ranks, mask=rank_mask, other=0.0)
    scaled = down.to(tl.float32) / N
    low = scaled * tl.sigmoid(scaled)
    tl.store(low_ptr + row * R + ranks, low.to(tl.bfloat16), mask=rank_mask)
    if HAS_WRITE:
        outs = tl.arange(0, BLOCK_N)
        out_mask = outs < N
        proj = tl.load(proj_ptr + row * N + outs, mask=out_mask, other=0.0)
        bias = tl.load(bias_ptr + outs, mask=out_mask, other=0.0)
        gate = 2.0 * tl.sigmoid(proj.to(tl.float32) / N + bias)
        tl.store(gate_ptr + row * N + outs, gate, mask=out_mask)


@triton.jit
def _contract_kernel(
    normed_ptr,  # [M, N * H] bf16
    logits_ptr,  # [M, N * H] bf16, read_up(low)
    out_ptr,  # [M, H] bf16, out: block input
    N: tl.constexpr,
    H: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pos = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = pos < H

    acc = tl.zeros([TILE], dtype=tl.float32)
    for branch in tl.static_range(N):
        offs = row * (N * H) + branch * H + pos
        logits = tl.load(logits_ptr + offs, mask=mask, other=0.0)
        normed = tl.load(normed_ptr + offs, mask=mask, other=0.0)
        acc += tl.sigmoid(logits.to(tl.float32)) * normed.to(tl.float32)
    tl.store(out_ptr + row * H + pos, (acc / N).to(tl.bfloat16), mask=mask)


def _next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _rows(stream: torch.Tensor, width: int) -> torch.Tensor:
    assert stream.dtype == torch.bfloat16 and stream.shape[-1] == width
    rows = stream.reshape(-1, width)
    assert rows.is_contiguous()
    return rows


def _write_norm(
    stream: torch.Tensor,
    num_streams: int,
    block_output: torch.Tensor | None = None,
    write_gate: torch.Tensor | None = None,
    gamma: torch.Tensor | None = None,
    eps: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Launch ``_write_norm_kernel`` on 2-D rows.

    Returns:
        The written stream (with ``block_output`` and ``write_gate``) and the
        normalized stream (with ``gamma``); ``None`` for the part not asked for.

    """
    num_rows, width = stream.shape
    hidden = width // num_streams
    written: torch.Tensor | None = None
    normed: torch.Tensor | None = None
    if block_output is not None:
        assert write_gate is not None
        assert block_output.dtype == torch.bfloat16 and block_output.is_contiguous()
        assert write_gate.dtype == torch.float32 and write_gate.is_contiguous()
        written = torch.empty_like(stream)
    if gamma is not None:
        assert gamma.dtype == torch.bfloat16 and gamma.is_contiguous()
        normed = torch.empty_like(stream)
    # Pointers of a part that is compiled out are never dereferenced.
    _write_norm_kernel[(num_rows, num_streams)](
        stream,
        block_output if block_output is not None else stream,
        write_gate if write_gate is not None else stream,
        gamma if gamma is not None else stream,
        written if written is not None else stream,
        normed if normed is not None else stream,
        N=num_streams,
        H=hidden,
        EPS=eps,
        HAS_WRITE=written is not None,
        HAS_NORM=normed is not None,
        BLOCK_H=_next_power_of_2(hidden),
        num_warps=16,
    )
    return written, normed


def _read_normed(
    normed: torch.Tensor,
    read_down: torch.Tensor,
    read_up: torch.Tensor,
    write_proj: torch.Tensor | None,
    write_bias: torch.Tensor | None,
    num_streams: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Block input and write gate from the normalized stream rows."""
    num_rows, width = normed.shape
    hidden, rank = width // num_streams, read_down.shape[0]
    assert read_down.shape == (rank, width) and read_down.is_contiguous()
    assert read_down.dtype == read_up.dtype == torch.bfloat16
    low = normed.new_empty(num_rows, rank)
    write_gate: torch.Tensor | None = None
    if write_proj is not None:
        assert write_bias is not None and write_bias.dtype == torch.float32
        assert write_proj.shape == (num_streams, width)
        assert write_proj.dtype == torch.bfloat16 and write_proj.is_contiguous()
        write_gate = normed.new_empty(num_rows, num_streams, dtype=torch.float32)

    if (
        num_rows <= PROJECT_MAX_ROWS
        and rank % _PROJECT_OUTPUTS == 0
        and num_streams <= _PROJECT_OUTPUTS
    ):
        programs = rank // _PROJECT_OUTPUTS + int(write_gate is not None)
        _project_kernel[(num_rows, programs)](
            normed,
            read_down,
            write_proj if write_proj is not None else read_down,
            write_bias if write_bias is not None else read_down,
            low,
            write_gate if write_gate is not None else low,
            N=num_streams,
            W=width,
            R=rank,
            OUTPUTS=_PROJECT_OUTPUTS,
            BLOCK_W=min(_PROJECT_BLOCK, _next_power_of_2(width)),
            num_warps=8,
        )
    else:
        down = F.linear(normed, read_down)
        _project_epilogue_kernel[(num_rows,)](
            down,
            F.linear(normed, write_proj) if write_proj is not None else down,
            write_bias if write_bias is not None else down,
            low,
            write_gate if write_gate is not None else low,
            N=num_streams,
            R=rank,
            HAS_WRITE=write_gate is not None,
            BLOCK_R=_next_power_of_2(rank),
            BLOCK_N=_next_power_of_2(num_streams),
            num_warps=1,
        )

    logits = F.linear(low, read_up)
    block_input = normed.new_empty(num_rows, hidden)
    tile = min(_CONTRACT_TILE, _next_power_of_2(hidden))
    _contract_kernel[(num_rows, triton.cdiv(hidden, tile))](
        normed,
        logits,
        block_input,
        N=num_streams,
        H=hidden,
        TILE=tile,
        num_warps=4,
    )
    return block_input, write_gate


def gated_residual_write(
    stream: torch.Tensor,
    block_output: torch.Tensor,
    write_gate: torch.Tensor,
    num_streams: int,
) -> torch.Tensor:
    """``stream + write_gate * block_output`` on every branch, rounded as bf16."""
    width = stream.shape[-1]
    written, _ = _write_norm(
        _rows(stream, width),
        num_streams,
        block_output.reshape(-1, width // num_streams),
        write_gate.reshape(-1, num_streams),
    )
    assert written is not None
    return written.view(stream.shape)


def gated_residual_read(
    stream: torch.Tensor,
    gamma: torch.Tensor,
    read_down: torch.Tensor,
    read_up: torch.Tensor,
    write_proj: torch.Tensor | None,
    write_bias: torch.Tensor | None,
    num_streams: int,
    eps: float,
    block_output: torch.Tensor | None = None,
    prev_write_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Read the stream, after an optional write of the previous block.

    With ``block_output`` and ``prev_write_gate`` the previous block's write is
    applied first, in the same kernel that normalizes the stream.

    Returns:
        The written stream (``None`` without a write), the block input
        ``[..., hidden_size]`` and the fp32 write gate ``[..., num_streams]``
        (``None`` without ``write_proj``).

    """
    width = stream.shape[-1]
    hidden = width // num_streams
    if block_output is not None:
        assert prev_write_gate is not None
        block_output = block_output.reshape(-1, hidden)
        prev_write_gate = prev_write_gate.reshape(-1, num_streams)
    written, normed = _write_norm(
        _rows(stream, width), num_streams, block_output, prev_write_gate, gamma, eps
    )
    assert normed is not None
    block_input, write_gate = _read_normed(
        normed, read_down, read_up, write_proj, write_bias, num_streams
    )
    lead = stream.shape[:-1]
    return (
        written.view(stream.shape) if written is not None else None,
        block_input.view(*lead, hidden),
        write_gate.view(*lead, num_streams) if write_gate is not None else None,
    )
