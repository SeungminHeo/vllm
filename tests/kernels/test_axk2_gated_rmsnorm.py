# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused A.X-K2 gated RMSNorm kernels.

The op is served by two CuTe DSL cluster kernels (a row kernel for decode
batches, a tensor-core tile kernel above) with the per-row Triton kernel as
fallback. All keep the pipeline in fp32 and round to bf16 once, so each is
compared against an fp32 PyTorch reference with rtol/atol=1e-2, the tolerance
the deepseek_v32 fused-kernel test uses for bf16 norm outputs, and the CuTe
kernels are compared with the Triton kernel.
"""

import pytest
import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.models.axk2 import gated_rmsnorm as grn
from vllm.models.axk2.gated_rmsnorm_cute import cute_gated_rmsnorm, make_wu_frag
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

EPS = 1e-6
RANK = 16

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the gated RMSNorm kernels require CUDA",
)
requires_cute = pytest.mark.skipif(
    cute_gated_rmsnorm() is None, reason="CuTe DSL not installed"
)


def _reference(x, residual, w_norm, w_down, w_up):
    h = x.float() if residual is None else x.float() + residual.float()
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * w_norm.float()
    gate = F.silu(y @ w_down.float().T) @ w_up.float().T
    return h, y * torch.sigmoid(gate)


def _inputs(num_tokens: int, hidden_size: int, with_residual: bool):
    set_random_seed(0)
    dev = "cuda"
    x = torch.randn(num_tokens, hidden_size, device=dev, dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    w_norm = torch.randn(hidden_size, device=dev, dtype=torch.bfloat16)
    w_down = torch.randn(RANK, hidden_size, device=dev) / hidden_size**0.5
    w_up = torch.randn(hidden_size, RANK, device=dev) / RANK**0.5
    return x, residual, w_norm, w_down.to(torch.bfloat16), w_up.to(torch.bfloat16)


def _check(out, x, x_in, residual, h, ref):
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # The first decoder layer aliases the residual to x, so x must stay intact.
    assert torch.equal(x, x_in)
    if residual is not None:
        torch.testing.assert_close(residual.float(), h, rtol=1e-2, atol=1e-2)


@requires_cuda
@pytest.mark.parametrize("hidden_size", [2048, 3072])
@pytest.mark.parametrize("num_tokens", [1, 17, 512])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gated_rmsnorm_triton_matches_fp32_reference(
    hidden_size: int, num_tokens: int, with_residual: bool
):
    """The per-row Triton kernel matches fp32 math and adds the residual in
    place. hidden_size=3072 is not a power of two, so it exercises the masked
    loads."""
    x, residual, w_norm, w_down, w_up = _inputs(num_tokens, hidden_size, with_residual)
    h, ref = _reference(x, residual, w_norm, w_down, w_up)
    x_in = x.clone()
    out, _ = grn.axk2_gated_rmsnorm_triton(x, residual, w_norm, w_down, w_up, EPS)
    _check(out, x, x_in, residual, h, ref)


@requires_cuda
@requires_cute
@pytest.mark.parametrize("hidden_size", [7168, 4096])
# 1..32 rows take the row kernel (clusters of 8 / 4) on B200; 64 and up the tile
# kernel, 149 / 1001 with a ragged last 16-row tile and multi-wave grids.
@pytest.mark.parametrize("num_tokens", [1, 4, 32, 64, 149, 1001, 2048])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gated_rmsnorm_cute_matches_triton(
    hidden_size: int, num_tokens: int, with_residual: bool
):
    """Both CuTe cluster kernels match the fp32 reference and the per-row
    Triton kernel (same fp32 pipeline, same single rounding) on the A.X-K2
    shape (H=7168, rank 16) and a smaller power-of-two H."""
    x, residual, w_norm, w_down, w_up = _inputs(num_tokens, hidden_size, with_residual)
    h, ref = _reference(x, residual, w_norm, w_down, w_up)
    res_tri = residual.clone() if residual is not None else None
    out_tri, _ = grn.axk2_gated_rmsnorm_triton(x, res_tri, w_norm, w_down, w_up, EPS)
    x_in = x.clone()
    kern = cute_gated_rmsnorm()
    assert kern is not None
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    if num_tokens <= 8:
        assert kern.kind(hidden_size, num_tokens, sms) == "row"
    elif num_tokens >= 64:
        assert kern.kind(hidden_size, num_tokens, sms) == "tile"
    out = kern(x, residual, w_norm, w_down, w_up, make_wu_frag(w_up), EPS)
    _check(out, x, x_in, residual, h, ref)
    torch.testing.assert_close(out.float(), out_tri.float(), rtol=1e-2, atol=1e-2)
    if residual is not None:
        torch.testing.assert_close(residual, res_tri)


@requires_cuda
@requires_cute
def test_make_wu_frag_is_a_permutation_of_w_up():
    """The fragment tensor holds every W_up entry exactly once (lane / tile
    bookkeeping of the tile kernel's B operand)."""
    w_up = torch.arange(7168 * RANK, device="cuda", dtype=torch.float32)
    w_up = w_up.reshape(7168, RANK).to(torch.bfloat16)
    frag = make_wu_frag(w_up)
    assert frag.shape == (7168 // 32, 32, RANK)
    assert torch.equal(
        frag.float().flatten().sort().values, w_up.float().flatten().sort().values
    )


@requires_cuda
@pytest.mark.parametrize("num_tokens", [4, 200, 3000])
def test_gated_rmsnorm_op_dispatch(num_tokens: int):
    """The registered op takes the CuTe kernels whenever they are available
    and every choice matches the reference."""
    x, residual, w_norm, w_down, w_up = _inputs(num_tokens, 7168, True)
    h, ref = _reference(x, residual, w_norm, w_down, w_up)
    backend = grn.pick_backend(x, w_down)
    assert backend == ("cute" if cute_gated_rmsnorm() is not None else "triton")
    x_in = x.clone()
    out = torch.ops.vllm.axk2_gated_rmsnorm(
        x, residual, w_norm, w_down, w_up, make_wu_frag(w_up), EPS
    )
    _check(out, x, x_in, residual, h, ref)


@requires_cuda
def test_gated_rmsnorm_op_forced_triton(monkeypatch):
    """VLLM_AXK2_GATED_RMSNORM_BACKEND=triton routes the op to the per-row
    Triton kernel even when CuTe DSL is available."""
    monkeypatch.setattr(envs, "VLLM_AXK2_GATED_RMSNORM_BACKEND", "triton")
    x, residual, w_norm, w_down, w_up = _inputs(64, 7168, True)
    h, ref = _reference(x, residual, w_norm, w_down, w_up)
    assert grn.pick_backend(x, w_down) == "triton"
    x_in = x.clone()
    out = torch.ops.vllm.axk2_gated_rmsnorm(
        x, residual, w_norm, w_down, w_up, make_wu_frag(w_up), EPS
    )
    _check(out, x, x_in, residual, h, ref)


def test_gated_rmsnorm_rejects_residual_needing_a_copy():
    """A reshaped copy of the residual would silently drop its in-place update."""
    hidden_size = 64
    x = torch.randn(4, 2, hidden_size, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, hidden_size, dtype=torch.bfloat16).transpose(0, 1)
    w_norm = torch.ones(hidden_size, dtype=torch.bfloat16)
    w_down = torch.zeros(RANK, hidden_size, dtype=torch.bfloat16)
    w_up = torch.zeros(hidden_size, RANK, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="view"):
        grn.axk2_gated_rmsnorm_triton(x, residual, w_norm, w_down, w_up, EPS)
