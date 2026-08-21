# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _axk2_gated_rmsnorm_kernel(
    # Pointers
    x_ptr,  # [M, H]
    residual_ptr,  # [M, H]  (may be unused if not HAS_RESIDUAL)
    out_ptr,  # [M, H]
    w_norm_ptr,  # [H]
    w_down_ptr,  # [R, H]  PyTorch Linear weight layout: [out, in]
    w_up_ptr,  # [H, R]  PyTorch Linear weight layout: [out, in]
    # Sizes
    M,
    stride_xm,  # stride to next row in x
    stride_rm,  # stride to next row in residual
    stride_om,  # stride to next row in out
    # Constants
    H: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)

    # Pointers to this row
    x_row_ptr = x_ptr + pid * stride_xm
    out_row_ptr = out_ptr + pid * stride_om

    # ---------- Step 1: Load x and (optionally) residual add ----------
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    x = tl.load(x_row_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    if HAS_RESIDUAL:
        res_row_ptr = residual_ptr + pid * stride_rm
        r = tl.load(res_row_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        x = x + r
        # Update residual in-place (cast back to original dtype, bf16)
        tl.store(res_row_ptr + offs_h, x.to(tl.bfloat16), mask=mask_h)

    # ---------- Step 2: RMSNorm ----------
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Load norm weight and apply
    w_norm = tl.load(w_norm_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    y = x * rstd * w_norm  # [BLOCK_H], fp32

    # ---------- Step 3: W_down GEMM: y @ W_down.T -> z[R] ----------
    offs_r = tl.arange(0, R)
    w_down_offsets = offs_r[:, None] * H + offs_h[None, :]
    w_down_mask = mask_h[None, :]
    w_down_block = tl.load(
        w_down_ptr + w_down_offsets,
        mask=w_down_mask,
        other=0.0,
    ).to(tl.float32)  # [R, BLOCK_H]

    z = tl.sum(w_down_block * y[None, :], axis=1)  # [R]

    # ---------- Step 4: SiLU ----------
    z = z * tl.sigmoid(z)

    # ---------- Step 5: W_up GEMM: z @ W_up.T -> gate[H] ----------
    w_up_offsets = offs_h[:, None] * R + offs_r[None, :]
    w_up_mask = mask_h[:, None]
    w_up_block = tl.load(
        w_up_ptr + w_up_offsets,
        mask=w_up_mask,
        other=0.0,
    ).to(tl.float32)  # [BLOCK_H, R]

    gate = tl.sum(w_up_block * z[None, :], axis=1)  # [BLOCK_H]

    # ---------- Step 6: Sigmoid gate and apply ----------
    out = y * tl.sigmoid(gate)  # [BLOCK_H], fp32

    # ---------- Step 7: Store ----------
    tl.store(out_row_ptr + offs_h, out.to(tl.bfloat16), mask=mask_h)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def axk2_gated_rmsnorm_triton(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    w_norm: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    orig_shape = x.shape
    H = orig_shape[-1]
    M = x.numel() // H

    x_2d = x.reshape(M, H)
    residual_2d = residual.reshape(M, H) if residual is not None else None

    R = w_down.shape[0]
    BLOCK_H = _next_power_of_2(H)

    out_2d = torch.empty_like(x_2d)

    stride_xm = x_2d.stride(0)
    stride_om = out_2d.stride(0)
    stride_rm = residual_2d.stride(0) if residual_2d is not None else 0

    grid = (M,)
    if BLOCK_H <= 2048:
        num_warps = 4
    elif BLOCK_H <= 4096:
        num_warps = 8
    else:
        num_warps = 16

    _axk2_gated_rmsnorm_kernel[grid](
        x_2d,
        residual_2d if residual_2d is not None else x_2d,
        out_2d,
        w_norm,
        w_down,
        w_up,
        M,
        stride_xm,
        stride_rm,
        stride_om,
        H=H,
        R=R,
        EPS=eps,
        HAS_RESIDUAL=residual is not None,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )

    out = out_2d.reshape(orig_shape)
    return out, residual


def _axk2_gated_rmsnorm_impl(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    w_norm: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out, _ = axk2_gated_rmsnorm_triton(x, residual, w_norm, w_down, w_up, eps)
    return out


def _axk2_gated_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    w_norm: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="axk2_gated_rmsnorm",
    op_func=_axk2_gated_rmsnorm_impl,
    mutates_args=["residual"],
    fake_impl=_axk2_gated_rmsnorm_fake,
)


class AXK2GatedRMSNorm(nn.Module):
    """Gated RMSNorm with low-rank bottleneck fused into Triton kernel."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        rank: int = 16,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.eps = eps

        self.norm = RMSNorm(hidden_size, eps=eps)
        self.W_down = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.W_down",
        )
        self.W_up = ReplicatedLinear(
            rank,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.W_up",
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.norm.weight

    @property
    def variance_epsilon(self) -> float:
        return self.norm.variance_epsilon

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            if residual is not None:
                y, residual = self.norm(x, residual)
            else:
                y = self.norm(x)
            z, _ = self.W_down(y)
            gate, _ = self.W_up(torch.nn.functional.silu(z))
            out = y * torch.sigmoid(gate)
            if residual is not None:
                return out, residual
            return out

        out = torch.ops.vllm.axk2_gated_rmsnorm(
            x,
            residual,
            self.norm.weight,
            self.W_down.weight,
            self.W_up.weight,
            self.eps,
        )
        if residual is not None:
            return out, residual
        return out
