# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Gated RMSNorm and Activation Quantization Triton Kernel.

This kernel fuses:
1. Optional residual addition: x = x + residual
2. Gated RMS Normalization:
   - If norm_before_gate: y = (RMSNorm(x) + bias) * act(z)
   - If not norm_before_gate: y = RMSNorm(x * act(z)) + bias
3. Activation Quantization in registers:
   - FP8 dynamic per-token quantization
   - FP8 dynamic per-block / group quantization
   - FP8 static quantization
   - INT8 dynamic per-token quantization
   - None / unquantized (returns floating-point tensor)

By keeping normalized and gated intermediate activations inside registers and SRAM,
this fused kernel eliminates redundant round-trips to DRAM / HBM, drastically
reducing memory bandwidth consumption for both prefill and decode phases.
"""

from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

logger = init_logger(__name__)


class GateActivation(IntEnum):
    SILU = 0
    SIGMOID = 1
    IDENTITY = 2
    GELU = 3

    @classmethod
    def from_str(cls, act: str) -> "GateActivation":
        act_lower = act.lower()
        if act_lower in ("silu", "swish"):
            return cls.SILU
        elif act_lower == "sigmoid":
            return cls.SIGMOID
        elif act_lower in ("identity", "none", "linear"):
            return cls.IDENTITY
        elif act_lower == "gelu":
            return cls.GELU
        else:
            raise ValueError(
                f"Unsupported gate activation '{act}'. "
                "Supported: silu, sigmoid, identity, gelu"
            )


class QuantMode(IntEnum):
    NONE = 0
    FP8_PER_TOKEN = 1
    FP8_PER_BLOCK = 2
    FP8_STATIC = 3
    INT8_PER_TOKEN = 4

    @classmethod
    def from_str(cls, mode: str) -> "QuantMode":
        mode_lower = mode.lower()
        if mode_lower in ("none", "unquantized", "fp16", "bf16", "fp32"):
            return cls.NONE
        elif mode_lower in (
            "fp8_per_token",
            "fp8_token",
            "fp8_dynamic_token",
            "fp8_dynamic",
        ):
            return cls.FP8_PER_TOKEN
        elif mode_lower in (
            "fp8_per_block",
            "fp8_block",
            "fp8_group",
            "fp8_dynamic_group",
        ):
            return cls.FP8_PER_BLOCK
        elif mode_lower in ("fp8_static", "static_fp8"):
            return cls.FP8_STATIC
        elif mode_lower in (
            "int8_per_token",
            "int8_token",
            "int8_dynamic_token",
            "int8_dynamic",
        ):
            return cls.INT8_PER_TOKEN
        else:
            raise ValueError(
                f"Unsupported quant_mode '{mode}'. Supported: none, "
                "fp8_per_token, fp8_per_block, fp8_static, int8_per_token"
            )


if HAS_TRITON and triton is not None:

    @triton.heuristics(
        {
            "HAS_BIAS": lambda args: args["B_ptr"] is not None,
            "HAS_RESIDUAL": lambda args: args["R_ptr"] is not None,
            "HAS_SCALE_UB": lambda args: args["scale_ub_ptr"] is not None,
            "HAS_SCALE_OUT": lambda args: args["Scale_ptr"] is not None,
        }
    )
    @triton.jit
    def _fused_gated_rmsnorm_quant_kernel(
        X_ptr,
        Z_ptr,
        W_ptr,
        B_ptr,
        R_ptr,
        Y_ptr,
        Scale_ptr,
        scale_ub_ptr,
        static_scale,
        stride_x_row,
        stride_z_row,
        stride_r_row,
        stride_y_row,
        stride_scale_row,
        M,
        N,
        eps,
        qmin,
        qmax,
        min_scaling_factor,
        BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        HAS_SCALE_UB: tl.constexpr,
        HAS_SCALE_OUT: tl.constexpr,
        NORM_BEFORE_GATE: tl.constexpr,
        GATE_ACT: tl.constexpr,
        QUANT_MODE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= M:
            return

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x_offset = row_idx * stride_x_row + cols
        x = tl.load(X_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)

        z_offset = row_idx * stride_z_row + cols
        z = tl.load(Z_ptr + z_offset, mask=mask, other=0.0).to(tl.float32)

        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        if HAS_RESIDUAL:
            r_offset = row_idx * stride_r_row + cols
            r = tl.load(R_ptr + r_offset, mask=mask, other=0.0).to(tl.float32)
            x = x + r
            tl.store(R_ptr + r_offset, x.to(R_ptr.dtype.element_ty), mask=mask)

        if GATE_ACT == 0:
            g = z * tl.sigmoid(z)
        elif GATE_ACT == 1:
            g = tl.sigmoid(z)
        elif GATE_ACT == 2:
            g = z
        else:
            g = z * 0.5 * (1.0 + tl.erf(z * 0.7071067811865475))

        if NORM_BEFORE_GATE:
            var = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = x * rstd * w
            if HAS_BIAS:
                b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = normed + b
            y = normed * g
        else:
            x_gated = x * g
            var = tl.sum(tl.where(mask, x_gated * x_gated, 0.0), axis=0) / N
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = x_gated * rstd * w
            if HAS_BIAS:
                b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = normed + b
            y = normed

        y_offset = row_idx * stride_y_row + cols

        if QUANT_MODE == 0:
            tl.store(Y_ptr + y_offset, y.to(Y_ptr.dtype.element_ty), mask=mask)
        elif QUANT_MODE == 1 or QUANT_MODE == 2:
            y_abs = tl.where(mask, tl.abs(y), 0.0)
            amax = tl.max(y_abs, axis=0)
            if HAS_SCALE_UB:
                sub = tl.load(scale_ub_ptr).to(tl.float32)
                amax = tl.minimum(amax, sub)
            scale = tl.maximum(amax / qmax, min_scaling_factor)
            if HAS_SCALE_OUT:
                tl.store(Scale_ptr + row_idx * stride_scale_row, scale)
            q = tl.clamp(y / scale, qmin, qmax)
            tl.store(Y_ptr + y_offset, q.to(Y_ptr.dtype.element_ty), mask=mask)
        elif QUANT_MODE == 3:
            q = tl.clamp(y / static_scale, qmin, qmax)
            tl.store(Y_ptr + y_offset, q.to(Y_ptr.dtype.element_ty), mask=mask)
        elif QUANT_MODE == 4:
            y_abs = tl.where(mask, tl.abs(y), 0.0)
            amax = tl.max(y_abs, axis=0)
            if HAS_SCALE_UB:
                sub = tl.load(scale_ub_ptr).to(tl.float32)
                amax = tl.minimum(amax, sub)
            scale = tl.maximum(amax / 127.0, min_scaling_factor)
            if HAS_SCALE_OUT:
                tl.store(Scale_ptr + row_idx * stride_scale_row, scale)
            q = tl.clamp(tl.floor(y / scale + 0.5), qmin, qmax)
            tl.store(Y_ptr + y_offset, q.to(Y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _fused_gate_act_quant_kernel(
        Y_ptr,
        Gate_ptr,
        Out_ptr,
        Scale_ptr,
        stride_y_row,
        stride_g_row,
        stride_out_row,
        M,
        N,
        qmin: tl.float32,
        qmax: tl.float32,
        min_scale: tl.float32,
        GATE_ACT: tl.constexpr,
        QUANT_MODE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= M:
            return

        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        y_offset = row_idx * stride_y_row + cols
        g_offset = row_idx * stride_g_row + cols

        y = tl.load(Y_ptr + y_offset, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(Gate_ptr + g_offset, mask=mask, other=0.0).to(tl.float32)

        if GATE_ACT == 0:
            g = z * tl.sigmoid(z)
        elif GATE_ACT == 1:
            g = tl.sigmoid(z)
        elif GATE_ACT == 2:
            g = z
        else:
            g = z * 0.5 * (1.0 + tl.erf(z * 0.7071067811865475))

        out = y * g

        out_offset = row_idx * stride_out_row + cols
        if QUANT_MODE == 0:
            tl.store(Out_ptr + out_offset, out.to(Out_ptr.dtype.element_ty), mask=mask)
        else:
            y_abs = tl.where(mask, tl.abs(out), 0.0)
            amax = tl.max(y_abs, axis=0)
            scale = tl.maximum(amax / qmax, min_scale)
            tl.store(Scale_ptr + row_idx, scale)
            q = tl.clamp(out / scale, qmin, qmax)
            tl.store(Out_ptr + out_offset, q.to(Out_ptr.dtype.element_ty), mask=mask)
else:
    _fused_gated_rmsnorm_quant_kernel = None
    _fused_gate_act_quant_kernel = None


def fused_gate_act_quant(
    y: torch.Tensor,
    gate: torch.Tensor,
    activation: str = "sigmoid",
    quant_type: str = "none",
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuses gate activation elementwise multiply with optional dynamic quantization."""
    quant_enum = QuantMode.from_str(quant_type)
    act_enum = GateActivation.from_str(activation)

    is_cuda_or_xpu = y.is_cuda or (
        hasattr(torch, "xpu") and torch.xpu.is_available() and y.is_xpu
    )

    if not (HAS_TRITON and _fused_gate_act_quant_kernel is not None and is_cuda_or_xpu):
        if act_enum == GateActivation.SIGMOID:
            g = torch.sigmoid(gate.float())
        elif act_enum == GateActivation.SILU:
            g = gate.float() * torch.sigmoid(gate.float())
        elif act_enum == GateActivation.IDENTITY:
            g = gate.float()
        else:
            g = F.gelu(gate.float())

        out = (y.float() * g).to(y.dtype)
        if quant_enum == QuantMode.NONE:
            return out, None
        elif quant_enum == QuantMode.FP8_PER_TOKEN:
            target_dtype = out_dtype or current_platform.fp8_dtype()
            finfo = torch.finfo(target_dtype)
            amax = torch.amax(torch.abs(out.float()), dim=-1, keepdim=True)
            scale = torch.clamp(amax / finfo.max, min=1.0 / (finfo.max * 512.0))
            q = torch.clamp(out.float() / scale, min=finfo.min, max=finfo.max).to(
                target_dtype
            )
            return q, scale

    orig_shape = y.shape
    hidden_size = orig_shape[-1]
    y_2d = y.reshape(-1, hidden_size).contiguous()
    gate_2d = gate.reshape(-1, hidden_size).contiguous()
    M, N = y_2d.shape

    if quant_enum == QuantMode.NONE:
        target_dtype = out_dtype or y.dtype
        qmin, qmax = 0.0, 0.0
        min_scale = 0.0
        scale_out = None
    else:
        target_dtype = out_dtype or current_platform.fp8_dtype()
        qmin, qmax = get_fp8_min_max()
        min_scale = 1.0 / (qmax * 512.0)
        scale_out = torch.empty((M, 1), device=y.device, dtype=torch.float32)

    out_2d = torch.empty((M, N), device=y.device, dtype=target_dtype)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_N // 256, 1), 8)

    _fused_gate_act_quant_kernel[(M,)](
        y_2d,
        gate_2d,
        out_2d,
        scale_out if scale_out is not None else out_2d,
        y_2d.stride(0),
        gate_2d.stride(0),
        out_2d.stride(0),
        M,
        N,
        qmin=qmin,
        qmax=qmax,
        min_scale=min_scale,
        GATE_ACT=int(act_enum),
        QUANT_MODE=int(quant_enum),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )

    final_out = out_2d.reshape(orig_shape)
    final_scale = (
        scale_out.reshape(*orig_shape[:-1], 1) if scale_out is not None else None
    )
    return final_out, final_scale


def fused_gated_rmsnorm_quant_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    scale_ub: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    gate_activation: str = "silu",
    quant_type: str = "fp8_per_token",
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """PyTorch reference implementation of Gated RMSNorm + Act Quantization."""
    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    z_2d = z.reshape(-1, hidden_size)
    M, N = x_2d.shape

    if residual is not None:
        res_2d = residual.reshape(-1, hidden_size)
        x_acc = x_2d.float() + res_2d.float()
        res_out = x_acc.to(residual.dtype).reshape(orig_shape)
    else:
        x_acc = x_2d.float()
        res_out = None

    act_enum = GateActivation.from_str(gate_activation)
    if act_enum == GateActivation.SILU:
        g = z_2d.float() * torch.sigmoid(z_2d.float())
    elif act_enum == GateActivation.SIGMOID:
        g = torch.sigmoid(z_2d.float())
    elif act_enum == GateActivation.IDENTITY:
        g = z_2d.float()
    elif act_enum == GateActivation.GELU:
        g = F.gelu(z_2d.float())

    w = weight.float()
    b = bias.float() if bias is not None else None

    if norm_before_gate:
        var = torch.mean(x_acc.pow(2), dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        normed = x_acc * rstd * w
        if b is not None:
            normed = normed + b
        y = normed * g
    else:
        x_gated = x_acc * g
        var = torch.mean(x_gated.pow(2), dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        normed = x_gated * rstd * w
        if b is not None:
            normed = normed + b
        y = normed

    quant_enum = QuantMode.from_str(quant_type)
    if quant_enum == QuantMode.NONE:
        target_dtype = out_dtype or x.dtype
        return y.to(target_dtype).reshape(orig_shape), None, res_out
    elif quant_enum == QuantMode.FP8_PER_TOKEN:
        target_dtype = out_dtype or current_platform.fp8_dtype()
        finfo = torch.finfo(target_dtype)
        qmin, qmax = finfo.min, finfo.max
        min_scaling_factor = 1.0 / (qmax * 512.0)
        amax = torch.amax(torch.abs(y), dim=-1, keepdim=True)
        if scale_ub is not None:
            amax = torch.minimum(amax, scale_ub.float())
        s = torch.clamp(amax / qmax, min=min_scaling_factor)
        q = torch.clamp(y / s, min=qmin, max=qmax).to(target_dtype)
        scale_out = s.reshape(*orig_shape[:-1], 1)
        return q.reshape(orig_shape), scale_out, res_out
    elif quant_enum == QuantMode.FP8_PER_BLOCK:
        target_dtype = out_dtype or current_platform.fp8_dtype()
        finfo = torch.finfo(target_dtype)
        qmin, qmax = finfo.min, finfo.max
        min_scaling_factor = 1.0 / (qmax * 512.0)
        gs = group_size or 128
        num_groups = N // gs
        y_grouped = y.view(M, num_groups, gs)
        amax = torch.amax(torch.abs(y_grouped), dim=-1, keepdim=True)
        if scale_ub is not None:
            amax = torch.minimum(amax, scale_ub.float())
        s = torch.clamp(amax / qmax, min=min_scaling_factor)
        q_grouped = torch.clamp(y_grouped / s, min=qmin, max=qmax).to(target_dtype)
        q = q_grouped.view(M, N)
        scale_out = s.view(*orig_shape[:-1], num_groups)
        return q.reshape(orig_shape), scale_out, res_out
    elif quant_enum == QuantMode.FP8_STATIC:
        target_dtype = out_dtype or current_platform.fp8_dtype()
        finfo = torch.finfo(target_dtype)
        qmin, qmax = finfo.min, finfo.max
        assert scale is not None
        q = torch.clamp(y / scale.float(), min=qmin, max=qmax).to(target_dtype)
        return q.reshape(orig_shape), scale, res_out
    elif quant_enum == QuantMode.INT8_PER_TOKEN:
        target_dtype = out_dtype or torch.int8
        qmin, qmax = -128.0, 127.0
        min_scaling_factor = 1.0 / (127.0 * 512.0)
        amax = torch.amax(torch.abs(y), dim=-1, keepdim=True)
        if scale_ub is not None:
            amax = torch.minimum(amax, scale_ub.float())
        s = torch.clamp(amax / 127.0, min=min_scaling_factor)
        q = torch.clamp(torch.round(y / s), min=qmin, max=qmax).to(target_dtype)
        scale_out = s.reshape(*orig_shape[:-1], 1)
        return q.reshape(orig_shape), scale_out, res_out


def fused_gated_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    scale_ub: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = True,
    gate_activation: str = "silu",
    quant_type: str = "fp8_per_token",
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
    out_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Fused Gated RMSNorm and Activation Quantization."""
    quant_enum = QuantMode.from_str(quant_type)
    act_enum = GateActivation.from_str(gate_activation)

    is_cuda_or_xpu = x.is_cuda or (
        hasattr(torch, "xpu") and torch.xpu.is_available() and x.is_xpu
    )
    has_triton_kernel = HAS_TRITON and _fused_gated_rmsnorm_quant_kernel is not None
    if not (has_triton_kernel and is_cuda_or_xpu):
        return fused_gated_rmsnorm_quant_ref(
            x=x,
            weight=weight,
            z=z,
            bias=bias,
            residual=residual,
            scale=scale,
            scale_ub=scale_ub,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            gate_activation=gate_activation,
            quant_type=quant_type,
            out_dtype=out_dtype,
        )

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x.reshape(-1, hidden_size).contiguous()
    z_2d = z.reshape(-1, hidden_size).contiguous()
    w_1d = weight.contiguous()
    b_1d = bias.contiguous() if bias is not None else None
    r_2d = (
        residual.reshape(-1, hidden_size).contiguous() if residual is not None else None
    )

    M, N = x_2d.shape

    if quant_enum == QuantMode.NONE:
        target_out_dtype = out_dtype or x.dtype
        qmin, qmax = 0.0, 0.0
        min_scaling_factor = 0.0
    elif quant_enum in (
        QuantMode.FP8_PER_TOKEN,
        QuantMode.FP8_PER_BLOCK,
        QuantMode.FP8_STATIC,
    ):
        target_out_dtype = out_dtype or current_platform.fp8_dtype()
        qmin, qmax = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qmax * 512.0)
    elif quant_enum == QuantMode.INT8_PER_TOKEN:
        target_out_dtype = out_dtype or torch.int8
        qmin, qmax = -128.0, 127.0
        min_scaling_factor = 1.0 / (127.0 * 512.0)

    if out is None:
        out_2d = torch.empty((M, N), device=x.device, dtype=target_out_dtype)
    else:
        out_2d = out.reshape(-1, hidden_size)

    if quant_enum == QuantMode.NONE:
        scale_2d = None
    elif quant_enum == QuantMode.FP8_STATIC:
        scale_2d = scale
    elif quant_enum == QuantMode.FP8_PER_BLOCK:
        gs = group_size or 128
        num_groups = N // gs
        if out_scale is None:
            scale_2d = torch.empty(
                (M, num_groups), device=x.device, dtype=torch.float32
            )
        else:
            scale_2d = out_scale.reshape(-1, num_groups)
    else:
        if out_scale is None:
            scale_2d = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        else:
            scale_2d = out_scale.reshape(-1, 1)

    static_scale_val = (
        float(scale.item())
        if (quant_enum == QuantMode.FP8_STATIC and scale is not None)
        else 1.0
    )

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_N // 256, 1), 8)

    grid = (M,)

    with torch.accelerator.device_index(x.device.index):
        _fused_gated_rmsnorm_quant_kernel[grid](
            x_2d,
            z_2d,
            w_1d,
            b_1d,
            r_2d,
            out_2d,
            scale_2d,
            scale_ub,
            static_scale_val,
            x_2d.stride(0),
            z_2d.stride(0),
            r_2d.stride(0) if r_2d is not None else 0,
            out_2d.stride(0),
            scale_2d.stride(0) if scale_2d is not None else 0,
            M,
            N,
            eps,
            qmin,
            qmax,
            min_scaling_factor,
            BLOCK_N=BLOCK_N,
            GROUP_SIZE=group_size or 128,
            NORM_BEFORE_GATE=norm_before_gate,
            GATE_ACT=int(act_enum),
            QUANT_MODE=int(quant_enum),
            num_warps=num_warps,
        )

    final_out = out_2d.reshape(orig_shape)
    final_res = r_2d.reshape(orig_shape) if r_2d is not None else None

    if scale_2d is not None:
        if quant_enum == QuantMode.FP8_STATIC:
            final_scale = scale
        elif quant_enum == QuantMode.FP8_PER_BLOCK:
            final_scale = scale_2d.reshape(*orig_shape[:-1], N // (group_size or 128))
        else:
            final_scale = scale_2d.reshape(*orig_shape[:-1], 1)
    else:
        final_scale = None

    return final_out, final_scale, final_res


@CustomOp.register("fused_gated_rmsnorm_quant")
class RMSNormGatedQuant(CustomOp):
    """Gated RMS Normalization fused with Activation Quantization layer."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "silu",
        quant_type: str = "fp8_per_token",
        has_bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        self.quant_type = quant_type

        factory_kwargs = {
            "device": device,
            "dtype": dtype or torch.get_default_dtype(),
        }
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward_native(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        residual: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return fused_gated_rmsnorm_quant_ref(
            x=x,
            weight=self.weight,
            z=z,
            bias=self.bias,
            residual=residual,
            scale=scale,
            scale_ub=scale_ub,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            gate_activation=self.activation,
            quant_type=self.quant_type,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        residual: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return fused_gated_rmsnorm_quant(
            x=x,
            weight=self.weight,
            z=z,
            bias=self.bias,
            residual=residual,
            scale=scale,
            scale_ub=scale_ub,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            gate_activation=self.activation,
            quant_type=self.quant_type,
        )
