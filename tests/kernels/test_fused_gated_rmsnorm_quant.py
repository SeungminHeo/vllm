# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.axk2.common.fused_gated_rmsnorm_quant import (
    RMSNormGatedQuant,
    fused_gated_rmsnorm_quant,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
QUANT_TYPES = [
    "fp8_per_token",
    "fp8_per_block",
    "fp8_static",
    "int8_per_token",
    "none",
]
GATE_ACTIVATIONS = ["silu", "sigmoid", "identity", "gelu"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_type", QUANT_TYPES)
@pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
def test_fused_gated_rmsnorm_quant_basic(
    dtype: torch.dtype, quant_type: str, gate_activation: str
):
    set_random_seed(42)
    batch_size = 8
    hidden_size = 256
    eps = 1e-5

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    z = torch.randn(batch_size, hidden_size, dtype=dtype)
    weight = torch.ones(hidden_size, dtype=dtype)
    scale = (
        torch.tensor([1.2], dtype=torch.float32) if quant_type == "fp8_static" else None
    )

    out, scale_out, res_out = fused_gated_rmsnorm_quant(
        x=x,
        weight=weight,
        z=z,
        scale=scale,
        eps=eps,
        group_size=64,
        gate_activation=gate_activation,
        quant_type=quant_type,
    )

    assert out.shape == (batch_size, hidden_size)
    assert res_out is None

    if quant_type == "none":
        assert out.dtype == dtype
        assert scale_out is None
    elif quant_type in ("fp8_per_token", "fp8_static"):
        assert out.dtype == current_platform.fp8_dtype()
        if quant_type == "fp8_per_token":
            assert scale_out.shape == (batch_size, 1)
            assert scale_out.dtype == torch.float32
    elif quant_type == "fp8_per_block":
        assert out.dtype == current_platform.fp8_dtype()
        assert scale_out.shape == (batch_size, hidden_size // 64)
        assert scale_out.dtype == torch.float32
    elif quant_type == "int8_per_token":
        assert out.dtype == torch.int8
        assert scale_out.shape == (batch_size, 1)
        assert scale_out.dtype == torch.float32


@pytest.mark.parametrize("gate_activation", GATE_ACTIVATIONS)
@pytest.mark.parametrize("norm_before_gate", [True, False])
def test_fused_gated_rmsnorm_quant_modes(gate_activation: str, norm_before_gate: bool):
    set_random_seed(123)
    batch_size = 4
    hidden_size = 128

    x = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    z = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    bias = torch.randn(hidden_size, dtype=torch.float32)
    residual = torch.randn(batch_size, hidden_size, dtype=torch.float32)

    out_unquant, _, res_out = fused_gated_rmsnorm_quant(
        x=x,
        weight=weight,
        z=z,
        bias=bias,
        residual=residual,
        norm_before_gate=norm_before_gate,
        gate_activation=gate_activation,
        quant_type="none",
    )

    assert out_unquant.shape == (batch_size, hidden_size)
    assert res_out is not None
    assert res_out.shape == (batch_size, hidden_size)
    # Check that residual is x + original residual
    torch.testing.assert_close(res_out, x + residual, rtol=1e-5, atol=1e-5)


def test_fused_gated_rmsnorm_quant_scale_ub():
    set_random_seed(999)
    batch_size = 16
    hidden_size = 128

    x = torch.randn(batch_size, hidden_size, dtype=torch.float32) * 10.0
    z = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    weight = torch.ones(hidden_size, dtype=torch.float32)

    scale_ub = torch.tensor(2.0, dtype=torch.float32)

    out, scale_out, _ = fused_gated_rmsnorm_quant(
        x=x,
        weight=weight,
        z=z,
        scale_ub=scale_ub,
        quant_type="fp8_per_token",
    )

    finfo = torch.finfo(current_platform.fp8_dtype())
    expected_max_scale = 2.0 / finfo.max
    # Verify scale is clamped below or equal to scale_ub / fp8_max + tolerance
    assert (scale_out <= expected_max_scale + 1e-6).all()


@pytest.mark.parametrize("hidden_size", [64, 128, 512, 1024, 2048])
@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_fused_gated_rmsnorm_quant_shapes(hidden_size: int, batch_size: int):
    set_random_seed(42)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    z = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    out, scale, _ = fused_gated_rmsnorm_quant(
        x=x,
        weight=weight,
        z=z,
        quant_type="fp8_per_token",
    )

    assert out.shape == (batch_size, hidden_size)
    assert scale.shape == (batch_size, 1)
    assert out.dtype == current_platform.fp8_dtype()


def test_rmsnorm_gated_quant_module(default_vllm_config):
    hidden_size = 256
    layer = RMSNormGatedQuant(
        hidden_size=hidden_size,
        eps=1e-5,
        norm_before_gate=True,
        activation="silu",
        quant_type="fp8_per_token",
        has_bias=True,
    )

    x = torch.randn(4, hidden_size)
    z = torch.randn(4, hidden_size)

    out, scale, res = layer(x, z)
    assert out.shape == (4, hidden_size)
    assert scale.shape == (4, 1)
    assert res is None
    assert out.dtype == current_platform.fp8_dtype()


def test_fused_gated_rmsnorm_numerical_precision():
    set_random_seed(100)
    batch_size = 8
    hidden_size = 512

    x = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    z = torch.randn(batch_size, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)
    residual = torch.randn(batch_size, hidden_size, dtype=torch.float32)

    # Step-by-step un-fused baseline
    x_acc = x + residual
    rstd = torch.rsqrt(torch.mean(x_acc**2, dim=-1, keepdim=True) + 1e-6)
    normed = x_acc * rstd * weight
    act_z = z * torch.sigmoid(z)
    y_ref = normed * act_z

    amax = torch.amax(torch.abs(y_ref), dim=-1, keepdim=True)
    finfo = torch.finfo(current_platform.fp8_dtype())
    qmax = finfo.max
    scale_ref = torch.clamp(amax / qmax, min=1.0 / (qmax * 512.0))
    q_ref = torch.clamp(y_ref / scale_ref, finfo.min, finfo.max).to(
        current_platform.fp8_dtype()
    )

    # Fused execution
    q_fused, scale_fused, res_fused = fused_gated_rmsnorm_quant(
        x=x,
        weight=weight,
        z=z,
        residual=residual.clone(),
        eps=1e-6,
        norm_before_gate=True,
        gate_activation="silu",
        quant_type="fp8_per_token",
    )

    torch.testing.assert_close(scale_fused, scale_ref, rtol=1e-5, atol=1e-5)
    assert torch.equal(q_fused, q_ref)
    torch.testing.assert_close(res_fused, x_acc, rtol=1e-5, atol=1e-5)
