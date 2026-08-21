# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for A.X K2 model architecture, configuration, and helpers."""

import torch
from torch import nn

import vllm.models.axk2.nvidia.model as axk2
from vllm.model_executor.models.axk2 import (
    PartialRMSNorm,
    _mla_pad_dims,
    _pad_fused_q_b_proj_native_to_eff,
    _pad_kv_b_proj_native_to_eff,
    _pad_q_b_proj_native_to_eff,
    _pick_mla_target,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.axk2 import (
    AXK2ForCausalLM,
    AXK2GatedRMSNorm,
)
from vllm.transformers_utils.configs.axk2 import AXK2Config
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


def test_axk2_config_init():
    config = AXK2Config(
        vocab_size=163840,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size=2048,
        num_hidden_layers=61,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_routed_experts=256,
        n_shared_experts=1,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attention_output_gate=True,
        attn_gate_fused=True,
        gated_norm=True,
        gated_norm_rank=16,
        index_topk=2048,
        index_n_heads=64,
        index_head_dim=128,
    )
    assert config.model_type == "axk2"
    assert config.hidden_size == 7168
    assert config.q_lora_rank == 1536
    assert config.kv_lora_rank == 512
    assert config.attention_output_gate is True
    assert config.attn_gate_fused is True
    assert config.gated_norm is True
    assert config.gated_norm_rank == 16
    assert hasattr(config, "index_topk")


def test_axk2_registry_and_convertor():
    assert "AXK2ForCausalLM" in ModelRegistry.get_supported_archs()
    model_cls = ModelRegistry._try_load_model_cls("AXK2ForCausalLM")
    assert model_cls is AXK2ForCausalLM

    config = AXK2Config(
        model_type="axk2",
        kv_lora_rank=512,
    )
    convertor = ModelArchConfigConvertorBase(config, config)
    assert convertor.is_deepseek_mla()


def test_axk2_pick_mla_target():
    dsa_config = AXK2Config(index_topk=2048)
    target_head_size, is_dsa = _pick_mla_target(dsa_config)
    assert target_head_size == 576
    assert is_dsa is True

    non_dsa_config = AXK2Config()
    target_head_size, is_dsa = _pick_mla_target(non_dsa_config)
    assert target_head_size == 576
    assert is_dsa is False


def test_axk2_mla_pad_dims():
    eff_nope, eff_rope, eff_v, eff_kv = _mla_pad_dims(
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kv_lora_rank=128,
        target_head_size=576,
        is_dsa=True,
    )
    assert eff_nope == 128
    assert eff_rope == 64
    assert eff_v == 128
    assert eff_kv == 512


def test_partial_rmsnorm(default_vllm_config):
    actual_size = 128
    full_size = 512
    norm = PartialRMSNorm(actual_size=actual_size, full_size=full_size, eps=1e-6)

    x = torch.randn(2, 4, full_size, dtype=torch.bfloat16)
    out = norm(x)
    assert out.shape == (2, 4, full_size)
    assert out.dtype == torch.bfloat16
    # Check that tail is unmodified
    assert torch.equal(out[..., actual_size:], x[..., actual_size:])

    # Residual path
    residual = torch.randn(2, 4, full_size, dtype=torch.bfloat16)
    out_res, new_res = norm(x, residual)
    assert out_res.shape == (2, 4, full_size)
    assert new_res.shape == (2, 4, full_size)
    assert torch.equal(
        new_res[..., actual_size:],
        x[..., actual_size:] + residual[..., actual_size:],
    )


def test_axk2_gated_rmsnorm(default_vllm_config, monkeypatch):
    class TestLinear(nn.Module):
        def __init__(self, input_size, output_size, **kwargs):
            super().__init__()
            self.linear = nn.Linear(
                input_size, output_size, bias=False, dtype=torch.bfloat16
            )

        def forward(self, x):
            return self.linear(x), None

    monkeypatch.setattr(axk2, "ReplicatedLinear", TestLinear)
    hidden_size = 64
    rank = 16
    gated_norm = AXK2GatedRMSNorm(
        hidden_size=hidden_size, eps=1e-6, rank=rank, prefix="test"
    )

    x = torch.randn(2, 8, hidden_size, dtype=torch.bfloat16)
    out = gated_norm(x)
    assert out.shape == (2, 8, hidden_size)
    assert out.dtype == torch.bfloat16

    residual = torch.randn(2, 8, hidden_size, dtype=torch.bfloat16)
    out_res, new_res = gated_norm(x, residual)
    assert out_res.shape == (2, 8, hidden_size)
    assert new_res.shape == (2, 8, hidden_size)


def test_pad_q_b_proj():
    num_heads = 4
    native_nope = 64
    native_rope = 32
    eff_nope = 128
    eff_rope = 64
    q_lora = 128

    native_head = native_nope + native_rope
    eff_head = eff_nope + eff_rope
    weight = torch.arange(num_heads * native_head * q_lora).reshape(
        num_heads * native_head, q_lora
    )

    padded = _pad_q_b_proj_native_to_eff(
        weight, num_heads, native_nope, native_rope, eff_nope, eff_rope
    )
    assert padded.shape == (num_heads * eff_head, q_lora)
    native = weight.view(num_heads, native_head, q_lora)
    padded = padded.view(num_heads, eff_head, q_lora)
    assert torch.equal(padded[:, :native_nope], native[:, :native_nope])
    assert torch.count_nonzero(padded[:, native_nope:eff_nope]) == 0
    assert torch.equal(
        padded[:, eff_nope : eff_nope + native_rope], native[:, native_nope:]
    )
    assert torch.count_nonzero(padded[:, eff_nope + native_rope :]) == 0


def test_pad_fused_q_b_proj():
    num_heads = 4
    native_nope = 64
    native_rope = 32
    eff_nope = 128
    eff_rope = 64
    v_head = 64
    q_lora = 128

    native_ph = native_nope + native_rope + v_head
    eff_ph = eff_nope + eff_rope + v_head
    weight = torch.arange(num_heads * native_ph * 2 * q_lora).reshape(
        num_heads * native_ph, 2 * q_lora
    )

    padded = _pad_fused_q_b_proj_native_to_eff(
        weight, num_heads, native_nope, native_rope, eff_nope, eff_rope, v_head
    )
    assert padded.shape == (num_heads * eff_ph, 2 * q_lora)
    native = weight.view(num_heads, native_ph, 2 * q_lora)
    padded = padded.view(num_heads, eff_ph, 2 * q_lora)
    native_q = native_nope + native_rope
    eff_q = eff_nope + eff_rope
    assert torch.equal(padded[:, :native_nope], native[:, :native_nope])
    assert torch.count_nonzero(padded[:, native_nope:eff_nope]) == 0
    assert torch.equal(
        padded[:, eff_nope : eff_nope + native_rope],
        native[:, native_nope:native_q],
    )
    assert torch.count_nonzero(padded[:, eff_nope + native_rope : eff_q]) == 0
    assert torch.equal(padded[:, eff_q:], native[:, native_q:])


def test_pad_kv_b_proj():
    num_heads = 4
    native_nope = 64
    native_v = 64
    native_kv_lora = 128
    eff_nope = 128
    eff_v = 128
    eff_kv_lora = 512

    native_out = native_nope + native_v
    eff_out = eff_nope + eff_v
    weight = torch.arange(num_heads * native_out * native_kv_lora).reshape(
        num_heads * native_out, native_kv_lora
    )

    padded = _pad_kv_b_proj_native_to_eff(
        weight,
        num_heads,
        native_nope,
        native_v,
        native_kv_lora,
        eff_nope,
        eff_v,
        eff_kv_lora,
    )
    assert padded.shape == (num_heads * eff_out, eff_kv_lora)
    native = weight.view(num_heads, native_out, native_kv_lora)
    padded = padded.view(num_heads, eff_out, eff_kv_lora)
    assert torch.equal(
        padded[:, :native_nope, :native_kv_lora], native[:, :native_nope]
    )
    assert torch.equal(
        padded[:, eff_nope : eff_nope + native_v, :native_kv_lora],
        native[:, native_nope:],
    )
    assert torch.count_nonzero(padded[:, native_nope:eff_nope]) == 0
    assert torch.count_nonzero(padded[:, eff_nope + native_v :]) == 0
    assert torch.count_nonzero(padded[..., native_kv_lora:]) == 0


def test_axk2_supports_eagle_and_dspark():
    from vllm.model_executor.models.interfaces import supports_eagle, supports_eagle3

    assert supports_eagle(AXK2ForCausalLM)
    assert supports_eagle3(AXK2ForCausalLM)
