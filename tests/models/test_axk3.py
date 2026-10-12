# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the A.X-K3 family additions on top of A.X-K2."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.axk3.nvidia.model as axk3
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.axk2 import AXK2ForCausalLM, AXK2Model
from vllm.models.axk3 import AXK3ForCausalLM, AXK3GatedResidualDecoderLayer
from vllm.models.axk3.gated_residual import AXK3GatedResidual
from vllm.transformers_utils.configs.axk2 import AXK2Config
from vllm.transformers_utils.configs.axk3 import AXK3Config
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

_INDEX_SHARE = dict(
    max_position_embeddings=1024,
    index_topk=2048,
    dsa_index_share=True,
    layer_types=["full_attention", "sliding_attention", "sliding_attention"] * 2,
    swa_window_size=1024,
)


def test_axk3_registry_and_convertor():
    assert ModelRegistry._try_load_model_cls("AXK3ForCausalLM") is AXK3ForCausalLM

    config = AXK3Config(kv_lora_rank=512)
    assert config.model_type == "axk3"
    assert ModelArchConfigConvertorBase(config, config).is_deepseek_mla()


def test_axk3_backbone_selection():
    """A.X-K3 runs on the A.X-K2 backbone; only A.X-K3 Flash swaps it."""
    k3 = AXK3Config(**_INDEX_SHARE, gated_norm=True)
    k3_flash = AXK3Config(**_INDEX_SHARE, gated_residual=True)
    assert axk3._backbone_cls(k3) is AXK2Model
    assert axk3._backbone_cls(k3_flash) is axk3.AXK3GatedResidualModel


def test_axk3_config_index_share_pattern():
    """Sliding layers of an index-share checkpoint reuse the full layer's top-k."""
    assert AXK3Config(**_INDEX_SHARE).index_topk_pattern == "FSSFSS"
    assert not hasattr(AXK3Config(index_topk=2048), "index_topk_pattern")


@pytest.mark.parametrize(
    "overrides",
    [
        {"dsa_index_share": False},
        {"swa_window_size": 512},
        {"softmax_type": "learnable"},
        {"use_hysparse": True},
    ],
)
def test_axk3_config_rejects_unsupported_attention(overrides):
    """A real window, a softmax sink and hysparse must not load silently."""
    with pytest.raises(ValueError):
        AXK3Config(**(_INDEX_SHARE | overrides))


@pytest.mark.parametrize("key", ["gated_residual", "dsa_index_share"])
def test_axk2_entry_rejects_axk3_checkpoint(key):
    """An A.X-K3 checkpoint labelled axk2 must point the user at axk3."""
    model_config = SimpleNamespace(hf_config=AXK2Config(**{key: True}))
    with pytest.raises(ValueError, match="axk3"):
        AXK2ForCausalLM(vllm_config=SimpleNamespace(model_config=model_config))


def _reference_gated_residual(module, stream, block):
    """Read -> block -> write, written out from the checkpoint's formulas."""
    n, h = module.num_streams, module.hidden_size
    branches = stream.float().view(-1, n, h)
    normed = branches * torch.rsqrt(branches.pow(2).mean(-1, keepdim=True) + 1e-6)
    normed = (normed * module.gamma.float().view(n, h)).to(stream.dtype).view(-1, n * h)
    low_rank = nn.functional.silu(normed @ module.read_down.weight.T / n)
    gate = torch.sigmoid((low_rank @ module.read_up.weight.T).float())
    block_input = (gate * normed.float()).view(-1, n, h).mean(1).to(stream.dtype)
    write_gate = 2.0 * torch.sigmoid(
        (normed @ module.write_proj.weight.T / n).float() + module.write_bias
    )
    block_output = block(block_input)
    written = stream.view(-1, n, h) + (
        write_gate[:, :, None] * block_output.float()[:, None, :]
    ).to(stream.dtype)
    return block_input, written.view(-1, n * h)


def _random_gated_residual(hidden_size, num_streams, has_write=True):
    module = AXK3GatedResidual(
        hidden_size, num_streams, rank=4, eps=1e-6, has_write=has_write
    )
    module = module.to(torch.bfloat16)
    with torch.no_grad():
        module.gamma.uniform_(0.3, 4.0)
        if has_write:
            module.write_bias.data = torch.randn(num_streams)
    return module


def test_axk3_gated_residual_matches_reference():
    """read/write reproduce the checkpoint's gated residual formulas."""
    torch.manual_seed(0)
    hidden_size, num_streams = 16, 4
    module = _random_gated_residual(hidden_size, num_streams)
    stream = torch.randn(5, num_streams * hidden_size, dtype=torch.bfloat16)

    def block(x):
        return x * 0.5 + 1.0

    block_input, write_gate = module.read(stream)
    written = module.write(stream, block(block_input), write_gate)

    expected_input, expected_written = _reference_gated_residual(module, stream, block)
    assert write_gate.dtype == torch.float32
    assert torch.equal(block_input, expected_input)
    assert torch.equal(written, expected_written)

    read_only = _random_gated_residual(hidden_size, num_streams, has_write=False)
    assert read_only.read(stream)[1] is None
    assert not any("write" in name for name, _ in read_only.named_parameters())


def test_axk3_gated_residual_layer_order(monkeypatch):
    """The layer reads before each block and writes its output back."""
    torch.manual_seed(0)
    hidden_size, num_streams = 16, 4
    monkeypatch.setattr(axk3, "tensor_model_parallel_all_reduce", lambda x: x)

    layer = AXK3GatedResidualDecoderLayer.__new__(AXK3GatedResidualDecoderLayer)
    nn.Module.__init__(layer)
    layer.attn_gated_residual = _random_gated_residual(hidden_size, num_streams)
    layer.mlp_gated_residual = _random_gated_residual(hidden_size, num_streams)
    layer.self_attn = lambda positions, hidden_states: hidden_states * 2.0
    layer.mlp = lambda hidden_states: hidden_states - 1.0

    stream = torch.randn(3, num_streams * hidden_size, dtype=torch.bfloat16)
    out = layer(torch.arange(3), stream)

    _, expected = _reference_gated_residual(
        layer.attn_gated_residual, stream, lambda x: x * 2.0
    )
    _, expected = _reference_gated_residual(
        layer.mlp_gated_residual, expected, lambda x: x - 1.0
    )
    assert torch.equal(out, expected)
