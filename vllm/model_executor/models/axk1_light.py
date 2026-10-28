# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K1-Light on top of the DeepSeek-V2/V3 implementation.

A.X-K1-Light is DeepSeek-V3 dense MLA + MoE with two differences:
- an RMSNorm (`post_mlp_layernorm`) on the output of every MoE layer;
- the latent is zero-padded from `kv_lora_rank_actual` to `kv_lora_rank`
  (kernel-supported size) in `kv_a_proj_with_mqa` and `kv_b_proj`, while
  `kv_a_layernorm` keeps the actual size and normalizes over it only.

Select it with `--hf-overrides '{"architectures": ["AXK1LightForCausalLM"]}'`.
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV2MoE,
)
from vllm.model_executor.utils import set_weight_attrs


def _load_padded_kv_a_layernorm(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    param.data.zero_()
    param.data[: loaded_weight.shape[0]].copy_(loaded_weight)


class AXK1LightMoE(DeepseekV2MoE):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        return self.post_mlp_layernorm(
            super().forward(hidden_states, already_sequence_parallel)
        )


class AXK1LightDecoderLayer(DeepseekV2DecoderLayer):
    moe_cls = AXK1LightMoE

    def __init__(self, vllm_config: VllmConfig, prefix: str, **kwargs) -> None:
        super().__init__(vllm_config, prefix, **kwargs)
        config = vllm_config.model_config.hf_config
        actual_rank = getattr(config, "kv_lora_rank_actual", None)
        norm = getattr(self.self_attn, "kv_a_layernorm", None)
        if norm is not None and actual_rank and actual_rank != norm.hidden_size:
            # In place: the MLA wrapper already holds a reference to this norm.
            norm.variance_size_override = actual_rank
            set_weight_attrs(
                norm.weight, {"weight_loader": _load_padded_kv_a_layernorm}
            )


class AXK1LightModel(DeepseekV2Model):
    decoder_layer_cls = AXK1LightDecoderLayer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())

        def remap(weights):
            for name, loaded_weight in weights:
                if ".post_mlp_layernorm." in name:
                    name = name.replace(
                        ".post_mlp_layernorm.", ".mlp.post_mlp_layernorm."
                    )
                    if name not in params:
                        # Dense layers ship this norm too but do not use it.
                        continue
                yield name, loaded_weight

        return super().load_weights(remap(weights))


class AXK1LightForCausalLM(DeepseekV2ForCausalLM):
    model_cls = AXK1LightModel
