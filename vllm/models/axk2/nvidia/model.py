# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K2 Model implementation on top of DeepSeek V3.2 DSA platform."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    DeepseekV2MoE,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.models.axk2.attention import AXK2Attention
from vllm.models.axk2.gated_rmsnorm import AXK2GatedRMSNorm
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
)
from vllm.models.deepseek_v32.nvidia.model import (
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM,
    DeepseekV32Model,
)


def _apply_layernorm(
    norm_module: nn.Module,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    use_sequence_parallel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(norm_module, AXK2GatedRMSNorm):
        if residual is None:
            residual = hidden_states
            out = norm_module(hidden_states)
            return out, residual
        if not use_sequence_parallel and get_tensor_model_parallel_world_size() > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        out, residual = norm_module(hidden_states, residual)
        return out, residual
    elif residual is None:
        residual = hidden_states
        hidden_states = norm_module(hidden_states)
    elif use_sequence_parallel:
        hidden_states, residual = norm_module(hidden_states, residual)
    else:
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states, residual, norm_module
        )
    return hidden_states, residual


class AXK2DecoderLayer(DeepseekV32DecoderLayer):
    """A.X-K2 Decoder Layer extending DeepseekV32DecoderLayer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)

        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.use_mha = False
        self.use_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )

        self.self_attn = AXK2Attention(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.mlp",
                apply_routed_scale_to_output=False,
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                reduce_results=False,
                is_sequence_parallel=self.use_sequence_parallel,
            )

        use_gated_norm = getattr(config, "gated_norm", False)
        gated_rank = getattr(config, "gated_norm_rank", 16)
        if use_gated_norm:
            self.input_layernorm = AXK2GatedRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                rank=gated_rank,
                prefix=f"{prefix}.input_layernorm",
            )
            if self._is_layer_sparse(config):
                self.post_attention_layernorm = AXK2GatedRMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                    rank=gated_rank,
                    prefix=f"{prefix}.post_attention_layernorm",
                )
            else:
                self.post_attention_layernorm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def _is_layer_sparse(self, config) -> bool:
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        return (
            getattr(config, "n_routed_experts", None) is not None
            and self.layer_idx >= first_k_dense_replace
            and self.layer_idx % moe_layer_freq == 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attn_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_num_tokens = positions.shape[0]

        hidden_states, residual = _apply_layernorm(
            self.input_layernorm,
            hidden_states,
            residual,
            self.use_sequence_parallel,
        )
        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        if self.use_sequence_parallel:
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states, residual = _apply_layernorm(
            self.post_attention_layernorm,
            hidden_states,
            residual,
            self.use_sequence_parallel,
        )

        if self.use_sequence_parallel and isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states, already_sequence_parallel=True)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class AXK2Model(DeepseekV32Model, EagleModelMixin):
    """A.X-K2 Model extending DeepseekV32Model with Eagle/DSpark support."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(DeepseekV32Model, self).__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        from vllm.platforms import current_platform

        self.device = current_platform.device_type
        self.vocab_size = config.vocab_size
        parallel_config = vllm_config.parallel_config
        self.use_sequence_parallel = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
        )
        self.is_v32 = True
        index_topk = getattr(config, "index_topk", None)
        if index_topk is not None:
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        # The gated input_layernorm cannot use DeepseekV32Model's fused
        # embed+norm gather, so keep the vocab-parallel embedding path.
        self.replicated_embed = False

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AXK2DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.aux_hidden_state_layers = tuple[int, ...]()
        self.num_redundant_experts = getattr(
            parallel_config.eplb_config, "num_redundant_experts", 0
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())

        def _remap_weights(w_iter):
            for name, loaded_weight in w_iter:
                if name not in params_dict:
                    for norm_name in ("input_layernorm", "post_attention_layernorm"):
                        old_suffix = f".{norm_name}.weight"
                        new_suffix = f".{norm_name}.norm.weight"
                        if old_suffix in name:
                            candidate = name.replace(old_suffix, new_suffix)
                            if candidate in params_dict:
                                name = candidate
                                break
                yield name, loaded_weight

        return super().load_weights(_remap_weights(weights))


class AXK2ForCausalLM(
    DeepseekV32ForCausalLM,
    SupportsPP,
    SupportsLoRA,
    SupportsEagle,
    SupportsEagle3,
):
    """A.X-K2 Causal Language Model inheriting DeepSeek V3.2 architecture."""

    model_cls = AXK2Model
