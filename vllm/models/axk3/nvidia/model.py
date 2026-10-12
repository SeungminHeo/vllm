# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K3 family on top of the A.X-K2 backbone.

A.X-K3 is A.X-K2 with the indexer top-k shared across layers, which the DSA
attention handles from the config alone. A.X-K3 Flash additionally runs on a
gated residual stream instead of the pre-norm residual.
"""

from itertools import islice

import torch

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.models.axk2.nvidia.model import (
    AXK2DecoderLayer,
    AXK2ForCausalLM,
    AXK2Model,
)
from vllm.models.axk3.gated_residual import AXK3GatedResidual
from vllm.sequence import IntermediateTensors


def _make_gated_residual(config, has_write: bool = True) -> AXK3GatedResidual:
    return AXK3GatedResidual(
        hidden_size=config.hidden_size,
        num_streams=config.gated_residual_num_streams,
        rank=config.gated_residual_rank,
        eps=config.rms_norm_eps,
        has_write=has_write,
    )


class AXK3GatedResidualDecoderLayer(AXK2DecoderLayer):
    """Decoder layer whose pre-norms and residual adds are gated residuals."""

    def _init_norms(self, config, prefix: str) -> None:
        if self.use_sequence_parallel:
            raise NotImplementedError(
                "A.X-K3 gated residual does not support sequence-parallel MoE."
            )
        self.attn_gated_residual = _make_gated_residual(config)
        self.mlp_gated_residual = _make_gated_residual(config)

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
    ) -> torch.Tensor:
        # self_attn and mlp leave their output un-reduced across TP ranks.
        hidden_states, write_gate = self.attn_gated_residual.read(stream)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        stream, hidden_states, write_gate = self.attn_gated_residual.write_read(
            stream, hidden_states, write_gate, self.mlp_gated_residual
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        return self.mlp_gated_residual.write(stream, hidden_states, write_gate)


class AXK3GatedResidualModel(AXK2Model):
    """A.X-K2 backbone running on a gated residual stream.

    The embedding is repeated into ``gated_residual_num_streams`` branches and
    a final write-less read contracts them before the LM head, so there is no
    final norm.
    """

    layer_cls = AXK3GatedResidualDecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "A.X-K3 gated residual does not support pipeline parallelism."
            )
        if config.gated_residual_final != "gated_read":
            raise NotImplementedError(
                "A.X-K3 gated_residual_final="
                f"{config.gated_residual_final!r} is not supported."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.num_streams = config.gated_residual_num_streams
        self.norm = None  # type: ignore[assignment]
        self.final_gated_residual = _make_gated_residual(config, has_write=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if self.aux_hidden_state_layers:
            raise NotImplementedError(
                "A.X-K3 gated residual does not expose auxiliary hidden states."
            )
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            assert input_ids is not None
            hidden_states = self.embed_input_ids(input_ids)

        stream = hidden_states.repeat(1, self.num_streams)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            stream = layer(positions, stream)
        hidden_states, _ = self.final_gated_residual.read(stream)
        return hidden_states


def _backbone_cls(config) -> type[AXK2Model]:
    """A.X-K3 keeps the A.X-K2 backbone; A.X-K3 Flash needs the gated residual."""
    return AXK3GatedResidualModel if config.gated_residual else AXK2Model


class AXK3ForCausalLM(AXK2ForCausalLM):
    """A.X-K3 family causal LM (A.X-K3, A.X-K3 Flash)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self.model_cls = _backbone_cls(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
