# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.eagle.config import wrap_eagle_hf_config
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DFlashSpeculatorConfig(DraftModelSpeculatorConfig):
    base_method = "dflash"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator.DFlashSpeculator"
    )
    uses_target_hidden_states = True
    parallel_drafting = True
    supports_watermarking = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        return "dflash" in spec.draft_model_config.model.lower()

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        assert cls.base_method is not None
        # DFlash / DSpark drafts name their auxiliary hidden state layers
        # `aux_hidden_state_layer_ids`; expose them under the EAGLE3 key that
        # the target-side readers use.
        hf = spec.draft_model_config.hf_config
        if getattr(hf, "eagle_aux_hidden_state_layer_ids", None) is None:
            aux_ids = getattr(hf, "aux_hidden_state_layer_ids", None)
            if aux_ids is not None:
                hf.eagle_aux_hidden_state_layer_ids = aux_ids
        wrap_eagle_hf_config(spec, cls.base_method)

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        # Fill-in drafting uses a bonus query plus one query per draft token.
        return self.num_speculative_tokens + 1

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int:
        # The draft pass has num_lookahead_tokens queries for every request.
        return max_num_batched_tokens // self.num_lookahead_tokens
