# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.transformers_utils.configs.axk2 import AXK2Config

_SLIDING_ATTENTION = "sliding_attention"


class AXK3Config(AXK2Config):
    r"""Configuration class for the AXK3 model family.

    Follows AXK2 and adds cross-layer sharing of the indexer top-k
    (``dsa_index_share``) and the gated residual stream that replaces the
    pre-norm residual (``gated_residual``, A.X-K3 Flash).
    """

    model_type = "axk3"

    def __init__(
        self,
        gated_residual: bool = False,
        gated_residual_num_streams: int = 4,
        gated_residual_rank: int = 64,
        gated_residual_final: str = "gated_read",
        dsa_index_share: bool = False,
        index_topk_pattern: str | None = None,
        layer_types: list[str] | None = None,
        swa_window_size: int | None = None,
        softmax_type: str = "vanilla",
        use_hysparse: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gated_residual = gated_residual
        self.gated_residual_num_streams = gated_residual_num_streams
        self.gated_residual_rank = gated_residual_rank
        self.gated_residual_final = gated_residual_final

        if softmax_type != "vanilla":
            raise ValueError(f"AXK3 softmax_type={softmax_type!r} is not supported.")
        if use_hysparse:
            raise ValueError("AXK3 use_hysparse is not supported.")
        self.dsa_index_share = dsa_index_share
        self.layer_types = layer_types
        self.swa_window_size = swa_window_size
        self.softmax_type = softmax_type
        self.use_hysparse = use_hysparse

        if layer_types is not None and _SLIDING_ATTENTION in layer_types:
            # Sliding layers are only supported as top-k consumers: they reuse
            # the preceding full layer's indexer selection over the whole
            # context. A real attention window is not implemented.
            if not dsa_index_share:
                raise ValueError(
                    "AXK3 sliding_attention layers require dsa_index_share=True."
                )
            if swa_window_size is not None and (
                self.max_position_embeddings is None
                or swa_window_size < self.max_position_embeddings
            ):
                raise ValueError(
                    f"AXK3 swa_window_size={swa_window_size} below "
                    f"max_position_embeddings={self.max_position_embeddings} is "
                    "not supported."
                )
            if index_topk_pattern is None:
                index_topk_pattern = "".join(
                    "S" if layer_type == _SLIDING_ATTENTION else "F"
                    for layer_type in layer_types
                )
        # Read by the DSA attention: "S" layers build no indexer.
        if index_topk_pattern is not None:
            self.index_topk_pattern = index_topk_pattern
