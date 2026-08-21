# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K2 Attention module extending DeepseekV32Attention."""

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.axk2.common.fused_gated_rmsnorm_quant import fused_gate_act_quant
from vllm.models.deepseek_v32.attention import DeepseekV32Attention
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q
from vllm.transformers_utils.configs.axk2 import AXK2Config


class AXK2Attention(DeepseekV32Attention):
    """A.X-K2 Attention extending DeepseekV32Attention.

    Key extensions:
    1. Fused q_b_proj + Output Gate GEMM (Qwen3-Next / A.X-K2 style):
       q_b_proj consumes [q_c_normed, q_c_prenorm] and produces per-head
       interleaved [q | gate] in a single GEMM.
    2. Sigmoid gate modulation on attention output before o_proj.
    3. Reuses the entire DeepSeek Sparse Attention (DSA) Indexer backend and
       FlashMLA / FlashInfer sparse execution paths from DeepSeek V3.2.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: AXK2Config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        attn_backend: type | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=attn_backend,
        )
        self.use_output_gate = getattr(config, "attention_output_gate", False)
        self.attn_gate_fused = self.use_output_gate and getattr(
            config, "attn_gate_fused", True
        )

        if self.attn_gate_fused:
            assert self.q_lora_rank is not None
            # Widened q_b_proj:
            # input:  2 * q_lora_rank -> [q_lora post-norm | q_lora pre-norm]
            # output: num_heads * (qk_head_dim + v_head_dim), per-head interleaved
            self.q_b_proj = ColumnParallelLinear(
                2 * self.q_lora_rank,
                config.num_attention_heads * (self.qk_head_dim + self.v_head_dim),
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        self._decode_q_b_buf: torch.Tensor | None = None
        self._decode_attn_out_buf: torch.Tensor | None = None

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.indexer is not None and not self.skip_topk:
            kw = self.indexer.wk_weights_proj(hidden_states)[0]
            index_k = kw[:, : self.indexer.head_dim]
            index_weights = kw[:, self.indexer.head_dim :]
        else:
            index_k = None
            index_weights = None

        num_tokens = hidden_states.shape[0]
        if num_tokens == 1:
            if self._decode_attn_out_buf is None:
                self._decode_attn_out_buf = torch.empty(
                    (1, self.num_local_heads * self.v_head_dim),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            output = self._decode_attn_out_buf
        else:
            output = torch.empty(
                (num_tokens, self.num_local_heads * self.v_head_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.layer_name)
        elif isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0].get(self.layer_name)
        else:
            attn_metadata = attn_metadata_raw

        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        mla_slot = slot_mapping.get(self.layer_name)

        if self.indexer is not None and not self.skip_topk:
            has_indexer = True
            indexer_k_norm_w = self.indexer.k_norm.weight
            indexer_k_norm_bias = self.indexer.k_norm.bias
            indexer_k_norm_eps = self.indexer.k_norm.eps
            indexer_k_rope_cos_sin_cache = self.indexer_rope_emb.cos_sin_cache
            indexer_k_cache = None if self.use_pcp else self.indexer.k_cache.kv_cache
            index_k_out = torch.empty_like(index_k) if self.use_pcp else None
            indexer_softmax_scale = self.indexer.softmax_scale
            indexer_n_head_scale = self.indexer.n_head**-0.5
        else:
            has_indexer = False
            indexer_k_norm_w = None
            indexer_k_norm_bias = None
            indexer_k_norm_eps = 1e-6
            indexer_k_rope_cos_sin_cache = None
            indexer_k_cache = None
            index_k_out = None
            indexer_softmax_scale = 0.0
            indexer_n_head_scale = 0.0

        if attn_metadata is None or self.use_pcp:
            mla_kv_cache = None
            mla_k_scale = None
            indexer_k_cache = None
            mla_slot = None
        else:
            mla_kv_cache = self.kv_cache
            mla_k_scale = self._k_scale

        kv_c_out = torch.empty_like(kv_c)
        k_pe_out = torch.empty_like(k_pe)
        q_c_prenorm = q_c
        q_c = fused_norm_rope(
            positions,
            q_c,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            k_pe,
            self.rotary_emb.cos_sin_cache,
            index_k,
            indexer_k_norm_w,
            indexer_k_norm_bias,
            indexer_k_norm_eps,
            indexer_k_rope_cos_sin_cache,
            self.topk_indices_buffer,
            slot_mapping=mla_slot,
            indexer_k_cache=indexer_k_cache,
            mla_kv_cache=mla_kv_cache,
            mla_kv_cache_dtype=self.kv_cache_dtype,
            mla_k_scale=mla_k_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
            kv_c_out=kv_c_out,
            k_pe_out=k_pe_out,
            index_k_out=index_k_out,
        )

        if self.attn_gate_fused:
            assert self.q_lora_rank is not None
            # Fused Q + Gate GEMM
            if num_tokens == 1:
                if self._decode_q_b_buf is None:
                    self._decode_q_b_buf = torch.empty(
                        (1, 2 * self.q_lora_rank),
                        dtype=q_c.dtype,
                        device=q_c.device,
                    )
                self._decode_q_b_buf[0, : self.q_lora_rank].copy_(q_c[0])
                self._decode_q_b_buf[0, self.q_lora_rank :].copy_(q_c_prenorm[0])
                q_b_input = self._decode_q_b_buf
            else:
                q_b_input = torch.cat([q_c, q_c_prenorm], dim=-1)
            q_b_out = self.q_b_proj(q_b_input)[0]
            per_head = self.qk_head_dim + self.v_head_dim
            q_gate = q_b_out.view(-1, self.num_local_heads, per_head)
            q = q_gate[:, :, : self.qk_head_dim].contiguous()
            gate = q_gate[:, :, self.qk_head_dim :].reshape(
                -1, self.num_local_heads * self.v_head_dim
            )
        else:
            q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            gate = None

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        if q_nope.shape[0] == 1:
            ql_nope = torch.matmul(q_nope.unsqueeze(-2), self.W_UK_T).squeeze(-2)
        else:
            ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)

        if self.indexer is not None and not self.skip_topk:
            index_q = self.indexer.wq_b(q_c)[0]
            index_q = index_q.view(-1, self.indexer.n_head, self.indexer.head_dim)
        else:
            index_q = None

        index_q_fp8, index_weights_out, mqa_q = fused_q(
            positions,
            q_pe,
            self.rotary_emb.cos_sin_cache,
            index_q,
            self.indexer_rope_emb.cos_sin_cache if has_indexer else None,
            ql_nope,
            self._q_scale,
            index_weights,
            indexer_softmax_scale,
            indexer_n_head_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
            quantize_mqa=self._fp8_query,
        )

        self._sparse_indexer_and_attn(
            positions,
            q_c,
            q_nope,
            q_pe,
            index_q_fp8,
            index_k_out,
            index_weights_out,
            kv_c_out,
            k_pe_out,
            ql_nope,
            mqa_q,
            output,
        )

        if gate is not None:
            output, _ = fused_gate_act_quant(
                y=output,
                gate=gate,
                activation="sigmoid",
                quant_type="none",
            )

        return self.o_proj(output)[0]
