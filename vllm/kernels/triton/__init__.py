# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel implementations."""

from vllm.kernels.triton.axk2_native_mla import (
    AXK2_KV_LORA_RANK,
    AXK2_NATIVE_HEAD_SIZE,
    AXK2_QK_NOPE_HEAD_DIM,
    AXK2_QK_ROPE_HEAD_DIM,
    AXK2_V_HEAD_DIM,
    FLASHMLA_PADDED_HEAD_SIZE,
    AXK2MLADims,
    axk2_native_mla_decode_reference,
    axk2_padded_flashmla_decode_reference,
    gather_paged_kv_tokens,
    matrix_absorption,
    pad_native_kv_to_flashmla,
    pad_native_query_to_flashmla,
    unpad_flashmla_kv_to_native,
    verify_axk2_native_mla_vs_flashmla,
)
from vllm.kernels.triton.fused_gated_rmsnorm_quant import (
    GateActivation,
    QuantMode,
    RMSNormGatedQuant,
    fused_gate_act_quant,
    fused_gated_rmsnorm_quant,
    fused_gated_rmsnorm_quant_ref,
)
from vllm.kernels.triton.qkv_padded_fp8_quant import (
    quantize_fp8_maybe_pad_head_dim,
    quantize_fp8_pad_head_dim_triton,
)

__all__ = [
    "quantize_fp8_pad_head_dim_triton",
    "quantize_fp8_maybe_pad_head_dim",
    "fused_gate_act_quant",
    "fused_gated_rmsnorm_quant",
    "fused_gated_rmsnorm_quant_ref",
    "RMSNormGatedQuant",
    "GateActivation",
    "QuantMode",
    "AXK2_KV_LORA_RANK",
    "AXK2_QK_ROPE_HEAD_DIM",
    "AXK2_QK_NOPE_HEAD_DIM",
    "AXK2_V_HEAD_DIM",
    "AXK2_NATIVE_HEAD_SIZE",
    "FLASHMLA_PADDED_HEAD_SIZE",
    "AXK2MLADims",
    "matrix_absorption",
    "pad_native_kv_to_flashmla",
    "unpad_flashmla_kv_to_native",
    "pad_native_query_to_flashmla",
    "gather_paged_kv_tokens",
    "axk2_native_mla_decode_reference",
    "axk2_padded_flashmla_decode_reference",
    "verify_axk2_native_mla_vs_flashmla",
]
