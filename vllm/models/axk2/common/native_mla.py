# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A.X-K2 Native Shape MLA (Multi-Head Latent Attention) Standalone Reference.

This module provides a pure PyTorch reference implementation of A.X-K2's native
160-dimensional MLA (128 kv_lora + 32 rope) decode attention, eliminating the
72.2% memory bandwidth waste caused by FlashMLA's 576-dimensional padding.

Architecture & Dimensions:
- Query: q_nope in [B, H, 64], q_rope in [B, H, 32]
- Key-Value Cache per Token: [c_kv (128) + k_pe (32)] = 160 dims
- Matrix Absorption: q_absorbed = q_nope @ W_uk_t in [B, H, 128]
- Score: S = (q_absorbed @ c_kv^T + q_rope @ k_pe^T) / sqrt(qk_head_dim)
- Softmax & Latent Accumulation: O_latent = Softmax(S) @ c_kv in [B, H, 128]
- Output Projection: Output = O_latent @ W_uv in [B, H, 64]
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# A.X-K2 Native Dimension Constants
AXK2_KV_LORA_RANK = 128
AXK2_QK_ROPE_HEAD_DIM = 32
AXK2_QK_NOPE_HEAD_DIM = 64
AXK2_V_HEAD_DIM = 64
AXK2_NATIVE_HEAD_SIZE = AXK2_KV_LORA_RANK + AXK2_QK_ROPE_HEAD_DIM  # 160
FLASHMLA_PADDED_HEAD_SIZE = 576  # 512 (kv_lora) + 64 (qk_rope)


@dataclass(frozen=True)
class AXK2MLADims:
    """Dimensions specification for A.X-K2 Native MLA."""

    num_heads: int = 32
    qk_nope_head_dim: int = AXK2_QK_NOPE_HEAD_DIM  # 64
    qk_rope_head_dim: int = AXK2_QK_ROPE_HEAD_DIM  # 32
    kv_lora_rank: int = AXK2_KV_LORA_RANK  # 128
    v_head_dim: int = AXK2_V_HEAD_DIM  # 64
    native_head_size: int = AXK2_NATIVE_HEAD_SIZE  # 160
    padded_head_size: int = FLASHMLA_PADDED_HEAD_SIZE  # 576

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim  # 96

    @property
    def default_scale(self) -> float:
        return 1.0 / math.sqrt(self.qk_head_dim)

    @property
    def bandwidth_reduction_ratio(self) -> float:
        """Memory bandwidth reduction: 1.0 - (160 / 576) = 72.22%."""
        return 1.0 - (self.native_head_size / self.padded_head_size)


def matrix_absorption(q_nope: torch.Tensor, w_uk_t: torch.Tensor) -> torch.Tensor:
    """Absorb W_uk into Query (q_nope @ W_uk_t).

    Args:
        q_nope: [B, H, 64] Query non-RoPE component
        w_uk_t: [H, 64, 128] Transposed Key un-absorption weight matrix

    Returns:
        q_absorbed: [B, H, 128] Absorbed query tensor in latent space
    """
    # q_nope: [B, H, 64], w_uk_t: [H, 64, 128] -> q_absorbed: [B, H, 128]
    return torch.einsum("bhd,hdm->bhm", q_nope, w_uk_t)


def pad_native_kv_to_flashmla(kv_native: torch.Tensor) -> torch.Tensor:
    """Pad Native KV tensor [..., 160] to FlashMLA layout [..., 576].

    Layout mapping:
    - [..., :128] (c_kv) -> [..., :128]
    - [..., 128:512] (zeros) -> [..., 128:512]
    - [..., 128:160] (k_pe) -> [..., 512:544]
    - [..., 544:576] (zeros) -> [..., 544:576]
    """
    orig_shape = kv_native.shape[:-1]
    out = kv_native.new_zeros(*orig_shape, FLASHMLA_PADDED_HEAD_SIZE)
    # c_kv: 128 dims
    out[..., :AXK2_KV_LORA_RANK] = kv_native[..., :AXK2_KV_LORA_RANK]
    # k_pe: 32 dims at offset 512
    out[..., 512 : 512 + AXK2_QK_ROPE_HEAD_DIM] = kv_native[
        ..., AXK2_KV_LORA_RANK:AXK2_NATIVE_HEAD_SIZE
    ]
    return out


def unpad_flashmla_kv_to_native(kv_padded: torch.Tensor) -> torch.Tensor:
    """Unpad FlashMLA layout [..., 576] back to Native KV [..., 160].

    Extracts:
    - [..., :128] (c_kv)
    - [..., 512:544] (k_pe)
    """
    c_kv = kv_padded[..., :AXK2_KV_LORA_RANK]
    k_pe = kv_padded[..., 512 : 512 + AXK2_QK_ROPE_HEAD_DIM]
    return torch.cat([c_kv, k_pe], dim=-1)


def pad_native_query_to_flashmla(
    q_absorbed: torch.Tensor,
    q_rope: torch.Tensor,
) -> torch.Tensor:
    """Pad absorbed query [B, H, 128] and q_rope [B, H, 32] to [B, H, 576]."""
    b, h, _ = q_absorbed.shape
    out = q_absorbed.new_zeros(b, h, FLASHMLA_PADDED_HEAD_SIZE)
    out[..., :AXK2_KV_LORA_RANK] = q_absorbed
    out[..., 512 : 512 + AXK2_QK_ROPE_HEAD_DIM] = q_rope
    return out


def gather_paged_kv_tokens(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Gather sequential KV tokens from paged KV cache blocks.

    Args:
        kv_cache: [num_blocks, block_size, 1, head_size] or
            [num_blocks, block_size, head_size]
        block_table: [num_blocks_for_seq]
        seq_len: Total context length to gather
        block_size: Block size (e.g. 16 or 128)

    Returns:
        gathered_kv: [seq_len, head_size]
    """
    head_size = kv_cache.shape[-1]
    gathered = kv_cache.new_empty(seq_len, head_size)

    num_full_blocks = seq_len // block_size
    remainder = seq_len % block_size

    for b_idx in range(num_full_blocks):
        phys_block = block_table[b_idx]
        block_data = kv_cache[phys_block].view(block_size, head_size)
        gathered[b_idx * block_size : (b_idx + 1) * block_size] = block_data

    if remainder > 0:
        phys_block = block_table[num_full_blocks]
        block_data = kv_cache[phys_block].view(block_size, head_size)[:remainder]
        gathered[num_full_blocks * block_size :] = block_data

    return gathered


def axk2_native_mla_decode_reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: list[int] | torch.Tensor | None = None,
    scale: float | None = None,
    attn_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch Reference for A.X-K2 Native Shape MLA Decode Attention.

    Args:
        q_nope: [B, H, 64] Query non-RoPE component
        q_rope: [B, H, 32] Query RoPE component
        w_uk_t: [H, 64, 128] Transposed Key un-absorption matrix
        w_uv: [H, 128, 64] Value un-absorption projection matrix
        kv_cache: [B, S, 160] (Dense KV cache) or list of [S_i, 160] tensors
        seq_lens: List/Tensor of sequence lengths per batch item
        scale: Softmax scaling factor (default: 1.0 / sqrt(96))
        attn_gate: Optional [B, H, 64] gate tensor for sigmoid modulation

    Returns:
        output: [B, H, 64] Final attention output per head
    """
    b, h, _ = q_nope.shape
    if scale is None:
        scale = 1.0 / math.sqrt(AXK2_QK_NOPE_HEAD_DIM + AXK2_QK_ROPE_HEAD_DIM)

    # 1. Matrix Absorption: [B, H, 64] -> [B, H, 128]
    q_absorbed = matrix_absorption(q_nope, w_uk_t)

    outputs = []
    for i in range(b):
        q_abs_i = q_absorbed[i]  # [H, 128]
        q_rope_i = q_rope[i]  # [H, 32]

        if isinstance(kv_cache, list):
            kv_i = kv_cache[i]  # [S, 160]
        elif kv_cache.ndim == 3:
            s_i = seq_lens[i] if seq_lens is not None else kv_cache.shape[1]
            kv_i = kv_cache[i, :s_i]  # [S_i, 160]
        else:
            raise ValueError(f"Unsupported kv_cache shape: {kv_cache.shape}")

        c_kv = kv_i[:, :AXK2_KV_LORA_RANK]  # [S_i, 128]
        k_pe = kv_i[:, AXK2_KV_LORA_RANK:]  # [S_i, 32]

        # 2. Score Calculation: S = scale * (q_abs @ c_kv.T + q_rope @ k_pe.T)
        # Using FP32 accumulation matching hardware attention kernel
        # semantics (FlashMLA / Triton)
        score_nope = torch.matmul(q_abs_i.float(), c_kv.float().T)
        score_rope = torch.matmul(q_rope_i.float(), k_pe.float().T)
        scores = (score_nope + score_rope) * scale  # [H, S_i]

        # 3. Softmax: [H, S_i]
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)

        # 4. Latent Accumulation: O_latent = attn_weights @ c_kv -> [H, 128]
        o_latent = torch.matmul(attn_weights, c_kv.float())  # [H, 128]

        # 5. Output Projection: Output = o_latent @ w_uv -> [H, 64]
        out_i = (
            torch.bmm(o_latent.unsqueeze(1), w_uv.float()).squeeze(1).to(q_nope.dtype)
        )

        # 6. Optional Attention Output Gate Modulation
        if attn_gate is not None:
            gate_i = attn_gate[i]  # [H, 64]
            out_i = (out_i.float() * torch.sigmoid(gate_i.float())).to(out_i.dtype)

        outputs.append(out_i)

    return torch.stack(outputs, dim=0)


def axk2_padded_flashmla_decode_reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    kv_cache_padded: torch.Tensor,
    seq_lens: list[int] | torch.Tensor | None = None,
    scale: float | None = None,
    attn_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference execution simulating FlashMLA's 576-dim padded layout."""
    b, h, _ = q_nope.shape
    if scale is None:
        scale = 1.0 / math.sqrt(AXK2_QK_NOPE_HEAD_DIM + AXK2_QK_ROPE_HEAD_DIM)

    q_absorbed = matrix_absorption(q_nope, w_uk_t)
    q_padded = pad_native_query_to_flashmla(q_absorbed, q_rope)  # [B, H, 576]

    outputs = []
    for i in range(b):
        q_p_i = q_padded[i]  # [H, 576]
        s_i = seq_lens[i] if seq_lens is not None else kv_cache_padded.shape[1]
        kv_p_i = kv_cache_padded[i, :s_i]  # [S_i, 576]

        # [H, 576] @ [576, S_i] -> [H, S_i] with FP32 accumulator
        scores = torch.matmul(q_p_i.float(), kv_p_i.float().T) * scale
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)

        # [H, S_i] @ [S_i, 576] -> [H, 576] with FP32 accumulator
        o_padded = torch.matmul(attn_weights, kv_p_i.float())
        # Strip padded extension to recover [H, 128]
        o_latent = o_padded[:, :AXK2_KV_LORA_RANK]

        out_i = (
            torch.bmm(o_latent.unsqueeze(1), w_uv.float()).squeeze(1).to(q_nope.dtype)
        )

        if attn_gate is not None:
            gate_i = attn_gate[i]
            out_i = (out_i.float() * torch.sigmoid(gate_i.float())).to(out_i.dtype)

        outputs.append(out_i)

    return torch.stack(outputs, dim=0)


def verify_axk2_native_mla_vs_flashmla(
    batch_size: int = 4,
    seq_len: int = 128,
    num_heads: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    atol: float | None = None,
    rtol: float | None = None,
    use_gate: bool = True,
) -> tuple[bool, float]:
    """Verify numerical equivalence between Native MLA and Padded FlashMLA.

    Returns:
        (is_identical, max_abs_diff)
    """
    if atol is None:
        atol = (
            1e-4
            if dtype == torch.float32
            else (1e-1 if dtype == torch.bfloat16 else 1e-2)
        )
    if rtol is None:
        rtol = 1e-4 if dtype == torch.float32 else 1.6e-2
    torch.manual_seed(42)
    q_nope = torch.randn(batch_size, num_heads, AXK2_QK_NOPE_HEAD_DIM, dtype=dtype)
    q_rope = torch.randn(batch_size, num_heads, AXK2_QK_ROPE_HEAD_DIM, dtype=dtype)
    w_uk_t = torch.randn(
        num_heads, AXK2_QK_NOPE_HEAD_DIM, AXK2_KV_LORA_RANK, dtype=dtype
    )
    w_uv = torch.randn(num_heads, AXK2_KV_LORA_RANK, AXK2_V_HEAD_DIM, dtype=dtype)

    # Native KV Cache: [B, S, 160]
    kv_native = torch.randn(batch_size, seq_len, AXK2_NATIVE_HEAD_SIZE, dtype=dtype)
    # Padded FlashMLA KV Cache: [B, S, 576]
    kv_padded = pad_native_kv_to_flashmla(kv_native)

    attn_gate = (
        torch.randn(batch_size, num_heads, AXK2_V_HEAD_DIM, dtype=dtype)
        if use_gate
        else None
    )

    out_native = axk2_native_mla_decode_reference(
        q_nope=q_nope,
        q_rope=q_rope,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        kv_cache=kv_native,
        attn_gate=attn_gate,
    )

    out_padded = axk2_padded_flashmla_decode_reference(
        q_nope=q_nope,
        q_rope=q_rope,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        kv_cache_padded=kv_padded,
        attn_gate=attn_gate,
    )

    diff = torch.max(torch.abs(out_native - out_padded)).item()
    is_close = torch.allclose(out_native, out_padded, atol=atol, rtol=rtol)

    return is_close, diff
