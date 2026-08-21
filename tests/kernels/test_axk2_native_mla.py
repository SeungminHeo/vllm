# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests and numerical equivalence verification for A.X-K2 Native MLA."""

import pytest
import torch

from vllm.models.axk2.common.native_mla import (
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
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_matrix_absorption(dtype: torch.dtype):
    set_random_seed(42)
    batch_size = 4
    num_heads = 32
    qk_nope = AXK2_QK_NOPE_HEAD_DIM  # 64
    kv_lora = AXK2_KV_LORA_RANK  # 128

    q_nope = torch.randn(batch_size, num_heads, qk_nope, dtype=dtype)
    w_uk_t = torch.randn(num_heads, qk_nope, kv_lora, dtype=dtype)

    q_absorbed = matrix_absorption(q_nope, w_uk_t)
    assert q_absorbed.shape == (batch_size, num_heads, kv_lora)
    assert q_absorbed.dtype == dtype

    # Check against explicit per-head loop
    for b in range(batch_size):
        for h in range(num_heads):
            expected = torch.matmul(q_nope[b, h], w_uk_t[h])
            torch.testing.assert_close(
                q_absorbed[b, h],
                expected,
                rtol=1e-5 if dtype == torch.float32 else 1e-3,
                atol=1e-5 if dtype == torch.float32 else 1e-3,
            )


def test_padding_roundtrip():
    set_random_seed(42)
    batch_size = 8
    seq_len = 64
    native_kv = torch.randn(batch_size, seq_len, AXK2_NATIVE_HEAD_SIZE)

    # Native (160) -> FlashMLA Padded (576)
    padded_kv = pad_native_kv_to_flashmla(native_kv)
    assert padded_kv.shape == (batch_size, seq_len, FLASHMLA_PADDED_HEAD_SIZE)

    # Verify zero segments
    assert torch.all(padded_kv[..., 128:512] == 0.0)
    assert torch.all(padded_kv[..., 544:576] == 0.0)

    # Verify data segments
    assert torch.all(padded_kv[..., :128] == native_kv[..., :128])
    assert torch.all(padded_kv[..., 512:544] == native_kv[..., 128:160])

    # FlashMLA Padded (576) -> Native (160)
    recovered_kv = unpad_flashmla_kv_to_native(padded_kv)
    assert recovered_kv.shape == native_kv.shape
    torch.testing.assert_close(recovered_kv, native_kv)


def test_pad_native_query():
    set_random_seed(42)
    batch_size = 4
    num_heads = 32
    q_absorbed = torch.randn(batch_size, num_heads, AXK2_KV_LORA_RANK)
    q_rope = torch.randn(batch_size, num_heads, AXK2_QK_ROPE_HEAD_DIM)

    q_padded = pad_native_query_to_flashmla(q_absorbed, q_rope)
    assert q_padded.shape == (batch_size, num_heads, FLASHMLA_PADDED_HEAD_SIZE)

    assert torch.all(q_padded[..., :128] == q_absorbed)
    assert torch.all(q_padded[..., 128:512] == 0.0)
    assert torch.all(q_padded[..., 512:544] == q_rope)
    assert torch.all(q_padded[..., 544:576] == 0.0)


def test_gather_paged_kv_tokens():
    set_random_seed(42)
    block_size = 16
    num_blocks = 20
    head_size = AXK2_NATIVE_HEAD_SIZE
    kv_cache = torch.randn(num_blocks, block_size, 1, head_size)

    seq_len = 50  # 3 full blocks (48) + 2 remainder
    block_table = torch.tensor([3, 7, 1, 12, 0])

    gathered = gather_paged_kv_tokens(kv_cache, block_table, seq_len, block_size)
    assert gathered.shape == (seq_len, head_size)

    # Verify first block
    torch.testing.assert_close(gathered[:16], kv_cache[3, :, 0, :])
    # Verify second block
    torch.testing.assert_close(gathered[16:32], kv_cache[7, :, 0, :])
    # Verify third block
    torch.testing.assert_close(gathered[32:48], kv_cache[1, :, 0, :])
    # Verify remainder
    torch.testing.assert_close(gathered[48:50], kv_cache[12, :2, 0, :])


@pytest.mark.parametrize("use_gate", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_numerical_equivalence_basic(use_gate: bool, dtype: torch.dtype):
    set_random_seed(42)
    is_identical, diff = verify_axk2_native_mla_vs_flashmla(
        batch_size=4,
        seq_len=64,
        num_heads=32,
        dtype=dtype,
        use_gate=use_gate,
    )
    assert is_identical, f"Failed numerical equivalence with max diff: {diff}"


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 64, 512, 2048])
def test_numerical_equivalence_batch_and_seqlen(batch_size: int, seq_len: int):
    set_random_seed(42)
    dtype = torch.bfloat16
    is_identical, diff = verify_axk2_native_mla_vs_flashmla(
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=32,
        dtype=dtype,
        use_gate=True,
    )
    assert is_identical, f"Failed for B={batch_size}, S={seq_len}: max diff={diff}"


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
def test_numerical_equivalence_varlen(batch_size: int):
    set_random_seed(42)
    dtype = torch.bfloat16
    max_len = 256
    num_heads = 32

    q_nope = torch.randn(batch_size, num_heads, AXK2_QK_NOPE_HEAD_DIM, dtype=dtype)
    q_rope = torch.randn(batch_size, num_heads, AXK2_QK_ROPE_HEAD_DIM, dtype=dtype)
    w_uk_t = torch.randn(
        num_heads, AXK2_QK_NOPE_HEAD_DIM, AXK2_KV_LORA_RANK, dtype=dtype
    )
    w_uv = torch.randn(num_heads, AXK2_KV_LORA_RANK, AXK2_V_HEAD_DIM, dtype=dtype)
    attn_gate = torch.randn(batch_size, num_heads, AXK2_V_HEAD_DIM, dtype=dtype)

    kv_native = torch.randn(batch_size, max_len, AXK2_NATIVE_HEAD_SIZE, dtype=dtype)
    kv_padded = pad_native_kv_to_flashmla(kv_native)

    seq_lens = torch.randint(1, max_len + 1, (batch_size,)).tolist()

    out_native = axk2_native_mla_decode_reference(
        q_nope=q_nope,
        q_rope=q_rope,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        kv_cache=kv_native,
        seq_lens=seq_lens,
        attn_gate=attn_gate,
    )

    out_padded = axk2_padded_flashmla_decode_reference(
        q_nope=q_nope,
        q_rope=q_rope,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        kv_cache_padded=kv_padded,
        seq_lens=seq_lens,
        attn_gate=attn_gate,
    )

    torch.testing.assert_close(out_native, out_padded, atol=1e-1, rtol=1.6e-2)


def test_paged_vs_dense_kv_cache():
    set_random_seed(42)
    batch_size = 2
    seq_len = 100
    block_size = 16
    num_heads = 32
    dtype = torch.bfloat16

    q_nope = torch.randn(batch_size, num_heads, AXK2_QK_NOPE_HEAD_DIM, dtype=dtype)
    q_rope = torch.randn(batch_size, num_heads, AXK2_QK_ROPE_HEAD_DIM, dtype=dtype)
    w_uk_t = torch.randn(
        num_heads, AXK2_QK_NOPE_HEAD_DIM, AXK2_KV_LORA_RANK, dtype=dtype
    )
    w_uv = torch.randn(num_heads, AXK2_KV_LORA_RANK, AXK2_V_HEAD_DIM, dtype=dtype)

    # Create dense KV cache
    kv_dense = torch.randn(batch_size, seq_len, AXK2_NATIVE_HEAD_SIZE, dtype=dtype)

    # Place dense tokens into paged blocks
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * batch_size + 10
    kv_paged_pool = torch.randn(
        total_blocks, block_size, AXK2_NATIVE_HEAD_SIZE, dtype=dtype
    )

    block_tables = []
    gathered_list = []
    for b in range(batch_size):
        b_table = torch.arange(b * num_blocks_per_seq, (b + 1) * num_blocks_per_seq)
        block_tables.append(b_table)
        for idx, block_id in enumerate(b_table):
            start = idx * block_size
            end = min((idx + 1) * block_size, seq_len)
            kv_paged_pool[block_id, : end - start] = kv_dense[b, start:end]

        gathered = gather_paged_kv_tokens(kv_paged_pool, b_table, seq_len, block_size)
        gathered_list.append(gathered)

    out_dense = axk2_native_mla_decode_reference(
        q_nope=q_nope,
        q_rope=q_rope,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        kv_cache=kv_dense,
    )

    out_paged = axk2_native_mla_decode_reference(
        q_nope=q_nope,
        q_rope=q_rope,
        w_uk_t=w_uk_t,
        w_uv=w_uv,
        kv_cache=gathered_list,
    )

    torch.testing.assert_close(out_dense, out_paged)


def test_bandwidth_reduction_calculation():
    dims = AXK2MLADims()
    assert dims.native_head_size == 160
    assert dims.padded_head_size == 576
    assert dims.bandwidth_reduction_ratio > 0.72  # 72.22%
