# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import solar_open2

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def clear_warmup_state():
    caches = (
        solar_open2._KDA_PREFILL_WARMED_SIGNATURES,
        solar_open2._KDA_PREFILL_FAILED_ATTEMPTS,
        solar_open2._KDA_PREFILL_STATE_DTYPE_FAILURES,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


def _make_layer(warmup_tokens: int = 2048):
    return SimpleNamespace(
        local_num_heads=2,
        head_dim=4,
        _kda_prefill_warmup_tokens=warmup_tokens,
        A_log=torch.ones(2, dtype=torch.float32),
        dt_bias=torch.ones(2, dtype=torch.float32),
        prefix="model.layers.0.self_attn",
        get_state_dtype=lambda: (None, torch.float32),
    )


def _warmup(layer, q_proj_states):
    solar_open2.SolarOpen2KimiDeltaAttention._warmup_prefill_kernels(
        layer, q_proj_states
    )


def test_kda_attention_op_is_a_graph_splitting_op():
    """The KDA op must split the piecewise graph.

    It reads `attn_metadata` from the forward context and writes the KDA
    caches, so capturing it inside a cudagraph segment freezes the metadata
    from capture time. The model then still produces fluent text while
    attending over the wrong state, which no cheap test would catch.
    """
    from vllm.config.compilation import CompilationConfig

    assert "vllm::solar_open2_kda_attention" in CompilationConfig._attention_ops


def test_kda_attention_op_registration_passes_opcheck(monkeypatch):
    """Opcheck the custom op: schema and mutation annotations, fake impl.

    The op resolves its layer through the forward context, so a stub layer
    that mutates only the declared-mutable ``core_attn_out`` stands in for
    the engine.
    """
    from vllm import forward_context

    layer_name = "opcheck.kda"

    class _StubLayer:
        def _forward(
            self,
            *,
            q_proj_states,
            k_proj_states,
            v_proj_states,
            g1,
            beta,
            core_attn_out,
        ):
            core_attn_out.add_(1.0)

    ctx = forward_context.ForwardContext(
        no_compile_layers={layer_name: _StubLayer()},
        attn_metadata=None,
        slot_mapping=None,
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)

    n, h, d = 4, 2, 8
    torch.library.opcheck(
        torch.ops.vllm.solar_open2_kda_attention.default,
        (
            torch.randn(n, h * d),
            torch.randn(n, h * d),
            torch.randn(n, h * d),
            torch.randn(1, n, h, d),
            torch.rand(1, n, h),
            torch.zeros(1, n, h, d),
            layer_name,
        ),
    )


def test_warmup_token_count_tracks_the_scheduler_batch_size():
    count = solar_open2._kda_warmup_token_count
    chunk = solar_open2.FLA_CHUNK_SIZE
    cap = solar_open2._KDA_WARMUP_MAX_TOKENS

    # Autotuning on a single chunk picks configs that lose ~20% on a real
    # prefill, so a realistic batch must not collapse back to one chunk.
    assert count(8192) == cap
    assert count(cap) == cap
    assert count(1024) == 1024
    assert count(chunk * 3 + 1) == chunk * 3
    assert cap % chunk == 0

    # An unknown batch size must not fall back to the one value measured to
    # select bad configs.
    assert count(None) == cap
    assert count(0) == cap

    # A genuinely tiny batch size is honoured, down to one runnable chunk.
    for tiny in (1, chunk - 1, chunk):
        assert count(tiny) == chunk


def test_warmup_runs_once_per_signature_at_the_configured_token_count(monkeypatch):
    calls = []

    def fake_kernel(**kwargs):
        calls.append((kwargs["q"].shape[1], kwargs["cu_seqlens"].tolist()))

    monkeypatch.setattr(solar_open2, "chunk_kda_with_fused_gate", fake_kernel)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    q_proj_states = torch.empty(1, dtype=torch.float32)
    _warmup(_make_layer(warmup_tokens=1024), q_proj_states)
    # A second layer with the same signature reuses the warmed kernels.
    _warmup(_make_layer(warmup_tokens=1024), q_proj_states)

    assert calls == [(1024, [0, 1024])]
    assert len(solar_open2._KDA_PREFILL_WARMED_SIGNATURES) == 1


def test_failed_warmup_retries_are_bounded_per_signature(monkeypatch):
    kernel_calls = 0

    def fail(**kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        raise RuntimeError("synthetic warmup failure")

    monkeypatch.setattr(solar_open2, "chunk_kda_with_fused_gate", fail)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    q_proj_states = torch.empty(1, dtype=torch.float32)
    for _ in range(3):
        _warmup(_make_layer(), q_proj_states)

    # Bounded attempts, not one retry per layer: the real model has 36 KDA
    # layers and a deterministic failure must not be retried by each of them.
    assert kernel_calls == solar_open2._KDA_PREFILL_MAX_FAILED_ATTEMPTS
    assert not solar_open2._KDA_PREFILL_WARMED_SIGNATURES


class _KernelSpy:
    """Stand-ins for the KDA kernels that tag their output by path."""

    SPEC_SCALE = 2.0
    PREFILL_SCALE = 3.0

    def __init__(self):
        self.spec_calls: list[dict] = []
        self.prefill_calls: list[dict] = []

    def conv_update(self, x, *args, out=None, **kwargs):
        return x if out is None else out.copy_(x)

    def conv_fn(self, x, *args, **kwargs):
        return x

    def gate(self, g, *args, **kwargs):
        return g

    def recurrent_fwd(self, **kwargs):
        self.spec_calls.append(kwargs)
        return kwargs["q"] * self.SPEC_SCALE, kwargs["initial_state"]

    def chunk(self, **kwargs):
        self.prefill_calls.append(kwargs)
        return kwargs["q"] * self.PREFILL_SCALE, kwargs["initial_state"]


def _spec_layer(num_heads: int, head_dim: int, num_blocks: int, num_spec: int):
    dim = num_heads * head_dim
    width = 4

    def conv(width):
        return SimpleNamespace(weight=torch.randn(dim, 1, width), bias=None)

    # Bypass __init__: it needs a full VllmConfig and distributed groups, while
    # the routing under test only reads these attributes.
    cls = solar_open2.SolarOpen2KimiDeltaAttention
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.1.self_attn"
    layer.head_dim = head_dim
    layer.local_num_heads = num_heads
    layer.A_log = torch.zeros(1, 1, num_heads, 1)
    layer.dt_bias = torch.zeros(dim)
    layer.q_conv1d = conv(width)
    layer.k_conv1d = conv(width)
    layer.v_conv1d = conv(width)
    layer.kv_cache = (
        torch.zeros(num_blocks, 3 * dim, width - 1 + num_spec),
        torch.zeros(num_blocks, num_heads, head_dim, head_dim),
    )
    return layer


def test_mixed_spec_and_plain_batch_routes_and_merges_by_token_order(monkeypatch):
    """A batch mixing a draft-carrying request with a plain one must run the
    draft tokens through the state-rewinding recurrent kernel and the plain
    tokens through the prefill kernel, then land every output at its original
    token position. Getting the merge wrong still yields fluent text (the
    tokens are only permuted between requests), so this is checked directly.
    """
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    num_heads, head_dim, num_spec = 2, 4, 2
    layer = _spec_layer(num_heads, head_dim, num_blocks=8, num_spec=num_spec)
    spy = _KernelSpy()
    monkeypatch.setattr(solar_open2, "causal_conv1d_update", spy.conv_update)
    monkeypatch.setattr(solar_open2, "causal_conv1d_fn", spy.conv_fn)
    monkeypatch.setattr(solar_open2, "fused_kda_gate", spy.gate)
    monkeypatch.setattr(solar_open2, "fused_recurrent_kda_fwd", spy.recurrent_fwd)
    monkeypatch.setattr(solar_open2, "chunk_kda_with_fused_gate", spy.chunk)
    monkeypatch.setattr(solar_open2, "is_conv_state_dim_first", lambda: True)

    # Token layout: [plain req (1 token), spec req (1 + num_spec tokens)].
    num_tokens = 1 + (1 + num_spec)
    spec_token_indx = torch.tensor([1, 2, 3])
    non_spec_token_indx = torch.tensor([0])
    num_accepted_tokens = torch.tensor([2], dtype=torch.int32)
    spec_state_indices = torch.tensor([[4, 5, 6]], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=1 + num_spec,
        num_actual_tokens=num_tokens,
        has_initial_state=torch.tensor([True]),
        spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        spec_state_indices_tensor=spec_state_indices,
        non_spec_state_indices_tensor=torch.tensor([2], dtype=torch.int32),
        spec_sequence_masks=torch.tensor([False, True]),
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=num_accepted_tokens,
    )
    monkeypatch.setattr(
        solar_open2,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
    )

    dim = num_heads * head_dim
    q = torch.arange(num_tokens * dim, dtype=torch.float32).view(num_tokens, dim)
    core_attn_out = torch.zeros(1, num_tokens, num_heads, head_dim)
    layer._forward(
        q_proj_states=q,
        k_proj_states=q.clone(),
        v_proj_states=q.clone(),
        g1=torch.zeros(1, num_tokens, num_heads, head_dim),
        beta=torch.ones(1, num_tokens, num_heads),
        core_attn_out=core_attn_out,
    )

    q_heads = q.view(num_tokens, num_heads, head_dim)
    expected = q_heads.clone()
    expected[spec_token_indx] *= spy.SPEC_SCALE
    expected[non_spec_token_indx] *= spy.PREFILL_SCALE
    torch.testing.assert_close(core_attn_out[0], expected)

    (spec_call,) = spy.spec_calls
    assert spec_call["num_accepted_tokens"] is num_accepted_tokens
    assert spec_call["ssm_state_indices"] is spec_state_indices
    assert spec_call["cu_seqlens"].tolist() == [0, 3]
    (prefill_call,) = spy.prefill_calls
    assert prefill_call["cu_seqlens"].tolist() == [0, 1]
