# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level engine warmup for AXIOM Dynamo workers.

A fresh worker still JIT-compiles kernels on its first requests that the
startup warmup does not cover (CuTeDSL FA4 MLA prefill / combine and Triton
sampling kernels on A.X-K2), so its first long prompt takes seconds and the KV
router steers traffic away from it. The AXIOM Dynamo images wrap Dynamo's
``configure_kv_event_block_size``, which every worker awaits right after the
engine is built and before it registers with the router, with
``with_engine_warmup``. With ``AXIOM_DYNAMO_WARMUP=1`` a few real requests run
through the engine first and the prefix cache is reset afterwards.
"""

import asyncio
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_FLAG = "AXIOM_DYNAMO_WARMUP"

# (prompt tokens, max tokens, sampling kwargs). The 20K prompt exceeds the
# usual 16K token budget, so it also runs chunked prefill.
_SINGLE_CASES: list[tuple[int, int, dict[str, Any]]] = [
    (128, 8, {"temperature": 0.0}),
    (8192, 16, {"temperature": 0.0}),
    (20480, 16, {"temperature": 0.7, "top_p": 0.9, "top_k": 50}),
    (4096, 16, {"temperature": 1.0}),
    (2048, 32, {"temperature": 0.6, "top_p": 0.95}),
]
# (index into _SINGLE_CASES whose prompt is reused as a cached prefix, extra
# tokens): prefills that read cached context. A few new tokens on a long cached
# context (the next turn of a conversation) take FA4's split-KV path.
_PREFIX_CASES = [(1, 1024), (4, 512), (2, 2048), (1, 16), (2, 32)]
_BATCH = 8
# Concurrent requests sharing one cached prefix, with longer and with very short
# suffixes: several prefills that read cached context in the same step.
_SHARED_PREFIX_TOKENS = 4096
_SHARED_PREFIX_REQUESTS = 16
# A larger decode batch than _BATCH (batch-size dependent DSA kernels).
_WIDE_BATCH = 32
_ROUNDS = 2


def _token_ids(seed: int, n: int, vocab: int) -> list[int]:
    span = max(1, vocab - 2000)
    return [1000 + (seed * 7919 + i * 104729) % span for i in range(n)]


async def _generate(
    engine: Any,
    ids: list[int],
    max_tokens: int,
    sampling: dict[str, Any],
    seed: int,
) -> float:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos="structured_outputs" not in sampling,
        seed=seed,
        **sampling,
    )
    start = time.perf_counter()
    async for _ in engine.generate(
        TokensPrompt(prompt_token_ids=ids),
        params,
        f"axiom-warmup-{uuid.uuid4().hex}",
    ):
        pass
    return time.perf_counter() - start


async def warm_up_engine(engine: Any, vllm_config: Any) -> None:
    """Run a small fixed set of requests through ``engine``."""
    model_config = vllm_config.model_config
    vocab = model_config.get_vocab_size()
    max_prompt = model_config.max_model_len - 64
    start = time.perf_counter()
    for rnd in range(_ROUNDS):
        times = []
        for i, (n, max_tokens, sampling) in enumerate(_SINGLE_CASES):
            if n > max_prompt:
                continue
            ids = _token_ids(rnd * 100 + i, n, vocab)
            times.append(await _generate(engine, ids, max_tokens, sampling, i))
        for base, extra in _PREFIX_CASES:
            n = _SINGLE_CASES[base][0]
            if n + extra > max_prompt:
                continue
            ids = _token_ids(rnd * 100 + base, n, vocab)
            ids += _token_ids(rnd * 100 + base + 1000, extra, vocab)
            times.append(await _generate(engine, ids, 16, {"temperature": 0.0}, 0))
        batch = await asyncio.gather(
            *(
                _generate(
                    engine,
                    _token_ids(rnd * 100 + 50 + j, 1024 + 256 * j, vocab),
                    32,
                    {"temperature": 0.7, "top_p": 0.9},
                    j,
                )
                for j in range(_BATCH)
            )
        )
        shared = _token_ids(rnd * 100 + 70, _SHARED_PREFIX_TOKENS, vocab)
        await _generate(engine, shared, 1, {"temperature": 0.0}, 0)
        shared_batch = await asyncio.gather(
            *(
                _generate(
                    engine,
                    shared + _token_ids(rnd * 100 + 71 + j, 256 + 64 * j, vocab),
                    16,
                    {"temperature": 0.7, "top_p": 0.9},
                    j,
                )
                for j in range(_SHARED_PREFIX_REQUESTS)
            )
        )
        short_suffix_batch = await asyncio.gather(
            *(
                _generate(
                    engine,
                    shared + _token_ids(rnd * 100 + 80 + j, 8 + 4 * j, vocab),
                    16,
                    {"temperature": 0.0},
                    j,
                )
                for j in range(_SHARED_PREFIX_REQUESTS)
            )
        )
        wide_batch = await asyncio.gather(
            *(
                _generate(
                    engine,
                    _token_ids(rnd * 100 + 90 + j, 512, vocab),
                    32,
                    {"temperature": 0.0},
                    j,
                )
                for j in range(_WIDE_BATCH)
            )
        )
        logger.info(
            "AXIOM Dynamo warmup round %d: %s, batch %.2fs, shared-prefix batch "
            "%.2fs / %.2fs, wide batch %.2fs",
            rnd,
            " ".join(f"{t:.2f}s" for t in times),
            max(batch),
            max(shared_batch),
            max(short_suffix_batch),
            max(wide_batch),
        )
    try:
        from vllm.sampling_params import StructuredOutputsParams

        schema = {
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
        }
        sampling = {
            "temperature": 0.0,
            "structured_outputs": StructuredOutputsParams(json=schema),
        }
        elapsed = await _generate(engine, _token_ids(999, 256, vocab), 16, sampling, 0)
        logger.info("AXIOM Dynamo warmup structured output %.2fs", elapsed)
    except Exception:
        logger.exception("AXIOM Dynamo warmup structured-output request failed")
    await engine.reset_prefix_cache()
    logger.info(
        "AXIOM Dynamo warmup done in %.1fs (prefix cache reset)",
        time.perf_counter() - start,
    )


def with_engine_warmup(
    configure: Callable[[Any, Any], Awaitable[int]],
) -> Callable[[Any, Any], Awaitable[int]]:
    """Wrap Dynamo's ``configure_kv_event_block_size`` to warm up first."""
    if os.environ.get(ENV_FLAG, "0") != "1":
        return configure

    async def configure_after_warmup(engine: Any, vllm_config: Any) -> int:
        try:
            await warm_up_engine(engine, vllm_config)
        except Exception:
            logger.exception("AXIOM Dynamo warmup failed; serving without it")
        return await configure(engine, vllm_config)

    return configure_after_warmup
