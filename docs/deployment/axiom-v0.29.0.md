# AXIOM v0.29.0 maintenance image

`axiom/release-v0.29.0` starts at `release/ax-k2-v0.29.0` and preserves its
history. Model, parser, routing and guidance changes from `axiom/main` are
cherry-picked as separate commits with their source hashes recorded.

The image carries A.X-K2, A.X-K3 / K3 Flash, A.X-K1 / K1-Light and Solar Open2,
including Solar DSpark support. It keeps vLLM 0.29.0, the release's PyTorch
and NIXL packages, and adds Dynamo 1.6.1 without its vLLM extra.

## Build

Build on Linux x86_64 with BuildKit, from the repository root:

```bash
image=axiom-vllm:v0.29.0-dynamo1.6.1
revision=$(git rev-parse HEAD)
docker build --platform linux/amd64 \
    -f docker/Dockerfile.axiom-dynamo \
    --build-arg VLLM_SRC_COMMIT="$revision" \
    -t "$image" .
```

The CUDA build stage rebuilds `_C_stable_libtorch` from this branch. This is
required by the inherited `dd140a3a12` SM100 FP8 cache-scale fix: copying only
`vllm/` onto the official release image does not include its two CUDA changes.
Other compiled extensions come from the official 0.29.0 image. The default
CUDA targets are SM90 and SM100 (Hopper and Blackwell); override
`torch_cuda_arch_list` when building for other GPUs. `max_jobs` defaults to 4 to bound CUDA compiler memory use.

`docker/dynamo/solar-runtime.requirements.txt` pins the custom runtime wheel
URL and SHA-256. The image installs it with `--no-deps --reinstall
--require-hashes` after `ai-dynamo==1.6.1`. Its release provenance is retained
at `/opt/axiom/dynamo/build-info.json` and in image labels.

## Compatibility decisions

- v0.29.0 retains `get_engine_zmq_addresses`, `Scheduler.skipped_waiting`,
  and `vllm.entrypoints.chat_utils`; the nightly Dockerfile's shims for those
  APIs are deliberately omitted.
- K3's function-tool adapter uses v0.29.0's Responses namespace flattening.
  Its structural-output conversion is kept within the K3 parser.
- The v0.29.0 guidance structural-tag backport comes from `a5f9aa7ba4`, the
  existing release-specific counterpart of `axiom/main`'s `6a59b05681`.
- Nightly changes to shared sparse-MLA index-group streams, NVFP4 sparse
  prefill, workspace allocation and in-place FlashInfer draft metadata are
  not applied to the older attention implementation. In particular,
  v0.29.0 has no FlashInfer `update_draft_decode_metadata` fast path that
  necessitates `abc89090c9` on nightly.

## Validation

CPU checks during the port used the repository's existing tests:

- K3 reasoning/tool parsers and Solar configuration: 159 passed, 1 skipped.
- Solar reasoning/tool parsers, Solar KDA dispatch, and K3 model unit tests:
  375 passed.
- Guidance backend tests: 16 passed with the CPU platform explicitly selected.

On macOS the tests' GPU cleanup fixture crashes in PyTorch's
`empty_host_cache`; CPU-only runs used the existing `skip_global_cleanup`
marker. These checks do not establish GPU model accuracy or serving
performance. Validate model generation and speculative acceptance on the
intended checkpoint before changing a serving workload.

## Published image

Built and pushed on 2026-11-06 from commit
`a2aac4babaf8e858f162c2a9b3179b179a4b877b`:

```text
533266999879.dkr.ecr.ap-northeast-2.amazonaws.com/axiom/serving/ai-dynamo/vllm-runtime:v0.29.0-axiom-a2aac4bab-dynamo1.6.1-solarrt526ee9c6df2f
sha256:17cf68d09061da15aade876c8c8371579143f60126bac5a2a33cc19f487d6767
```

The pushed image was pulled and tested on NVIDIA B200 (SM100, driver
580.95.05). Both Dynamo frontend and worker `--help` checks passed. Installed
versions were vLLM 0.29.0, PyTorch 2.13.0+cu130, Dynamo/runtime 1.6.1,
NIXL 1.3.2, and NCCL 2.30.7; the Solar reasoning/tool parsers were registered.

The following existing tests ran against the installed image package,
without a source-tree vLLM override: **70 passed, 3 skipped** in 98.54 seconds.
The three skips are unsupported dtype/rank combinations in the MLA cache test.

```bash
.venv/bin/python -m pytest -q \
    tests/kernels/test_axk3_gated_residual.py \
    tests/kernels/moe/test_grouped_topk.py::test_bias_free_grouped_topk \
    tests/kernels/attention/test_cache.py::test_concat_and_cache_ds_mla \
    tests/kernels/attention/test_kimi_k3_mla_key_concat_kv_cache.py \
    tests/kernels/attention/test_flashmla_sparse.py::test_sparse_flashmla_decode_matches_cache_writer_scales
```

These cover the rebuilt FP8 cache-scale kernels, their FlashMLA consumer,
K3 gated residuals, and bias-free grouped routing. Full checkpoint generation,
model evaluations, and serving throughput benchmarks were not run. No serving
workload was updated; the delivery scope was image build and registry push.

### Solar tool-marker fix

Commit `eb9aba4edf` exposes `tool_call_start_token` and `tool_call_end_token`
on `SolarOpen2ToolParser`, matching the production overlay. Dynamo's vLLM
postprocessor uses these attributes to identify parser-owned boundaries,
including tool calls immediately after reasoning ends.

The replacement image reuses the previous image by digest and replaces only
`vllm/tool_parsers/solar_open2_tool_parser.py`; CUDA binaries and dependencies
are unchanged:

```text
533266999879.dkr.ecr.ap-northeast-2.amazonaws.com/axiom/serving/ai-dynamo/vllm-runtime:v0.29.0-axiom-eb9aba4ed-dynamo1.6.1-solarrt526ee9c6df2f
sha256:52f079c766b84ab33732e4fcca5f6d2336d7cfbc08bdd2ebec62dd9fd036d18d
```

The existing Solar tool-parser suite passed all 235 tests. In the built image,
the installed Dynamo `StreamingPostProcessor` reproduced the named-tool failure
with the two attributes removed. With the fix, all 8 combinations of named/auto,
streaming/non-streaming, and complete/split tool-call output passed, including
reasoning-end and tool-start in the same chunk. This is a deterministic
postprocessor check, not live model generation or an HTTP serving test.
