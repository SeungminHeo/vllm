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
`torch_cuda_arch_list` when building for other GPUs. `max_jobs` defaults to 12.

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
