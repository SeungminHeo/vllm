# A.X-K2 TPOT 벤치마크 리포트 (2026-09-13)

> 환경: `ipp-axiom-h2013` (H200×8 전용), TP=8, `--kv-cache-dtype fp8_ds_mla`,
> `--max-model-len 262144`, spec decode 없음.
> 벤치: `vllm bench serve` random, input 1024 / output 256 (`--ignore-eos`),
> seed 777, temp 0.0, 3회 반복 (아래 수치는 안정 run 2·3회차 평균, ramp run 제외).
> 원시 데이터: NFS `/t1data/axk2-bench/kernelab/` (조건별 run JSON + 서버 로그 + torch profiler 트레이스).
> 상세 리포트(차트 포함): claude.ai/code/artifact/778853ef-b9fe-4652-9baa-9282df9499bd

## 핵심 결론

1. **지배 요인은 실행 모드**: 같은 코드에서 eager 94.4ms ↔ breakable graph 11.3ms (8.3배, conc1).
   AXK2는 DSV32 기반이라 torch-compile 비대상 → breakable CUDA graph가 유일한 graph 경로.
   `fix/axk2-breakable-cudagraph-default` (nightly), `fix/axk2-breakable-default-v0.28.0` (release)
   브랜치가 env 없이 자동활성화하는 1줄 수정.
2. **MoE shared-expert overlap**: conc1~16에서 약 1ms 이득, conc64+에서는 SM 포화로 노이즈 수준.
   upstream 내장 기능으로 기본 ON — graph만 켜지면 별도 구현 없이 작동.
3. **FlashMLA pad 커밋**: 전 동시성에서 무효과 (평균 ±0.3ms 이내). graph 모드에서는
   per-step 할당이 capture에 박제되어 제거 이득이 원천적으로 0. revert 권고.
4. **v0.23 → 현행 −2.4ms(conc1)의 실체** (torch profiler 분해): 커널 시간 감소 −0.68ms
   (routed MoE −0.51, mnnvl allreduce −0.32, indexer −0.19, quant −0.07; attention 코어와
   dense GEMM은 동일) + **동시 실행 증가 −1.70ms**. 개선의 70%는 동시성의 실현.
5. **세대 격차는 고동시성에서 수렴** (conc1 −18% → conc128 −3%, GEMM-bound화)하지만
   **p99 꼬리 격차는 유지** — v0.23은 conc64에서 34–46ms 스파이크가 일상적.
6. **다음 타깃**: `_axk2_gated_rmsnorm_kernel` 1.29ms/tok (TPOT의 11%) + 동반 glue 복사 0.25ms.

## 전체 수치표

| 조건 | conc | TPOT avg | p50 | p99 (ms) | TPS out | TPS total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.23 공식 (compile+piecewise 기본) | 1 | 13.77 | 13.77 | 13.78 | 72 | 360 |
| | 16 | 19.59 | 19.51 | 22.36¹ | 769 | 3,842 |
| | 64 | 26.68 | 26.15 | 39.95¹ | 1,950 | 9,751 |
| | 128 | 33.03 | 32.67 | 38.00¹ | 3,464 | 17,318 |
| 현행 · overlap 켬 · pad 없음 (breakable) | 1 | 11.31 | 11.31 | 11.31 | 88 | 438 |
| | 16 | 16.35 | 16.29 | 16.75 | 948 | 4,740 |
| | 64 | 24.90 | 24.11 | 29.91¹ | 2,235 | 11,174 |
| | 128 | 32.16 | 31.77 | 39.91¹ | 3,579 | 17,892 |
| 현행 · overlap 끔 · pad 없음 | 1 | 12.37 | 12.37 | 12.37 | 80 | 401 |
| | 16 | 17.41 | 17.34 | 19.49 | 841 | 4,202 |
| | 64 | 24.85 | 24.76 | 25.86 | 2,430 | 12,150 |
| | 128 | 33.00 | 32.92 | 35.11 | 3,656 | 18,281 |
| **현행 · overlap 켬 · pad 포함 (=이 브랜치)** | 1 | 11.40 | 11.40 | 11.41 | 87 | 435 |
| | 16 | 16.56 | 16.55 | 16.95 | 937 | 4,683 |
| | 64 | 24.63 | 24.62 | **25.37** | **2,481** | **12,402** |
| | 128 | 32.13 | 32.17 | **34.39** | **3,755** | **18,775** |

¹ 두 run 중 한 번만 p99 스파이크 (prefill 웨이브 경계 간섭 여부에 따라 갈림).
p50≈avg — 분포는 대칭이며 차이는 전부 꼬리에서 발생. TPS total은 prefill 포함(≈ out × 5).

## conc1 보조 측정

| 조건 | TPOT (ms) |
| --- | ---: |
| 현행 eager (`-cc.cudagraph_mode=NONE`) | 94.35 |
| v0.23 eager (`--enforce-eager`) | 99.28 |
| v0.23 + breakable 강제 (compile 상실) | 17.08 |
| v0.23 + overlap 끔 | 14.65 |
| ECR `v0.28.0-ax-k2` + breakable 강제 | 11.47 |

## 권고

1. 프로덕션(release 이미지)에 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` env 추가 (즉효: 13.8→11.5ms급).
2. `fix/axk2-breakable-*` 브랜치 머지로 기본값 영구화.
3. FlashMLA pad 커밋 revert (`bench/axk2-v2-nopad` 상태).
4. `_axk2_gated_rmsnorm_kernel` 최적화 착수 (~1.5ms/tok 여지).
5. 자체 overlap 커밋 재도입 불필요 — upstream 공식(event 기반, capture-safe)이 상위 호환.

## 재현 커맨드

```bash
# 서빙 (조건 공통)
vllm serve /models/skt/A.X-K2 --served-model-name A.X-K2 \
  --tensor-parallel-size 8 --trust-remote-code --kv-cache-dtype fp8_ds_mla \
  --gpu-memory-utilization 0.90 --max-model-len 262144 --port 8000
# graph: VLLM_USE_BREAKABLE_CUDAGRAPH=1 / eager: -cc.cudagraph_mode=NONE
# overlap 끔: VLLM_DISABLE_SHARED_EXPERTS_STREAM=1

# 벤치 (warmup 후 3회, conc는 16/64/128로 변경)
vllm bench serve --base-url http://127.0.0.1:8000 --model A.X-K2 \
  --tokenizer /models/skt/A.X-K2 --trust-remote-code --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --num-prompts 64 \
  --max-concurrency 16 --ignore-eos --seed 777 --temperature 0.0 --save-result
```
