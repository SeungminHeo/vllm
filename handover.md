# A.X-K2 vLLM Handover (2026-09-04)

> 환경: k8s context **axiom-ipp**, namespace `model-serving`. 결과/스크립트는 NFS `/t1data/axk2-bench/`
> (모든 서빙 pod에 `/t1data`로 마운트; NFS: `172.27.7.122:/axiom_ipp_pvc_47890b91_f2a7_4a63_9ebc_9436024d8a1a`).
> Claude 세션 메모리: `~/.claude/projects/-Users-1112991-orca-vllm/memory/` (axk2-pod-context, deepgemm-driver-regression 등).

## 1. 브랜치 / 이미지 / 태그 규칙

| 브랜치 | 내용 |
| --- | --- |
| `feature/ax-k2` | upstream main 추적 내부 브랜치 (nightly 플로우). Dockerfile.axk2 기본 base = 바이너리 호환 nightly sha |
| `release/ax-k2-v0.28.0` | v0.28.0 태그 + cherry-pick + **AXK2Attention 다운포트**(v0.28.0은 PCP 이전 6-인자 API) |

- 이미지 규칙: repo 고정(`axiom/serving/vllm-openai`, `axiom/serving/ai-dynamo/vllm-runtime`), 태그가 플로우 표현
  — nightly: `<sha9>-ax-k2`, release: `v<tag>-ax-k2[-dynamo는 vllm-runtime repo]`.
- 빌드: `docker/Dockerfile.axk2`(vllm serve용) / `docker/Dockerfile.axk2-dynamo`(ai-dynamo 1.4.2, `[vllm]` extra 미사용).
  Mac(arm64)에서 podman 교차빌드 시 dynamo는 `--build-arg INSTALLER="python3 -m pip install"`(uv가 qemu에서 segfault).
- rebase 시 필수: `git diff --stat <base_old> <base_new> -- csrc cmake CMakeLists.txt setup.py requirements docker/Dockerfile`
  로 호환 nightly 재선정 + **merge-base 이미지 스모크 기동**(부모 DSV3.2 API drift가 조용히 깨짐 — replicated_embed/PCP 시그니처 전례).

## 2. 핵심 사건: h2015 노드 불량 (9/3–9/4)

- 증상: A.X-K2가 h2015에서만 어려운 프롬프트에서 퇴행 반복 생성(runaway) → IFBench strict 56.7~65.7%.
- 오판 이력: 드라이버(580.126→580.178) → DeepGEMM 순으로 의심했으나 **전부 h2015 교란**이었음.
  `VLLM_USE_DEEP_GEMM=0`은 불필요(그 노드 결함을 부분 은폐했을 뿐).
- 확정 근거: 동일 이미지·설정·프롬프트 A/B — h2001 **34/34 정상** vs h2015 문제 key 전원 runaway.
- DCGM `diag -r 3` 양 노드 하드웨어 전항목 Pass(= 표준 진단으로 안 잡힘; software Fail은 persistence mode 경고일 뿐).
  정황: h2015 stress level 전 GPU ~5–7% 낮음, 유휴 온도 +10°C. r4는 사용자 지시로 보류(인프라 작업 선행).
- **h2015에서 품질 벤치/서빙 금지.** 수리 후 검증: 문제 key A/B 재실행(`/t1data/axk2-bench/ab_keys.json`, probe_keys.py).

## 3. IFBench 스코어보드 (300문항, thinking on, 262K, temp 0, 65K캡, 동일 채점기)

| 구성 | 노드 | strict / loose | finish |
| --- | --- | --- | --- |
| 검증선 8/24 (구이미지 base) | h2012 | 72.3 / 75.0 | 293 stop |
| 구이미지 base, DG on | h2009 | **75.3 / 78.0** (296) | 296 stop |
| 구이미지 base, DG on | h2007 | 73.7 / 75.3 | 295/5 |
| 구이미지 base, DG off | h2012 | 71.7 / 74.7 | 299/1 |
| 구이미지 DSpark k5 | h2003 | 73.4 / 74.8 (290) | 289/1 |
| release EAGLE3 k3 (DG off) | h2001 | 74.0 / 76.3, acc 2.68/56% | 297/3 |
| h2015 각종 | h2015 | 56.7~65.7 | 무효(노드 불량) |

- 8/24 검증 데이터(base/EAGLE3/DSpark, acceptance 포함): `/t1data/axk2-bench/IFBench/data/*newpod_10f6db502|eagle3|dspark*`
- EAGLE3/DSpark 특성(8/24): EAGLE3 k3 고동시성 우세(sonnet c128 1668 tok/s), DSpark k5 저지연 우세(conc1 4.9ms).
  DSpark k=3은 **CUDA graph 버그**(동시≥8에서 acceptance 붕괴, eager로 회복 확인) — 사용 금지, upstream 이슈감.

## 4. 진행 중 (이 문서 작성 시점)

- h2007에서 **release 라인 순차 검증** 자동 진행: ① rel-base(기본설정; "release 코드 요인" 최종판정) →
  ② rel-e3(EAGLE3 k3 기본설정) → ③ rel-ds(DSpark k5, gpu-mem 0.85 — 로드 OOM 회피 시도).
  태그: `rel_base_h2007 / rel_e3_h2007 / rel_ds_h2007`; 완료 마커 `/t1data/axk2-bench/par/<tag>/done`.
- 판정: rel-base ≥ ~72% → release 코드 무죄 확정. rel-ds OOM 재발 시 → release 라인 DSpark 로드 순서/메모리 버그 확정
  (v0.28.0 트리, drafter 로드 시점 OOM: "6GB 할당 실패, 4.25GB free" ×2 재현).
- 사용자 측: Dynamo 배포(`ax-k2-bench-d-*`)가 h2001/03/09/12/13/nh2002 점유 중.

## 5. 런북

```bash
# 서빙 pod (manifest 샘플: scratchpad 소실 시 이 문서 기준으로 재작성)
# 필수: nodeName(모델은 노드로컬 /DATA/models/skt/*), hostPath /DATA/models, NFS /t1data,
#       shm 64Gi, privileged, ecr-global-secret, HOME=/tmp, readiness /health(failureThreshold 400)
# args: /DATA/models/skt/A.X-K2 --tensor-parallel-size=8 --trust-remote-code --kv-cache-dtype=fp8_ds_mla
#       --gpu-memory-utilization=0.90 --max-model-len=262144 --port=8000
# spec: --spec-model=... --spec-tokens=3|5 --spec-method=eagle3|dspark (+ --served-model-name)

# IFBench (pod 안에서; 건별 저장·재개 가능, runaway 대비 65K캡)
cd /t1data/axk2-bench && HOME=/tmp python3 ifbench_resume.py --tag <TAG> --cap 65536 --workers 24
# acceptance 스냅샷 포함 실행: bash par_launch.sh <TAG>  → par/<TAG>/{m.before,m.after,done}
# 문제 key 단독 probe: python3 probe_keys.py "42,77,52" / A/B keys: ab_keys.json

# 채점 (Mac, GPU 불필요)
cd <scratch>/ifbench_local && uv venv --python 3.12 .venv && uv pip install -r IFBench/requirements.txt
NLTK_DATA=./nltk_data .venv/bin/python -m run_eval --input_data=... --input_response_data=... --output_dir=...
# acceptance 계산: accsum.py <TAG> (m.before/after diff)
```

주의(스크립트 함정): `pgrep/pkill -f`는 kubectl exec의 bash -c 문자열에 자기매칭됨 → `P=foo; pgrep -f "${P}_bar"` 패턴 사용.
`vllm bench serve`는 temperature 미지정 시 서버 기본(0.6/0.95) 사용 — acceptance 측정은 `--temperature` 명시 + 고정 num-prompts.

## 6. 열린 항목

1. h2015 하드웨어 조치(인프라) → 수리 후 A/B 재검
2. §4 순차 검증 결과 회수/채점 (다른 세션이 이어받을 것)
3. release 라인 DSpark 로드 OOM 원인 수정 (0.85 회피가 실패하면 코드 diff: v0.28.0 dspark utils vs 8/31)
4. acceptance k-스윕 마무리 (EAGLE3 k2/4, DSpark k4; k3-DSpark 금지)
5. DSpark k3 CG 버그 upstream 이슈 제기 (데이터: /t1data/axk2-bench/dspark_spec3/retest*)
