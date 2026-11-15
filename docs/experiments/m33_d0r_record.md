# M3-3 D0′ 기록 — 개정 실행기로 D0 재실행, C0b와 대조

- 작성: 2026-11-15, cv-detection-expert. 규칙은 [PP] [`m33_pp_spec.md`](m33_pp_spec.md) v0.2.4 부록 A-1, 순서는 [PLAN] [`m33_cross_domain_plan.md`](m33_cross_domain_plan.md) v0.2.8 §4 D0′ 행이다.
- **C1 감사 참고**: 부록 A-1의 "S5 제외(해석 결정 1)" 출처는 C0b 판정서 [`m33_c0b_d0_judgment.md`](m33_c0b_d0_judgment.md) §3 1이다(처음 실패한 $S_4$ 뒤의 $S_5$는 계산·기록하지만 해석·선택에 쓰지 않음).
- 이 문서는 C0b 판정서를 바꾸지 않는다. C0b 결과(`runs/eval/m33_d0/`)와 선택 S3는 그대로이고, 여기서는 그 선택이 개정 실행기로도 재현되는지만 기록한다.

## 1. 결론

**D0′가 C0b 선택(S3)을 확인했다.**
- 선택 `pp_set_id`, `stop_reason`, 평가 집합, 집합별 G1–G4 통과·실패가 C0b와 모두 같다.
- 수치 비교 183칸은 모두 차이 0이다(허용 폭 안 차이도 0건).
- 기록 전용 칸(A2 40칸)도 모두 같다.
- C0b 주 1의 경계 박스: C0b 판정서가 추론한 "실행기 값이 규칙값이고 0.25를 넘었다"는 재실행 점수(0.25003907…)와 두 실행의 G4 S2 = 70 일치로 확인됐다(C0b 실행의 원래 점수는 저장되지 않음).

## 2. 실행 정체성

| 항목 | 값 |
|---|---|
| 명령 | 저장소 루트에서 `conda run -n gpr-dev python -I -B analysis/detection/m33_d0_gate.py --out-root runs/eval/m33_d0r` |
| argv | `[analysis/detection/m33_d0_gate.py, --out-root, runs/eval/m33_d0r]` |
| 시각·시간 | 2026-11-15 22:38 무렵 시작, 295.3 s |
| 종료·status | 0, `ok` |
| 영상 수 | `n_images` 16,919 |
| `judgment` | `judgeable` true, `blockers` [], `folds_run` 0–4, `tree_stable` true, `tree_changes` [] |
| `sealed_pp_assertion` | `pass`(checks 5개 `pp_set_sha256`·`pp_sets_sha256`·`src_tree_id`·`pipeline_blob`·`selected_set_sha256` 모두 true) |
| git HEAD (시작·끝) | `882e881a7cdcc3cf3ff0431f472301df76d723a8`, `dirty` false, `tree_problems` [] |
| src tree id | `164f09909ad51c1a8d26545cadb0417b1da3acc6` (C0a 53c4e16과 같음) |
| 설정 | device 0, full_run true, smoke false, limit null, 모델 20, GPU·패키지 버전 C0b와 같음(메인 세션 확인) |
| 출력 | `runs/eval/m33_d0r/d0_gate.json` sha256 `16d7d2ade37a79ef24f00603628d864b5ddd396bd468738dbe9ab3f1970aedf1` |

- 메인 세션은 실행 뒤 정체성 키만 보았고 게이트 값은 열지 않았다. 게이트 값은 대조 스크립트 출력으로 이 문서에서 처음 해석했다.
- 실패 폴더나 재실행은 없었다. 예비 폴더 `runs/eval/m33_d0r2`는 쓰지 않았다.

## 3. 대조

### 3-1. 명령과 결과

```
conda run -n gpr-dev python -I -B analysis/detection/m33_d0r_compare.py ^
  --base runs/eval/m33_d0/d0_gate.json --new runs/eval/m33_d0r/d0_gate.json ^
  --out runs/eval/m33_d0r/compare_c0b.json
```

| 항목 | 값 |
|---|---|
| 종료·status | 0, **`match`** |
| 대조 스크립트 | `analysis/detection/m33_d0r_compare.py` blob `ee802e8b8e665264ccc95dc0f9944a8f661b4097` (882e881에 커밋된 것 그대로, 출력의 `script blob`과 같음) |
| base | `runs/eval/m33_d0/d0_gate.json` sha256 `107d7b53f30667684baf7c2236bdf338c861309531ff17497e2836193bea08c6` (C0b 판정서 기록값과 같음). 대조 스크립트는 base sha를 기록만 하고 단언하지 않는다(A-1 "먼저 단언"과 다름). 실행 뒤 판정서 기록값과 대조해 같음을 확인했다. |
| new | `runs/eval/m33_d0r/d0_gate.json` sha256 `16d7d2ad…` |
| 출력 | `compare_c0b.json` sha256 `39fac6049d632addfd8328605c2425caa40f96e675cf785fcb7c0c7d9db57058`, `compare_c0b.md` sha256 `e2abcd27039e57290a82e30072c33125bed8dcdb1ea317be39113635c6511bc6` |
| 비교 칸 | 183(폭 밖 0, 폭 안 차이 0), 기록 전용 40 |

### 3-2. 칸 수 독립 점검

부록 A-1의 판정 범위(집합 S1–S4, 게이트 조건 A0·A1·A3, 조건·fold별 칸)에서 기대 칸 수를 따로 세었다. 모두 스크립트 값과 같다.

| 구분 | 기대 | 스크립트 | 근거 |
|---|---|---|---|
| selection | 3 | 3 | `pp_set_id`, `stop_reason`, `evaluated` |
| checks | 20 | 20 | 4 집합 × (G1–G4 + pass) |
| G1 | 60 | 60 | 4 집합 × 3 조건 × 5 fold |
| G2 | 60 | 60 | 같음 |
| G3 | 12 | 12 | 4 집합 × ($H(S_0)$, $H(S_j)$, 노출 수) |
| G4 | 8 | 8 | 4 집합 × (증분, 누적) |
| G4 규칙·조건 | 20 | 20 | `by_rule`·`by_condition` 키 합집합: S1 2, S2 4, S3 6, S4 8. 자료 키에서 센 값, 규칙 고정 기대값 아님(스크립트 미검사) |
| 합 | 183 | 183 | |
| 기록 전용 A2 | 40 | 40 | 4 집합 × (G1, G2) × 5 fold |

- G4 `by_condition`의 Y0 값은 두 json 모두 빈 사전(`{}`)이다. Y0 병합 박스 3개(fold 1에 1개, fold 2에 2개)가 S1–S5 어디에서도 처리되지 않았다(C0b 판정서 §7 2와 같음). 키 합집합 규칙(없는 키 = 0)으로 0↔0이라 칸이 생기지 않는다. 비교 전체는 183칸이므로 "0칸 = 입력 오류"에는 걸리지 않는다.

### 3-3. 선택 필드 직접 확인

두 `d0_gate.json`의 `selection`을 직접 열어 확인했다.

| 필드 | C0b | D0′ |
|---|---|---|
| `pp_set_id` | S3 | S3 |
| `pp_config_sha256` | `eb2796dd…080602f` | 같음 |
| `evaluated` | S1, S2, S3, S4 | 같음 |
| `stopped_at` | S4 | S4 |
| `stop_reason` | `gate_failed` | `gate_failed` |
| `s1_failed` | false | false |

### 3-4. 집합별 게이트(같은 값, 참고)

| 집합 | G1 | G2 | G3 | G4 증분 / 누적 | 결과 |
|---|---|---|---|---|---|
| S1 | 통과 | 통과 | 통과 ($H$ 3 → 3, 노출 0) | 25 / 25 | 통과 |
| S2 | 통과 | 통과 (A3 fold 0 0.072 %p, fold 1 2.401 %p) | 통과 | 70 / 95 | 통과 |
| S3 | 통과 | 통과 (S2와 같음) | 통과 | 336 / 431 | 통과 |
| S4 | 통과 | **실패** (A1 fold별 1.006·0.925·1.424·3.392·2.081 %p) | 통과 | 68 / 499 | 실패 |

- 칸 값은 C0b와 비트 단위로 같다(부동소수 표기까지 같음). GPU 비결정성의 흔적은 없었다.

### 3-5. 폭 안 차이

**없음.** 183칸 모두 차이 0이다. 그래서 추정 원인을 적을 차이가 없다.

## 4. 기록 전용 항목

### 4-1. A2(보고 조건) 40칸
- 모두 차이 0이다.
- S1–S3의 G1·G2는 0이다.
- S4 G2 A2 fold별 값은 1.063·1.461·1.369·3.223·1.344 %p다. 게이트 대상이 아니라 보고 값이고, C0b와 같다.

### 4-2. C0b 주 1 경계 박스
crop `NM_20221102(142627)_N00_B10_g1.0_x(10892-11148)`, fold 1, A3, `pipe_band`. 같은 crop에 박스 두 개가 있다.

| 박스 | C0b 점수(4자리) | D0′ 점수(전체 정밀도) | > 0.25 | 범주 | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.25 | **0.25003907084465027** | 예 | d | none | removed_d | removed_d | removed_d | removed_d |
| 2 | 0.2216 | 0.22161488234996796 | 아니오 | 없음 | none | none | none | none | none |

- 박스 1은 0.25를 3.9e-5 넘는다. 4자리 기록에서는 "0.25"로 보여 재집계가 0.25 이하로 잘못 읽었다(C0b 판정서 §6-2 주 1의 G4 S2 70 ↔ 69, G2 A3 0.495 ↔ 0.458 %p 차이의 원인). 실행기 값(70, 0.495)이 규칙값이라는 C0b 판정서의 추론은 재실행 점수와 두 실행의 G4 S2 = 70 일치로 확인됐다(C0b 실행의 원래 점수는 저장되지 않음).
- 박스 2는 문턱 아래이고 범주가 없어 어느 집합에서도 처리되지 않는다.
- 부록 A-1에 따라 이 항목은 기록만 하고 불일치 사유로 쓰지 않는다. 처리도 C0b와 같다.

### 4-3. Y1(보고 조건, 기록만)
- 스크립트는 Y1을 대조하지 않는다. `report.synthetic_apex_hits`·`report.rule_counts_outside_gate`의 Y1 항목과 `synthetic_apex.csv` 500행이 C0b와 같다(verif 판정 후 감사 재집계).

## 5. 쓴 문서·코드 (882e881 기준 blob)

| 대상 | blob |
|---|---|
| `analysis/detection/m33_d0_gate.py` (개정, D0′ 출처) | `1775f9a27f9e798e8fc36364fb723cb836fb22ee` |
| `analysis/detection/m31_d1_common.py` (개정) | `0ba0675c8fc6bd632ed2b1e6794702b60f9b5d72` |
| `analysis/detection/m33_d0r_compare.py` | `ee802e8b8e665264ccc95dc0f9944a8f661b4097` |
| `src/gpr_detection/pipeline.py` | `dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1` (C0a와 같음) |
| `docs/experiments/m33_pp_spec.md` (v0.2.4) | `49d048058dad90b5d4cf7fcc15618874178da3d8` |
| `docs/experiments/m33_cross_domain_plan.md` (v0.2.8) | `c4090534983e25fb94afa6dd2fe42b08ae34f4a4` |
| `docs/experiments/m33_runner_spec.md` (v0.3.6) | `68c657b56d6eb295819258ef0fa6948f6e7d3987` |
| `docs/experiments/m33_c0b_d0_judgment.md` (봉인, 변경 없음) | `2b6c94cd39632382c3be8da83fa0182aa3a1c01b` |

## 6. C1 sha 표에 들어갈 값

| 대상 | C0a·C0b 쪽 (53c4e16, C0b 결과의 출처) | D0′ 쪽 (882e881) |
|---|---|---|
| D0 실행기 `m33_d0_gate.py` blob | `a2741259bddbeaa26daf3b091ac76ea753679918` | `1775f9a27f9e798e8fc36364fb723cb836fb22ee` |
| `m31_d1_common.py` blob | `bdfb7f0f2ff08ced10b7f281d646d717dee34621` | `0ba0675c8fc6bd632ed2b1e6794702b60f9b5d72` |
| `pipeline.py` blob | `dd5e7bcba4397efd732e7bf3ca1c5c92fb2404a1` | 같음 |
| src tree id | `164f09909ad51c1a8d26545cadb0417b1da3acc6` | 같음 |
| `d0_gate.json` sha256 | `107d7b53f30667684baf7c2236bdf338c861309531ff17497e2836193bea08c6` | `16d7d2ade37a79ef24f00603628d864b5ddd396bd468738dbe9ab3f1970aedf1` |
| 대조 | – | `m33_d0r_compare.py` blob `ee802e8b…`, `compare_c0b.json` sha256 `39fac604…`, status `match` |
| `PPConfig` 해시(S3) | `eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f` | 같음 |

## 7. 감사와 남은 일
- verif 판정 후 감사 2026-11-15 완료(차단 0, 기록 6, gates·report 전체와 CSV 전 행 독립 재집계 일치). 결론 "D0′가 S3를 확인했다"는 유지된다.
- C1에서 위 §6 값을 sha 표에 옮긴다. §6의 D0′ 쪽 blob은 882e881 값 그대로 쓴다. C1 1단계(2026-11-16) 도구 보강으로 `m33_d0_gate.py`·`m33_d0r_compare.py`가 다시 바뀌었으므로, C1 커밋 blob은 별도 열로 적는다([PP] 부록 A-1).
- 대조 스크립트를 다시 쓸 일이 있으면(감사 권고, 이번 결론과 무관): base sha 단언, Y1 기록 칸 추가, `judgment` 없는 입력 거부, 칸 수 어긋남을 `mismatch`로 처리. G4 규칙·조건 칸의 기대 수를 규칙으로 정할지(지금은 자료 키에서 셈)도 그때 정한다.
