# ADR-002: 관로 깊이는 학습 회귀가 아니라 물리 기반 쌍곡선 피팅으로 추정한다

- 상태: Accepted (2026-10-29, 사용자 확정)
- 날짜: 2026-10-29
- 관련 드라이버: QA-2(합성 깊이), QA-3(4TU 제186조), QA-4(단위 오용 방지), FR-2, FR-3, CON-5 — 정의는 [README](README.md) §1
- 반복: 1회차 (체크리스트 M0-12)

## 맥락

- 목표는 관로마다 지표에서의 수직 거리(m)다. 모델은 t(x) = (2/v)·√(d² + (x−x0)²), d = v·t0/2이다(CLAUDE.md).
- 깊이 정답이 있는 실측은 4TU뿐이고, 그것도 트레이스가 아니라 **활동 단위** 깊이표(OCR, 잡음 있음)다. 주 검출 데이터인 AIHub·모로코는 물리 스케일이 없다(CON-5, AIHub permittivity 6.0은 자리채움 — `docs/datasets/aihub_71327.md` §2).
- gprMax 합성 20개로 확인한 물리(`docs/research/depth_physics.md` §0, `docs/processing/spec_depth_v1.md` §12):
  - 영점이 가장 큰 단일 오차원이다. 0.5 ns 어긋나면 εr 9에서 2.5 cm.
  - 공기결합(4TU)에서는 지표 반사 피크가 맞는 영점이다(M20 −0.9 cm). 직접파·송신·4TU 메타 규칙은 +8.6~+10.3 cm 깊다.
  - 공기결합에서 직선 광선 곡률로 v를 피팅하면 v'/v 1.11–1.82, 깊이 +0.11~+0.50 m(M20 +47.8 cm)이고 RMS가 작아 **잔차로 실패를 알 수 없다**.
  - 반경을 함께 추정하면 잡음에서 반경 오차가 최대 500 %다(Giannakis 2022, prior_art [10]). R=0으로 원통에 맞추면 v가 1.01–1.14배 과대.
- 판정 기준은 공공측량 작업규정 제186조 깊이 ±30 cm(3 m 이하)다(`docs/datasets/readme_4tu.md` 157행).

## 고려한 대안

| 대안 | 장점 | 단점 | 드라이버 충족 |
|---|---|---|---|
| **A. 물리 기반: 박스 안 점 추출 → RANSAC(x²–t²) → least_squares(R 고정), 데이터 유형별 영점·v 결정표** (채택) | 학습 데이터 불필요, 합성 정답으로 단계별 검증 가능, 편향 부호와 불확도를 설명 가능, 데이터셋이 바뀌어도 메타만 있으면 동작 | 영점·εr 메타 품질에 민감, 규칙이 많음, 박스 품질에 의존 | QA-2 ○, QA-3 ○(보고), QA-4 ◎ |
| B. 학습 회귀(박스·패치 → 깊이) | 실측 사례 R² 0.963, MAE 0.041 m(Štifanić 2025) | 스케일 있는 깊이 라벨이 거의 없음(4TU 활동 단위만), 장비·εr이 바뀌면 재학습, 단위 오용(스케일 없는 영상에서 m 출력) 위험 | QA-3 ✕(라벨 부족), QA-4 △ |
| C. 하이브리드: 학습으로 꼭짓점·점 추출, 깊이는 물리식 | 점 추출 견고성 향상 가능 | 점 라벨이 없음. v1 범위 밖 | 후속 개선 후보 |
| D. 모든 데이터에 곡률 피팅으로 v 추정 | 메타 εr 불필요 | 공기결합에서 +47.8 cm(M20) 실패를 잔차로 못 잡음 | QA-3 ✕ |
| E. (x0, t0, v, R) 동시 추정 | 관경까지 얻음 | 잡음에서 반경 오차 최대 500 % | QA-2 ✕ |

## 결정

1. 깊이는 **물리 기반 쌍곡선 피팅**으로만 낸다. 학습 회귀는 쓰지 않는다. 세부 알고리즘·기본값·시험은 [`spec_depth_v1.md`](../processing/spec_depth_v1.md)가 기준 문서다.
2. 깊이는 **관 상단** 깊이 d_top이다. R을 알면 d_center = d_top + R도 낸다.
3. **스케일 없는 입력**(dt_ns 또는 dx_m 없음: AIHub·모로코)은 m 단위 깊이를 내지 않는다. `status="no_scale"`과 꼭짓점 (trace, sample)만 낸다.
4. 영점과 꼭짓점은 같은 위상 기준(기본 Hilbert 포락선 피크)이다. first break 영점은 쓰지 않는다.
5. **공기결합(4TU)**: 영점은 지표 반사(배경 제거 전 자료로 검출). v는 메타 εr로 고정(`meta_fixed`)하거나 굴절 모델(`refraction_qc`, 공기층 h 포함)로만 구한다. **직선 광선 곡률 피팅은 금지**하고 호출 시 오류로 막는다.
6. **반경 R은 피팅 미지수로 넣지 않는다.** 0 또는 기지 관경/2로 고정한다.
7. 피팅은 gain을 적용하지 않은 갈래에서 한다(README §2-3 두 갈래).
8. 깊이 피팅 대상 클래스는 `pipe_hyperbola`뿐이다. `pipe_band` 깊이는 M2-6 명세로 따로 정한다(ADR-003).
9. 결과에 사용한 규칙(`zero_rule`, `v_rule`), 불확도(`sigma_depth_m`), 편향 메모(`bias_notes`), 플래그(`air_single_event`, `meta_eps_unverified`, `zero_unverified` 등)를 남긴다. 4TU는 메타 t0 두 값으로 낸 깊이를 `alt_depths`에 함께 넣는다.

## 결과

얻는 것
- 합성 정답으로 영점·꼭짓점·v·깊이를 단계별로 검증할 수 있다(spec §11 U1–U8, R1–R13).
- 실패 유형이 상태 값과 플래그로 드러난다. 스케일 없는 입력에서 m 단위가 새지 않는다(QA-4).
- 장비·데이터셋이 바뀌어도 메타(dt, dx, air_gap, εr)만 채우면 재학습 없이 동작한다.

잃는 것 / 비용
- 4TU 정확도가 메타 εr 품질에 묶인다. `eps_r_used`가 곡률 피팅값이면 깊이가 한쪽(깊게)으로 편향된다(spec §9, `sigma_eps_rel=0.15` `[가정]`).
- 지면결합에서 직접파 영점은 1.9–3.9 cm 얕은 편향이 남는다(보정표는 기본 끔).
- 규칙 표(영점 §4, v §7)가 길어 구현·시험 부담이 크다.
- 다중 관(겹친 쌍곡선) 분리는 v1에서 하지 않는다.

위험
- 4TU 정답 기준점(상단/중심) 미정 → 두 가정 모두 보고(ADR-003, U-2).
- 4TU Tx–Rx 간격 s 미확인 → 공기층 h 추정에서 s=0 가정.
- GPR-SD는 영점 미확정(`meta_sample0`, `zero_unverified`), dt 가설 A/B는 곡률 깊이를 바꾸지 않는다(U5).
- 목표 판정률은 미정(U-3) — M2-8 1차 결과를 보고 정한다.
- 4TU 영점 규약은 M2-8 전체 검증 뒤 재검토할 예정이다(근거: [`docs/processing/fourtu_explore.md`](../processing/fourtu_explore.md) — 4TU 영점 사건이 단일 띠, 공기층 ≈ 0).

검증 방법(측정으로 판정)
- M2-5: `tests/depth/` 단위·회귀 시험 전부 통과(시드 0). 판정은 gpr-processing-expert.
- M2-8: 4TU MAE와 제186조 ±30 cm 판정률, t0 두 값 × 기준점 두 가정. 깊이 파라미터(`DepthConfig`)는 4TU로 튜닝하지 않고, 합성에서 정한 기본값으로 13개 현장 전체를 시험에만 쓴다(사용자 결정 2026-10-29).

## 근거 문서

- [`docs/processing/spec_depth_v1.md`](../processing/spec_depth_v1.md) §0–§12 — 계약, 전처리, 영점, 피팅, v 결정표, 시험, analyze 결과
- [`docs/research/depth_physics.md`](../research/depth_physics.md) §0–§4 — gprMax 물리 검토(영점·공기층·관경)
- [`docs/research/prior_art.md`](../research/prior_art.md) §2 — 피팅 방법 비교, R 동시 추정 금지 근거
- [`docs/datasets/readme_4tu.md`](../datasets/readme_4tu.md) — 4TU 메타·깊이표·제186조 판정
- [`docs/datasets/aihub_71327.md`](../datasets/aihub_71327.md) §2 — AIHub 스케일 없음
