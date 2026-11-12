# M2-10 (b) v3 판정 — 선정기 검출 정규화 바닥 재시험 (gprMax 새 실현 9모델, 시드 40)

**판정: 결정 순위 5 — 판정 후보 B_full·B_min 모두 불충분. U1 = 현행(current) 바닥 규칙 유지.**
- R(결함 재현)은 성립했다. R1·R2 8짝, e 0.775–0.925.
- B_full·B_min은 F1·F2·F4·F4b·F5-B를 통과했고 **F3 하나만** 실패했다. 실패 조건은 S4N_D13_S042 ρ 20이고, 성공률은 0.75 대 current 0.90이다.
- **F3-S(사전 등록 진단): B_full·B_min 모두 label H-C.** 실패 조건의 gap_S는 −0.05다. 곧 기준 성질(능선 이동 시작점 수)로 분류된다. 판정과는 무관하다.
- 보고 후보 C_none은 F5-B 실패로 순위 5다.
- 기준은 봉인 명세 [`m210b_selector_floor_v3.md`](m210b_selector_floor_v3.md)(유효 봉인 `77f5f96`)와 그것이 이어받는 [`m210b_selector_floor_v2.md`](m210b_selector_floor_v2.md) v2.1·[`m210b_selector_floor.md`](m210b_selector_floor.md) v1.4·[`m210b_harness_interpretation.md`](m210b_harness_interpretation.md)다.
- **결과를 본 뒤 기준·셈법·결정표를 바꾸지 않았다(v3 §7-2).**
- 수치는 `docs/processing/m210b_v3_judgment.json`(판정 스크립트 출력), `data/processing/m210b_s4s6_v3/runs_v3.csv`·`run_info.json`, source_guard 상태 파일, `h5_compare.py` 보고서에서만 옮겼다. 판정서 대조용 수집은 `analysis/processing/m210b_v3_judgment_check.py` → `docs/processing/m210b_v3_judgment_check.json`(읽기 전용)이다.
- 판정자: gpr-processing-expert, 2026-11-12. 4TU 자료는 쓰지 않았다.

## 1. 판정서 첫머리 목록 (v3 §7-3 X7, 11항목 — 모두 있음·맞음)

| # | 항목 | 값 | 근거 |
|---|---|---|---|
| 1 | v3 봉인 커밋과 파일별 blob | `77f5f961d8d05d5bcccef026de9ea70a265fc332`(2026-11-12 08:51:45 +09:00). run_info `seal.sealed_files` 113개 = `git ls-tree -r 77f5f96`(불일치 0). 실행 시작·끝 모두 113/113 일치. 하네스 `V3_SEALED_FILES` 18개가 모두 113개에 들어 있다. 파일별 blob 표는 아래 | `m210b_v3_judgment_check.json` |
| 2 | `--expected-seal-commit` | `77f5f96` = 봉인 커밋(`rank1_identity.seal_commit_is_expected` 참) | 판정 json |
| 3 | 판정 스크립트 blob | `git hash-object analysis/processing/m210b_v3_judge.py` = `a843b5c0906d685b9ee7072d9cd6e902cf3a5a82` = 봉인 blob | check json |
| 4 | 실행 HEAD·dirty 전후 | 시작(09:35:21)·끝(run_info `seal_at_end.taken_at` 10:12:01, `finished_at` 10:12:02) HEAD `77f5f961…`, dirty 0, git 조회 실패 0 | run_info `seal`·`seal_at_end`(아래 공개 참고) |
| 5 | `runs_v3.csv` SHA-256 | `f29aa345962204151da7115c5a6f240ab534ed8c6c4522035640e7113ec50411`(run_info 기록 = 판정 시 재계산), 12,880행 | check json |
| 6 | 시각 순서 | 봉인 08:51:45 < 하네스 로그 폴더 생성 08:51:59 < gprMax 러너 로그 첫 생성 08:53:18(`runner_logs/BG_N_S012.log`·`BG_PF_S012.log`) < 마지막 `.done` 09:32:57(`S6P_NP_S038_J6`, 17단계) < 참조 h5 snapshot 09:33:50 < 내보내기 09:33:58–09:34:29 < 하네스 시작 09:35:21(판정 스크립트 단언 `seal_before_harness_start` 참) < 하네스 끝 10:12:01–10:12:02 | 파일 시각(수기 대조), run_info |
| 7 | 시드 겹침 | 0. 묶음 1(101)·v2(201)·시제품(1,710) 시드 집합과 단언. v3 잡음 시드 481개(20281120–20286259), 공식 밖 행 0, 순잡음 20281199 | run_info `seeds`, `rank1_seeds` |
| 8 | 참조 h5 출처·SHA-256 | 아래 표(11개). BG 8개 배열 대조: **8/8 `bit_identical`**(`rxs/rx1/Ez`, dt·Iterations 같음). 파일 SHA-256은 6개가 같고 2개가 다르다(BG_PF_S042 점검 `03878B33…`/새 `4F1016A4…`, BG_PF_S048 `AAC19FC7…`/`8B18C824…`). 원인은 확인하지 않았다(배열은 비트 동일; HDF5 메타데이터 차이로 추정, 6/8은 파일 해시도 같음). 공개 항목이며 순위 1 사유가 아니다 | `h5_compare.py` 보고서, source_guard 상태 |
| 9 | `rank2_checked_models` | S4N_D14_S038·S4N_D13_S042·S4N_D13_S012(짝 대조 3개). 관 없는 줄은 S6P_NP_S038_J6 하나(`roles_fixed.pass` 참). F 표적 보고값(`rank2_f_targets_reported`): S4N_D16_S045 0.675, S4N_D15_S048 0.675 | 판정 json |
| 10 | `blank_strong_start_in_current_s4n_rows` | **0**(공개 항목) | 판정 json `F3S_by_condition` |
| 11 | 실행 전 감사 기록 1–11 | §2 | – |

- **공개(항목 4 관련 운영 실수, 판정 영향 없음)**:
  - `post_run.txt`는 하네스가 10:12:01에 쓴 끝 정체성 파일이다. 메인 세션이 10:12:22에 이를 다른 형식(`HEAD=… dirty=0 at=…` 한 줄)으로 덮어썼다. 그래서 하네스판 끝 기록은 run_info `seal_at_end`에만 남는다(HEAD = 봉인, dirty 없음, git 실패 0).
  - `pre_run.txt`는 메인 세션이 09:34:58에 먼저 썼고, 하네스가 시작 때 자기 형식으로 덮어썼다.
  - 판정 json `identity.post_run`의 값은 메인 세션 기록이다.
  - 판정 단언(`head_same_start_end`, `dirty_0_end`)은 run_info `seal_at_end`로 이뤄졌으므로 판정에는 영향이 없다.
  - **다음 판 운영 메모**: 하네스 출력 폴더의 `pre_run.txt`·`post_run.txt`는 메인 세션이 쓰지 않는다(메인 세션 기록은 다른 파일 이름으로).

참조 h5 11개(source_guard snapshot 09:33:50, 하네스 뒤 verify pass, 변경 0):

| 파일 | 출처 | SHA-256 |
|---|---|---|
| `synthetic_gprmax_s4s6/h5/FS.h5` | 묶음 1 | `ae2c3844077f9ba566c443a7215a8f911f1950261386c0a1597b1b599f943bfa` |
| `synthetic_gprmax_s4s6_v2/h5/BG_N_S045_merged.h5` | v2 | `6ebd7fd449c7f65134c6aaa6a50c9112733d90ffa0219e6ee06220dad67b3eac` |
| `synthetic_gprmax_s4s6_v2/h5/BG_PF_S045_merged.h5` | v2 | `37201d18363abeb12969525ccaabdc1d85d8a4f40e00b884ca96bceb1bed6896` |
| `synthetic_gprmax_s4s6_v3/h5/BG_N_S012_merged.h5` | v3 전체 실행 | `e19934d5c0d9e5a0548ecdfd0d12dfb7c43892971862f5e8e7e48660fc908a80` |
| `synthetic_gprmax_s4s6_v3/h5/BG_N_S038_merged.h5` | v3 | `791f1baa6c79cfe0358aed4ba479b24b2082ce3bde814c08a9ed78cdb2ed8a36` |
| `synthetic_gprmax_s4s6_v3/h5/BG_N_S042_merged.h5` | v3 | `4f4cb6a82283ffb3b54197b60297cd3238bbded79d68318e8a3fac95a84a344d` |
| `synthetic_gprmax_s4s6_v3/h5/BG_N_S048_merged.h5` | v3 | `ea52c2f1903609a37f8debb7be26d27373769fc2ae022c2e98e9fde8d74fbf99` |
| `synthetic_gprmax_s4s6_v3/h5/BG_PF_S012_merged.h5` | v3 | `8a341def0826cd1d14f60465e67756f93a42cee3d062cc2c55bb7690543f9e4b` |
| `synthetic_gprmax_s4s6_v3/h5/BG_PF_S038_merged.h5` | v3 | `8dac119d0308a5ca00822dc922c5f678eb0bf86f249aa1f26973350b87ab4ccf` |
| `synthetic_gprmax_s4s6_v3/h5/BG_PF_S042_merged.h5` | v3 | `4f1016a414a9e066d12542b777e1a698330379c38fb0333ac0e9e038125e7346` |
| `synthetic_gprmax_s4s6_v3/h5/BG_PF_S048_merged.h5` | v3 | `8b18c824da07029ea2bc8a27cb9c25cff811afa8a3af5e2ee66faf77875c6c05` |

- 점검 산출 BG 8개 h5·`.log` 8개는 전체 실행 전 `data/synthetic_gprmax_s4s6_v3/pilot_prerun_h5/`로 이동해 보관했다(mtime 01:44:56–01:47:48 유지, X8).
- 내보낸 입력 38개(npy·json, `pilot_run.log`·`manifest.csv` 포함)는 source_guard snapshot(09:34:40) 뒤 하네스가 끝나고 verify했다(pass, 변경 0).

봉인 파일 113개 — `77f5f96` 시점 blob(run_info `seal.sealed_files.<경로>.blob_at_seal` 그대로; `git ls-tree -r 77f5f96`과 대조해 불일치 0):

| 경로 | blob(SHA-1, LF 기준) |
|---|---|
| `analysis/processing/fourtu_apex_candidate_select.py` | `7cc2bf2b3db80a626bd6f81fabc00e53b9805d8a` |
| `analysis/processing/m210b_floor_proto.py` | `860cd3ada5b898e611ea24b3e16d4725105a7776` |
| `analysis/processing/m210b_floor_rules.py` | `bad0f95cdcebe3d193fb7aad6511dbbafcac8327` |
| `analysis/processing/m210b_lambda_pilot.py` | `aee1b0a24ffde10ef37c3f1c65a43ca52347d4aa` |
| `analysis/processing/m210b_s4s6_pipeline.py` | `ae63d3f04327628ad36425a10b31d8c9ba09798a` |
| `analysis/processing/m210b_v3_judge.py` | `a843b5c0906d685b9ee7072d9cd6e902cf3a5a82` |
| `analysis/synthetic/s4s6/inputs/BG_N_S005.in` | `0375e7039dd4c91badf58bf315e6e55e2e4d1b95` |
| `analysis/synthetic/s4s6/inputs/BG_N_S010.in` | `1d249114f069ba0067efc6c79a2a814052e92130` |
| `analysis/synthetic/s4s6/inputs/BG_N_S030.in` | `ff07589b3827ac7043f6e4816eb5185aaf6e8828` |
| `analysis/synthetic/s4s6/inputs/BG_N_S040.in` | `e8c317eb3c10e0f36f0506e20536c1c1e4c5fcb6` |
| `analysis/synthetic/s4s6/inputs/BG_PF_S010.in` | `0577b1897c9809761351ea5920097deb75ac9831` |
| `analysis/synthetic/s4s6/inputs/BG_PF_S030.in` | `a54c95d7ecfcee21fd1c8dbd1944a5ee2f9e90d1` |
| `analysis/synthetic/s4s6/inputs/BG_PF_S040.in` | `194a8b238a595d606c8d2e33f63adab0058bcf97` |
| `analysis/synthetic/s4s6/inputs/FS.in` | `e826698de4e44c0a357e722802dde5121629d2df` |
| `analysis/synthetic/s4s6/inputs/S4N_D12_S005.in` | `6b15f781bb1ef425f932e4a07372b310fb31e4af` |
| `analysis/synthetic/s4s6/inputs/S4N_D12_S010.in` | `c5c02a274190f7d94fd4b96198b588924444fb24` |
| `analysis/synthetic/s4s6/inputs/S4N_D12_S030.in` | `bb56632450fd07fd17608c0291c45d1f4523643e` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S005.in` | `2eb4f0c91b2bf204674a2ac5f29e5e9ab4758e7a` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S010.in` | `ad36b3e971d36088c9f8c63ecbc7d2f974a9fe35` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S030.in` | `23faa51021c361bc9dc661018cabf1b2529b17c0` |
| `analysis/synthetic/s4s6/inputs/S4N_D15_S040.in` | `d84eb3099a22fe3adc9bf5d4bc32609e5e3b64e3` |
| `analysis/synthetic/s4s6/inputs/S4S_D12_S005.in` | `77b7322f5c3191891af894b98e61504ebf457d25` |
| `analysis/synthetic/s4s6/inputs/S4S_D12_S010.in` | `5a1d5a55b44522ec4fe7cef7a9021259999b1d0c` |
| `analysis/synthetic/s4s6/inputs/S4S_D12_S030.in` | `6ebdaa966c04bfbf686a26c222f929f0fdadc16d` |
| `analysis/synthetic/s4s6/inputs/S4S_D15_S005.in` | `628c11300bd6e576f28a3abfbaae186fd0f32db1` |
| `analysis/synthetic/s4s6/inputs/S4S_D15_S010.in` | `90aad372bd44ddf7933d2e7ff46baaa69e2f49b3` |
| `analysis/synthetic/s4s6/inputs/S4S_D15_S030.in` | `e8a8fc532e777b3e2bf25d56aa4abcbc5eab3f17` |
| `analysis/synthetic/s4s6/inputs/S6P_D12_S010.in` | `9c28520312dbd954637909fb6cc287bc5695eefd` |
| `analysis/synthetic/s4s6/inputs/S6P_D15_S010.in` | `dbcd7758aa2d552bfac501aae50db8fe555f4357` |
| `analysis/synthetic/s4s6/inputs/S6P_D15_S030.in` | `46931dc4b75a65700fca7fa7cdb1f2dca58f9de8` |
| `analysis/synthetic/s4s6/inputs/S6P_D15_S040.in` | `fcbbe53cbcc351872d5f8d34f4267628bcdadea5` |
| `analysis/synthetic/s4s6/inputs/S6P_NP_S010.in` | `0f7a85ddf3a7e75ba676d0978df305812a44045d` |
| `analysis/synthetic/s4s6/inputs_v2/BG_N_S045.in` | `756a03b9171f01b4fe5e5e8102fdfae6fe342e57` |
| `analysis/synthetic/s4s6/inputs_v2/BG_PF_S045.in` | `9eabcafac85157cfc88e74f6bd1924e22be0886b` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D13_S010.in` | `9debe0018303ad31a017d634231ae1744366b37a` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D14_S040.in` | `94a91bba2afba8ea6d7b92adf09106077c8c5a2b` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D15_S045.in` | `a0192ef0fd6eb3cc4ff26545e50e5b588958c5a3` |
| `analysis/synthetic/s4s6/inputs_v2/S4N_D16_S040.in` | `21c4bf67adbc7c4b26373e0c7d96e94a316f9cd3` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D13_S010_J5.in` | `ce404228b5665052cfb03239c2a1b1c05695e7ab` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D14_S040_J2.in` | `f557f9f29d4c46ee46f518e2db312454b263e163` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D15_S045_J4.in` | `f896f180b027e8944bbf1b33fd7ccc19249eae85` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_D16_S040_J3.in` | `b7e2bf6779b893ee4a22084efd7ca682bc13798d` |
| `analysis/synthetic/s4s6/inputs_v2/S6P_NP_S040_J2.in` | `5fe1c3c23cf911bc3472fb97fe56778102f9adb6` |
| `analysis/synthetic/s4s6/inputs_v3/BG_N_S012.in` | `9e064b890aab39ed1718ad29a6e86abef324baae` |
| `analysis/synthetic/s4s6/inputs_v3/BG_N_S038.in` | `ecdca8a98dd972431fca4fc5fd2bfb8abff793e2` |
| `analysis/synthetic/s4s6/inputs_v3/BG_N_S042.in` | `e4c5d0b14a2115190576dff0b81bf119fbb07d82` |
| `analysis/synthetic/s4s6/inputs_v3/BG_N_S048.in` | `b9f0c5dbc866c705c3eaffdd6717998979eae854` |
| `analysis/synthetic/s4s6/inputs_v3/BG_PF_S012.in` | `1720ce117ed7057b274c1c54bc273b725506c712` |
| `analysis/synthetic/s4s6/inputs_v3/BG_PF_S038.in` | `9c943095eba3a45a6cc7b18db3d494319c5dae39` |
| `analysis/synthetic/s4s6/inputs_v3/BG_PF_S042.in` | `def42e2803bb9632808b2f68b1c5cb1a52898d33` |
| `analysis/synthetic/s4s6/inputs_v3/BG_PF_S048.in` | `45c88a2914dbb003bb74006d675cb16dac00cba4` |
| `analysis/synthetic/s4s6/inputs_v3/S4N_D13_S012.in` | `bc4a064196a42fb8df0296262caa4ef91199ad94` |
| `analysis/synthetic/s4s6/inputs_v3/S4N_D13_S042.in` | `13417410893dfd29c2b6ecb94f42ccb656b66c8a` |
| `analysis/synthetic/s4s6/inputs_v3/S4N_D14_S038.in` | `56db90c530bb5264a651ee5b50ff48764d794327` |
| `analysis/synthetic/s4s6/inputs_v3/S4N_D15_S048.in` | `20239631e144a39772ff8af61ebd18686293f78c` |
| `analysis/synthetic/s4s6/inputs_v3/S4N_D16_S045.in` | `7488cb856d086b2bc5a37bf97cec3c760efdfb70` |
| `analysis/synthetic/s4s6/inputs_v3/S6P_D13_S012_J8.in` | `23514e81f83444178e8a46bae9f46898aecfbb7c` |
| `analysis/synthetic/s4s6/inputs_v3/S6P_D13_S042_J7.in` | `b96cf4f1bf98bb9a52209a6763bbc55c4bbb5b91` |
| `analysis/synthetic/s4s6/inputs_v3/S6P_D14_S038_J6.in` | `a17d196730093f0ca2c23fdece954c9cd6acbb6c` |
| `analysis/synthetic/s4s6/inputs_v3/S6P_NP_S038_J6.in` | `eaa065377f47b43576b2aa92806bc77754ab942d` |
| `analysis/synthetic/s4s6/models_s4s6.json` | `73d8bdce152577359cfe30a51aeed2e1fa24d9fa` |
| `analysis/synthetic/s4s6/models_s4s6_v2.json` | `d1d9f52315ec636756208563d791af38b94867a0` |
| `analysis/synthetic/s4s6/models_s4s6_v3.json` | `97d2c84e1a40e29beb5d2f5caef4cf6ea77c8a1a` |
| `analysis/synthetic/s4s6/pilot/L_S6P_NP_S040_T15.in` | `887c18142d76b293b07d72c7b5ff1002df3bd043` |
| `analysis/synthetic/s4s6/pilot/P_S4N_D15_S030.in` | `8e4ca906468692776bca137baa4b7d509284bcaa` |
| `analysis/synthetic/s4s6/pilot/P_S4N_D15_S040.in` | `65337f98a2631b63bb396f0d1af4e8da5fcd4dcb` |
| `analysis/synthetic/s4s6/pilot/P_S6P_D15_S030.in` | `66c299964a86ce339f1bf5cede7bb8aa2f1782a7` |
| `analysis/synthetic/s4s6/pilot/P_S6P_D15_S040.in` | `690bcd669cc6a7a7383624ca379c7584bc2ca9d8` |
| `analysis/synthetic/s4s6/pilot/P_S6P_NP_S040.in` | `5f3f70dc8b85fbe4a2a8ad3329e56fa7790fac01` |
| `analysis/synthetic/s4s6/pilot/lambda_S6P_D15_S040.json` | `8c6018e9b9370f0da3af17d1819833b0caeac3c1` |
| `analysis/synthetic/s4s6/pilot/pilot_check.json` | `fa75f7b4477353a0696b76b42dff0d05a24fba34` |
| `analysis/synthetic/s4s6/pilot/pilot_envelopes.png` | `ca0c0499756ca6b237e2ee8f191c2f86306f4d1d` |
| `analysis/synthetic/s4s6/pilot_v2/L_S6P_NP_S040_J2_T15.in` | `c058e7f252d60009553f3fdf5b5f98fb09dfa405` |
| `analysis/synthetic/s4s6/pilot_v2/L_S6P_NP_S040_J3_T15.in` | `aa378c13ca9319f55553a522dab58c16daf8ea0a` |
| `analysis/synthetic/s4s6/pilot_v2/L_S6P_NP_S045_J4_T15.in` | `2e890a5d722d6e4fdfbece822523bb9e2bd8f159` |
| `analysis/synthetic/s4s6/pilot_v2/P_S4N_D14_S040.in` | `c9d5dde5f1167f7084ad2b0247de93b18857008f` |
| `analysis/synthetic/s4s6/pilot_v2/P_S4N_D15_S045.in` | `9f3626acf9614d7cc207467d99a56195c302a9ce` |
| `analysis/synthetic/s4s6/pilot_v2/P_S4N_D16_S040.in` | `8fe761451a6203e44c38708ac69554a46eb6ce93` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_D14_S040_J2.in` | `0f0bfc2f253997efa3c4a0cd3015179f6431db02` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_D15_S045_J4.in` | `fe57879d2a5df2083f69ae442e0ec7dab9218ac6` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_D16_S040_J3.in` | `840e48bcfaa9637f12effe1f7a33bb5b2ab848c8` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_NP_S040_J2.in` | `8333256903b3b9b2149a92998210cb43a9352e76` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_NP_S040_J3.in` | `eb199c538616a20749a09c62bdd96f42e00b37ba` |
| `analysis/synthetic/s4s6/pilot_v2/P_S6P_NP_S045_J4.in` | `8a81376d1d1322a56a9149ea49ebf30205c67092` |
| `analysis/synthetic/s4s6/pilot_v2/lambda_S6P_D14_S040_J2.json` | `cfda7bb74aa4ffc4a15da3dbb028bcc7ceb12049` |
| `analysis/synthetic/s4s6/pilot_v2/lambda_S6P_D15_S045_J4.json` | `f9d403f5cc9b10d9b7bf178c93ac157e2bf6744e` |
| `analysis/synthetic/s4s6/pilot_v2/lambda_S6P_D16_S040_J3.json` | `db142c3a36f3342da4f662fa70e55c2968e01a90` |
| `analysis/synthetic/s4s6/pilot_v3/L_S6P_NP_S038_J6_T15.in` | `64e7f8a6720729c9621299edb30ab9f783b02645` |
| `analysis/synthetic/s4s6/pilot_v3/L_S6P_NP_S042_J7_T15.in` | `fcc91ae1a8e1b06041e4e366e2bc6bb5f04604f6` |
| `analysis/synthetic/s4s6/pilot_v3/P_S4N_D13_S042.in` | `8dea8ad243601da3a79aed6b034cc017ba7b7ff0` |
| `analysis/synthetic/s4s6/pilot_v3/P_S4N_D14_S038.in` | `7d51eecbfe8cfabefedad1d785f8f8da910c3063` |
| `analysis/synthetic/s4s6/pilot_v3/P_S4N_D15_S048_X5.in` | `4526eb8526e0781daddeca400a0e01e6037b47d6` |
| `analysis/synthetic/s4s6/pilot_v3/P_S4N_D16_S045_X5.in` | `c446d91cb038ed883c12ea29952a65082db4faef` |
| `analysis/synthetic/s4s6/pilot_v3/P_S6P_D13_S042_J7.in` | `77dd18649f192e29c5f8020f8ff997388899441e` |
| `analysis/synthetic/s4s6/pilot_v3/P_S6P_D14_S038_J6.in` | `f36fe669d56d977750a42051be7596106781d391` |
| `analysis/synthetic/s4s6/pilot_v3/P_S6P_NP_S038_J6.in` | `f966166cfd950268c67f54840548126b095f2487` |
| `analysis/synthetic/s4s6/pilot_v3/P_S6P_NP_S042_J7.in` | `df074989af2adaddd9bab21c393e238f8313d248` |
| `analysis/synthetic/s4s6/pilot_v3/lambda_S6P_D13_S042_J7.json` | `8c4f7f91c8f55f07269a52595e5d999ba219d5f3` |
| `analysis/synthetic/s4s6/pilot_v3/lambda_S6P_D14_S038_J6.json` | `5cdd7d49884f8fc48647708b6cb7e8f6f2d159bc` |
| `analysis/synthetic/s4s6/run_s4s6.cmd` | `b561467dbd7346bc1f5e74bc24592a106ec13870` |
| `analysis/synthetic/s4s6/run_s4s6_v2.cmd` | `2c3d95e361a07b358c0a83831d0d4b97a84c927f` |
| `analysis/synthetic/s4s6/run_s4s6_v3.cmd` | `701941019d6ea807d513a64044658a5d424dc413` |
| `docs/processing/m210b_a0_S4N_D15_S030.json` | `ce4f33b5ae20ff9400ee3aaf82157d9d1f0cfc70` |
| `docs/processing/m210b_a0_S6P_D15_S030.json` | `f6ff56e0e43432049b5e386aea03faf2c5e27db1` |
| `docs/processing/m210b_floor_proto.json` | `a6e9db068d28e788677760fadd7b4ed23ef95b33` |
| `docs/processing/m210b_harness_interpretation.md` | `1598818a6a76128a722e1e55da411a97b56bf719` |
| `docs/processing/m210b_selector_floor.md` | `64004278548f4091f0801035745af03fca1541ec` |
| `docs/processing/m210b_selector_floor_v2.md` | `72e1ee0e31b76c4e1dde2243412eb2cab215b5cd` |
| `docs/processing/m210b_selector_floor_v3.md` | `7d1fd6c46add3edc3973e65bcf2c2e1ea2a26e4e` |
| `scripts/synthetic/export_dataset.py` | `51d8cf2ae802e41d27d5ddf739b0740c162b1b81` |
| `tests/test_m210b_s4s6_pipeline.py` | `882217a66e155df8cad75cec2f4147a042ea7732` |
| `tests/test_m210b_v3_judge.py` | `cfa1a598ad04bbb905e6609fcadfdf12dd0994e6` |
| `tests/test_synthetic_export.py` | `1afc1b131a751074e54323ca7a6a5656044297d9` |

## 2. 실행 전 감사 기록 1–11 (v3 §7-3 항목 11)

| # | 분류 | 기록 | 이 판정에서 |
|---|---|---|---|
| 1 | S | seal_check `seal_missing` 4건(git blob 봉인이라 설계상) | 봉인은 blob 대조로 했다(항목 1). 4문서(v3·v2.1·v1.4·해석 문서) blob이 기대값과 같다(`rank1_identity`: v3 문서·v1.4·해석 문서·v2 명세 blob 모두 참) |
| 2 | S | CRLF·LF 작업 사본(`pilot_v3` lambda json 2개 CRLF, `run_s4s6_v3.cmd` LF) | blob 대조 결과 일치(봉인 파일 113/113) |
| 3 | E | gate_scan `empty_pass_not`(하네스 1977행) 오탐 | 오탐이다. 하네스 가드 `guards.pass` 참, failures 없음 |
| 4 | I | placeholder_scan `{S}` 5건은 LaTeX `\text{S}` 오탐 | 오탐 |
| 5 | E | F3-S 빈칸은 불충족이라 $r_\text{cur,S}$를 낮춘다(H-C 쪽으로 기움, 보수적이지 않음). 잠재 위험 | `blank_strong_start_in_current_s4n_rows` = **0**. 영향 없음 |
| 6 | I | 판정 스크립트는 자기 blob·`protocol`·`sealed_files ⊇ V3_SEALED_FILES`를 단언하지 않음 | 수기 확인 셋: 판정 스크립트 blob = 봉인 blob(항목 3), run_info `protocol` = `"v3"`, `V3_SEALED_FILES` 18개가 모두 봉인 파일 113개에 들어 있음(빠진 것 0) |
| 7 | I | MKL_CBWR는 기록만 | run_info `mkl_cbwr` = `COMPATIBLE`(numpy 2.5.4) |
| 8 | T | X6(결정 순위 2의 적용 범위)은 F 표적의 점검 진폭($A_0$ 원신호 포락선 0.0047–0.0050)을 본 뒤 확정됐다. 성공률은 보지 않았다 | 공개한다. 이 결정으로 순위 2는 짝 대조 3개로만 판단됐다. F 표적의 ρ 20 Δ 0 current 성공률은 0.675·0.675로 하한 0.5를 넘는다. 따라서 X6이 없었어도 순위 2는 해당하지 않았다 |
| 9 | T | R1의 $\Lambda_\text{est}(0)$ 15.79, 여유 약 5 % | 전체 실행 M0에서 R1 $\Lambda(\Delta_e)$는 Δ 0이 15.77, Δ −1이 13.70이다(> 10). $\Lambda(+2.35)$ 3.37 ≤ 10이고 바닥 비율 1.0이라 **G 성립**(R 짝 4개) |
| 10 | I | `pilot_run.log`가 입력 폴더 맨 위에 있어 입력 해시 대상 | 입력 해시가 실행 전후 같다(`inputs_changed_during_run` 빈 목록). source_guard verify(38개)도 pass |
| 11 | I | 줄눈 시드 J6–J9(20261116–20261119)는 시제품 잡음 시드 범위 안이나 별도 난수 흐름 | 줄눈 시드는 잡음 시드 겹침 대상이 아니다. 잡음 시드 겹침은 0이다(항목 7) |

## 3. 순위 1 단언과 P-F1 (모두 참)

- `rank1_identity`의 모든 항목이 참이다: 봉인 파일 목록·blob 일치, 봉인 커밋 기대값, 실행 HEAD = 봉인, v3 문서 blob, git 조회, v1.4·해석 문서·v2 명세·바닥 규칙·래퍼 blob, HEAD 시작 = 끝, dirty 0(시작·끝), 봉인 < 하네스 시작, runs SHA-256, 시드 겹침 0, 하네스 가드, 래퍼 회귀(4TU 두 csv 바이트 동일).
- P-F1(v3): 판정 행 11,520, 칸 288개 빠짐·남음 없음, 칸마다 시드 정확히 40개(F4b 칸 포함). `n_runs` 12,880 = 기대 = run_info, 영점 실패 0. 모델 집합 = 기대 9개 ID. 관 모델 `L_noise_free_current`·`apex_trace` 이상 0. 관 모델 판정 행 `accepted_apex`·`floor_held_at_apex` 빈칸 0. 역할 고정(`roles_fixed`) 참.
- **W10형 보정**: NP의 `calibration_pair` = D14_S038이고, `sigma_raw`(ρ 10 0.012918, ρ 20 0.0064589)가 R1 보정값과 같다.

## 4. R — Q·G·C·e (current, 수락 단계, 판정 열 `rho_real`·`Lambda`)

| S6P 조건 | ρ | π | 성공률 e→l | e | Q(ρ_real 최소) | Λ(Δe) / Λ(Δl) | Δe 바닥 비율 | G | C(대조 차) | R 짝 |
|---|---|---|---|---|---|---|---|---|---|---|
| **S6P_D14_S038_J6**(R1) | 10 | (0, +2.35) / (−1, +2.35) | 0.000→0.825 | **0.825** | 참(17.99) | 15.77 · 13.70 / 3.37 | 1.0 | **참** | 참(0) | **예** |
| 〃 | 20 | 두 짝 | 0.000→0.875 | **0.875** | 참(35.72) | 〃 | 1.0 | 참 | 참 | **예** |
| **S6P_D13_S042_J7**(R2) | 10 | 두 짝 | 0.000→0.775 | **0.775** | 참(19.24) | 17.93 · 15.58 / 3.75 | 1.0 | 참 | 참 | **예** |
| 〃 | 20 | 두 짝 | 0.000→0.925 | **0.925** | 참(38.07) | 〃 | 1.0 | 참 | 참 | **예** |
| S6P_D13_S012_J8(P3) | 10 / 20 | 두 짝 | 0.525→0.525 / 0.80→0.80 | 0 | 참(12.06 / 24.00) | 0.20 · 0.18 / 0.06 | 0.0 | 거짓(`lambda_e_le_10`, `floor_frac_lt_0.5`) | 참 | 아니오 |

- R 성립(R 짝 8개) → 결정 순위 4/5(규칙별).
- 순위 2(짝 대조 S4N ρ 20 current Δ 0 < 0.5)는 해당 없다. D14_S038 0.85, D13_S042 0.90, D13_S012 0.85다.
- 순위 3도 해당 없다.
- `G_excluded_no_escape`·`lambda_rounding_check`는 빈 목록이다. 모든 바닥 영역 짝이 늦은 뮤트에서 바닥을 벗어났고(V9), |Λ − 10| < 1e-3인 짝은 없다.
- current는 R1·R2에서 Δ +1.0에서도 이미 회복한다(0.825·0.875·0.775·0.925). R1·R2의 Λ(+1.0)은 8.41·9.91 ≤ 10이고, 이는 기작(§2-2)과 같은 방향이다(보고).

## 5. 후보별 F1–F5·F4b (수락 단계; 기준 = current)

| 규칙 | F1 분리 | F2 회복(R 짝) | F3 대조 무손실 | F4 거짓(Δ 0) | F4b 경사 잡음(Δ −1 / 0 / +1 / +2.35, X 대 cur) | F5-B(S1) | 순위 |
|---|---|---|---|---|---|---|---|
| **B_full**(판정) | 통과 | 통과(R1 0.825·0.875, R2 0.775·0.925 ≥ current Δl − 0.1) | **실패**: S4N_D13_S042 ρ 20 0.75 대 0.90 | 통과 | 0.05/0.05/0.05/0.05 대 0.00/0.00/0.00/0.05(통과) | 통과: 1,156쌍 악화 36·개선 100 | **5** |
| **B_min**(판정) | 통과 | 통과(R2 ρ 10 0.80) | **실패**(같은 조건, 0.75 대 0.90) | 통과 | 0.15 ×4 대 0.00/0.00/0.00/0.05(통과) | 통과: 1,154쌍 악화 32·개선 100 | **5** |
| C_none(보고) | 통과 | 통과 | 통과 | 통과 | 0.425 ×4 대 0.00/0.00/0.00/0.05(통과, 문턱 +0.5 안) | **실패**: 1,232쌍 악화 26·개선 6 | 5 |

- F4b는 Δ 0으로 판정했다(봉인 코드의 읽기, 묶음 1·v2와 같음). 네 Δ 모두 같은 결론이다.
- F3 경계(보고): S4N_D14_S038 ρ 20은 B 0.75 대 current 0.85로 차가 정확히 0.10이다. 문턱(≤ 0.1)이라 통과다.
- 나머지 짝 대조·F 표적 S4N의 ρ 20 Δ 0(B_full = B_min 대 current):

  | 조건 | B | current |
  |---|---|---|
  | D13_S012 | 0.825 | 0.85 |
  | D16_S045 | 0.65 | 0.675 |
  | D15_S048 | 0.70 | 0.675 |

- F5-B 범위별(보고, B_full):

  | 범위 | 쌍 | 악화 | 개선 |
  |---|---|---|---|
  | S2(R 조건) | 272 | 0 | 0 |
  | S4N만 | 728 | 16 | 72 |
  | G 밖 S6P | 156 | 20 | 28 |

  판정 json의 읽기(`reading`)는 "B_full better"다.
- U1: B_full 5, B_min 5 → **current**.

## 6. F3-S — 시작점 수를 맞춘 진단 (사전 등록 해석, 판정 무관)

| S4N 조건 | ρ | $r_\text{cur}$ | $r_\text{cur,S}$ | $r_X$(B_full = B_min) | gap | gap_S | F3 | 분류 |
|---|---|---|---|---|---|---|---|---|
| S4N_D14_S038 | 10 / 20 | 0.45 / 0.85 | 0.425 / 0.725 | 0.375 / 0.75 | 0.075 / 0.10 | 0.05 / −0.025 | 통과 | – |
| **S4N_D13_S042** | 10 / **20** | 0.45 / **0.90** | 0.375 / **0.70** | 0.35 / **0.75** | 0.10 / **0.15** | 0.025 / **−0.05** | 통과 / **실패** | – / **H-C** |
| S4N_D16_S045(F 표적) | 10 / 20 | 0.475 / 0.675 | 0.45 / 0.625 | 0.45 / 0.65 | 0.025 / 0.025 | 0.00 / −0.025 | 통과 | – |
| S4N_D15_S048(F 표적) | 10 / 20 | 0.375 / 0.675 | 0.35 / 0.60 | 0.375 / 0.70 | 0.00 / −0.025 | −0.025 / −0.10 | 통과 | – |
| S4N_D13_S012 | 10 / 20 | 0.45 / 0.85 | 0.45 / 0.775 | 0.425 / 0.825 | 0.025 / 0.025 | 0.025 / −0.05 | 통과 | – |

- 규칙 단위 요약(판정 스크립트 `label`): **B_full H-C, B_min H-C**(실패 조건 1개, 그 분류 H-C). C_none은 F3 실패가 없어 H-B다.
- **뜻(사전 등록 규칙대로)**:
  - B의 F3 실패는 H-C(기준 성질)로 분류된다.
  - current의 성공률을 강한 시작점($E\ge5\hat\sigma_n$)만으로 다시 세면 0.90 → 0.70으로 떨어진다. 그러면 B(0.75)가 오히려 0.05 높다.
  - 곧 이 조건에서 current가 B보다 높았던 몫은 잡음 수준 피크가 주는 추가 시작점에서 왔다. 강한 시작점끼리 비교하면 차가 없다(H-A가 아님).
- **한계(v3 §3)**:
  - F3-S는 시작점의 **수**만 맞춘다. 강한 시작점은 current의 피크 가운데 $E\ge5\hat\sigma_n$인 부분 집합이고, B의 피크와 위치까지 같지는 않다. 그래서 "위치까지 같은 조건에서 B가 current와 같다"는 증거는 아니다.
  - 표본은 조건당 40시드이고, 분해능은 0.025다.
  - H-C는 진단 분류일 뿐 판정을 바꾸지 않는다. 순위와 U1은 v2.1 결정표로만 정했다.

## 7. 보고 항목 (판정 무관)

- **F5-C 깊이**(관 상단, 최근접 kept 힌트), \|d − d_true\| 10/50/90 % 분위:

  | 규칙 | X | current |
  |---|---|---|
  | B_full | 0.0127 / 0.0171 / 0.0224 m | 0.0128 / 0.0171 / 0.0224 m |
  | C_none | 0.0128 / 0.0166 / 0.0223 m | 같음 |

  깊이 정확도에는 실질 차가 없다.
- **F6**:
  - kept 단계 성공률(관 모델 판정 행): current 0.494, B_full 0.633, B_min 0.630, C_none 0.588
  - lost_reason(current): no_peak 596, not_accepted 496, other_nms 204
  - lost_reason(B_full): no_peak 468, not_accepted 456, other_nms 16
  - 상한 300 도달 0행
- **W5 깊은 쪽 감도**: 해당 없음(빈 목록).
- **M0**:
  - 잡음 없는 argmax(뮤트 아래 주기, Δ −1 / 0 / +1 / +2.35): R1·R2는 0.784 / 0 / 0 / 0.539, P3는 0.784 / 0 / 0 / 11.711다.
  - Λ_cal(Δ 0 / +2.35): R1 29.55 / 6.32, R2 35.77 / 7.48, P3 0.25 / 0.08.
- **P-F3 민감도**(Λ_cal·ρ_cal): R 성립, 결정 4/5로 같다(판정 json `P_F3`).
- **원 Δt 보조**(720행, current·B_full, ρ 10, Δ 0): 원 격자에서 S6P_D13_S042_J7의 current는 0.00, B_full은 0.725다. current는 S4N 5개와 S6P_D13_S012_J8에서 상한 300에 40/40행 걸렸다(판정 json `original_dt`).

## 8. v2 대비

| 항목 | v2(새 실현 1) | v3(새 실현 2, 시드 40) |
|---|---|---|
| 결함 재현(R) | P0 하나(e 0.55·1.00) | R1·R2 둘(e 0.775–0.925) |
| B의 F3 실패 조건 | S4N_D15_S045 ρ 20(0.55 대 0.85) | **S4N_D13_S042 ρ 20(0.75 대 0.90)**. 실패 조건이 바뀌었다. 깊고 평평한 꼭짓점 F 표적 2개(D16_S045·D15_S048)는 통과했다(차 0.025·−0.025) |
| F3 실패의 성질 | 사후 진단(실현 7개 추적): 시작점 수 가설 | **사전 등록 F3-S로 H-C**(gap_S −0.05) |
| F5-B | 통과(악화 0·개선 60) | 통과(악화 36·개선 100) |
| 순위·U1 | 5, current | 5, current |

- 해석(진단): v2에서 F3 실패의 표적처럼 보였던 "깊고 평평한 꼭짓점"은 v3에서 통과했다. 실패는 다른 짝 대조 조건(D13_S042 ρ 20)으로 옮겨 갔고, 그 실패도 시작점 수로 설명된다.
- 이는 "특정 깊이 조건의 규칙 결함"(H-A)보다는 "F3가 시작점 수 차이를 함께 재는 성질"(H-C)과 맞는다.
- 다만 실패 조건이 실현마다 바뀐다는 점은 F3 차가 문턱 0.1 근처에서 흔들린다는 뜻이기도 하다(v3 D14_S038 ρ 20도 차 0.10). 이것은 결과를 본 뒤의 읽기다.

## 9. 결론과 한계

- **결정 순위 5의 귀결**: U1에 따라 현행 바닥 규칙($0.05\max\ell$)을 유지한다. 4TU 후보 재생성·재평가(U4)는 하지 않는다.
- 세 번의 합성 판정(묶음 1·v2·v3)이 모두 같은 결론을 냈다.
  - 결함은 재현됐다.
  - 잡음 기준 바닥(B)은 결함을 고친다(분리 F1·회복 F2 통과).
  - 거짓 후보(F4·F4b)와 위치(F5-B)는 나빠지지 않는다.
  - 그러나 봉인 기준 한 항목에서 채택 문턱을 넘지 못했다(묶음 1 F5, v2·v3 F3).
- **한계**:
  1. 2D 합성(원통 확산)이라 성공률 크기는 정성 판단이다.
  2. F3 문턱 0.1은 40시드에서 4시드 차다. 실패 차 0.15는 6시드 차다.
  3. F3-S는 시작점 수만 맞춘다(§6).
  4. §8의 v2 대비 해석은 결과를 본 뒤의 읽기다.
  5. 4TU 영향은 재지 않았다.

## 10. 재현

아래 두 명령 모두 git 제외 폴더 `data/`(`data/synthetic_gprmax_s4s6_v3/`, `data/processing/m210b_s4s6_v3/`)가 필요하다.

수집 스크립트는 scratchpad 폴더를 인자로 받아 다음 세 가지를 읽는다. **이 파일들은 저장소에 없다**(세션 scratchpad). 생성 명령은 python-dev `-I -B`로 실행했다.

| 입력 | 생성 명령 |
|---|---|
| `s4s6v3_refh5_guard.json` | `source_guard.py snapshot <참조 h5 11개> --state <scratchpad>/s4s6v3_refh5_guard.json`(내보내기 전), 하네스 뒤 `source_guard.py verify --state …` |
| `s4s6v3_export_guard.json` | `source_guard.py snapshot <내보낸 npy·json 38개> --state <scratchpad>/s4s6v3_export_guard.json`(하네스 전), 하네스 뒤 `source_guard.py verify --state …` |
| `bgcmp_v3/BG_*.json` | BG 8개마다 `h5_compare.py <pilot_prerun_h5/BG_*_merged.h5> <h5/BG_*_merged.h5> --expect bit --all-datasets --format json --out <scratchpad>/bgcmp_v3/BG_*.json` |

(도구 경로: `~/.claude/tools/integrity/source_guard.py`, `~/.claude/tools/gprmax/h5_compare.py`. 정확한 인자 형식은 각 도구 README를 따른다.)

```
conda run -n gpr-dev python -I analysis/processing/m210b_v3_judge.py --expected-seal-commit 77f5f96   # 판정(봉인판)
conda run -n gpr-dev python -I analysis/processing/m210b_v3_judgment_check.py <scratchpad 폴더>      # 판정서 대조용 수집(읽기 전용)
```
