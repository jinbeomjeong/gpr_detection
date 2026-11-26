# M3-3 마감: PP 운용 기본값 S3 전환 감사 (verif-protocol-auditor, 전환 후, 좁힌 범위)

## 감사 요약
- 대상: 작업 트리(미커밋, HEAD 00ab7da)의 `src/gpr_detection/pipeline.py`(blob dd5e7bcb → cbc7b847), `tests/test_pp_default.py`(새 파일), `tests/m33_close_blob.py`(새 파일), 시험 파일 7개의 fixture·단언 변경, `analysis/detection/m33_pp_off_identity.py`
- 시점: 전환 후. 사용자 결정으로 범위를 좁힌 가벼운 감사다.
- 기준 문서
  - 계획 `docs/experiments/m33_cross_domain_plan.md` v0.2.8 §3-4 6(:194), §4 채택 결정 행(:270)
  - J1 판정서 `docs/processing/m33_j1_judgment.md` §8(:244–248)
  - J1 사전 등록 `docs/processing/m33_j1_preregistration.md:147`
  - 봉인 상수 `analysis/detection/m33_inputs.py:52–60`(`C0B_PP_SET_ID` S3, `C0B_PP_CONFIG_SHA256` eb2796dd, `C0A_PIPELINE_BLOB` dd5e7bcb)
- 본 결과: 없음. 판정 결과 파일을 열지 않았다. 블라인드 경로(`runs/eval/m33/e3x/{uninjected001,injected,judgment}/`)와 그 상위 폴더 `runs/eval/m33/`·`runs/`는 Read·Grep·Glob의 범위로 쓰지 않았다.
- 판정: 차단 0 / 실행 전 수정 1 / 기록 7. **커밋해도 된다.** Critical·High에 해당하는 막는 지적은 없다. 실행 전 수정 1건은 이번 커밋에 넣는 것이 좋지만, 다음 커밋으로 미뤄도 이번 결론은 바뀌지 않는다.

## 발견 사항 (심각도순)

### [실행 전 수정][S·E] `accept_close_pipeline_blob`이 작업 트리의 `pipeline.py` blob을 그대로 봉인값으로 받는다. 그래서 러너 끝단 시험이 하던 "pipeline.py 불변" 감시가 사라진다
- 근거: `tests/m33_close_blob.py:34–35`
  - `blob = working_tree_pipeline_blob()` 다음에 `monkeypatch.setattr(m33_inputs, "C0A_PIPELINE_BLOB", blob)`를 한다.
  - 고정값이 없다. 바꾸기 전에는 5개 파일의 끝단 fixture(`offline_run`·`_patch_offline`·`offline_r1`·`offline_r2`·`offline_band`)가 실제 `C0A_PIPELINE_BLOB`으로 `assert_sealed_pp`를 지나갔다. 그래서 `pipeline.py`가 조금이라도 바뀌면 그 시험들이 실패했다.
  - 이제는 `pipeline.py`가 어떻게 바뀌든(예: 다음에 `apply_pp`나 `PP_SETS`를 고쳐도) 이 fixture가 그 blob을 받아들인다.
  - 거부 확인 시험 `tests/test_m32_recipe_eval.py:1018–1024`는 "작업 트리 blob ≠ dd5e7bcb"만 본다. 바뀐 내용이 무엇이든 참이므로 감시 역할을 하지 못한다.
- 실패 시나리오: 다음에 `pipeline.py`의 PP 규칙이 바뀌는 경우를 생각한다.
  - `PPConfig` sha가 같은 채 동작만 바뀌면 R 러너 시험은 통과한다.
  - 그것을 잡는 것은 `tests/test_pp.py`·`tests/test_pp_default.py`의 동작 시험뿐이다. 블롭 단위의 "결과 뒤 PP 수정 금지"(계획 §3-4 6) 감시는 없어진다.
- 권장 조치(python-expert)
  - `m33_close_blob.py`에 `CLOSE_PIPELINE_BLOB = "cbc7b84784f26f3dac1c10f62bd757fd74e7df4d"`를 둔다.
  - `accept_close_pipeline_blob`가 setattr 전에 `working_tree_pipeline_blob() == CLOSE_PIPELINE_BLOB`을 단언하게 한다. 다르면 "pipeline.py가 마감 뒤 바뀜, 의도한 변경이면 상수 갱신"으로 실패시킨다.
  - `pipeline.py`는 `eol=lf`이고 `git hash-object`가 정규화하므로, 체크아웃 줄바꿈과 무관하게 이 값은 안정적이다.
  - 이것은 사용자 결정 (A)(작업 트리 blob 수용)를 다듬는 것이다. 결정 자체를 뒤집지 않는다.

### [기록][R] 비트 동일성 시험은 J1 §8의 조건을 충족한다. 근거는 세 가지다
1. 코드 불변: `git diff`의 `pipeline.py` hunk 4개는 모듈 docstring(:26–29), `DEFAULT_PP_SET` 상수와 그 docstring(:704–710), `detect_line` 기본 인자(:928), 인자 설명(:937–939)뿐이다. `apply_pp`·`PP_SETS`·`PPConfig`·`merge_line_boxes`·`content_rows_for`·`line_zero`·`kept_after_pp` 본문은 바뀌지 않았다. HEAD blob이 dd5e7bcb임은 `git ls-tree HEAD`로 확인했다.
2. 설정 sha: 64자리가 세 곳에서 글자 그대로 같다.
   - `tests/test_pp_default.py:49` `SEALED_S3_SHA256`
   - `m33_inputs.py:55`
   - J1 판정서 :246의 `eb2796ddfaa9611e5030342c4ed66b00c91ff2ed6a6698ee5979b9afe080602f`
   - 시험 `test_default_set_is_the_sealed_c0b_s3`(:101–106)는 `PP_SETS[DEFAULT_PP_SET].sha256()`가 이 값과 같고 `DEFAULT_PP_SET is S3 is C0B_PP_SET_ID`임을 단언한다. 이번에 통과했다.
3. 경로 동일: 시험의 참조 `_r2_reference`(:68–86)는 R2 러너와 같은 함수를 같은 순서로 부른다.
   - 시험: `runner.uninjected_jobs` → `runner.run_jobs` → `r2._geometry`(= `content_rows_for(image, data.zero)`, `m33_r2_eval.py:592–597`) → `merge_line_boxes(…, DEFAULT_MERGE_CONFIG)` → `apply_pp(…, PP_SETS[S3])` → `r2.pp_versions[post]` → `r2._grid`
   - R2 본 경로: `m33_r2_eval.py:631–634` `merge_line_boxes(tile_boxes, job.image, DEFAULT_MERGE_CONFIG)`, `apply_pp(merged, geometry.pp, shared.config)`
   - 비교 `_assert_bit_identical`(:89–98)은 값 동등(`==`)과 `repr` 동등을 함께 본다. float의 repr은 왕복 최단 표기라 -0.0과 0.0을 구별하므로 비트 수준 비교다(NaN payload는 예외).
   - fixture 시험(:109–140)은 S3의 모든 작용(`NONE`·`REMOVED_A`·`CLIPPED_B`·`REMOVED_D`)이 실제로 일어남을 단언한다. 합성 측선 시험(:143–157)은 영상과 영점을 패치하지 않은 경로다. 영점 동일성의 근거: `runner.load_line`(`m32_recipe_eval.py:500–504`)과 `pipeline.line_zero`(`pipeline.py:913–919`)는 같은 `find_time_zero(radargram, prepare_fitting_branch(radargram).filtered)`다.
- 한계(판정서에 이미 일부 적혀 있음, `m33_closing_judgment.md:165`)
  - (a) fixture와 가짜 검출기만 쓴다.
  - (b) 검출기 호출 경로가 다르다. `detect_line`은 `detect_tiles`로 타일마다 부르고, R2는 `runner.run_jobs`로 batch를 돌린다(시험은 batch_size=1). 실제 YOLO에서 batch 추론과 단건 추론이 비트로 같은지는 이 시험 범위 밖이다. 이 차이는 PP 단계 앞에 있고 이번 전환으로 생긴 것이 아니다. §8 조건("그 설정이 같은 함수에서")은 PP 설정에 대한 것이므로 위 세 근거로 충족된다고 판단한다.
  - (c) J1 행은 PP 뒤에 `score > 0.25`로 거른다(`m33_r2_eval.py:417`). `detect_line`은 거르기 전 집합을 낸다. 시험은 두 쪽 모두 거르기 전 집합으로 비교하므로 정합하다.
- 권장: 마감 판정서 §4-2 한계 문장에 (b)를 한 줄 덧붙이는 것을 권한다(cv-detection-expert 또는 판정서 작성자).

### [기록][S·E] 봉인 보호는 유지된다. fixture 교체는 함수 범위와 with 블록 안에서만 일어난다
- 봉인 러너와 상수는 바뀌지 않았다. `m33_inputs.py`·`m32_recipe_eval.py`·`m33_r2_eval.py`·`m33_r3_change_report.py`·`m33_e3x.py`·`m33_j1_judge.py`는 `git status`에서 바뀌지 않은 것으로 나온다.
- `assert_sealed_pp`(`m33_inputs.py:358–411`)는 호출 시점에 모듈 전역 `C0A_PIPELINE_BLOB`을 읽는다(:398). 다음 호출처는 모두 작업 트리의 실측 `code_blobs`를 넘기므로, 패치가 없으면 새 트리를 `pipeline_blob_is_c0a`로 거부한다.
  - `m32_recipe_eval.py:2523`(R 러너)
  - `m33_morocco_eval.py:1125`
  - `m33_e3x.py:502`
  - `m33_j1_judge.py:426`
- 이 확인은 마감 판정서 `m33_closing_judgment.md:167`의 주장과 맞는다. J1 판정 도구는 HEAD `src` tree(:418–423)에서도 따로 거부한다.
- 거부 확인 시험 `test_sealed_pp_still_refuses_the_working_tree_pipeline`은 `sealed_c0b`만 쓰고 `accept_close_pipeline_blob`를 부르지 않는다. 그래서 패치 없이 돌고, 이번 실행에서 통과했다.
- 끝단 거부 경로(`runner.main` 종료 2)는 기존 매개변수 시험 `test_a_pp_other_than_the_sealed_one_refuses_with_exit_2`(:992–1007)가 덮는다. 이 시험은 `offline_run`이 바꿔 끼운 값을 `"0"*40`으로 다시 덮는다.
- 누수 경로를 확인했다.
  - `offline_run`·`offline_r1`·`offline_r2`·`offline_band`·`offline`은 함수 범위 `monkeypatch`를 쓴다.
  - 모듈 범위 `synthetic_r3`(`tests/test_m33_change_defs.py:263–276`)는 `pytest.MonkeyPatch.context()`의 with 블록 안에서만 패치한다.
  - `report.C1_CODE_BLOBS | {…}`(:230)는 새 dict를 만들어 setattr하므로 원래 dict를 바꾸지 않는다.
  - 모듈 범위로 새는 길은 찾지 못했다.

### [기록][I] `m33_pp_off_identity.py`는 두 트리 모두에서 S0(꺼짐)으로 동작한다(코드 읽기)
- `pp_off_options`(:52–62)를 읽었다.
  - `inspect.signature(detect_line)`에 `pp_mode`가 없으면 `{}`를 돌려준다. a0dc987의 `detect_line`(`git grep a0dc987` :562–567)에는 `pp_mode`가 없으므로 항상 꺼짐이다.
  - `pp_mode`가 있으면 `PPSetId.S0`을 돌려준다.
- import는 `--src` 트리를 `sys.path[0]`에 넣은 뒤의 같은 `gpr_detection.pipeline` 모듈에서 온다(:69, :75, :60).
- 공백: `pp_off_options`를 직접 부르는 단위 시험이 없다. 바뀐 스크립트를 새 트리에서 실행한 기록도 없다. 마감 판정서 :170은 "고쳤다"고만 쓰고 실행했다고 주장하지 않는다.

### [기록][E] 수정된 opt-in 실측 꺼짐 시험은 이번 실행과 python-expert 실행 모두에서 돌지 않았다
- `tests/test_pp.py::test_1_pp_off_is_identical_on_the_r0_eleven_lines`(:173~)는 이번 변경에서 기본값 비교 줄을 지웠다(S0 명시만 남음). 이 시험은 `-m realdata` opt-in이라 건너뛰었다. `junit_skips` 결과: skip 2건 모두 opt-in이고, 필수 시험의 건너뜀은 0이다.
- S0 경로 코드(`pipeline.py:944–948`)와 `PP_SETS[S0]`이 바뀌지 않았으므로 위험은 낮다. 다만 PP 명세 §5 1(1a·1b, 지금 :156–158)이 요구하는 실측 확인은 마감 트리에서 다시 하지 않았다.
- 권장: 아래 메인 세션 권고 1.

### [기록][S] 작업 트리 CRLF 2건은 커밋 때 LF로 정규화된다
- 근거: `git ls-files --eol` 결과 `analysis/detection/m33_pp_off_identity.py`와 `tests/test_pp.py`가 `i/lf w/crlf attr/text=auto eol=lf`이고, `core.autocrlf=true`다.
- `eol=lf`이므로 `git add` 때 LF로 들어간다. 해시가 고정된 파일도 아니다(docs·analysis·tests에 두 파일 blob을 고정한 곳 없음, Grep). 영향은 없다.

### [기록][R] 문서 표기 2건 (integrity-checker)
- `docs/processing/spec_detection_image_v1.md:442`(감사 중 다른 쪽이 고친 줄)의 `detect_line` 서명 행에는 두 가지가 빠졌다.
  - `pp_mode`가 키워드 전용(`*`, `pipeline.py:927`)이라는 표시
  - 흐름 열의 PP 단계(`merge_line_boxes` → `apply_pp` → `line_boxes_to_grid`)
- `m33_closing_judgment.md` §4-2의 한계 문장에 검출기 호출 경로 차이((b), 위 R 항목)가 없다.

### [기록][S] E3x 쪽 영향 확인(범위 밖 한 줄)
- 미커밋 `analysis/detection/m33_e3x_judge.py`에는 `pipeline.py`·`PIPELINE_PATH`·`assert_sealed_pp`·`src_tree` 참조가 없다(Grep 0건). 그래서 봉인 대조에는 영향이 없다.
- E3x 러너 `m33_e3x.py`는 새 트리를 `:502`(`assert_sealed_pp`)와 `:800`(`src_tree_id`)에서 거부한다. 이는 마감 판정서 §4-3 "다음 E3x 판은 봉인 r2부터"와 맞는다.

## 규칙↔도구 대조표

| 규칙 | 규칙 위치 | 코드·시험 위치 | 일치 |
|---|---|---|---|
| J1 판정 구성 = R2가 적용한 PP, `PPConfig` eb2796dd | `m33_j1_judgment.md:246`, `m33_j1_preregistration.md:147` | `test_pp_default.py:49,104,139,157`; `m33_inputs.py:55` | 같음(64자리 글자 대조) |
| 기본값을 S3로 바꾸면 같은 함수에서 비트 동일 | `m33_j1_judgment.md:247` | `test_pp_default.py:109–157`(값과 repr) | 같음(한계는 위에 적음) |
| 결과 뒤 PP 수정 금지(PP v2로 공개) | 계획 :194 | `git diff`의 `pipeline.py` hunk 4개가 docstring·상수·기본 인자·설명뿐 | 규칙 불변 |
| 채택 결정: 켜면 python-expert가 기본값 변경과 결정 기록 | 계획 :270 | `pipeline.py:704–710,928`, 마감 판정서 §4-2 | 같음 |
| 봉인 러너는 C0a blob 단언 | `m33_inputs.py:398` | 바뀌지 않음, 거부 시험 :1018–1024 | 유지 |
| 꺼짐 비트 동일성(S0 명시) | PP 명세 :41, :156 | `test_pp.py:152–170`(fixture), :173~(opt-in, 미실행) | fixture만 확인 |

## 확인 필요
- 없음. 실행 전 수정 1건의 고정 상수를 둘지는 사용자 결정 (A)를 다듬는 일이므로, 메인이 python-expert에 넘기면서 사용자에게 알리면 된다.

## 위임 필요
- python-expert: `tests/m33_close_blob.py:30–36`. `CLOSE_PIPELINE_BLOB` 상수를 두고, setattr 전에 작업 트리 blob이 이 값과 같은지 단언한다. 다르면 사유와 함께 실패시킨다. 위 실행 전 수정 항목 참고.
- integrity-checker:
  - `docs/processing/spec_detection_image_v1.md:442`의 키워드 전용 표시와 PP 단계 표기
  - `docs/experiments/m33_closing_judgment.md:165`에 검출기 경로 한계 덧붙임(판정서 작성자 확인)
- tools/verif 후보: 없음.

## 메인 세션 권고
1. (선택, 권장) 마감 트리에서 opt-in 실측 꺼짐 시험을 한 번 돌린다. 데이터셋 11측선을 읽고 가짜 검출기를 쓰며 판정 결과를 만들지 않는다. 출력은 scratchpad에 둔다.
   - `conda run -n gpr-dev python -I -B -m pytest tests/test_pp.py -m realdata -q -p no:cacheprovider`
   - 필요하면 PP 명세 §5 1(1b)의 비교 스크립트를 두 트리에서 실행한다: `m33_pp_off_identity.py --src <a0dc987 archive>/src --out <scratchpad>\a.json`와 `--src src --out <scratchpad>\b.json`. 그다음 `sha256`이 같은지 확인한다.

## 검사하지 못한 것
- 실제 YOLO 가중치로 `detect_line` 기본값과 R2 batch 경로의 비트 동일성(범위 밖, 위 한계 (b))
- 전체 시험 2256건 재실행(지시대로 하지 않음), opt-in `realdata` 시험 2건
- `pp_off_options`를 실행해 본 확인(코드 읽기만 했다)

## 실행한 검증
- `~/.claude/tools/cpu-scheduler/status.py`: P 평균 11.5 %, 예약 0, GPU 1 %. 여유 상태로 판단하고 pytest 1프로세스를 Normal로 돌렸다.
- depcheck: STAMP(이 세션에서 이미 점검).
- git 조회: `git diff --stat`, `git diff`(범위 파일), `git hash-object src/gpr_detection/pipeline.py`(cbc7b847, 보고 직전 다시 확인해도 같음), `git ls-tree HEAD`(dd5e7bcb), `git ls-files --eol`, `git check-attr`, `git config core.autocrlf`, `git grep a0dc987 -- pipeline.py`, `git blame -L`(PP 명세 :41·:156)
- 코드 읽기:
  - `pipeline.py:885–958`
  - `m33_r2_eval.py:290–439, 585–667`
  - `m33_inputs.py:50–61, 245–279, 355–411`
  - `m33_r3_change_report.py:128–152, 510–529`
  - `m32_recipe_eval.py:498–505`
  - `m33_j1_judge.py:355–430`
  - `m33_pp_off_identity.py` 전체
  - `test_pp_default.py` 전체
  - `m33_close_blob.py` 전체
  - 시험 fixture 위치들
- 시험: `conda run -n gpr-dev python -I -B -m pytest <8개 파일> -q -p no:cacheprovider -rs --junitxml=<scratchpad>\junit.xml`(PowerShell, `PYTHONDONTWRITEBYTECODE=1`)
  - 결과: **210 passed, 2 skipped**(둘 다 opt-in `realdata`), 0 failed, 263 s
  - `test_pp_default` 4건과 거부 확인 시험이 junit에 있음을 확인했다.
  - `junit_skips.py --critical "test_pp_default|test_sealed_pp_still_refuses|test_1_pp_off_is_the_pre_pp_path|test_a_pp_other_than_the_sealed_one"` → `status ok`, `critical_skipped 0`, xfail 0
  - 실행 뒤 `tests/__pycache__`·`analysis/detection/__pycache__`에 새 파일 없음
- 인터프리터: 전역 도구는 python-dev `-I -B`, 프로젝트 시험은 gpr-dev(`conda run`, `-I -B`)
- 만든 뒤 지운 것: scratchpad `verif-pp-default\`(guard.json, junit.xml). 폴더째 지웠다.
- 감사 중 바뀐 파일(다른 에이전트, 나는 저장소에 쓰지 않았다)
  - `docs/experiments/m33_pp_spec.md` :41·:156(git blame "Not Committed Yet 17:06:41", 내 스냅샷 약 17:00 뒤)
  - `docs/processing/spec_detection_image_v1.md` :442(새로 수정 상태가 됨)
  - `docs/experiments/m33_closing_judgment.md` :316 등
  - 세 파일 모두 다시 읽었다. 최종 판정은 다시 읽은 내용 기준이다. 판정 근거인 코드·시험 파일(`pipeline.py`와 시험 10개)은 감사 중 바뀌지 않았다(source_guard에서 해당 파일 위반 없음, blob 재확인).
- 내 절차 위반 2건: Bash 도구에서 파이프 한 번(`git diff … | grep -nE … | head`, 결과 0줄), `> /dev/null` 한 번을 썼다. 둘 다 조회 명령이고 파일을 쓰지 않았다.
- 원본 불변: 위반 — 원본 내용 변경 1건(m33_pp_spec.md); 형제 항목 내용 변경 4건(m33_closing_judgment.md, m33_e3x_cv_postjudgment_review.md, m33_reservation_overlap_20261120.md 외 1건)
  - 감사자가 바꾼 것이 아니다. 감사 중 다른 세션·에이전트가 문서를 고쳤다(위 목록).
  - 판정 근거 코드·시험 파일은 위반 목록에 없다.
- 사용한 스킬: 없음
