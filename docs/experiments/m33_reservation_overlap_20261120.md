> 성격: compute-scheduler-expert의 조사 보고 원문(2026-11-26, hand-back 그대로 저장, 판정 아님). 다른 세션의 경로·프로젝트 이름은 공개 저장소라 메인 세션이 가렸다. 본문의 "내 소유"는 compute-scheduler-expert 기준이다.

의존성 점검: STAMP(생략). 조사 보고(읽기 전용, 프로세스·예약 변경 없음). 저장소 파일은 쓰지 않았고 scratchpad 하위 폴더 `compute-scheduler-breach`는 지웠다.

## 1. 그 시각의 예약 (사실)
- 우리 예약: `r-20261120142657-4cb2a7`, owner `gpr_detection:m33_e3x`, mask 0x7FFF(논리 0–14), GPU 0, TTL 480분. 근거는 러너 사본 `run_e3x.ps1`(`$ReserveOwner`·`$ReserveTtlMin`·`$ReserveMaskHex`)과 `run_e3x.out`의 `[reserve]` 줄이다. 14:26:57에 등록되어 단계 시작마다 renew됐고, 끝에 released됐다(`run_e3x.out` 1·317행).
- 등록 때 `[reserve] overlaps ...` 줄이 없다. 14:26:57에는 겹치는 다른 세션 유효 예약이 없었다는 뜻이다.
- 그 시점의 예약 목록 전체는 복원할 수 없다. `reservations.json`은 현재 상태만 담고(지금은 `reservations: []`), 추가·갱신·해제 이력 로그가 없다. 락 파일은 10-30 이후 갱신되지 않았다. `history.jsonl`에도 11-20 13~16시 기록이 없다(그날 기록은 09:44 leak_dvcm 4건뿐이다).
- 다른 세션 예약이 있었다는 증거는 못 찾았다. 있었다면 다른 세션 쪽이 우리와 겹쳤을 때 conflicts를 받았을 것이다. 하지만 아래 2번 스크립트에는 `reserve.py` 호출이 없다.

## 2. 겹친 작업의 정체
- **사실**: 다른 세션 scratchpad (경로·프로젝트 이름은 공개 저장소라 가림, python-expert)의 Chrono-dev pytest 실행이다. 인계 메모의 "dynamics-dev"는 아니고 `Chrono-dev`다.
  - `done_wt.txt`: `exit=1 start=2026-11-20T15:17:23 end=15:28:42`(11분 18초, 7,398 passed·2 failed). 로그에 "bringing up nodes"가 있어 xdist 병렬 실행이다. 작업 대상은 worktree `z4369-jp3`의 `05_tests/unit`으로 보인다.
  - 이어서 `done_master.txt` 15:22:53–15:24:25, `done_master_flow` 15:29:47–15:30:10, 그리고 15:31–15:34의 mainwt 두 번이 있다.
  - 남아 있는 `run_unit.ps1`(15:22:48 판)과 `run_main_data_wt_src.ps1`(15:30:59)은 둘 다 `$me.ProcessorAffinity = [IntPtr]0xFFFF`로 자기 셸을 P 전체(논리 0–15)에 고정하고, pytest `-n 12`(기본값, mainwt는 `-n 8`) 작업자가 이를 상속한다. 이 스크립트에 `reserve.py`·`wait_until`·`status.py` 호출은 없다.
  - 15:17:23 시작분(`wt`)의 스크립트 원본은 남아 있지 않다. 같은 형태의 `run_unit.ps1`(`-Tag wt`)이었다는 점과 작업자 12는 **추정**이다. 우리가 본 "15:19 0xFFFF 고정"은 이 `wt` 실행이다. 시작은 15:17:23이다.
- 예약 등록은 안 했다(스크립트에 없음). 11분짜리라 전역 규칙의 30분 초과 예약 기준에도 해당하지 않는다. 사용한 도구는 `reserve.py`·`wait_until --no-reservation-overlap` 모두 아니다.
- 겹침은 논리 0–14(우리 mask 전체)다. GPU는 쓰지 않았다(CPU 테스트).
- 우리 쪽 겹침 시간: 15:17:23–15:20:58(우리 injected 종료) ≈ 215 s(injected 3,119 s의 약 7 %). 15:20:58 이후는 겹치지 않는다. 15:22 이후의 master·mainwt 실행은 우리 종료 뒤다.

## 3. 우리 실행에 준 영향
- 사실: 같은 injected 단계 wall은 1회차 3,119 s(mask 0x7FFF, 작업자 5, e3x_defect 종료)이고, 재시도는 2,911.4 s(P 0xFFFF, 작업자 5, ok, 2026-11-26 13:00 시작, `m33_e3x_judgment.md` 58행)다. 차이는 +208 s(+7 %)이고, 겹침 215 s와 숫자만 비슷하다.
- 이 비교는 근거가 약하다. 마스크 크기(15 대 16)·날짜·종료 상태가 다르고, 재시도 때 다른 부하 유무는 확인하지 않았다. 1회차의 겹침 영향은 **추정으로도 확정하지 못한다**.
- GPU 카운터 5분 구간 평균(`e3x-injected.gpucounters.csv`): 트리 busy 10–20 %, GPU 사용률 20–39 %로 CPU 쪽 병목이다. 겹침 구간(15:15–15:20대)은 GPU 20.5 %로 낮지만, 14:55 구간도 21.6 %이고 구간별 변동이 크다. 전력 스로틀 기록은 없다(`throttle_limiting` 비어 있음). 구간 평균만으로는 겹침 영향을 구분할 수 없다.
- 1회차 uninjected001(122 s, 14:26:57–14:28:59)은 겹침 이전이라 영향이 없다. 1회차는 e3x_defect로 판정에 쓰지 않으므로 판정값에는 영향이 없다. 재시도 판정은 실측 시간만 기록하고 결과를 바꾸지 않는다.

## 4. 재발 방지 권고
원인은 도구 설계에 있다. 예약은 "나중에 등록하는 쪽만 conflicts를 받는" 알림 전용 방식이고, 먼저 등록된 쪽에는 나중에 들어온 비등록 작업이 보이지 않는다. 예약 이력도 안 남는다.
1. (도구, 내 소유) `reserve.py`에 추가·갱신·해제·prune 이력 로그(`reservations.log.jsonl`, 시각·owner·mask·PID)를 덧붙인다. 지금은 조사에 필요한 이력이 사라진다.
2. (도구, 내 소유) `make_runner.py`의 `Update-Reservation`(단계마다 renew)에서 renew 뒤 `affinity_map.py`로 우리 마스크에 걸친 타 세션 명시 고정을 확인해 `[reserve] foreign-pin ...` 경고를 `_runner.log`에 남긴다. 지금은 등록 시점만 점검한다. 선택으로 러너에 몇 분 간격 감시 훅(`contention.py`)을 붙일 수 있다.
3. (도구, 내 소유) `plan.py`·`make_runner.py` 기본값은 이미 예약을 지원한다. 15분 이상 점유하는 단계가 있으면 보고에서 "다른 세션 기록 요구"를 `메인 세션 권고`에 고정 문구로 넣는다.
4. **위임/정의 반영안(메인 세션)**: 다른 세션의 병렬 테스트 실행(`pytest -n 12` + 자기 고정 0xFFFF)이 예약을 보지 않는다. 전역 `CLAUDE.md` §4 또는 python-expert 호출 지시문에 "pytest -n N 같은 병렬 실행도 시작 직전 `status.py`로 타 세션 예약을 보고 mask를 빈 논리로 한정, 유효 예약과 겹치면 대기 또는 `wait_until --no-reservation-overlap`"을 넣는 것을 권한다(`위임 필요(python-expert)`가 아니라 지시문 규칙 변경이므로 메인 반영 사항). 11분 pytest라도 장시간 GPU·CPU 학습 예약과 겹치면 같은 문제가 생긴다.
5. 그 PC에 한해: 1회차 러너가 쓴 mask 0x7FFF는 P 논리 15를 남기는 설계였으나 겹친 세션이 0xFFFF를 걸어 무효였다. 장시간 판정 실행은 겹침이 의심되면 `contention.py watch --victim-match`를 병행하는 편이 낫다(SMT 감속은 못 잡는 한계 있음).

## 사용자 조치 필요 / 위임 필요
- 사용자 조치 필요: 없음. 실행 중인 겹침은 이미 끝났다.
- 위임 필요: 위 4번(메인 세션이 지시문·`CLAUDE.md` 반영). 1~3번은 이 에이전트가 다음 호출에서 직접 고칠 수 있다(지금은 조사 전용이라 수정하지 않았다).

## 마감 판정서에 적을 2–3줄
- 2026-11-20 1회차 본 실행(우리 예약 `gpr_detection:m33_e3x`, mask 0x7FFF)의 injected 단계 중 15:17:23–15:20:58(약 215 s)에 다른 세션(다른 프로젝트, python-expert의 Chrono-dev pytest, 자기 고정 0xFFFF, 작업자 12 추정)이 우리 마스크와 겹쳐 돌았다. 그 세션은 `reserve.py`를 쓰지 않았고(스크립트 확인) 예약 알림은 구조상 우리에게 가지 않았다.
- 1회차 injected wall 3,119 s는 재시도의 2,911 s보다 7 % 길지만, 마스크·종료 상태가 달라 겹침의 영향으로 확정하지 못한다(GPU 사용률 20–39 %, CPU 병목, 스로틀 없음). 1회차는 e3x_defect로 판정에 쓰지 않았으므로 판정값에는 영향이 없다.
- 재발 방지로 예약 이력 로그·러너의 단계별 타 세션 고정 감시를 도구에 추가하고(스케줄러 소유), 병렬 테스트 실행도 예약 확인을 하도록 지시문 규칙을 바꾸는 안을 낸다.

## 확인되지 않은 것
- 11-20 15:17 시점의 `reservations.json` 전체 내용(이력 없음).
- 15:17:23 시작분(`wt`)의 실제 스크립트·작업자 수(원본 없음, 같은 스크립트 형태 추정).
- 재시도(11-26 13:00)가 겹침 없는 상태였는지.
