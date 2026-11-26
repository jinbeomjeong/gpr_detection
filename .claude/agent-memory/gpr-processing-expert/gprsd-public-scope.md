---
name: gprsd-public-scope
description: GPR-SD (licence unknown) committed outputs may hold summary stats only - no per-scatterer trace/sample coordinates, no label (pipe-mark) window positions; full tables go to git-ignored data/derived/
metadata:
  type: project
---

GPR-SD analysis outputs in the public repo: summary statistics only. Per-scatterer tables/picks (u0, cand_trace, picks...) go to `data/derived/<task>/` (git-ignored) and the public json carries a `public_split` pointer (M3-6 pattern: `m36_gprsd_common.write_json_split` / `load_full_json`). Label (pipe-mark) window positions go nowhere. Mean waveforms / per-window spectra without images were allowed (user decision 2026-11-26).

**Why:** licence unknown (`docs/datasets/cards/gprsd.md`); user decided 2026-11-26 before the M3-6 commit.

**How to apply:** design any new GPR-SD script with the split from the start. Watch indirect leaks: grid windows that skip mark +-400 reveal mark positions by their gaps; `m36_gprsd_pilot.json` mark windows (`xlsx_row`, `trace`) were pushed before the rule; split 2026-11-26 (history not rewritten; marks are public in the source repo's Mark.xlsx). Grid-window lists in period/migration_focus json were kept by user decision. Windows `Path.write_text` writes CRLF, so byte checks must translate newlines.
- 사용자 결정(2026-11-26): 원 저장소 Mark.xlsx에 공개된 개별 표식 트레이스 번호의 문서 인용은 허용한다(도로 독립성 근거 등). 창 목록·창 좌표·라벨·산란체 픽 좌표는 계속 공개본에서 뺀다.
