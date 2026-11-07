---
name: fourtu-duplicate-radargrams
description: 4TU has 39 byte-identical radargram pairs (SHA-256, 759 npy) - within 09.5 and across whole activities (03.5=03.6, 010.11=010.12, ...); breaks cross-line support and double-counts B-scans
metadata:
  type: project
---

Found 2026-11-07 (M2-9 D/E prep) by hashing `05_radargrams_npy/*/*.npy`.

- Within one activity: 09.5 Path7=12, 8=13, 9=14, 10=15, 11=16 (no GT). A duplicate line "supports" itself in the >= 2-line support filter (spec_eval §4-3 `n_lines`).
- Across activities (34 pairs): 03.5=03.6 (10 paths), 010.11=010.12 (10), 02.2=02.4 (8), 013.1=013.2 (3), 010.15=010.16, 09.6=09.7, 012.2/Path7=012.4/Path1. GT activities among them: 03.5/03.6, 010.11/12, 010.15/16, 012.2/012.4, 013.1/2 -> the same B-scan is matched against two GT sets.
- Identical zero-event times on different paths are the quick tell (m29e_zero_survey_lines.csv).

**Why:** silently inflates support counts and weights in 4TU depth evaluation; not in any dataset doc as of 2026-11-07 (spec_eval_4tu_v1 §10 row, §11-1 condition 4).
**How to apply:** before any 4TU re-evaluation or line-count statistic, dedupe by file hash (rule decided without GT depth); ask gpr-dataset-curator to confirm against source metadata. Related: [[fourtu-zero-band]].
