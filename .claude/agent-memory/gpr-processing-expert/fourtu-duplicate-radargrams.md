---
name: fourtu-duplicate-radargrams
description: 4TU has 39 byte-identical radargram pairs (SHA-256, 759 npy) - within 09.5 and across whole activities (03.5=03.6, 010.11=010.12, ...); handling rules (spec_eval v1.7 4-5), keep_one "strict superset" pitfall
metadata:
  type: project
---

Found 2026-11-07 (M2-9 D/E prep) by hashing `05_radargrams_npy/*/*.npy`. Curator confirmed all 39 are source-zip duplicates (docs/datasets/fourtu_duplicates.md).

- Within one activity: 09.5 Path7=12, 8=13, 9=14, 10=15, 11=16 (no GT). A duplicate line "supports" itself in the >= 2-line support filter (spec_eval §4-3 `n_lines`).
- Across activities (34 pairs): 03.5=03.6 (10 paths), 010.11=010.12 (10), 02.2=02.4 (8), 013.1=013.2 (3), 010.15=010.16, 09.6=09.7, 012.2/Path7=012.4/Path1. GT activities among them: 03.5/03.6, 010.11/12, 010.15/16, 012.2/012.4, 013.1/2 -> the same B-scan is matched against two GT sets.
- Candidate rows on both files of every pair are identical (39/39), so dropping a copy loses nothing. Full SHA-256 scan of 759 files (470 MB) costs ~1 s warm.
- keep_one pitfall: "keep the line-set superset" must mean STRICT superset with lower-activity-number fallback; "more lines wins" keeps 09.7 instead of 09.6 and contradicts the curator list. 012.2 is kept by the number fallback, not as a superset. Compare activity ids numerically ((10, 11)), never as strings.
- Removing k GT items changes pass_b by sampling noise alone: SD ~ (k/N) sqrt(p(1-p)(1/(N-k)+1/k)) -> 0.6-0.9 %p for k 25-46 of 388 (basis of the 2 %p caveat proposal).
- Identical zero-event times on different paths are the quick tell (m29e_zero_survey_lines.csv).

**Why:** silently inflates support counts and weights in 4TU depth evaluation.
**How to apply:** use the committed list + runtime hash check (spec_eval v1.7 §4-5-2); count within copies once; keep across duplicates in the main result and report keep_one/drop_all as auxiliaries. Related: [[fourtu-zero-band]], [[eval-4tu-verify-method]].
