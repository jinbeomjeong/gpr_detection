---
name: eval-4tu-verify-method
description: How to verify eval_depth_4tu.py changes without 4TU truth — scratch mutation runner, HEAD-vs-tree fixture compare, NaN baseline cells, untested pass-predicate boundary
metadata:
  type: project
---

Verifying `scripts/eval_depth_4tu.py` changes (v1.5.1, v1.6.1 judgments, 2026-11-07):

- Mutation runner: copy 7 files (pyproject.toml, the script, tests/conftest.py, tests/synthetic_bscan.py, tests/test_eval_depth_4tu.py, spec_eval_4tu_v1.md, m29c_qc_constants.json) to scratchpad, one exact-string mutation at a time, `pytest tests/test_eval_depth_4tu.py`. ~55 s per mutation (175 tests at v1.6.1) -> 20 mutations exceed the 10 min foreground cap; run in background. `conda run` buffers stdout until exit (log looks empty meanwhile).
- HEAD regression: `git show <rev>:scripts/eval_depth_4tu.py` to scratch, load both with importlib (register in sys.modules — dataclasses need it), import fixtures from `tests/test_eval_depth_4tu.py` (put tests/ on sys.path). Compare summary rows NaN-aware.
- Pre-existing since v1.2: summary rows whose scope has n_est = 0 carry `nan` (not blank) in baseline_mae_* / baseline_pass_a_* (6 cols). Gate columns (v1.6) use blank. Plain `==` row compare fails on these.
- Test gap (HEAD onward): pass-rate tolerance boundary (0.45-0.15 = 0.30000000000000004) is not asserted in `summarize`; reverting only that call site to exact `<= 0.30` survives all tests. Proposed G1b (spec 9-2-3).
- Single-line mutations that a runtime guard blocks (ValueError/RuntimeError) are not "untestable": write the multi-line mutation that bypasses the guard and check the targeted test still catches it (done for G9).

**Why:** these cost time to rediscover and decide whether a survivor blocks a verdict.
**How to apply:** reuse the runner layout for the next verify of this script; treat `nan` baseline cells as pre-existing, not a regression. Related: [[apex-candidate-qc-facts]].
