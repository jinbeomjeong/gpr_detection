---
name: eval-4tu-verify-method
description: How to verify eval_depth_4tu.py changes without 4TU truth — scratch mutation runner, HEAD-vs-tree fixture compare, NaN baseline cells, untested pass-predicate boundary
metadata:
  type: project
---

Verifying `scripts/eval_depth_4tu.py` changes (v1.5.1, v1.6.1 judgments, 2026-11-07):

- Mutation runner: copy 7 files (pyproject.toml, the script, tests/conftest.py, tests/synthetic_bscan.py, tests/test_eval_depth_4tu.py, spec_eval_4tu_v1.md, m29c_qc_constants.json) to scratchpad, one exact-string mutation at a time, `pytest tests/test_eval_depth_4tu.py`. ~55 s per mutation (175 tests at v1.6.1) -> 20 mutations exceed the 10 min foreground cap; run in background. `conda run` buffers stdout until exit (log looks empty meanwhile).
- HEAD regression: `git show <rev>:scripts/eval_depth_4tu.py` to scratch, load both with importlib (register in sys.modules — dataclasses need it), import fixtures from `tests/test_eval_depth_4tu.py` (put tests/ on sys.path). Compare summary rows NaN-aware.
- v1.2–v1.6.1: summary rows whose scope has n_est = 0 carried `nan` in baseline_mae_* / baseline_pass_a_* (6 cols); since v1.7 (user decision b) they are blank. When comparing against a pre-v1.7 commit, blank those `nan` cells first (test N1 does this via a digest).
- Pass-rate tolerance boundary (0.45-0.15 = 0.30000000000000004) is now asserted (G1b, v1.7).
- Single-line mutations that a runtime guard blocks (ValueError/RuntimeError) are not "untestable": write the multi-line mutation that bypasses the guard and check the targeted test still catches it (done for G9).

- v1.7.1 (2026-11-07): the runner must also copy `docs/datasets/fourtu_duplicate_radargrams.csv` and `docs/processing/fourtu_lines_v1.csv` (tests read them via PROJECT_ROOT). Use `-k` to run only the version's tests (~25 s per mutant instead of ~85 s for 210 tests). Launching `envs\gpr-dev\python.exe` directly (Start-Process for BelowNormal) crashes with 0xc06d007f in numpy lstsq unless the env's `Library\bin` etc. are prepended to PATH.
- Known v1.7 test gaps (non-blocking, code correct by review): caveat scan restricted to the detected zero survives (D8 only edits detected/top); `main` passing all candidates (copies) to `site11_diagnostic` survives (no site-11 copy in fixtures; none in 4TU either).

- Zero-rule (timezero) mutations without copying files: a pytest plugin module in scratchpad that monkeypatches `gpr_detection.preprocess.timezero` private functions from env GPR_MUT, run via `python -I -c "sys.path.insert(...); pytest.main(['-p','mutplug',...])"` in a subprocess per mutant (~5 s each for 76 zero tests). Module-global lookups make the patch effective through `find_time_zero`.

**Why:** these cost time to rediscover and decide whether a survivor blocks a verdict.
**How to apply:** reuse the runner layout for the next verify of this script; treat `nan` baseline cells as pre-existing, not a regression. Related: [[apex-candidate-qc-facts]].
