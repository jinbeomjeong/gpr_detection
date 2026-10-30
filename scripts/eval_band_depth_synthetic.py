"""Pseudo-band depth regression on the gprMax set (CHECKLIST M2-6b, band spec 9-2).

A pipe parallel to the line and directly below it gives on every trace the
response of a crossing pipe's apex trace. As in
``analysis/processing/pipe_band_check.py`` Part A, each model's apex trace
(json ``apex.trace_index``) is tiled over the middle 40 % or 100 % of a
160-trace section, the other traces get the no-pipe trace (``_bg.npy`` middle
column), and white noise (0.05 x max |pipe-only apex trace|, one generator
with seed 0 over all models in order) is added.

The radargram carries survey metadata only (``load_synthetic``); the zero is
detected with the band traces excluded, the velocity is ``meta_fixed`` and
truth is used for the box and for judging BR1-BR7. Truth-dependent code stays
in this script (not in ``src``). ``tests/depth/test_band_synthetic.py``
imports it.

Outputs (``--out-dir``, default ``docs/processing``): ``<stem>.csv``,
``<stem>_checks.csv`` and ``<stem>.md``.

Example:
    conda run -n gpr-dev python -I scripts/eval_band_depth_synthetic.py
"""

import argparse
import csv
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from gpr_detection.depth.band import band_exclude_traces, estimate_band_depths
from gpr_detection.depth.config import DEFAULT_BAND_CONFIG
from gpr_detection.depth.estimate import depth_from_time, prepare_depth_inputs
from gpr_detection.depth.result import DepthResult
from gpr_detection.depth.velocity import meta_velocity
from gpr_detection.io.paths import DEFAULT_SYNTHETIC_ROOT
from gpr_detection.io.radargram import GridBox, Radargram
from gpr_detection.io.synthetic import (
    SyntheticTruth,
    list_synthetic_models,
    load_synthetic,
    load_synthetic_background,
    load_synthetic_truth,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "processing"
DEFAULT_STEM = "depth_band_synthetic_v1"

N_TRACES = 160
NOISE_REL = 0.05
SEED = 0
COVERAGES = (0.4, 1.0)
BACKGROUND_COLUMN = 1
"""``_bg.npy`` middle column (json x 1.5 m, trace 55)."""

BOX_UP_PERIODS = 0.4
BOX_DOWN_PERIODS = 0.8
"""Emulated detection box: ``[t_apex - 0.4/f_c, t_apex + 0.8/f_c]`` (band spec 9-2)."""

SINGLE_EVENT_GAP_M = 0.1
TWO_EVENT_GAP_M = 0.3
EXPECTED_ZERO_RULE = {
    0.0: "ground_direct_peak",
    SINGLE_EVENT_GAP_M: "air_single_event",
    TWO_EVENT_GAP_M: "air_surface_late",
}

LOGGER = logging.getLogger("eval_band_depth_synthetic")


# ----------------------------------------------------------------- pseudo band


def pseudo_band(
    model_id: str, coverage: float, rng: np.random.Generator, root: Path
) -> tuple[Radargram, GridBox, SyntheticTruth]:
    """Build one pseudo-band radargram and its detection-like box."""
    truth = load_synthetic_truth(model_id, root=root)
    base = load_synthetic(model_id, root=root)
    data = base.data.astype(np.float64)
    pipe_trace = data[:, truth.apex_trace_index]
    background = load_synthetic_background(model_id, root=root).astype(np.float64)
    background_trace = background[:, BACKGROUND_COLUMN]
    amplitude_ref = float(np.abs(pipe_trace - background_trace).max())

    n_band = round(coverage * N_TRACES)
    i0 = (N_TRACES - n_band) // 2
    i1 = i0 + n_band - 1
    section = np.repeat(background_trace[:, None], N_TRACES, axis=1)
    section[:, i0 : i1 + 1] = pipe_trace[:, None]
    section = section + rng.normal(0.0, NOISE_REL * amplitude_ref, section.shape)

    radargram = replace(
        base, data=section.astype(np.float32), line_id=f"{model_id}_band{coverage}"
    )
    assert radargram.dt_ns is not None and radargram.f_c_hz is not None
    period_ns = 1e9 / radargram.f_c_hz
    box = GridBox(
        i0=i0,
        i1=i1,
        k0=int(
            (truth.apex_envelope_peak_ns - BOX_UP_PERIODS * period_ns) / radargram.dt_ns
        ),
        k1=int(
            (truth.apex_envelope_peak_ns + BOX_DOWN_PERIODS * period_ns)
            / radargram.dt_ns
        ),
    )
    return radargram, box, truth


@dataclass(frozen=True)
class BandRun:
    """One model at one coverage."""

    truth: SyntheticTruth
    coverage: float
    result: DepthResult
    apex_rule_depth_m: float | None
    seconds: float

    @property
    def model_id(self) -> str:
        return self.truth.model_id

    @property
    def depth_error_m(self) -> float | None:
        if self.result.depth_top_m is None:
            return None
        return self.result.depth_top_m - self.truth.pipe_top_depth_m

    @property
    def band_minus_apex_rule_m(self) -> float | None:
        if self.result.depth_top_m is None or self.apex_rule_depth_m is None:
            return None
        return self.result.depth_top_m - self.apex_rule_depth_m


def evaluate_band(
    model_id: str, coverage: float, rng: np.random.Generator, root: Path
) -> BandRun:
    """Zero (band traces excluded) -> band depth, and the apex-rule depth."""
    started = time.perf_counter()
    radargram, box, truth = pseudo_band(model_id, coverage, rng, root)
    excluded = band_exclude_traces([box], radargram.n_traces, radargram.dx_m)
    inputs = prepare_depth_inputs(radargram, exclude_traces=excluded)
    (result,) = estimate_band_depths(
        radargram, [box], inputs.zero, cfg=DEFAULT_BAND_CONFIG, branch=inputs.branch
    )
    apex_rule = None
    v_meta = meta_velocity(radargram)
    if inputs.zero.t_zero_ns is not None and v_meta is not None:
        apex_rule = depth_from_time(
            v_meta,
            truth.apex_envelope_peak_ns - inputs.zero.t_zero_ns,
            air_gap=radargram.air_gap,
            offset_m=radargram.antenna_offset_m,
        )
    return BandRun(truth, coverage, result, apex_rule, time.perf_counter() - started)


def evaluate_all(root: Path = DEFAULT_SYNTHETIC_ROOT) -> list[BandRun]:
    """All models x coverages 0.4 and 1.0, one noise generator (seed 0)."""
    rng = np.random.default_rng(SEED)
    runs = []
    for model_id in list_synthetic_models(root):
        for coverage in COVERAGES:
            run = evaluate_band(model_id, coverage, rng, root)
            LOGGER.info("%s cover %.1f %s", model_id, coverage, run.result.status)
            runs.append(run)
    return runs


# ----------------------------------------------------------------- checks


@dataclass(frozen=True)
class Check:
    check_id: str
    model_id: str
    coverage: float
    measured: str
    criterion: str
    passed: bool


def _cm(value: float | None) -> str:
    return "None" if value is None else f"{value * 100:+.2f} cm"


def _depth_checks(run: BandRun) -> list[Check]:
    """BR1-BR4."""
    error = run.depth_error_m
    gap = run.truth.air_gap_m
    checks = [
        Check(
            "BR1",
            run.model_id,
            run.coverage,
            _cm(run.band_minus_apex_rule_m),
            "|d_band - d_apex_rule| <= 0.5 cm",
            run.band_minus_apex_rule_m is not None
            and abs(run.band_minus_apex_rule_m) <= 0.005,
        )
    ]
    if gap == 0:
        checks.append(
            Check(
                "BR2",
                run.model_id,
                run.coverage,
                _cm(error),
                "-5 <= d - d_top < 0 cm",
                error is not None and -0.05 <= error < 0,
            )
        )
    elif gap == SINGLE_EVENT_GAP_M:
        checks.append(
            Check(
                "BR3",
                run.model_id,
                run.coverage,
                _cm(error),
                "0 < d - d_top <= 5 cm",
                error is not None and 0 < error <= 0.05,
            )
        )
    elif gap == TWO_EVENT_GAP_M:
        checks.append(
            Check(
                "BR4",
                run.model_id,
                run.coverage,
                _cm(error),
                "|d - d_top| <= 2 cm",
                error is not None and abs(error) <= 0.02,
            )
        )
    return checks


def _band_checks(run: BandRun) -> list[Check]:
    """BR5-BR7."""
    band = run.result.band
    checks = []
    if run.coverage == 1.0:
        mode = None if band is None else band.bg_mode
        checks.append(
            Check(
                "BR5",
                run.model_id,
                run.coverage,
                f"bg_mode {mode}",
                'bg_mode == "none" (< 30 outside traces)',
                mode == "none",
            )
        )
    coverage = None if band is None else band.coverage
    checks.append(
        Check(
            "BR6",
            run.model_id,
            run.coverage,
            "None" if coverage is None else f"{coverage:.3f}",
            "coverage >= 0.95",
            coverage is not None and coverage >= 0.95,
        )
    )
    expected = EXPECTED_ZERO_RULE.get(run.truth.air_gap_m)
    checks.append(
        Check(
            "BR7",
            run.model_id,
            run.coverage,
            f"{run.result.zero_rule} (expected {expected})",
            "zero_rule as spec_depth_v1 R1-R3 (by air gap)",
            run.result.zero_rule == expected,
        )
    )
    return checks


def regression_checks(runs: Sequence[BandRun]) -> list[Check]:
    checks: list[Check] = []
    for run in runs:
        checks.extend(_depth_checks(run))
        checks.extend(_band_checks(run))
    return checks


# ----------------------------------------------------------------- outputs


def _round(value: float | None, digits: int = 4) -> float | str:
    return "" if value is None else round(float(value), digits)


def run_row(run: BandRun) -> dict[str, object]:
    result, band = run.result, run.result.band
    return {
        "model_id": run.model_id,
        "pipe_material": run.truth.pipe_material or "",
        "air_gap_m": run.truth.air_gap_m,
        "eps_r": run.truth.eps_r,
        "coverage": run.coverage,
        "status": result.status.value,
        "zero_rule": result.zero_rule or "",
        "t_zero_ns": _round(result.t_zero_ns),
        "d_true_m": run.truth.pipe_top_depth_m,
        "d_band_m": _round(result.depth_top_m),
        "d_err_cm": _round(
            None if run.depth_error_m is None else run.depth_error_m * 100, 2
        ),
        "band_minus_apex_rule_cm": _round(
            None
            if run.band_minus_apex_rule_m is None
            else run.band_minus_apex_rule_m * 100,
            2,
        ),
        "v_rule": result.v_rule or "",
        "model": "" if band is None else band.model,
        "bg_mode": "" if band is None else band.bg_mode,
        "band_coverage": "" if band is None else round(band.coverage, 3),
        "sigma_res_ns": "" if band is None else round(band.sigma_res, 4),
        "confidence": "" if band is None else round(band.confidence, 3),
        "flags": " ".join(result.flags),
        "seconds": round(run.seconds, 2),
    }


def check_row(check: Check) -> dict[str, object]:
    return {
        "check": check.check_id,
        "model_id": check.model_id,
        "coverage": check.coverage,
        "measured": check.measured,
        "criterion": check.criterion,
        "passed": check.passed,
    }


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _table(rows: Sequence[dict[str, object]], columns: Sequence[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _summary(checks: Sequence[Check]) -> list[dict[str, object]]:
    rows = []
    for check_id in dict.fromkeys(check.check_id for check in checks):
        group = [check for check in checks if check.check_id == check_id]
        failed = [f"{c.model_id}@{c.coverage}" for c in group if not c.passed]
        rows.append(
            {
                "check": check_id,
                "criterion": group[0].criterion,
                "passed": f"{len(group) - len(failed)}/{len(group)}",
                "verdict": "PASS" if not failed else "FAIL",
                "failed": ", ".join(failed) or "-",
            }
        )
    return rows


def write_markdown(
    path: Path, runs: Sequence[BandRun], checks: Sequence[Check], elapsed_s: float
) -> None:
    from gpr_detection.run_info import git_state, local_timestamp

    git = git_state(PROJECT_ROOT)
    br1 = [
        run.band_minus_apex_rule_m
        for run in runs
        if run.band_minus_apex_rule_m is not None
    ]
    lines = [
        "# 의사 띠 깊이 회귀 v1 (M2-6b, spec_band_depth_v1 §9-2)",
        "",
        f"- 생성: `scripts/eval_band_depth_synthetic.py`, {local_timestamp()}, "
        f"commit {git['commit']} (dirty {git['dirty']})",
        f"- 구성: 20개 모델 x 덮음 {COVERAGES}, 측선 {N_TRACES} 트레이스, "
        f"잡음 {NOISE_REL} x max|관 단독 꼭짓점 트레이스|, 시드 {SEED}",
        "- 영점: 띠 트레이스(+여유) 제외 검출값, v: `meta_fixed`(json eps_r)",
        f"- 실행 시간: {elapsed_s:.1f} s (단일 프로세스)",
        f"- BR1 범위(띠 - 꼭짓점 규칙): {min(br1) * 100:+.2f} ~ "
        f"{max(br1) * 100:+.2f} cm",
        "",
        "## BR1-BR7 판정",
        "",
        *_table(
            _summary(checks), ("check", "criterion", "passed", "verdict", "failed")
        ),
        "",
        "## 모델 x 덮음 결과",
        "",
        *_table(
            [run_row(run) for run in runs],
            (
                "model_id",
                "coverage",
                "air_gap_m",
                "status",
                "zero_rule",
                "d_err_cm",
                "band_minus_apex_rule_cm",
                "model",
                "bg_mode",
                "band_coverage",
                "confidence",
                "flags",
            ),
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate, write csv/md; exit 1 when any check failed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    started = time.perf_counter()
    runs = evaluate_all(args.root)
    checks = regression_checks(runs)
    elapsed_s = time.perf_counter() - started
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / f"{args.stem}.csv", [run_row(run) for run in runs])
    write_csv(args.out_dir / f"{args.stem}_checks.csv", [check_row(c) for c in checks])
    write_markdown(args.out_dir / f"{args.stem}.md", runs, checks, elapsed_s)
    failed = [check for check in checks if not check.passed]
    LOGGER.info("checks: %d passed, %d failed", len(checks) - len(failed), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
