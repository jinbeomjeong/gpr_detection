"""Judge M3-1 D1 r2 with the pre-registered rules of
docs/experiments/m31_detection_input_tiling.md v0.5 §8-2 (3)(4); rules P and O
and the 3.0 %p threshold are those of v0.3.3, the baseline check is the v0.5
"same path" check G1-G3 (spec of this script §8-4 (i)).

Reads runs/detect/<prefix><k>/eval/<subdir>/metrics.json and objects.csv for
k = 0..4 (written by m31_d1_shift_scale.py), the M1-10
eval/last_crop/metrics.json and the M1-10 validator detections runs/analysis/m110_band_aux/fold<k>_crop/, and writes a
JSON verdict (--out). CPU only, read-only on inputs.

Order: G2 identity (weights and eval-list sha256, image count, validator image
set; v0.5.1 G2-a/b/c: run-recorded weights sha, one run_id, unchanged G1 input
sha) -> G3 n_gt (g1.00_dx+0 per class equals M1-10 with nothing ignored; the
judged conditions keep every pipe_hyperbola GT, gp6_dx+-64 lose exactly the
expected number) -> G1 same-path scoring (the
validator detections scored with the D1 code, 15 AP50 values within 1.0 %p of
M1-10) -> rule P -> rule O. Any gate failure, or a recall / n_gt the decision
needs being None, writes the gate results only (no P or O) and exits 1.

Aggregation (§8-2 (3)(a)-(e)):
  - units %p (fractions x 100);
  - every paired difference, min/max and "worse of two series" per fold first,
    then the unweighted 5-fold mean is compared with the 3.0 %p threshold;
  - shift deltas Delta(d) on the GT set each shifted condition keeps, paired
    by (image, gt_index) from objects.csv (§8-2 (3)(f), v0.5.1);
  - rule O clause order: both fail -> 50 % -> 75 %;
  - compression loss (report only): recall at g = 0.79 and 0.53 by linear
    interpolation in log g between neighbouring S points, per fold, then mean.

Stdout carries progress only; its last line is {"status", "exit_code", "out"}.
Exit codes: 0 gates passed and judged, 1 gate failed, 2 input missing or
--out exists (without --overwrite), 3 internal error.
"""

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

THRESHOLD_PP = 3.0
G1_TOL_PP = 1.0
G1_CLASSES = ("pipe_hyperbola", "cavity", "manhole")
HYP = "pipe_hyperbola"
BASE = "g1.00_dx+0"
S_POINTS = (
    ("g0.50_dx+0", 0.50),
    ("g0.74_dx+0", 0.74),
    ("g0.87_dx+0", 0.87),
    ("g1.00_dx+0", 1.00),
    ("g1.13_dx+0", 1.13),
    ("g1.28_dx+0", 1.28),
    ("gp6_dx+0", 6.0 / 4.06),
    ("g1.67_dx+0", 1.67),
)
# The 13 conditions of G3 that must keep every pipe_hyperbola GT (§8-2 (3)).
JUDGED = (
    "g0.50_dx+0",
    "g0.74_dx+0",
    "g1.00_dx+0",
    "gp6_dx+0",
    "g1.67_dx+0",
    "g1.00_dx-32",
    "g1.00_dx+32",
    "g1.00_dx-64",
    "g1.00_dx+64",
    "gp6_dx-32",
    "gp6_dx+32",
    "g1.00_dx+0_fillL64",
    "g1.00_dx+0_fillR64",
)
# Rule O also reads X6 delta(64) (P* 6, or X1 contaminated by fill).
DECISION_CONDITIONS = (*JUDGED, "gp6_dx-64", "gp6_dx+64")
ALL_CONDITIONS = (
    "g0.50_dx+0",
    "g0.74_dx+0",
    "g0.87_dx+0",
    "g1.00_dx+0",
    "g1.13_dx+0",
    "g1.28_dx+0",
    "gp6_dx+0",
    "g1.67_dx+0",
    "g1.00_dx-96",
    "g1.00_dx-64",
    "g1.00_dx-32",
    "g1.00_dx+32",
    "g1.00_dx+64",
    "g1.00_dx+96",
    "gp6_dx-64",
    "gp6_dx-32",
    "gp6_dx+32",
    "gp6_dx+64",
    "g1.00_dx+0_fillL64",
    "g1.00_dx+0_fillR64",
)
METRICS_SCHEMA = "m31_d1/2"
VERDICT_SCHEMA = "m31_d1_verdict/2"
RULES = "m31_detection_input_tiling.md v0.5.1 §8-2 (3)(4), §8-4 (i)(k)"
# Shifted conditions per series: (base, delta d, +d name, -d name), §8-2 (3)(f).
SHIFT_PAIRS = {
    "X1": ("g1.00_dx+0", {d: (f"g1.00_dx+{d}", f"g1.00_dx-{d}") for d in (32, 64, 96)}),
    "X6": ("gp6_dx+0", {d: (f"gp6_dx+{d}", f"gp6_dx-{d}") for d in (32, 64)}),
}
INTERNAL_TOL_PP = 1e-9
SECONDARY_CLASSES = ("cavity", "manhole")
MAP50_CONDITIONS = ("g1.00_dx+0", "gp6_dx+0", "g1.67_dx+0", "g0.50_dx+0", "g0.74_dx+0")

logger = logging.getLogger("m31_d1_judge")


@dataclass(frozen=True)
class FoldInputs:
    """Everything one fold contributes to the verdict.

    Attributes:
        fold: Fold index.
        metrics: r2 ``metrics.json`` of m31_d1_shift_scale.py.
        m110: M1-10 ``eval/last_crop/metrics.json``.
        weights: ``weights/last.pt`` of this fold.
        eval_list: ``eval/last_crop/eval_images.txt`` of this fold.
        export: M1-10 validator detections (G1 input).
        hyperbola_hits: Condition -> {(image, gt_index): hit at conf > 0.25}
            for the pipe_hyperbola GT kept in that condition (``objects.csv``).
    """

    fold: int
    metrics: dict
    m110: dict
    weights: Path
    eval_list: Path
    export: common.ValExport
    hyperbola_hits: dict[str, dict[tuple[str, int], int]]


class InternalCheckError(RuntimeError):
    """An internal consistency assertion failed; the judge exits with code 3."""


# ------------------------------------------------------------ arithmetic


def pp(x: float | None) -> float | None:
    return None if x is None else 100.0 * x


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _minus_pair_mean(
    base: float | None, a: float | None, b: float | None
) -> float | None:
    if base is None or a is None or b is None:
        return None
    return base - 0.5 * (a + b)


def _mean_or_none(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else float(np.mean(values))


def interp_log_g(values: dict[float, float], g: float) -> tuple[float, float, float]:
    """Linear interpolation in log g; returns (value, nearest_g, nearest_value)."""
    gs = sorted(values)
    lo = max(x for x in gs if x <= g)
    hi = min(x for x in gs if x >= g)
    if lo == hi:
        v = values[lo]
    else:
        t = (math.log(g) - math.log(lo)) / (math.log(hi) - math.log(lo))
        v = values[lo] + t * (values[hi] - values[lo])
    nearest = min(gs, key=lambda x: abs(math.log(x) - math.log(g)))
    return v, nearest, values[nearest]


def summarize(xs: list[float | None]) -> dict:
    """Per-fold values and 5-fold mean / SD (ddof 1), rounded to 3 decimals."""
    if any(x is None for x in xs):
        return {
            "per_fold": [None if x is None else round(x, 3) for x in xs],
            "mean": None,
            "sd": None,
        }
    arr = np.asarray(xs, dtype=float)
    return {
        "per_fold": [round(float(x), 3) for x in arr],
        "mean": round(float(arr.mean()), 3),
        "sd": round(float(arr.std(ddof=1)), 3) if arr.size > 1 else None,
    }


# ---------------------------------------------------------------- inputs


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise common.InputError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fold(run_root: Path, export_root: Path, subdir: str, fold: int) -> FoldInputs:
    """Read one fold's inputs.

    Raises:
        common.InputError: If a file is missing, is not a full r2 metrics file,
            or lacks a condition.
    """
    run = run_root / f"{common.RUN_PREFIX}{fold}"
    metrics = _read_json(run / "eval" / subdir / "metrics.json")
    if metrics.get("schema") != METRICS_SCHEMA:
        raise common.InputError(
            f"fold {fold}: not an r2 metrics file ({metrics.get('schema')})"
        )
    missing = [name for name in ALL_CONDITIONS if name not in metrics["conditions"]]
    if missing:
        raise common.InputError(f"fold {fold}: conditions missing {missing}")
    m110 = _read_json(run / "eval" / "last_crop" / "metrics.json")
    weights = run / "weights" / "last.pt"
    eval_list = run / "eval" / "last_crop" / "eval_images.txt"
    for path in (weights, eval_list):
        if not path.is_file():
            raise common.InputError(f"missing {path}")
    export = common.load_val_export(common.export_folder(export_root, fold))
    hits = read_hyperbola_hits(run / "eval" / subdir / "objects.csv")
    return FoldInputs(fold, metrics, m110, weights, eval_list, export, hits)


def read_hyperbola_hits(path: Path) -> dict[str, dict[tuple[str, int], int]]:
    """pipe_hyperbola rows of ``objects.csv`` keyed by (image, gt_index).

    Raises:
        common.InputError: If the file is missing, lacks ``gt_index`` (a 1st-run
            file) or repeats a key within one condition.
    """
    if not path.is_file():
        raise common.InputError(f"missing {path}")
    hits: dict[str, dict[tuple[str, int], int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "gt_index" not in (reader.fieldnames or []):
            raise common.InputError(f"{path}: no gt_index column")
        for row in reader:
            if row["class"] != HYP:
                continue
            per_condition = hits.setdefault(row["condition"], {})
            key = (row["image"], int(row["gt_index"]))
            if key in per_condition:
                raise common.InputError(f"{path}: repeated object {key}")
            per_condition[key] = int(row["hit_conf025"])
    return hits


# ----------------------------------------------------------------- gates


def check_identity(inputs: FoldInputs, is_single_run: bool) -> dict:
    """G2: weights, eval list and validator image set are those of M1-10.

    v0.5.1 adds G2-a (weights sha recorded by the run = recomputed), G2-b (one
    ``run_id`` over the five folds, passed in) and G2-c (G1 input sha256 at
    run time = at judge time).
    """
    k = inputs.fold
    run_identity = inputs.metrics.get("run_identity", {})
    recorded_g1 = run_identity.get("g1_inputs_sha256", {}).get(str(k))
    weights_sha = common.sha256_file(inputs.weights)
    eval_sha = common.sha256_file(inputs.eval_list)
    eval_stems = common.eval_list_stems(inputs.eval_list)
    recorded = (
        inputs.metrics.get("run_identity", {}).get("weights_sha256", {}).get(str(k))
    )
    checks = {
        "weights_match_m110_metrics": weights_sha == inputs.m110["weights"]["sha256"],
        "weights_match_audit": weights_sha.startswith(common.EXPECTED_WEIGHTS_SHA8[k]),
        "eval_list_match_m110_metrics": eval_sha
        == inputs.m110["eval_images"]["sha256"],
        "n_images_match_m110": inputs.metrics["n_images"]
        == inputs.m110["eval_images"]["count"],
        "eval_list_count_match_m110": len(eval_stems)
        == inputs.m110["eval_images"]["count"],
        "validator_images_equal_eval_list": set(inputs.export.stems) == set(eval_stems),
        "g2a_weights_match_run_record": recorded == weights_sha,
        "g2b_single_run_id": is_single_run,
        "g2c_g1_inputs_match_run_record": recorded_g1 == inputs.export.sha256,
    }
    return {
        "fold": k,
        "pass": all(checks.values()),
        "checks": checks,
        "weights": str(inputs.weights),
        "weights_sha256": weights_sha,
        "weights_sha8_expected": common.EXPECTED_WEIGHTS_SHA8[k],
        "weights_sha256_recorded_by_run": recorded,
        "g1_inputs_sha256_recorded_by_run": recorded_g1,
        "eval_list": str(inputs.eval_list),
        "eval_list_sha256": eval_sha,
        "n_images": inputs.metrics["n_images"],
        "n_images_m110": inputs.m110["eval_images"]["count"],
    }


def check_n_gt(inputs: FoldInputs) -> dict:
    """G3: g1.00_dx+0 keeps every GT; judged conditions keep every hyperbola GT.

    ``gp6_dx-64`` / ``gp6_dx+64`` must lose exactly the expected number of
    pipe_hyperbola GT (v0.5.1).
    """
    k = inputs.fold
    conditions = inputs.metrics["conditions"]
    base = conditions[BASE]["classes"]
    m110_instances = {pc["name"]: pc["instances"] for pc in inputs.m110["per_class"]}
    base_rows = {}
    for cls in common.CLASS_NAMES:
        expected = common.EXPECTED_N_GT[cls][k]
        row = {
            "n_gt": base[cls]["n_gt"],
            "n_ignored": base[cls]["n_ignored"],
            "expected": expected,
            "m110_instances": m110_instances.get(cls),
        }
        row["pass"] = (
            row["n_gt"] == expected == row["m110_instances"] and row["n_ignored"] == 0
        )
        base_rows[cls] = row
    judged_failures = []
    for name in (*JUDGED, *common.EXPECTED_GP6_64_HYP_IGNORED):
        hyp = conditions[name]["classes"][HYP]
        lost = common.EXPECTED_GP6_64_HYP_IGNORED.get(name, (0,) * 5)[k]
        expected_kept = common.EXPECTED_N_GT[HYP][k] - lost
        if hyp["n_ignored"] != lost or hyp["n_gt"] != expected_kept:
            judged_failures.append(name)
    ignored = {}
    for name, cond in conditions.items():
        counts = {
            cls: v["n_ignored"] for cls, v in cond["classes"].items() if v["n_ignored"]
        }
        if counts:
            ignored[name] = counts
    passed = all(row["pass"] for row in base_rows.values()) and not judged_failures
    return {
        "fold": k,
        "pass": passed,
        "g1.00_dx+0": base_rows,
        "judged_conditions_losing_hyperbola_gt": judged_failures,
        "ignored_by_condition": ignored,
    }


def check_same_path(inputs: FoldInputs) -> dict:
    """G1: validator detections scored with the D1 code equal M1-10 AP50."""
    k = inputs.fold
    eval_stems = common.eval_list_stems(inputs.eval_list)
    assertions = common.check_val_export(inputs.export, eval_stems, inputs.weights)
    scored = common.score_val_export(inputs.export, common.LABEL_DIR)
    m110_ap = {pc["name"]: pc["ap50"] for pc in inputs.m110["per_class"]}
    values = []
    for cls in common.CLASS_NAMES:
        ap_d1code = pp(scored[cls]["ap50"])
        ap_m110 = pp(m110_ap.get(cls))
        diff = _minus(ap_d1code, ap_m110)
        is_gated = cls in G1_CLASSES
        values.append(
            {
                "class": cls,
                "gated": is_gated,
                "ap50_d1code_pp": ap_d1code,
                "ap50_m110_pp": ap_m110,
                "diff_pp": diff,
                "pass": diff is not None and abs(diff) <= G1_TOL_PP,
            }
        )
    gated_pass = all(v["pass"] for v in values if v["gated"])
    return {
        "fold": k,
        "pass": all(assertions.values()) and gated_pass,
        "input_assertions": assertions,
        "input_sha256": inputs.export.sha256,
        "values": values,
    }


def missing_decision_values(folds: dict[int, FoldInputs]) -> list[str]:
    """Decision inputs that are None: hyperbola recall / n_gt of the used conditions."""
    missing = []
    for k, inputs in folds.items():
        for name in DECISION_CONDITIONS:
            hyp = inputs.metrics["conditions"][name]["classes"][HYP]
            if hyp["recall_conf025"] is None or hyp["n_gt"] in (None, 0):
                missing.append(f"fold {k} {name}")
    return missing


# --------------------------------------------------------- rules P and O


def _recalls(conditions: dict, cls: str = HYP) -> dict[str, float | None]:
    return {
        name: pp(cond["classes"][cls]["recall_conf025"])
        for name, cond in conditions.items()
    }


def _compression_loss(r: dict[str, float | None]) -> dict[str, float | None]:
    sv = {g: r[name] for name, g in S_POINTS}
    out: dict[str, float | None] = {}
    for g, key in ((0.79, "079"), (0.53, "053")):
        if any(v is None for v in sv.values()):
            out[f"R_g{key}_interp"], out[f"R_g{key}_nearest"] = None, None
            continue
        value, _, nearest_value = interp_log_g(sv, g)
        out[f"R_g{key}_interp"], out[f"R_g{key}_nearest"] = value, nearest_value
    return out


def paired_shift_delta(
    hits: dict[str, dict[tuple[str, int], int]], base: str, plus: str, minus: str
) -> float | None:
    """Delta(d) in %p on the GT set each shifted condition keeps (§8-2 (3)(f)).

    ``0.5 * [(R0|S+ - R(+d)|S+) + (R0|S- - R(-d)|S-)]`` where ``S+-`` are the
    pipe_hyperbola GT kept in the shifted condition and ``R0`` is the base
    condition (every GT kept, G3) read on the same objects. None when a
    shifted condition keeps no GT (possible only for report-only shifts).

    Raises:
        InternalCheckError: If a shifted condition keeps a GT the base lacks.
    """
    halves = []
    for shifted in (plus, minus):
        kept = hits.get(shifted, {})
        base_hits = hits.get(base, {})
        if not kept:
            return None
        if not set(kept) <= set(base_hits):
            raise InternalCheckError(f"{shifted}: objects not in {base}")
        base_recall = 100.0 * sum(base_hits[key] for key in kept) / len(kept)
        shifted_recall = 100.0 * sum(kept.values()) / len(kept)
        halves.append(base_recall - shifted_recall)
    return 0.5 * (halves[0] + halves[1])


def _check_objects_match_metrics(inputs: FoldInputs) -> None:
    """objects.csv must hold the same hyperbola GT count and recall as metrics.json."""
    for name in ALL_CONDITIONS:
        hyp = inputs.metrics["conditions"][name]["classes"][HYP]
        kept = inputs.hyperbola_hits.get(name, {})
        if len(kept) != hyp["n_gt"]:
            raise InternalCheckError(f"fold {inputs.fold} {name}: objects != n_gt")
        if kept and abs(sum(kept.values()) / len(kept) - hyp["recall_conf025"]) > 1e-12:
            raise InternalCheckError(f"fold {inputs.fold} {name}: objects != recall")


def shift_deltas(inputs: FoldInputs) -> dict[str, dict[int, float | None]]:
    """Paired Delta(d) per series; equal to the metrics formula when nothing is lost."""
    _check_objects_match_metrics(inputs)
    conditions = inputs.metrics["conditions"]
    r = _recalls(conditions)
    deltas: dict[str, dict[int, float | None]] = {}
    for series, (base, shifts) in SHIFT_PAIRS.items():
        deltas[series] = {}
        for d, (plus, minus) in shifts.items():
            paired = paired_shift_delta(inputs.hyperbola_hits, base, plus, minus)
            is_lossless = all(
                conditions[name]["classes"][HYP]["n_ignored"] == 0
                for name in (plus, minus)
            )
            unpaired = _minus_pair_mean(r[base], r[plus], r[minus])
            is_comparable = paired is not None and unpaired is not None
            if (
                is_lossless
                and is_comparable
                and abs(paired - unpaired) > INTERNAL_TOL_PP
            ):
                raise InternalCheckError(
                    f"fold {inputs.fold} {series} d{d}: k3 mismatch"
                )
            deltas[series][d] = paired
    return deltas


def fold_quantities(inputs: FoldInputs) -> dict[str, float | None]:
    """Per-fold hyperbola quantities in %p (decision ones are never None here)."""
    r = _recalls(inputs.metrics["conditions"])
    r1, r6, r167 = r["g1.00_dx+0"], r["gp6_dx+0"], r["g1.67_dx+0"]
    deltas = shift_deltas(inputs)
    x1, x6 = deltas["X1"], deltas["X6"]
    q = {
        "R_g1": r1,
        "R_g1478": r6,
        "R_g167": r167,
        "dC_best": r1 - max(r6, r167),
        "dC_worst": r1 - min(r6, r167),
        "dB": r["g0.74_dx+0"] - r["g0.50_dx+0"],
        "X1_d32": x1[32],
        "X1_d64": x1[64],
        "X1_d96": x1[96],
        "X6_d32": x6[32],
        "X6_d64": x6[64],
        "dF": _minus_pair_mean(r1, r["g1.00_dx+0_fillL64"], r["g1.00_dx+0_fillR64"]),
        "worse_d32": max(x1[32], x6[32]),
        "worse_d64": max(x1[64], x6[64]),
        "E406": _mean_or_none([r["g0.50_dx+0"], r["g0.87_dx+0"], r["g1.13_dx+0"]]),
        "E6": _mean_or_none([r["g0.74_dx+0"], r["g1.28_dx+0"], r["g1.67_dx+0"]]),
    }
    q.update(_compression_loss(r))
    return q


def secondary_quantities(conditions: dict, cls: str) -> dict[str, float | None]:
    """Report-only differences of cavity / manhole recall (%p)."""
    rc = _recalls(conditions, cls)
    base = rc["g1.00_dx+0"]
    worst_scale = (
        None
        if rc["gp6_dx+0"] is None or rc["g1.67_dx+0"] is None
        else min(rc["gp6_dx+0"], rc["g1.67_dx+0"])
    )
    return {
        "dC_worst": _minus(base, worst_scale),
        "X1_d32": _minus_pair_mean(base, rc["g1.00_dx+32"], rc["g1.00_dx-32"]),
        "X1_d64": _minus_pair_mean(base, rc["g1.00_dx+64"], rc["g1.00_dx-64"]),
        "dF": _minus_pair_mean(
            base, rc["g1.00_dx+0_fillL64"], rc["g1.00_dx+0_fillR64"]
        ),
    }


def rule_p(mean: dict[str, float]) -> dict:
    """Rule P: P* 6 only if the cost is small and the (lower-bound) gain large."""
    dc_worst, d_b = mean["dC_worst"], mean["dB"]
    p_star = 6.0 if (dc_worst <= THRESHOLD_PP and d_b >= THRESHOLD_PP) else 4.06
    return {
        "dC_worst_mean_pp": dc_worst,
        "dB_mean_pp": d_b,
        "p_star": p_star,
        "small_hyperbola_followup": p_star == 4.06 and d_b >= THRESHOLD_PP,
    }


def rule_o(mean: dict[str, float], p_star: float) -> dict:
    """Rule O on the series of the chosen P*; clause order both fail -> 50 % -> 75 %."""
    if p_star == 4.06:
        series, dF_mean = "X1", mean["dF"]
        contaminated = dF_mean > THRESHOLD_PP
        key32, key64 = (
            ("worse_d32", "worse_d64") if contaminated else ("X1_d32", "X1_d64")
        )
    else:
        series, dF_mean, contaminated = "X6", None, False
        key32, key64 = "X6_d32", "X6_d64"
    d32, d64 = mean[key32], mean[key64]
    if d32 > THRESHOLD_PP:
        overlap = "both_fail (provisional 75 %)"
    elif d64 <= THRESHOLD_PP:
        overlap = "50 % (O 128)"
    else:
        overlap = "75 % (O 192)"
    return {
        "series": series,
        "dF_mean_pp": dF_mean,
        "fill_contaminated": contaminated,
        "delta32_used_pp": d32,
        "delta64_used_pp": d64,
        "overlap": overlap,
    }


def decide(folds: dict[int, FoldInputs]) -> dict:
    """Aggregate per fold, then mean over folds, then apply rules P and O."""
    per_fold: dict[str, list] = {}
    s_curve: dict[str, list] = {name: [] for name, _ in S_POINTS}
    secondary: dict[str, dict[str, list]] = {cls: {} for cls in SECONDARY_CLASSES}
    map50: dict[str, list] = {name: [] for name in MAP50_CONDITIONS}
    for inputs in folds.values():
        conditions = inputs.metrics["conditions"]
        for key, value in fold_quantities(inputs).items():
            per_fold.setdefault(key, []).append(value)
        recalls = _recalls(conditions)
        for name, _ in S_POINTS:
            s_curve[name].append(recalls[name])
        for cls in SECONDARY_CLASSES:
            for key, value in secondary_quantities(conditions, cls).items():
                secondary[cls].setdefault(key, []).append(value)
        for name in map50:
            map50[name].append(pp(conditions[name]["map50"]))
    # Rules compare the exact 5-fold mean; the rounded summaries are for reading.
    exact_mean = {key: _mean_or_none(values) for key, values in per_fold.items()}
    p_rule = rule_p(exact_mean)
    o_rule = rule_o(exact_mean, p_rule["p_star"])
    return {
        "per_fold_and_mean_pp": {
            key: summarize(values) for key, values in per_fold.items()
        },
        "hyperbola_recall_S_curve_pp": {
            name: summarize(v) for name, v in s_curve.items()
        },
        "secondary_classes_pp": {
            cls: {key: summarize(v) for key, v in d.items()}
            for cls, d in secondary.items()
        },
        "map50_pp": {key: summarize(v) for key, v in map50.items()},
        "rule_p": p_rule,
        "rule_o": o_rule,
        "verdict": {"p_star": p_rule["p_star"], "overlap": o_rule["overlap"]},
    }


def path_difference(folds: dict[int, FoldInputs], g1: list[dict]) -> list[dict]:
    """Report item: D1 g1.00_dx+0 AP50 minus the G1 value (predict vs val path)."""
    g1_by_fold = {row["fold"]: {v["class"]: v for v in row["values"]} for row in g1}
    rows = []
    for k, inputs in folds.items():
        base = inputs.metrics["conditions"][BASE]["classes"]
        for cls in G1_CLASSES:
            d1 = pp(base[cls]["ap50"])
            same_path = g1_by_fold[k][cls]["ap50_d1code_pp"]
            rows.append(
                {
                    "fold": k,
                    "class": cls,
                    "ap50_d1_predict_pp": d1,
                    "ap50_val_detections_pp": same_path,
                    "diff_pp": _minus(d1, same_path),
                }
            )
    return rows


# ------------------------------------------------------------------ main


def _source_runs(folds: dict[int, FoldInputs]) -> dict:
    runs = {}
    for k, inputs in folds.items():
        identity = inputs.metrics.get("run_identity", {})
        runs[str(k)] = {
            "run_id": identity.get("run_id"),
            "git_head": identity.get("git", {}).get("head"),
        }
    run_ids = {row["run_id"] for row in runs.values()}
    return {"per_fold": runs, "single_run": len(run_ids) == 1 and None not in run_ids}


def _gate_failures(gates: dict[str, list[dict]]) -> list[str]:
    failures = []
    for gate, rows in gates.items():
        failures.extend(f"{gate} fold {row['fold']}" for row in rows if not row["pass"])
    return failures


def judge(args: argparse.Namespace, identity: dict) -> int:
    """Evaluate gates, then rules; write the verdict; return the exit code."""
    if args.out.exists() and not args.overwrite:
        raise common.InputError(f"--out exists (use --overwrite): {args.out}")
    folds = {
        k: load_fold(args.run_root, args.export_root, args.subdir, k)
        for k in common.FOLDS
    }
    print("inputs read", flush=True)
    source_runs = _source_runs(folds)
    gates: dict[str, list[dict]] = {"G2": [], "G3": [], "G1": []}
    gates["G2"] = [check_identity(f, source_runs["single_run"]) for f in folds.values()]
    print("G2 done", flush=True)
    for gate, check in (("G3", check_n_gt), ("G1", check_same_path)):
        gates[gate] = [check(inputs) for inputs in folds.values()]
        print(f"{gate} done", flush=True)
    failures = _gate_failures(gates)
    failures.extend(f"None value: {item}" for item in missing_decision_values(folds))
    out = {
        "schema": VERDICT_SCHEMA,
        "rules": RULES,
        "run_identity": identity,
        "source_runs": source_runs,
        "gate_order": ["G2", "G3", "G1"],
        "gates": gates,
        "gates_pass": not failures,
        "failures": failures,
    }
    if failures:
        out["verdict"] = "no verdict: gate failed"
        identity["finished_at"] = common.local_now()
        common.write_json(args.out, out)
        return 1
    out.update(decide(folds))
    out["path_difference_report_pp"] = path_difference(folds, gates["G1"])
    print("rules applied", flush=True)
    identity["finished_at"] = common.local_now()
    common.write_json(args.out, out)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subdir", default="m31_d1_r2")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-root", type=Path, default=common.RUN_ROOT)
    parser.add_argument("--export-root", type=Path, default=common.EXPORT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns 0 judged, 1 gate failed, 2 input / output refused, 3 error."""
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    identity = common.new_run_identity()
    identity["gpu"] = "cpu"
    try:
        code = judge(args, identity)
    except common.InputError as error:
        logger.error("input error: %s", error)
        return common.print_final_line(2, args.out)
    except Exception:
        logger.exception("internal error")
        return common.print_final_line(3, args.out)
    return common.print_final_line(code, args.out)


if __name__ == "__main__":
    sys.exit(main())
