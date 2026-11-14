"""Judge B and B-p of M3-1 N2-a″ (pre-registered, blind).

Plan docs/experiments/m31_n2a3_retrain_plan.md v1.0 §4-1 (thresholds) and §6-3
(gates, output). Reads the N2-a″ band auxiliary metrics
(``n2a3_band_aux.json``, from ``n2a3_band_aux.py``), the N2-a' reference
(``n2a2_band_aux.json``, sha256 fixed in the plan) and the A1 verdict of the
same runs (``n2a3_a1_verdict.json``, from ``n2a_a1_judge.py``), and writes a
JSON verdict (--out). CPU only, read-only on inputs.

Gates (any failure: no verdict values, exit 1):
  G-b1  sha256 of --reference = --reference-sha256;
  G-b2  ``units.crop.fold_ids`` = [0, 1, 2, 3, 4] in both band files;
  G-b3  every fold of --band-aux has ``checks.confusion_matches`` and
        ``checks.ap50_matches`` true (the predictions reproduced the run's
        own evaluation, plan §6-2);
  G-b4  the A1 verdict has ``gates_pass`` true and was made for --run-prefix.

Criteria (crop unit, unweighted 5-fold means, exact ``fractions.Fraction`` of
the decimal values in the JSON, no rounding):
  B    mean recall >= 641/1000 and recall > the N2-a' recall of the same fold
       in at least 4 of 5 folds (equal is not better);
  B-p  mean precision >= 497/1000.
``verdict`` is "B and B-p met" only when both pass, else "B or B-p not met";
``failed`` lists the criteria that did not pass, in the order ["B", "B-p"].
The A1 result is copied from the A1 verdict, not recomputed.
``adoption_pending_a2`` = A1 and B and B-p (A2' is judged later, §6-4).

Blind (§7-1): stdout carries only {"status", "exit_code", "out"}. Exit codes:
0 judged (met or not), 1 gate failed, 2 input missing / --out exists (without
--overwrite) / malformed input, 3 internal error.
"""

import argparse
import hashlib
import json
import logging
import math
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

FOLDS = [0, 1, 2, 3, 4]
UNIT = "crop"
RECALL_THRESHOLD = Fraction(641, 1000)
PRECISION_THRESHOLD = Fraction(497, 1000)
MIN_FOLDS_BETTER = 4
VERDICT_SCHEMA = "n2a3_band_verdict/1"
RULES = "m31_n2a3_retrain_plan.md v1.0 §4-1 (B, B-p), §6-3 (gates G-b1..G-b4)"
VERDICT_MET = "B and B-p met"
VERDICT_NOT_MET = "B or B-p not met"
"""The pre-registered judging unit is B and B-p together (plan v1.0 §6-3)."""
EXACT_DECIMAL_DIGITS = 60

logger = logging.getLogger("n2a3_band_judge")


# ---------------------------------------------------------------- inputs


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise common.InputError(f"missing {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise common.InputError(f"{path} is not JSON: {error}") from error


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crop_block(document: dict, name: str) -> dict:
    try:
        return document["units"][UNIT]
    except (KeyError, TypeError) as error:
        raise common.InputError(f"{name}: no units.{UNIT} block") from error


def fold_values(document: dict, name: str, statistic: str) -> list[Fraction]:
    """``units.crop.statistics.<statistic>.folds`` as exact fractions.

    Raises:
        common.InputError: If the list is missing, not five numbers, or not finite.
    """
    try:
        values = _crop_block(document, name)["statistics"][statistic]["folds"]
    except (KeyError, TypeError) as error:
        raise common.InputError(f"{name}: no {statistic} fold values") from error
    if not isinstance(values, list) or len(values) != len(FOLDS):
        raise common.InputError(f"{name}: {statistic} needs {len(FOLDS)} fold values")
    return [_exact(value, f"{name} {statistic}") for value in values]


def _exact(value: Any, what: str) -> Fraction:
    """Exact fraction of the decimal text JSON wrote (``repr`` of the float)."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value):
        raise common.InputError(f"{what}: {value!r} is not a finite number")
    return Fraction(repr(float(value)))


# ----------------------------------------------------------------- gates


def gate_reference_sha(reference: Path, expected_sha256: str) -> dict:
    actual = _file_sha256(reference)
    return {
        "pass": actual == expected_sha256.lower(),
        "expected": expected_sha256.lower(),
        "actual": actual,
    }


def gate_fold_ids(band_aux: dict, reference: dict) -> dict:
    found = {}
    for name, document in (("band_aux", band_aux), ("reference", reference)):
        block = document.get("units", {}).get(UNIT, {}) if document else {}
        found[name] = block.get("fold_ids") if isinstance(block, dict) else None
    return {"pass": all(ids == FOLDS for ids in found.values()), "fold_ids": found}


def gate_reproduction(band_aux: dict) -> dict:
    block = band_aux.get("units", {}).get(UNIT, {})
    folds = block.get("folds") if isinstance(block, dict) else None
    rows = []
    for row in folds if isinstance(folds, list) else []:
        checks = row.get("checks", {})
        rows.append(
            {
                "fold": row.get("fold"),
                "confusion_matches": checks.get("confusion_matches") is True,
                "ap50_matches": checks.get("ap50_matches") is True,
            }
        )
    has_all_folds = [row["fold"] for row in rows] == FOLDS
    is_reproduced = all(
        row["confusion_matches"] and row["ap50_matches"] for row in rows
    )
    return {"pass": has_all_folds and is_reproduced, "per_fold": rows}


def gate_a1(a1_verdict: dict, run_prefix: str) -> dict:
    recorded_prefix = a1_verdict.get("inputs", {}).get("run_prefix")
    gates_pass = a1_verdict.get("gates_pass") is True
    return {
        "pass": gates_pass and recorded_prefix == run_prefix,
        "gates_pass": gates_pass,
        "run_prefix_recorded": recorded_prefix,
        "run_prefix_expected": run_prefix,
    }


# -------------------------------------------------------------- criteria


def exact_decimal_text(value: Fraction) -> str:
    """Decimal text of a mean of decimals (it always terminates)."""
    with localcontext() as context:
        context.prec = EXACT_DECIMAL_DIGITS
        text = Decimal(value.numerator) / Decimal(value.denominator)
    return format(text.normalize(), "f")


def _mean(values: list[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def _mean_view(values: list[Fraction], threshold: Fraction) -> dict:
    mean = _mean(values)
    return {
        "per_fold": [float(value) for value in values],
        "mean": float(mean),
        "mean_exact": exact_decimal_text(mean),
        "threshold": float(threshold),
        "threshold_exact": f"{threshold.numerator}/{threshold.denominator}",
        "pass": mean >= threshold,
    }


def criterion_b(recall: list[Fraction], reference_recall: list[Fraction]) -> dict:
    """B: mean recall >= 0.641 and strictly better than N2-a' in >= 4 folds."""
    mean_part = _mean_view(recall, RECALL_THRESHOLD)
    is_better = [new > old for new, old in zip(recall, reference_recall, strict=True)]
    folds_better = sum(is_better)
    return {
        "mean_recall": mean_part,
        "reference_recall_per_fold": [float(value) for value in reference_recall],
        "fold_better_than_reference": is_better,
        "folds_better": folds_better,
        "folds_better_needed": MIN_FOLDS_BETTER,
        "pass": mean_part["pass"] and folds_better >= MIN_FOLDS_BETTER,
    }


def paired_differences(values: list[Fraction], reference: list[Fraction]) -> list[dict]:
    return [
        {"fold": fold, "difference": float(new - old)}
        for fold, new, old in zip(FOLDS, values, reference, strict=True)
    ]


def decide(band_aux: dict, reference: dict, a1_verdict: dict) -> dict:
    """B, B-p, the copied A1 result and ``adoption_pending_a2``."""
    recall = fold_values(band_aux, "band_aux", "recall")
    precision = fold_values(band_aux, "band_aux", "precision")
    reference_recall = fold_values(reference, "reference", "recall")
    reference_precision = fold_values(reference, "reference", "precision")
    b = criterion_b(recall, reference_recall)
    b_p = {"mean_precision": _mean_view(precision, PRECISION_THRESHOLD)}
    b_p["pass"] = b_p["mean_precision"]["pass"]
    a1 = {
        "verdict": a1_verdict.get("verdict"),
        "a1": a1_verdict.get("a1"),
        "pass": (a1_verdict.get("a1") or {}).get("pass") is True,
    }
    failed = [name for name, result in (("B", b), ("B-p", b_p)) if not result["pass"]]
    return {
        "verdict": VERDICT_NOT_MET if failed else VERDICT_MET,
        "failed": failed,
        "b": b,
        "b_p": b_p,
        "paired_differences_vs_reference": {
            "recall": paired_differences(recall, reference_recall),
            "precision": paired_differences(precision, reference_precision),
        },
        "a1": a1,
        "adoption_pending_a2": a1["pass"] and b["pass"] and b_p["pass"],
    }


# ------------------------------------------------------------------ main


def _run_identity(args: argparse.Namespace) -> dict:
    git = common.git_state()
    return {
        "started_at": common.local_now(),
        "finished_at": None,
        "argv": list(sys.argv),
        "git": {"head": git["head"], "dirty": git["dirty"], "error": git["error"]},
        "inputs_sha256": {
            "band_aux": _file_sha256(args.band_aux),
            "reference": _file_sha256(args.reference),
            "a1_verdict": _file_sha256(args.a1_verdict),
        },
    }


def judge(args: argparse.Namespace) -> int:
    """Check the gates, then B and B-p; write the verdict; return the exit code."""
    if args.out.exists() and not args.overwrite:
        raise common.InputError(f"--out exists (use --overwrite): {args.out}")
    band_aux = _read_json(args.band_aux)
    reference = _read_json(args.reference)
    a1_verdict = _read_json(args.a1_verdict)
    identity = _run_identity(args)
    gates = {
        "G-b1_reference_sha256": gate_reference_sha(
            args.reference, args.reference_sha256
        ),
        "G-b2_fold_ids": gate_fold_ids(band_aux, reference),
        "G-b3_reproduction": gate_reproduction(band_aux),
        "G-b4_a1_verdict": gate_a1(a1_verdict, args.run_prefix),
    }
    failures = [name for name, gate in gates.items() if not gate["pass"]]
    out: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "rules": RULES,
        "run_identity": identity,
        "inputs": {
            "band_aux": str(args.band_aux),
            "reference": str(args.reference),
            "a1_verdict": str(args.a1_verdict),
            "run_prefix": args.run_prefix,
        },
        "gates": gates,
        "gates_pass": not failures,
        "failures": failures,
    }
    exit_code = 1
    if failures:
        out["verdict"] = "no verdict: gate failed"
    else:
        out.update(decide(band_aux, reference, a1_verdict))
        exit_code = 0
    identity["finished_at"] = common.local_now()
    common.write_json(args.out, out)
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--band-aux", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--a1-verdict", type=Path, required=True)
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="run folder name without the fold digit, "
        "e.g. yolo11m_aihubv1n2a3_<YYYYMMDD>_fold",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; 0 judged, 1 gate failed, 2 input / output refused, 3 error."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
    args = _parse_args(argv)
    try:
        code = judge(args)
    except common.InputError as error:
        logger.error("input error: %s", error)
        return common.print_final_line(2, args.out)
    except Exception:
        logger.exception("internal error")
        return common.print_final_line(3, args.out)
    return common.print_final_line(code, args.out)


if __name__ == "__main__":
    sys.exit(main())
