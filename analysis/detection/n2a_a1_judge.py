"""Judge A1 of M3-1 N2-a with the pre-registered rule of
docs/experiments/m31_n2a_retrain_plan.md v0.3.3 §3-1 (gates and thresholds),
implementation spec §5-5.

Reads, for folds k = 0..4,
  <run-root>/<run-prefix><k>/eval/last_crop/metrics.json, eval_images.txt,
  weights/last.pt, args.yaml, gpr_train_settings.yaml, train_record.json
and <run-root>/<m110-prefix><k>/eval/last_crop/metrics.json, and writes a JSON
verdict (--out). CPU only, read-only on inputs.

Gates (all must hold, else no verdict, exit 1):
  - evaluation list sha256 = the M1-10 run of the same fold (and the list file
    still has the recorded hash);
  - last.pt sha256 = the hash the evaluation recorded = the hash the training
    recorded (train_record.json, written by scripts/train_detector.py);
  - config sha256 = the blob of --config at --seal-commit (git show), for the
    file now and as recorded by every fold's training;
  - args.yaml: scale 0, translate 0.1, flipud 0, epochs 50, seed 0;
  - identity: the metrics are last.pt on the crop unit of fold k;
  - N2-a' only (the sealed config has a ``gpr_context`` block,
    docs/experiments/m31_n2a2_retrain_plan.md v1.0 §5 A1, §7-3): every fold's
    recorded ``gpr_context_sha256`` = the canonical-JSON sha256 of the sealed
    block, and all folds recorded the same context index file sha256.

A1 (§3-1): unweighted 5-fold mean of overall mAP50 >= 76.8 and of
pipe_hyperbola AP50 >= 90.9, in %, without rounding. The means are computed
exactly (``fractions.Fraction``) from the decimal values written in
metrics.json, so a mean of exactly 76.8 passes and 76.79999 fails.

Always blind (§4-2): stdout carries only the last line
{"status", "exit_code", "out"}; --blind is accepted so runner commands look
alike. Exit codes: 0 judged (A1 met or not), 1 gate failed, 2 input missing or
--out exists (without --overwrite), 3 internal error.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m31_d1_common as common

from gpr_detection.augment_geometry import settings_block_sha256

REPO = Path(__file__).resolve().parents[2]
FOLDS = (0, 1, 2, 3, 4)
DEFAULT_RUN_ROOT = REPO / "runs" / "detect"
DEFAULT_M110_PREFIX = "yolo11m_aihubv1_20261101_fold"
EVAL_SUBDIR = Path("eval") / "last_crop"
HYPERBOLA = "pipe_hyperbola"
MAP50_THRESHOLD_PP = Fraction("76.8")
HYPERBOLA_AP50_THRESHOLD_PP = Fraction("90.9")
PERCENT = 100
REQUIRED_ARGS = {"scale": 0, "translate": 0.1, "flipud": 0, "epochs": 50, "seed": 0}
"""args.yaml values the gate requires (§3-1)."""
VERDICT_SCHEMA = "n2a_a1_verdict/1"
RULES = "m31_n2a_retrain_plan.md v0.3.3 §3-1 (A1, gates), §5-5"
CONTEXT_RULES = "; m31_n2a2_retrain_plan.md v1.0 §5 (A1 + gpr_context gate), §7-3"
CONTEXT_BLOCK = "gpr_context"
GIT_TIMEOUT_S = 30
EXACT_DECIMAL_DIGITS = 60

logger = logging.getLogger("n2a_a1_judge")


@dataclass(frozen=True)
class FoldFiles:
    """Parsed inputs of one fold."""

    fold: int
    run_dir: Path
    metrics: dict
    m110_metrics: dict
    args_yaml: dict
    train_settings: dict
    train_record: dict
    weights: Path
    eval_list: Path


# ---------------------------------------------------------------- inputs


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise common.InputError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise common.InputError(f"missing {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise common.InputError(f"missing {path}")
    return path


def load_fold(
    run_root: Path, run_prefix: str, m110_prefix: str, fold: int
) -> FoldFiles:
    """Read one fold's files.

    Raises:
        common.InputError: If a file is missing.
    """
    run_dir = run_root / f"{run_prefix}{fold}"
    m110_dir = run_root / f"{m110_prefix}{fold}"
    return FoldFiles(
        fold=fold,
        run_dir=run_dir,
        metrics=_read_json(run_dir / EVAL_SUBDIR / "metrics.json"),
        m110_metrics=_read_json(m110_dir / EVAL_SUBDIR / "metrics.json"),
        args_yaml=_read_yaml(run_dir / "args.yaml"),
        train_settings=_read_yaml(run_dir / "gpr_train_settings.yaml"),
        train_record=_read_json(run_dir / "train_record.json"),
        weights=_require_file(run_dir / "weights" / "last.pt"),
        eval_list=_require_file(run_dir / EVAL_SUBDIR / "eval_images.txt"),
    )


def sealed_config_sha256(repo: Path, seal_commit: str, config: Path) -> str:
    """sha256 of the config blob at ``seal_commit`` (``git show``).

    The repository stores text with LF (``.gitattributes`` eol=lf), so the blob
    bytes equal the checked-out file bytes.

    Raises:
        common.InputError: If git cannot show the file at that commit.
    """
    return hashlib.sha256(sealed_config_blob(repo, seal_commit, config)).hexdigest()


def sealed_context_sha256(blob: bytes) -> str | None:
    """Canonical sha256 of the sealed ``gpr_context`` block, ``None`` without one."""
    sealed = yaml.safe_load(blob.decode("utf-8")) or {}
    block = sealed.get(CONTEXT_BLOCK) if isinstance(sealed, dict) else None
    return None if block is None else settings_block_sha256(block)


def sealed_config_blob(repo: Path, seal_commit: str, config: Path) -> bytes:
    """Bytes of the config blob at ``seal_commit`` (``git show``).

    Raises:
        common.InputError: If git cannot show the file at that commit.
    """
    try:
        relative = config.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as error:
        raise common.InputError(f"--config {config} is outside {repo}") from error
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "show", f"{seal_commit}:{relative}"],
            capture_output=True,
            timeout=GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise common.InputError(
            f"cannot read {relative} at {seal_commit}: {error!r}"
        ) from error
    return completed.stdout


# ----------------------------------------------------------------- gates


def _args_checks(args_yaml: dict) -> dict[str, bool]:
    return {
        f"args_{key}_is_{expected}": args_yaml.get(key) == expected
        for key, expected in REQUIRED_ARGS.items()
    }


def _context_checks(files: FoldFiles, sealed_context_sha: str | None) -> dict:
    """N2-a' gate: the fold trained with the sealed ``gpr_context`` block."""
    if sealed_context_sha is None:
        return {}
    recorded = files.train_settings.get("gpr_context_sha256")
    return {"gpr_context_matches_seal": recorded == sealed_context_sha}


def _context_index_sha256(files: FoldFiles) -> dict | None:
    index = files.train_settings.get("gpr_context_index") or {}
    return index.get("file_sha256")


def check_fold(
    files: FoldFiles, seal_sha256: str, sealed_context_sha: str | None = None
) -> dict:
    """All §3-1 gates of one fold, plus the identity of the evaluated target.

    ``sealed_context_sha`` (N2-a') adds the ``gpr_context`` gate.
    """
    k = files.fold
    weights_sha = common.sha256_file(files.weights)
    eval_list_sha = common.sha256_file(files.eval_list)
    recorded_eval_sha = files.metrics["eval_images"]["sha256"]
    checks = {
        "eval_list_equals_m110": recorded_eval_sha
        == files.m110_metrics["eval_images"]["sha256"],
        "eval_list_file_matches_record": eval_list_sha == recorded_eval_sha,
        "weights_match_eval_record": weights_sha == files.metrics["weights"]["sha256"],
        "weights_match_train_record": weights_sha
        == files.train_record.get("weights_sha256", {}).get("last"),
        "config_matches_seal": files.train_settings.get("config_sha256") == seal_sha256,
        "evaluated_last_pt": files.metrics["weights"]["name"] == "last",
        "evaluated_crop_unit": files.metrics["unit"] == "crop",
        "evaluated_fold_k": str(files.metrics["fold"]) == str(k),
        "trained_fold_k": str(files.train_settings.get("fold")) == str(k),
        **_args_checks(files.args_yaml),
        **_context_checks(files, sealed_context_sha),
    }
    row = {
        "fold": k,
        "pass": all(checks.values()),
        "checks": checks,
        "run_dir": str(files.run_dir),
        "weights_sha256": weights_sha,
        "eval_list_sha256": eval_list_sha,
        "config_sha256_recorded": files.train_settings.get("config_sha256"),
        "args": {key: files.args_yaml.get(key) for key in REQUIRED_ARGS},
    }
    if sealed_context_sha is not None:
        row["gpr_context_sha256_recorded"] = files.train_settings.get(
            "gpr_context_sha256"
        )
        row["gpr_context_index_sha256_recorded"] = _context_index_sha256(files)
    return row


# ----------------------------------------------------------- A1 arithmetic


def exact_pp(value: float) -> Fraction:
    """Fraction value x 100, exact from the decimal text JSON wrote (``repr``)."""
    return Fraction(repr(float(value))) * PERCENT


def exact_decimal_text(value: Fraction) -> str:
    """Decimal text of a mean of decimals (it always terminates)."""
    with localcontext() as context:
        context.prec = EXACT_DECIMAL_DIGITS
        text = Decimal(value.numerator) / Decimal(value.denominator)
    return format(text.normalize(), "f")


def _hyperbola_ap50(metrics: dict) -> float | None:
    for row in metrics["per_class"]:
        if row["name"] == HYPERBOLA:
            return row["ap50"]
    return None


def _criterion(values: list[Fraction], threshold: Fraction) -> dict:
    mean = sum(values, Fraction(0)) / len(values)
    return {
        "per_fold_pp": [float(value) for value in values],
        "mean_pp": float(mean),
        "mean_pp_exact": exact_decimal_text(mean),
        "threshold_pp": float(threshold),
        "pass": mean >= threshold,
    }


def decide(folds: list[FoldFiles]) -> dict:
    """A1 criteria; the caller has checked that no value is ``None``."""
    map50 = [exact_pp(files.metrics["map50"]) for files in folds]
    hyperbola = []
    for files in folds:
        ap50 = _hyperbola_ap50(files.metrics)
        assert ap50 is not None  # missing_decision_values() ran first
        hyperbola.append(exact_pp(ap50))
    map50_result = _criterion(map50, MAP50_THRESHOLD_PP)
    hyperbola_result = _criterion(hyperbola, HYPERBOLA_AP50_THRESHOLD_PP)
    is_met = map50_result["pass"] and hyperbola_result["pass"]
    return {
        "a1": {
            "map50": map50_result,
            "pipe_hyperbola_ap50": hyperbola_result,
            "pass": is_met,
        },
        "verdict": "A1 met" if is_met else "A1 not met",
    }


def missing_decision_values(folds: list[FoldFiles]) -> list[str]:
    missing = []
    for files in folds:
        if files.metrics.get("map50") is None:
            missing.append(f"fold {files.fold} map50")
        if _hyperbola_ap50(files.metrics) is None:
            missing.append(f"fold {files.fold} {HYPERBOLA} ap50")
    return missing


def report_only(folds: list[FoldFiles]) -> dict:
    """Values §3-1 lists as report-only: per-class AP50, mAP50-95, paired diffs."""
    rows = []
    for files in folds:
        m110_hyperbola = _hyperbola_ap50(files.m110_metrics)
        hyperbola = _hyperbola_ap50(files.metrics)
        rows.append(
            {
                "fold": files.fold,
                "ap50_by_class": {
                    row["name"]: row["ap50"] for row in files.metrics["per_class"]
                },
                "map50_95": files.metrics.get("map50_95"),
                "map50_minus_m110_pp": _difference_pp(
                    files.metrics.get("map50"), files.m110_metrics.get("map50")
                ),
                "pipe_hyperbola_ap50_minus_m110_pp": _difference_pp(
                    hyperbola, m110_hyperbola
                ),
            }
        )
    return {"per_fold": rows, "values_are": "fractions unless named _pp"}


def _difference_pp(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return PERCENT * (value - baseline)


# ------------------------------------------------------------------ main


def judge(args: argparse.Namespace, identity: dict) -> int:
    """Check the gates, then A1; write the verdict; return the exit code."""
    if args.out.exists() and not args.overwrite:
        raise common.InputError(f"--out exists (use --overwrite): {args.out}")
    folds = [
        load_fold(args.run_root, args.run_prefix, args.m110_prefix, k) for k in FOLDS
    ]
    sealed_blob = sealed_config_blob(args.repo, args.seal_commit, args.config)
    seal_sha256 = hashlib.sha256(sealed_blob).hexdigest()
    sealed_context_sha = sealed_context_sha256(sealed_blob)
    config_sha256 = common.sha256_file(_require_file(args.config))
    identity["config_sha256"] = config_sha256
    identity["seal_commit"] = args.seal_commit
    identity["sealed_config_sha256"] = seal_sha256

    fold_gates = [check_fold(files, seal_sha256, sealed_context_sha) for files in folds]
    for row in fold_gates:
        identity["weights_sha256"][str(row["fold"])] = row["weights_sha256"]
        identity["eval_list_sha256"][str(row["fold"])] = row["eval_list_sha256"]
    config_gate = {"config_file_matches_seal": config_sha256 == seal_sha256}
    if sealed_context_sha is not None:
        identity["sealed_gpr_context_sha256"] = sealed_context_sha
        index_hashes = [_context_index_sha256(files) for files in folds]
        identity["gpr_context_index_sha256"] = index_hashes[0]
        config_gate["context_index_same_in_all_folds"] = index_hashes[
            0
        ] is not None and all(hashes == index_hashes[0] for hashes in index_hashes)
    failures = [f"fold {row['fold']}" for row in fold_gates if not row["pass"]]
    if not all(config_gate.values()):
        failures.append("config file")
    failures.extend(f"None value: {item}" for item in missing_decision_values(folds))

    out: dict[str, Any] = {
        "schema": VERDICT_SCHEMA,
        "rules": RULES if sealed_context_sha is None else RULES + CONTEXT_RULES,
        "run_identity": identity,
        "inputs": {
            "run_root": str(args.run_root),
            "run_prefix": args.run_prefix,
            "m110_prefix": args.m110_prefix,
            "config": str(args.config),
        },
        "gates": {"per_fold": fold_gates, "config": config_gate},
        "gates_pass": not failures,
        "failures": failures,
    }
    exit_code = 1
    if failures:
        out["verdict"] = "no verdict: gate failed"
    else:
        out.update(decide(folds))
        out["report_only"] = report_only(folds)
        exit_code = 0
    identity["finished_at"] = common.local_now()
    common.write_json(args.out, out)
    return exit_code


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="N2-a run folder name without the fold digit, e.g. "
        "yolo11m_aihubv1n2a_<YYYYMMDD>_fold",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--seal-commit",
        required=True,
        help="commit whose --config blob the runs must have used",
    )
    parser.add_argument("--m110-prefix", default=DEFAULT_M110_PREFIX)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="git repository of --seal-commit (default: this repository)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="accepted for uniform runner commands; this judge is always blind",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns 0 judged, 1 gate failed, 2 input / output refused, 3 error."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
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
