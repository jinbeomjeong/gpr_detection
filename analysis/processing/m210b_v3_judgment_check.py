"""M2-10 (b) v3: read-only collection for the judgment report (no judgment, no recount of ranks).

Prints the v3 judgment json items the report quotes and the identity facts the report
must state (sealed blobs vs git ls-tree, judge blob, run_info protocol and sealed set,
reference-h5 hashes from the source_guard state files, BG array comparisons, runner-log
first time). Writes docs/processing/m210b_v3_judgment_check.json.

Run: conda run -n gpr-dev python -I analysis/processing/m210b_v3_judgment_check.py <scratchpad dir>
"""

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEAL = "77f5f961d8d05d5bcccef026de9ea70a265fc332"
RUN = ROOT / "data" / "processing" / "m210b_s4s6_v3"
OUT = ROOT / "docs" / "processing" / "m210b_v3_judgment_check.json"


EZ_DATASET = "rxs/rx1/Ez"
H5_SIDE_KEYS = ("path", "size_bytes", "Iterations", "dt", "ntraces", "gprMax")
CORE_ATTRS_QUOTED = ("Iterations", "dt")
"""h5_compare attrs the report quotes by name (all core attrs are summarised too)."""
ABSOLUTE_PATH = re.compile(r"[A-Za-z]:(?:\\\\|/)|Users(?:\\\\|/)", re.IGNORECASE)
"""Searched in the json text (backslashes doubled there): a drive-letter path or a
user folder left after ``portable_paths``."""
HOME_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\"\s]+", re.IGNORECASE)


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def h5_compare_summary(report: dict) -> dict:
    """The parts of one h5_compare report the judgment quotes (never truncated).

    ``ez`` is the ``rxs/rx1/Ez`` dataset entry (``None`` if the report has none);
    ``core_attrs`` maps every core attribute to its ``equal`` flag.
    """
    datasets = report.get("datasets", [])
    ez = next((d for d in datasets if d.get("path") == EZ_DATASET), None)
    attrs = {a["name"]: a for a in report.get("attrs", [])}
    core = {name: a.get("equal") for name, a in attrs.items() if a.get("core")}

    def side(key: str) -> dict:
        record = report.get(key) or {}
        return {k: record.get(k) for k in H5_SIDE_KEYS}

    def quoted(name: str) -> dict:
        attr = attrs.get(name) or {}
        return {key: attr.get(key) for key in ("equal", "ref", "cand")}

    return {
        "verdict": report.get("verdict"),
        "verdict_label": report.get("verdict_label"),
        "expect": report.get("expect"),
        "aborted": report.get("aborted"),
        "mismatch_reasons": report.get("mismatch_reasons"),
        "reference": side("reference"),
        "candidate": side("candidate"),
        "ez": None if ez is None else {
            "path": ez.get("path"),
            "bit_equal": ez.get("bit_equal"),
            "verdict": ez.get("verdict"),
            "shape_ref": ez.get("shape_ref"),
            "shape_cand": ez.get("shape_cand"),
            "dtype_ref": ez.get("dtype_ref"),
            "dtype_cand": ez.get("dtype_cand"),
        },
        "attrs_quoted": {name: quoted(name) for name in CORE_ATTRS_QUOTED},
        "core_attrs": core,
        "core_attrs_all_equal": bool(core) and all(core.values()),
        "auxiliary_bit_equal": {
            a.get("path"): a.get("bit_equal") for a in report.get("auxiliary", [])
        },
    }


def _portable(text: str, prefixes: list[tuple[str, str]]) -> str:
    """Replace known absolute prefixes (either slash style, any case) in ``text``.

    A string that contained one is written with forward slashes afterwards.
    """
    original = text
    for prefix, label in prefixes:
        for variant in {prefix, prefix.replace("\\", "/")}:
            pattern = re.compile(re.escape(variant) + r"[\\/]*", re.IGNORECASE)
            text = pattern.sub(lambda _match, lab=label: lab, text)
    text = HOME_PATH.sub("<home>/", text)
    return text.replace("\\", "/") if text != original else text


def portable_paths(value, scratch: Path):
    """Copy of ``value`` with local absolute paths made portable (v3 check, audit).

    Paths inside the repository become repository-relative, paths in the
    scratchpad become ``<scratchpad>/...``, any other user folder ``<home>/...``.
    """
    prefixes = [
        (str(scratch.resolve()), "<scratchpad>/"),
        (str(ROOT.resolve()), ""),
    ]
    if isinstance(value, dict):
        return {
            portable_paths(k, scratch): portable_paths(v, scratch)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [portable_paths(v, scratch) for v in value]
    if isinstance(value, str):
        return _portable(value, prefixes)
    return value


def main():
    scratch = Path(sys.argv[1])
    j = json.loads((ROOT / "docs" / "processing" / "m210b_v3_judgment.json").read_text(encoding="utf-8"))
    info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
    out = {"note": "read-only collection for the v3 judgment report"}
    out["judgment_keys"] = list(j.keys())
    sealed = info["seal"]["sealed_files"]
    tree = {}
    for line in git("ls-tree", "-r", SEAL).splitlines():
        meta, path = line.split("\t", 1)
        tree[path] = meta.split()[2]
    out["sealed_n"] = len(sealed)
    out["sealed_blob_vs_tree_mismatch"] = [p for p, v in sealed.items() if tree.get(p) != v.get("blob_at_seal")]
    out["sealed_all_match_run"] = all(v.get("matches") for v in sealed.values())
    out["sealed_at_end"] = {k: v for k, v in (info.get("seal_at_end") or {}).items() if k != "sealed_files"}
    out["sealed_at_end_n_match"] = sum(1 for v in (info.get("seal_at_end") or {}).get("sealed_files", {}).values() if v.get("matches"))
    out["sealed_table"] = {p: v.get("blob_at_seal") for p, v in sorted(sealed.items())}
    out["judge_blob_now"] = git("hash-object", "analysis/processing/m210b_v3_judge.py")
    out["judge_blob_seal"] = git("rev-parse", f"{SEAL}:analysis/processing/m210b_v3_judge.py")
    out["spec_blob_seal"] = git("rev-parse", f"{SEAL}:docs/processing/m210b_selector_floor_v3.md")
    out["spec_blob_now"] = git("hash-object", "docs/processing/m210b_selector_floor_v3.md")
    out["seal_time"] = git("show", "-s", "--format=%cI", SEAL)
    out["run_info"] = {k: info.get(k) for k in ("protocol", "started_at", "finished_at", "n_runs", "n_runs_without_zero",
                                                  "seed_overlap", "mkl_cbwr", "numpy", "inputs_changed_during_run",
                                                  "runs_csv_sha256", "models_run")}
    out["run_info"]["guards"] = info.get("guards")
    out["run_info"]["seeds"] = info.get("seeds")
    out["run_info"]["seal_head"] = info["seal"].get("head_commit")
    out["run_info"]["dirty_start"] = info["seal"].get("dirty_files")
    out["run_info"]["git_failures"] = info["seal"].get("git_failures")
    out["run_info"]["wrapper_regression"] = info.get("wrapper_regression")
    out["runs_sha_now"] = hashlib.sha256((RUN / "runs_v3.csv").read_bytes()).hexdigest()
    out["pre_run"] = (RUN / "pre_run.txt").read_text(encoding="utf-8").strip()
    out["post_run"] = (RUN / "post_run.txt").read_text(encoding="utf-8").strip()
    # V3_SEALED_FILES subset check
    sys.path.insert(0, str(ROOT / "analysis" / "processing"))
    import m210b_s4s6_pipeline as hp  # noqa: E402
    v3list = list(getattr(hp, "V3_SEALED_FILES", []))
    out["V3_SEALED_FILES_n"] = len(v3list)
    out["V3_SEALED_FILES_missing_in_run_info"] = [p for p in v3list if p not in sealed]
    # guards
    for name in ("s4s6v3_refh5_guard.json", "s4s6v3_export_guard.json"):
        p = scratch / name
        if p.exists():
            g = json.loads(p.read_text(encoding="utf-8"))
            out[name] = g
    # BG compare
    bg = {}
    for p in sorted((scratch / "bgcmp_v3").glob("BG_*.json")):
        bg[p.stem] = h5_compare_summary(json.loads(p.read_text(encoding="utf-8")))
    out["bg_compare"] = bg
    # runner logs
    rl = ROOT / "data" / "synthetic_gprmax_s4s6_v3" / "runner_logs"
    times = sorted((datetime.fromtimestamp(x.stat().st_ctime).isoformat(timespec="seconds"), x.name) for x in rl.glob("*.log"))
    dones = sorted((datetime.fromtimestamp(x.stat().st_mtime).isoformat(timespec="seconds"), x.name) for x in rl.glob("*.done"))
    out["runner_first_log"] = times[:2]
    out["runner_last_done"] = dones[-1:] if dones else None
    out["n_runner_done"] = len(dones)
    pre = ROOT / "data" / "synthetic_gprmax_s4s6_v3" / "pilot_prerun_h5"
    out["pilot_prerun_files"] = sorted((x.name, datetime.fromtimestamp(x.stat().st_mtime).isoformat(timespec="seconds")) for x in pre.iterdir()) if pre.exists() else None
    out["log_dir_ctime"] = datetime.fromtimestamp(RUN.stat().st_ctime).isoformat(timespec="seconds")
    err = RUN / "harness.err"
    out["harness_err_bytes"] = err.stat().st_size if err.exists() else None
    out = portable_paths(out, scratch)
    text = json.dumps(out, ensure_ascii=False, indent=1, default=str)
    leftovers = ABSOLUTE_PATH.findall(text)
    if leftovers:
        raise SystemExit(f"absolute paths left in the output: {leftovers[:5]}")
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
