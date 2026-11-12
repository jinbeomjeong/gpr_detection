"""Read-only collection for the M2-10 K7 judgment record (spec m210_k7_generator.md v1.2 section 8).

Gathers the facts the judgment document must state from files that already
exist; it computes no verdict and changes nothing:

* git: seal commits (first seal d6aaf0f, re-seal 8642e29) and their times,
  blobs of the sealed files at both commits, the sealed folder tree.
* run_info / judgment json of the K7 harness and judge (identity, seeds, G0,
  tests, ranks, K-R, K-F items, reports).
* runs_k7.csv SHA-256 recomputed; diagnostic tallies from its rows (noise-free
  rows, cur loss reasons in the defect models) - diagnostics only.
* runner logs (.done) of the re-run, identity text files of the harness and the
  main session.
* h5_compare reports (scratchpad ``h5cmp_k7``) and source_guard states
  (snapshots of run-1 h5, re-run h5 + FS, export) with a fresh ``verify``.
* SHA-256 of the archived pilot h5 (``pilot_prerun_h5``) for the S001-S003
  comparison record.

Paths in the output are repository-relative or ``<scratchpad>/...``; no absolute
paths. Output: ``docs/processing/m210_k7_judgment_check.json`` (LF).

Run (single process, BelowNormal, E cores 24-27):
    conda run -n gpr-dev python -I analysis/processing/m210_k7_judgment_check.py --scratchpad <dir>
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "processing" / "m210_k7_judgment_check.json"
FIRST_SEAL = "d6aaf0f"
SEAL = "8642e29"
RUN_DIR = ROOT / "data" / "processing" / "m210_k7"
K7_DATA = ROOT / "data" / "synthetic_gprmax_k7"
PY_DEV = Path(r"C:\ProgramData\miniforge3\envs\python-dev\python.exe")
SOURCE_GUARD = Path.home() / ".claude" / "tools" / "integrity" / "source_guard.py"
DEFECT_MODELS = ("K7_P2_D070", "K7_P2_D070_L", "K7_P3_D080")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace"
    ).stdout.strip()


def blob_at(commit: str, path: str) -> str | None:
    try:
        return git("rev-parse", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str | Path, scratch: Path) -> str:
    p = str(path)
    for base, label in ((str(scratch), "<scratchpad>"), (str(ROOT), "")):
        if p.lower().startswith(base.lower()):
            tail = p[len(base):].lstrip("\\/").replace("\\", "/")
            return f"{label}/{tail}" if label else tail
    return p.replace("\\", "/")


def seal_part(run_info: dict) -> dict:
    out = {}
    for c in (FIRST_SEAL, SEAL):
        out[c] = {
            "full": git("rev-parse", c),
            "commit_time": git("log", "-1", "--format=%cI", c),
            "subject": git("log", "-1", "--format=%s", c),
        }
    out["head"] = git("rev-parse", "HEAD")
    out["dirty_now"] = git("status", "--porcelain", "--untracked-files=no")
    files = sorted(run_info["sealed_files"])
    rows = []
    for path in files:
        b_seal, b_first = blob_at(SEAL, path), blob_at(FIRST_SEAL, path)
        work = git("hash-object", path) if (ROOT / path).exists() else None
        rows.append(
            {
                "path": path,
                "blob_8642e29": b_seal,
                "blob_d6aaf0f": b_first,
                "changed_since_first_seal": b_seal != b_first,
                "blob_working_tree": work,
                "working_equals_seal": work == b_seal,
            }
        )
    out["sealed_files"] = rows
    out["n_sealed_files"] = len(rows)
    out["changed_since_first_seal"] = [r["path"] for r in rows if r["changed_since_first_seal"]]
    out["working_mismatch"] = [r["path"] for r in rows if not r["working_equals_seal"]]
    tree = {c: git("rev-parse", f"{c}:analysis/synthetic/k7") for c in (FIRST_SEAL, SEAL)}
    out["sealed_folder_tree"] = {
        "analysis/synthetic/k7": tree,
        "same": tree[FIRST_SEAL] == tree[SEAL],
        "n_files_8642e29": len(git("ls-tree", "-r", "--name-only", SEAL, "analysis/synthetic/k7").splitlines()),
    }
    return out


def identity_files() -> dict:
    out = {}
    for name in ("pre_run.txt", "post_run.txt", "main_session_pre.txt", "main_session_post.txt"):
        p = RUN_DIR / name
        out[name] = p.read_text(encoding="utf-8").strip() if p.exists() else None
    err = RUN_DIR / "harness.err"
    out["harness.err_bytes"] = err.stat().st_size if err.exists() else None
    return out


def runner_logs() -> dict:
    d = K7_DATA / "runner_logs"
    done = {}
    for p in sorted(d.glob("*.done")):
        done[p.stem] = p.read_text(encoding="utf-8").strip()
    first_lines = {}
    for p in sorted(d.glob("*.log")):
        with p.open(encoding="utf-8", errors="replace") as f:
            first_lines[p.stem] = f.readline().strip()[:160]
    return {"done": done, "log_first_line": first_lines, "n_done": len(done)}


def h5_reports(scratch: Path) -> dict:
    out = {}
    for p in sorted((scratch / "h5cmp_k7").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = {
            "report": rel(p, scratch),
            "reference": rel(r["reference"]["path"], scratch),
            "candidate": rel(r["candidate"]["path"], scratch),
            "expect": r.get("expect"),
            "verdict": r.get("verdict"),
            "verdict_label": r.get("verdict_label"),
            "mismatch_reasons": r.get("mismatch_reasons"),
            "n_datasets": len(r.get("datasets") or []),
        }
    return out


def guard_states(scratch: Path) -> dict:
    out = {}
    for name in ("k7_run1_h5_guard.json", "k7v12_h5_guard.json", "k7v12_export_guard.json"):
        p = scratch / name
        if not p.exists():
            out[name] = None
            continue
        state = json.loads(p.read_text(encoding="utf-8"))
        vout = scratch / "gpr-processing-expert" / f"reverify_{p.stem}.json"
        proc = subprocess.run(
            [str(PY_DEV), "-I", "-B", str(SOURCE_GUARD), "verify", "--state", str(p),
             "--out", str(vout), "--format", "json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        v = json.loads(vout.read_text(encoding="utf-8")) if vout.exists() else {}
        out[name] = {
            "state": rel(p, scratch),
            "created_at": state.get("created_at"),
            "n_inputs": len(state.get("inputs", [])),
            "inputs": [
                {"path": rel(i["path"], scratch), "sha256": i["sha256"]} for i in state.get("inputs", [])
            ],
            "reverify_exit": proc.returncode,
            "reverify_pass": v.get("pass"),
            "reverify_at": v.get("verified_at"),
            "reverify_report_line": v.get("report_line"),
        }
    return out


def h5_sha_compare(guards: dict) -> dict:
    run1 = {Path(i["path"]).name: i["sha256"] for i in (guards["k7_run1_h5_guard.json"] or {}).get("inputs", [])}
    new = {Path(i["path"]).name: i["sha256"] for i in (guards["k7v12_h5_guard.json"] or {}).get("inputs", [])}
    same = sorted(n for n in run1 if new.get(n) == run1[n])
    diff = sorted(n for n in run1 if n in new and new[n] != run1[n])
    pilot = {}
    for name in ("K7_REF_S001", "K7_REF_S002", "K7_REF_S003"):
        cands = list((K7_DATA / "pilot_prerun_h5").rglob(f"{name}_merged.h5"))
        sha = sha256(cands[0]) if cands else None
        pilot[name] = {
            "pilot_path": rel(cands[0], Path(str(ROOT) + "_no_scratch")) if cands else None,
            "pilot_sha256": sha,
            "rerun_sha256": new.get(f"{name}_merged.h5"),
            "file_sha_same": sha is not None and sha == new.get(f"{name}_merged.h5"),
        }
    return {"run1_vs_rerun_file_sha_same": same, "run1_vs_rerun_file_sha_differ": diff, "pilot_vs_rerun": pilot}


def runs_diagnostics(runs_csv: Path) -> dict:
    import pandas as pd

    r = pd.read_csv(runs_csv, dtype={"model_id": str, "variant": str})
    nf = r[r["noise_free"] == True]  # noqa: E712
    noise_free = [
        {
            "model": row.model_id, "variant": row.variant,
            "pipe": None if pd.isna(row.pipe_index) else int(row.pipe_index),
            "accepted": None if pd.isna(row.accepted_apex) else bool(row.accepted_apex),
            "kept": None if pd.isna(row.kept_apex) else bool(row.kept_apex),
            "lost_reason_detail": None if pd.isna(row.lost_reason_detail) else row.lost_reason_detail,
            "start_is_local_max": None if pd.isna(row.c_start_is_local_max) else bool(row.c_start_is_local_max),
        }
        for row in nf.itertuples()
    ]
    noisy = r[(r["noise_free"] == False) & (r["variant"] == "cur")]  # noqa: E712
    cur = {}
    for (model, rho), g in noisy[noisy["model_id"].isin(DEFECT_MODELS)].groupby(["model_id", "rho"]):
        acc = g["accepted_apex"] == True  # noqa: E712
        kept = g["kept_apex"] == True  # noqa: E712
        lost = g[acc & ~kept]
        cur[f"{model}|{int(rho)}"] = {
            "n_rows": int(len(g)),
            "accepted": int(acc.sum()),
            "kept": int(kept.sum()),
            "accepted_not_kept": int(len(lost)),
            "accepted_not_kept_detail": {str(k): int(v) for k, v in lost["lost_reason_detail"].value_counts().items()},
            "not_accepted_reason": {str(k): int(v) for k, v in g[~acc]["lost_reason"].value_counts().items()},
            "kept_item_start_is_local_max_frac": (
                round(float((g.loc[kept, "c_start_is_local_max"] == True).mean()), 4)  # noqa: E712
                if kept.any() else None
            ),
        }
    r["kept_offset_ns"] = r["kept_apex_sample"] * 0.098 - r["t_a0_ns"]
    by_variant = {}
    noisy_all = r[(r["noise_free"] == False) & r["model_id"].isin(DEFECT_MODELS + ("K7_P1_D060",))]  # noqa: E712
    for (model, rho, variant), g in noisy_all.groupby(["model_id", "rho", "variant"]):
        acc = g["accepted_apex"] == True  # noqa: E712
        kept = g["kept_apex"] == True  # noqa: E712
        off = g.loc[kept, "kept_offset_ns"]
        by_variant[f"{model}|{int(rho)}|{variant}"] = {
            "n_rows": int(len(g)),
            "accepted": int(acc.sum()),
            "kept": int(kept.sum()),
            "lost_reason": {str(k): int(v) for k, v in g["lost_reason"].value_counts().items()},
            "kept_offset_ns_q10_q50_q90": [round(float(x), 3) for x in off.quantile([0.1, 0.5, 0.9])] if len(off) else None,
            "kept_offset_below_minus_0p5ns_frac": round(float((off < -0.5).mean()), 3) if len(off) else None,
            "kept_start_is_local_max_frac": (
                round(float((g.loc[kept, "c_start_is_local_max"] == True).mean()), 3) if kept.any() else None  # noqa: E712
            ),
        }
    nf2 = r[r["noise_free"] == True]  # noqa: E712
    nf_offsets = {
        f"{row.model_id}|{row.variant}|{'' if pd.isna(row.pipe_index) else int(row.pipe_index)}": (
            None if pd.isna(row.kept_offset_ns) else round(float(row.kept_offset_ns), 3)
        )
        for row in nf2.itertuples()
    }
    return {
        "note": "kept_offset_ns = kept_apex_sample * 0.098 ns - t_a0_ns (resampled grid 0.098 ns)",
        "noise_free_rows": noise_free,
        "noise_free_kept_offset_ns": nf_offsets,
        "cur_noisy_defect_models": cur,
        "by_variant": by_variant,
    }


def export_zeros() -> dict:
    out = {}
    for p in sorted(K7_DATA.glob("K7_*.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        out[p.stem] = meta.get("time_zero_candidates_ns")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratchpad", required=True, type=Path)
    args = ap.parse_args()
    scratch = args.scratchpad.resolve()
    run_info = json.loads((RUN_DIR / "run_info.json").read_text(encoding="utf-8"))
    judgment = json.loads((ROOT / "docs/processing/m210_k7_judgment.json").read_text(encoding="utf-8"))
    runs_csv = RUN_DIR / "runs_k7.csv"
    guards = guard_states(scratch)
    out = {
        "note": "read-only collection for docs/processing/m210_k7_judgment.md; no verdict computed here",
        "seal": seal_part(run_info),
        "judge_blob": {
            "working_tree": git("hash-object", "analysis/processing/m210_k7_judge.py"),
            "at_seal": blob_at(SEAL, "analysis/processing/m210_k7_judge.py"),
        },
        "run_info": {
            k: run_info.get(k)
            for k in ("spec", "protocol", "started_at", "finished_at", "mkl_cbwr", "seeds", "seed_overlap",
                      "tests", "g0", "guards", "generator_blob", "n_runs", "n_runs_without_zero",
                      "inputs_changed_during_run", "runs_csv_sha256")
        },
        "run_info_seal_at_end": {
            k: v for k, v in run_info["seal_at_end"].items() if k != "sealed_files"
        },
        "run_info_sealed_files_all_match": all(
            v.get("matches") for v in run_info["seal_at_end"]["sealed_files"].values()
        ),
        "runs_csv_sha256_recomputed": sha256(runs_csv),
        "judgment_json_sha256": sha256(ROOT / "docs/processing/m210_k7_judgment.json"),
        "judgment": {k: judgment.get(k) for k in ("ranks", "adopted", "rank1_reasons", "K_R", "K_R_table",
                                                   "K_F2_trivial_pass_cells", "K_F4_not_applicable_cells")},
        "judgment_rank1_pass": {k: judgment[k]["pass"] for k in ("rank1_identity", "rank1_seeds", "rank1_fixed_sets")},
        "identity_files": identity_files(),
        "runner_logs": runner_logs(),
        "h5_compare": h5_reports(scratch),
        "source_guard": guards,
        "h5_file_sha": h5_sha_compare(guards),
        "diagnostics": runs_diagnostics(runs_csv),
        "export_time_zero_candidates_ns": export_zeros(),
        "harness_detected_zero_ns": {m: v.get("t_zero_ns") for m, v in run_info["models"].items()},
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    OUT.write_text(text + "\n", encoding="utf-8", newline="\n")
    abs_hits = [s for s in ("C:\\\\Users", "C:/Users", "C:\\\\workspace") if s in text]
    print(json.dumps({"out": "docs/processing/m210_k7_judgment_check.json", "absolute_path_hits": abs_hits}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
