"""Run the synthetic gprMax model set on GPU and merge A-scans into B-scans.

Usage (from any directory):
  conda run -n gprmax-acceleration python -I scripts/synthetic/run_models.py [--only M01 M02] [--ntraces N]
      [--skip-bg] [--raw-dir DIR] [--tag TAG] [--workers 4]

For each model: pipe B-scan (n_traces A-scans, --geometry-fixed) and background
(3 A-scans, no pipe). Raw per-trace files are merged with gprMax's
``outputfiles_merge.merge_files`` and removed. A timing/log line per run is
appended to ``<raw-dir>/run_log.jsonl``.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_LOG_LOCK = threading.Lock()

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DEFAULT_RAW = PROJECT / "data" / "synthetic_gprmax" / "_raw"


def run_one(input_file, out_dir, name, ntraces, log_path, tag):
    from gprMax.toolboxes.Utilities.outputfiles_merge import merge_files

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{name}.h5"
    cmd = [sys.executable, "-m", "gprMax", str(input_file), "-gpu", "0", "-n", str(ntraces), "-o", str(base)]
    if ntraces > 1:
        cmd.append("--geometry-fixed")
    t_start = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, stdin=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t_start
    (out_dir / f"{name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gprMax failed for {name} (exit {proc.returncode}); see {out_dir / (name + '.log')}")
    if ntraces > 1:
        files = sorted(out_dir.glob(f"{name}[0-9]*.h5"), key=lambda p: int(p.stem[len(name):]))
        if len(files) != ntraces:
            raise RuntimeError(f"{name}: expected {ntraces} trace files, found {len(files)}")
        merged = out_dir / f"{name}_merged.h5"
        if merged.exists():
            merged.unlink()
        merge_files(files, merged, removefiles=True)
    else:
        merged = base
    entry = {"name": name, "tag": tag, "ntraces": ntraces, "wall_s": round(elapsed, 2), "merged": str(merged),
             "command": " ".join(cmd)}
    with _LOG_LOCK, open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"{name}: {ntraces} traces, {elapsed:.1f} s -> {merged}", flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--ntraces", type=int, default=None, help="override trace count (benchmark)")
    parser.add_argument("--skip-bg", action="store_true")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--tag", default="production")
    parser.add_argument("--workers", type=int, default=1, help="concurrent gprMax processes on GPU 0")
    parser.add_argument("--fs-only", action="store_true",
                        help="run only the 1-trace free-space references (air-gap models), sequentially")
    args = parser.parse_args()

    specs = json.loads((HERE / "model_specs.json").read_text(encoding="utf-8"))
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    log_path = raw / "run_log.jsonl"
    selected = [s for s in specs if not args.only or s["model_id"] in args.only]
    # Longest models first so the pool finishes evenly.
    selected.sort(key=lambda s: -s["est_iterations"] * s["n_traces"])

    if args.fs_only:
        for spec in selected:
            if spec.get("fs_reference"):
                mid = spec["model_id"]
                run_one(HERE / "inputs" / f"{mid}_fs.in", raw / mid, f"{mid}_fs", 1, log_path, args.tag)
        return

    def job(spec):
        mid = spec["model_id"]
        out_dir = raw / mid
        n = args.ntraces or spec["n_traces"]
        run_one(HERE / "inputs" / f"{mid}.in", out_dir, mid, n, log_path, args.tag)
        if not args.skip_bg:
            run_one(HERE / "inputs" / f"{mid}_bg.in", out_dir, f"{mid}_bg", len(spec["bg_trace_positions_m"]),
                    log_path, args.tag)
        return mid

    # Small 2D grids are kernel-launch bound on the GPU; several gprMax processes share it well.
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(job, spec): spec["model_id"] for spec in selected}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # report all failures at the end
                failures.append(f"{futures[fut]}: {exc}")
                print(f"FAILED {futures[fut]}: {exc}", flush=True)
    if failures:
        raise SystemExit("Failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
