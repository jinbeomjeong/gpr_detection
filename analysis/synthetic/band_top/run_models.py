"""Run the band-top experiment inputs on GPU 0 (single A-scan each).

  conda run -n gprmax-acceleration python -I analysis/synthetic/band_top/run_models.py --stage calib [--workers 4]

Reads models_<stage>.json, runs inputs/<id>.in -> data/synthetic_gprmax_band_top/h5/<id>.h5,
appends one JSON line per run to data/synthetic_gprmax_band_top/run_log.jsonl. Existing outputs are skipped.
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

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
DATA = PROJECT / "data" / "synthetic_gprmax_band_top"
LOCK = threading.Lock()


def run_one(mid, log_path):
    out_dir = DATA / "h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{mid}.h5"
    if out.exists():
        return mid, 0.0, "skip"
    cmd = [sys.executable, "-m", "gprMax", str(HERE / "inputs" / f"{mid}.in"), "-gpu", "0", "-o", str(out)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, stdin=subprocess.DEVNULL)
    wall = time.perf_counter() - t0
    (out_dir / f"{mid}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{mid}: gprMax exit {proc.returncode}")
    with LOCK, open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": mid, "wall_s": round(wall, 2), "command": " ".join(cmd)}) + "\n")
    return mid, wall, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    models = json.loads((HERE / f"models_{args.stage}.json").read_text(encoding="utf-8"))["models"]
    ids = [m["id"] for m in models if not args.only or m["id"] in args.only]
    DATA.mkdir(parents=True, exist_ok=True)
    log = DATA / "run_log.jsonl"
    t0 = time.perf_counter()
    fails = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(run_one, mid, log): mid for mid in ids}
        for f in as_completed(futs):
            try:
                mid, wall, st = f.result()
                print(f"{mid}: {st} {wall:.1f} s", flush=True)
            except Exception as exc:
                fails.append(str(exc))
                print(f"FAILED {exc}", flush=True)
    print(json.dumps({"stage": args.stage, "n": len(ids), "failed": len(fails),
                      "total_s": round(time.perf_counter() - t0, 1)}))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
