"""4TU duplicate radargram scan (read-only on the 4TU root of configs/paths.yaml).

Stage 1: SHA-256 of every npy/json/png-raw in 05_radargrams_npy, grouped by npy hash.
Stage 2: for every npy, read the source SEG-Y from the original zip (zipfile, no extract),
         hash the whole member, trace-data-only bytes, text header, first trace header,
         and zip directory info (CRC, size, date_time).
Stage 3: trace-level (column) hashes to find partial overlaps between different files.
Output: data/analysis/fourtu_dup/fourtu_dup_scan.json (derived, git-ignored; no GT
depth values are read). Seed: none (deterministic).

Run order: fourtu_dup_scan -> fourtu_dup_analyze -> fourtu_dup_extra ->
fourtu_dup_build_csv (writes docs/datasets/fourtu_duplicate_radargrams.csv).
Example (from the project root, PowerShell):
    $log = "data/analysis/fourtu_dup/fourtu_dup_scan.log"
    conda run -n gpr-dev python -I analysis/data/fourtu_dup_scan.py *> $log
"""
import hashlib
import json
import struct
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from gpr_detection.io.paths import load_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "analysis" / "fourtu_dup"
ROOT = load_dataset_paths().fourtu
NPY = ROOT / "05_radargrams_npy"
ZIPS = ROOT / "00_원본"
OUT = OUT_DIR / "fourtu_dup_scan.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def parse_sgy(buf):
    info = {}
    info["text_sha"] = sha(buf[:3200])
    info["text_head"] = buf[:3200].decode("cp037", "replace")[:0]
    try:
        info["text_ascii"] = buf[:3200].decode("ascii", "replace")
    except Exception:
        info["text_ascii"] = ""
    bh = buf[3200:3600]
    info["bin_sha"] = sha(bh)
    ns = struct.unpack(">H", bh[20:22])[0]
    fmt = struct.unpack(">H", bh[24:26])[0]
    ext = struct.unpack(">H", bh[96:98])[0] if len(bh) >= 98 else 0
    info["ns"], info["fmt"] = ns, fmt
    bps = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}.get(fmt, 4)
    tl = 240 + ns * bps
    body = buf[3600:]
    ntr = len(body) // tl
    info["ntr"] = ntr
    info["rem"] = len(body) - ntr * tl
    data = bytearray()
    hdr0 = body[:240]
    info["hdr0_sha"] = sha(hdr0)
    # time fields (try big endian)
    be = struct.unpack(">5H", hdr0[156:166])
    info["time_be"] = list(be)
    le = struct.unpack("<5H", hdr0[156:166])
    info["time_le"] = list(le)
    hdrs = []
    cols = []
    for i in range(ntr):
        t = body[i * tl: (i + 1) * tl]
        hdrs.append(t[:240])
        cols.append(t[240:])
    info["data_sha"] = sha(b"".join(cols))
    info["hdrs_sha"] = sha(b"".join(hdrs))
    info["col_hashes"] = [hashlib.blake2b(c, digest_size=8).hexdigest() for c in cols]
    return info


def main():
    metas = {}
    for js in sorted(NPY.glob("*/*.json")):
        metas[(js.parent.name, js.stem)] = json.loads(js.read_text(encoding="utf-8"))
    print("n json", len(metas), file=sys.stderr)

    zcache = {}
    rows = []
    for (act, path), m in metas.items():
        npy = NPY / act / f"{path}.npy"
        nb = npy.read_bytes()
        arr = np.load(npy)
        rec = {
            "activity": act, "path": path,
            "npy_sha": sha(nb), "npy_size": len(nb),
            "json_sha": sha((NPY / act / f"{path}.json").read_bytes()),
            "shape": list(arr.shape), "dtype": str(arr.dtype),
            "arr_sha": sha(np.ascontiguousarray(arr).tobytes()),
            "arr_flip_sha": sha(np.ascontiguousarray(arr[:, ::-1]).tobytes()),
            "arr_zero_frac": float((arr == 0).mean()),
            "source_zip": m["source_zip"], "source_member": m["source_member"],
        }
        zn = m["source_zip"]
        if zn not in zcache:
            zcache[zn] = zipfile.ZipFile(ZIPS / zn)
        z = zcache[zn]
        zi = z.getinfo(m["source_member"])
        rec["zip_crc"] = zi.CRC
        rec["zip_size"] = zi.file_size
        rec["zip_dt"] = "%04d-%02d-%02d %02d:%02d:%02d" % zi.date_time
        buf = z.read(m["source_member"])
        rec["sgy_sha"] = sha(buf)
        rec.update({"sgy_" + k: v for k, v in parse_sgy(buf).items()})
        rows.append(rec)
    # png raw hash for pair info
    for r in rows:
        p = ROOT / "03_radargrams_png_raw" / r["activity"] / f"{r['path']}.png"
        r["png_raw_sha"] = sha(p.read_bytes()) if p.exists() else None
    # all zip members names for activities (ground-truth png / survey map hashes), hashing only
    extra = {}
    for zn in zcache:
        z = zcache[zn]
        for zi in z.infolist():
            if zi.filename.lower().endswith(".png"):
                extra[zi.filename] = {"crc": zi.CRC, "size": zi.file_size,
                                      "sha": sha(z.read(zi))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"files": rows, "zip_pngs": extra}, ensure_ascii=False), encoding="utf-8")
    print("done", len(rows), len(extra))


if __name__ == "__main__":
    main()
