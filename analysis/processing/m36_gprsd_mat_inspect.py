"""M3-6 follow-up: inspect the original GPR-SD .mat files and compare with the processed npy.

Inputs (all read-only):
  argv[1] folder with the 12 extracted .mat files (extracted from GPR-SD.zip.001-.033 of
          github.com/ubriustc/GPR_data, commit 88a9540, into a new folder outside the repo)
  argv[2] processed npy folder (GPR_Data/GPR-SD_..._261016/05_radargrams_npy)
  argv[3] extraction report json (zip part sizes, joined sha256, entry list) from the
          one-off extraction step (scratch)
  argv[4] folder with tools/pyinspect/data_inspect.py json reports (<line>.json)

Output: analysis/processing/m36_gprsd_mat_inspect.json
  Public summary only: variable list, shapes, dtypes, MAT header scalars, file sha256,
  whole-array equality flags. No amplitude values, no coordinates, no mark positions
  (M3-4 Q8 label-window blind: comparison is whole-array equality/hash only).

Usage: python -I analysis/processing/m36_gprsd_mat_inspect.py <mat_dir> <npy_dir> <extract_report> <inspect_dir>
"""

import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import scipy.io as sio

OUT = Path(__file__).resolve().parent / "m36_gprsd_mat_inspect.json"
MI = {14: "miMATRIX", 15: "miCOMPRESSED"}
MX_CLASS = {6: "double", 7: "single"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for blk in iter(lambda: fp.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def mat_top_elements(path: Path) -> dict:
    """Walk the top-level MAT v5 data elements and peek the array name/dims of each."""
    raw = path.read_bytes()
    out = {
        "header_text": raw[:116].decode("latin-1").rstrip(),
        "subsys_data_offset_zero": raw[116:124] == b"\x00" * 8,
        "version_word": struct.unpack("<H", raw[124:126])[0],
        "endian": raw[126:128].decode("latin-1"),
        "elements": [],
    }
    pos = 128
    while pos + 8 <= len(raw):
        dtype, nbytes = struct.unpack("<II", raw[pos:pos + 8])
        rec = {"type": MI.get(dtype, str(dtype)), "nbytes": nbytes}
        inner = zlib.decompressobj().decompress(raw[pos + 8:pos + 8 + nbytes], 4096) if dtype == 15 else raw[pos:pos + 4096]
        itype = struct.unpack("<I", inner[:4])[0]
        if itype == 14:
            cls = inner[16]
            p = 8 + 16
            dlen = struct.unpack("<I", inner[p + 4:p + 8])[0]
            dims = list(struct.unpack("<" + "i" * (dlen // 4), inner[p + 8:p + 8 + dlen]))
            p += 8 + dlen + ((8 - dlen % 8) % 8)
            ntag = struct.unpack("<I", inner[p:p + 4])[0]
            if ntag >> 16:
                name = inner[p + 4:p + 4 + (ntag >> 16)].decode("latin-1")
            else:
                nlen = struct.unpack("<I", inner[p + 4:p + 8])[0]
                name = inner[p + 8:p + 8 + nlen].decode("latin-1")
            rec.update({"inner": "miMATRIX", "name": name, "dims": dims, "class": MX_CLASS.get(cls, str(cls))})
        out["elements"].append(rec)
        pos += 8 + nbytes
        if dtype != 15:
            pos += (8 - pos % 8) % 8
    out["walk_covers_file"] = pos == len(raw)
    return out


def main() -> None:
    mat_dir, npy_dir, ext_rep, insp_dir = (Path(a) for a in sys.argv[1:5])
    ext = json.loads(ext_rep.read_text(encoding="utf-8"))
    res = {
        "source": {
            "repo": "github.com/ubriustc/GPR_data",
            "commit": "88a9540efc49f36362d5b89a1bc7fe701d5ada98",
            "zip_parts": len(ext["parts"]),
            "zip_parts_total_bytes": ext["joined_size"],
            "joined_zip_sha256": ext["joined_sha256"],
            "zip_testzip_first_bad": ext["testzip_first_bad"],
            "zip_entries": [{"name": e["name"], "file_size": e["file_size"], "stored_uncompressed": e["file_size"] == e["compress_size"],
                             "zip_date_time": e["date_time"], "sha256": e.get("sha256")} for e in ext["entries"]],
            "zip_rejected_entries": len(ext["rejected"]),
            "other_repo_files": ["Introduce.txt", "Mark.xlsx", "README.md"],
            "note": "extracted outside the repo; .mat files are not committed",
        },
        "lines": {},
    }
    for mp in sorted(mat_dir.glob("*.mat")):
        name = mp.stem
        rec = {"mat_size": mp.stat().st_size, "mat_sha256": sha256_file(mp)}
        rec["whosmat"] = [{"name": w[0], "shape": list(w[1]), "class": w[2]} for w in sio.whosmat(str(mp))]
        rec["mat_structure"] = mat_top_elements(mp)
        md = sio.loadmat(str(mp))
        rec["loadmat_keys"] = sorted(md.keys())
        rec["loadmat_version"] = md.get("__version__")
        rec["loadmat_globals_empty"] = len(md.get("__globals__", [])) == 0
        P = md["P"]
        rec["P_shape"] = list(P.shape)
        rec["P_dtype"] = str(P.dtype)
        rec["P_all_finite"] = bool(np.isfinite(P).all())
        rec["P_rows_all_zero"] = int((np.abs(P).max(axis=1) == 0).sum())
        rec["P_cols_all_zero"] = int((np.abs(P).max(axis=0) == 0).sum())
        D = np.load(npy_dir / f"{name}.npy")
        js = json.loads((npy_dir / f"{name}.json").read_text(encoding="utf-8"))
        rec["npy_shape"] = list(D.shape)
        rec["npy_dtype"] = str(D.dtype)
        rec["shape_equal"] = list(D.shape) == list(P.shape)
        rec["npy_equals_P_cast_float32_exact"] = bool(rec["shape_equal"] and np.array_equal(D, P.astype(np.float32)))
        if rec["shape_equal"]:
            rec["max_abs_diff_over_max_abs_P"] = float(np.abs(D.astype(np.float64) - P).max() / np.abs(P).max())
        rec["npy_json_n_samples"] = js.get("n_samples")
        rec["npy_json_source_file"] = js.get("source_file")
        insp = json.loads((insp_dir / f"{name}.json").read_text(encoding="utf-8"))
        rec["data_inspect_variables"] = [{"path": c["path"], "shape": c.get("shape"), "dtype": c.get("dtype")}
                                         for c in insp["tree"].get("children", [])]
        res["lines"][name] = rec
    L = res["lines"].values()
    res["summary"] = {
        "n_mat": len(res["lines"]),
        "variables_union": sorted({w["name"] for r in L for w in r["whosmat"]}),
        "n_samples_set": sorted({r["P_shape"][0] for r in L}),
        "n_lines_npy_equal_exact": sum(r["npy_equals_P_cast_float32_exact"] for r in L),
        "max_rel_diff_all": max(r.get("max_abs_diff_over_max_abs_P", 0.0) for r in L),
        "has_time_axis_or_header_fields": False,
        "mat_created_range": [min(r["mat_structure"]["header_text"][-24:] for r in L),
                              max(r["mat_structure"]["header_text"][-24:] for r in L)],
    }
    res["summary"]["has_time_axis_or_header_fields"] = res["summary"]["variables_union"] != ["P"]
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    s = res["summary"]
    print(json.dumps({"status": "ok", "n_mat": s["n_mat"], "variables": s["variables_union"],
                      "n_samples": s["n_samples_set"], "n_equal": s["n_lines_npy_equal_exact"], "out": str(OUT)}))


if __name__ == "__main__":
    main()
