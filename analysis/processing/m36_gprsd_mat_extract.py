"""M3-6 follow-up (one-off): concatenate GPR-SD split zip parts (.zip.001 ...) into a temp file and extract safely.

usage: python -I analysis/processing/m36_gprsd_mat_extract.py <clone_dir> <tmp_dir> <out_dir> <report_json>
- refuses entries with absolute paths, drive letters, or '..' components
- never executes anything from the archive
"""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import PurePosixPath, PureWindowsPath


def sha256_file(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_name(name):
    if name.startswith(("/", "\\")):
        return False
    if PureWindowsPath(name).drive:
        return False
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return ".." not in parts


def main(clone, tmp, out, report):
    parts = sorted(p for p in os.listdir(clone) if p.startswith("GPR-SD.zip."))
    idx = [int(p.rsplit(".", 1)[1]) for p in parts]
    if idx != list(range(1, len(parts) + 1)):
        print(json.dumps({"status": "error", "reason": "part numbering gap"}))
        return 2
    os.makedirs(tmp, exist_ok=True)
    joined = os.path.join(tmp, "GPR-SD.joined.zip")
    part_info = []
    with open(joined, "wb") as w:
        for p in parts:
            fp = os.path.join(clone, p)
            part_info.append({"name": p, "size": os.path.getsize(fp)})
            with open(fp, "rb") as r:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    w.write(b)
    rep = {"parts": part_info, "joined_size": os.path.getsize(joined), "joined_sha256": sha256_file(joined)}
    if os.path.exists(out) and os.listdir(out):
        print(json.dumps({"status": "error", "reason": "out dir not empty"}))
        return 2
    os.makedirs(out, exist_ok=True)
    entries, rejected = [], []
    with zipfile.ZipFile(joined) as z:
        bad = z.testzip()
        rep["testzip_first_bad"] = bad
        for zi in z.infolist():
            ok = safe_name(zi.filename)
            rec = {"name": zi.filename, "file_size": zi.file_size, "compress_size": zi.compress_size,
                   "crc": f"{zi.CRC:08x}", "date_time": list(zi.date_time), "is_dir": zi.is_dir(), "safe": ok}
            if not ok:
                rejected.append(rec)
                continue
            z.extract(zi, out)
            if not zi.is_dir():
                rec["sha256"] = sha256_file(os.path.join(out, *PurePosixPath(zi.filename).parts))
            entries.append(rec)
    rep["entries"] = entries
    rep["rejected"] = rejected
    with open(report, "w", encoding="utf-8") as fp:
        json.dump(rep, fp, indent=1)
    print(json.dumps({"status": "ok", "n_entries": len(entries), "n_rejected": len(rejected),
                      "testzip": bad, "report": report}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
