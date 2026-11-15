"""M3-3 X7 item 4: assert that evaluation/dev input lists contain 0 AIHub test images.

Usage: python -I m33_assert_no_aihub_test.py [--expect-hits] LIST [LIST ...]
LIST is a .txt (one image path per line), a .yaml (train/val/test entries, resolved against `path:`
and the .txt lists it names) or a .csv (column file_name or crop_key or file). Matching is done on three keys
against the test rows of data/splits/aihub_yolo_v1.csv (subset == "test"):
  1. file name (basename), 2. gain-stripped crop key, 3. session (XML source, via the manifest row of the file).
Exit code 0 = no hit (or hits when --expect-hits, used as positive control), 1 = hits, 2 = bad input.
Prints one JSON line per list and a final summary line. Reads only the manifest and the lists.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

MAN = Path(r"C:\workspace\gpr_detection\data\splits\aihub_yolo_v1.csv")
GAIN = re.compile(r"_g[0-9.]+_x")


def names_from(p: Path, depth=0):
    sfx = p.suffix.lower()
    if sfx == ".txt":
        return [Path(l.strip().replace("\\", "/")).name for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if sfx == ".csv":
        d = pd.read_csv(p, encoding="utf-8")
        for c in ("file_name", "file", "crop_key"):
            if c in d.columns:
                return [Path(str(v).replace("\\", "/")).name for v in d[c]]
        raise ValueError(f"{p}: no file_name/file/crop_key column")
    if sfx in (".yaml", ".yml"):
        out = []
        root = None
        cur = None
        for l in p.read_text(encoding="utf-8").splitlines():
            s = l.strip()
            if s.startswith("path:"):
                root = Path(s.split(":", 1)[1].strip())
            elif re.match(r"^(train|val|test):", s):
                cur = s.split(":", 1)[0]
                v = s.split(":", 1)[1].strip()
                if v:
                    out += names_from((root or p.parent) / v, depth + 1)
            elif s.startswith("- ") and cur:
                out += names_from((root or p.parent) / s[2:].strip(), depth + 1)
            elif re.match(r"^\w+:", s):
                cur = None
        return out
    raise ValueError(f"{p}: unsupported list type")


def main(argv):
    expect = "--expect-hits" in argv
    lists = [Path(a) for a in argv if not a.startswith("--")]
    if not lists:
        print("usage: m33_assert_no_aihub_test.py [--expect-hits] LIST...", file=sys.stderr)
        return 2
    m = pd.read_csv(MAN, encoding="utf-8")
    test = m[m.subset == "test"]
    t_files, t_crops, t_sess = set(test.file_name), set(test.crop_key), set(test.session)
    by_file = m.set_index("file_name")
    total_hits = 0
    for lp in lists:
        try:
            names = names_from(lp)
        except Exception as e:  # noqa: BLE001
            print(f"input error: {e}", file=sys.stderr)
            return 2
        stems = [n if n.endswith(".jpg") else n + ".jpg" for n in names]
        crops = [GAIN.sub("_x", Path(n).stem) for n in names]
        f_hit = sum(n in t_files for n in stems)
        c_hit = sum(c in t_crops for c in crops)
        s_hit = sum(
            (by_file.loc[n, "session"] in t_sess) for n in stems if n in by_file.index
        )
        unknown = sum(n not in by_file.index for n in stems)
        total_hits += f_hit + c_hit + s_hit
        print(json.dumps(dict(list=str(lp), n=len(names), unique=len(set(names)), file_hits=f_hit, crop_key_hits=c_hit,
                              session_hits=s_hit, not_in_manifest=unknown), ensure_ascii=False))
    ok = (total_hits > 0) if expect else (total_hits == 0)
    print(json.dumps(dict(total_hits=total_hits, expect_hits=expect, assertion_ok=ok, manifest_test_files=len(t_files))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
