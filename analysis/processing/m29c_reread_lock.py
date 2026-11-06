"""M2-9 C: lock the blinded re-read sheet (copy + SHA-256 + timestamp).

Run: conda run -n gpr-dev python -I analysis/processing/m29c_reread_lock.py
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

GAL = Path(__file__).resolve().parents[2] / "data" / "processing" / "m29c_gallery"
VALID_TYPES = set("TWLBCXANU")
VALID_CONF = {"high", "medium", "low"}


def main() -> None:
    src = GAL / "reread_sheet.csv"
    dst = GAL / "reread_sheet_locked.csv"
    if dst.exists():
        raise SystemExit("locked sheet already exists; refusing to overwrite")
    sheet = pd.read_csv(src)
    assert len(sheet) == 30 and sheet["reread_id"].is_unique
    assert set(sheet["type"]) <= VALID_TYPES, set(sheet["type"]) - VALID_TYPES
    assert set(sheet["confidence"]) <= VALID_CONF
    shutil.copyfile(src, dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest().upper()
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    (GAL / "reread_lock.json").write_text(
        json.dumps({"file": dst.name, "sha256": digest, "locked_at": stamp}, indent=2),
        encoding="utf-8",
    )
    print(f"locked {dst.name} sha256={digest} at {stamp}")


if __name__ == "__main__":
    main()
