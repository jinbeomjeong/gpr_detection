"""One-off split of existing M3-6 outputs (D2 fits, pilot mark windows) into public / local-full copies (no re-run).

Why: GPR-SD licence is unknown (docs/datasets/cards/gprsd.md); the public repository gets summary
statistics only, without per-diffractor trace/sample coordinates (user decision 2026-11-26).

For each file in m36_gprsd_common.PRIVATE_KEYS:
  1. read analysis/processing/<name> (the pre-split full output) and record its sha256;
  2. check that dump_json(parsed), with write_text newline translation, reproduces the file byte
     for byte (same serialisation);
  3. copy the original bytes unchanged to data/derived/m36_gprsd/<name> (git-ignored);
  4. write the public view (withheld keys removed + "public_split" block) to analysis/processing/<name>;
  5. verify: full copy sha256 == original sha256; every retained key serialises identically in
     public and full; withheld set == PRIVATE_KEYS; no coordinate key (COORD_KEYS) anywhere in the
     public copy.
Idempotent: if the public file already has "public_split", only the verification runs (against the
local full copy). A manifest with all sha256 values is written to data/derived/m36_gprsd/.

Usage: python -I m36_gprsd_split_public.py
Exit 0 ok, 1 verification failed, 2 input missing.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m36_gprsd_common import (COORD_KEYS, DERIVED_DIR, OUT_DIR, PRIVATE_KEYS, dump_json,  # noqa: E402
                              public_view)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def as_written(text: str) -> bytes:
    """Bytes that Path.write_text(text, encoding="utf-8") produces (newline translation to os.linesep)."""
    return text.replace("\n", os.linesep).encode("utf-8")


def keys_anywhere(obj, found=None):
    found = set() if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(k)
            keys_anywhere(v, found)
    elif isinstance(obj, list):
        for v in obj:
            keys_anywhere(v, found)
    return found


def verify(name: str, pub_b: bytes, full_b: bytes) -> list[str]:
    errs = []
    pub, full = json.loads(pub_b.decode("utf-8")), json.loads(full_b.decode("utf-8"))
    withheld = set(full) - set(pub)
    if withheld != set(PRIVATE_KEYS[name]):
        errs.append(f"withheld {sorted(withheld)} != {sorted(PRIVATE_KEYS[name])}")
    if set(pub) - set(full) != {"public_split"}:
        errs.append(f"extra public keys {sorted(set(pub) - set(full))}")
    for k in pub:
        if k == "public_split":
            continue
        if json.dumps(pub[k], ensure_ascii=False) != json.dumps(full[k], ensure_ascii=False):
            errs.append(f"value differs: {k}")
    if list(k for k in pub if k != "public_split") != [k for k in full if k not in withheld]:
        errs.append("key order differs")
    leaked = keys_anywhere({k: v for k, v in pub.items() if k != "public_split"}) & set(COORD_KEYS)
    if leaked:
        errs.append(f"coordinate keys in public copy: {sorted(leaked)}")
    # public file must be exactly the canonical public view of the full file
    if as_written(dump_json(public_view(name, full))) != pub_b:
        errs.append("public file != public_view(full)")
    return errs


def main() -> int:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"note": "M3-6 public/full split (no re-run); sha256 of UTF-8 bytes",
                "files": {}}
    status = 0
    for name in PRIVATE_KEYS:
        src, dst = OUT_DIR / name, DERIVED_DIR / name
        if not src.is_file():
            print(json.dumps({"status": "input_missing", "file": str(src)}))
            return 2
        src_b = src.read_bytes()
        src_obj = json.loads(src_b.decode("utf-8"))
        entry = {}
        if "public_split" in src_obj:            # already split -> verify only
            if not dst.is_file():
                print(json.dumps({"status": "input_missing", "file": str(dst)}))
                return 2
            full_b = dst.read_bytes()
            entry["mode"] = "verify_only"
        else:
            if as_written(dump_json(src_obj)) != src_b:
                print(json.dumps({"status": "serialisation_mismatch", "file": str(src)}))
                return 1
            entry["original_sha256"] = sha(src_b)
            entry["original_bytes"] = len(src_b)
            dst.write_bytes(src_b)
            full_b = dst.read_bytes()
            if sha(full_b) != entry["original_sha256"]:
                print(json.dumps({"status": "copy_mismatch", "file": str(dst)}))
                return 1
            src.write_text(dump_json(public_view(name, src_obj)), encoding="utf-8")
            entry["mode"] = "split"
        pub_b = src.read_bytes()
        errs = verify(name, pub_b, full_b)
        entry.update({"full_sha256": sha(full_b), "full_bytes": len(full_b),
                      "public_sha256": sha(pub_b), "public_bytes": len(pub_b),
                      "withheld_keys": list(PRIVATE_KEYS[name]), "errors": errs})
        manifest["files"][name] = entry
        if errs:
            status = 1
    mpath = DERIVED_DIR / "split_manifest.json"
    old = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else None
    if old is not None:  # keep the original sha256 recorded at split time
        for name, e in manifest["files"].items():
            prev = old.get("files", {}).get(name, {})
            for k in ("original_sha256", "original_bytes"):
                if k not in e and k in prev:
                    e[k] = prev[k]
    mpath.write_text(dump_json(manifest), encoding="utf-8")
    print(json.dumps({"status": "ok" if status == 0 else "verify_failed",
                      "files": {n: {"mode": e["mode"], "public_bytes": e["public_bytes"],
                                    "full_bytes": e["full_bytes"], "n_errors": len(e["errors"])}
                                for n, e in manifest["files"].items()},
                      "manifest": str(mpath)}))
    return status


if __name__ == "__main__":
    sys.exit(main())
