"""M1-10 pre-training leak re-assertion and per-fold counts (CPU only, read-only).

Checks on data/splits/aihub_yolo_v1.csv (manifest) and data/aihub_yolo_v1:
1. No `session`, `session_from_name`, `crop_key`, `split_group` value is shared
   between any two of fold 0-4 and test (fold -1): 15 pairs x 4 keys.
2. Every crop_key (gain copies) lives in exactly one fold.
3. fold{k}_images.txt / test_images.txt equal the manifest file sets, and
   fold{k}.yaml trains on the other four folds only (no val file in train).
4. Counts per fold: images, crops, sessions, groups, instances per class at
   image level and at crop level (one gain copy per crop, 1.0 > 0.5 > 0.25,
   the M1-7 crop-level rule).

Writes analysis/detection/m110_leak_check.json; prints a one-line JSON summary.
Exit 0 = pass, 1 = leak or mismatch found.

Run: conda run -n gpr-dev python -I analysis/detection/m110_leak_check.py
"""

import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "splits" / "aihub_yolo_v1.csv"
DATASET = ROOT / "data" / "aihub_yolo_v1"
OUT = ROOT / "analysis" / "detection" / "m110_leak_check.json"
KEYS = ("session", "session_from_name", "crop_key", "split_group")
FOLDS = (-1, 0, 1, 2, 3, 4)
NAMES = {0: "pipe_hyperbola", 1: "pipe_band", 2: "cavity", 3: "manhole"}
GAIN_PRIORITY = {"1.0": 0, "0.5": 1, "0.25": 2}


def fold_label(fold: int) -> str:
    return "test" if fold == -1 else f"fold{fold}"


def read_list(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {Path(line.strip()).name for line in lines if line.strip()}


def class_counts(file_names: list[str]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for name in file_names:
        label = DATASET / "labels" / (Path(name).stem + ".txt")
        if not label.exists():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return {NAMES[k]: counts.get(k, 0) for k in NAMES}


def main() -> int:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_fold: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_fold[int(row["fold"])].append(row)

    problems: list[str] = []

    # 1. pairwise key overlap
    overlaps = {}
    for a, b in itertools.combinations(FOLDS, 2):
        for key in KEYS:
            va = {r[key] for r in by_fold[a]}
            vb = {r[key] for r in by_fold[b]}
            n = len(va & vb)
            overlaps[f"{fold_label(a)}|{fold_label(b)}|{key}"] = n
            if n:
                problems.append(f"overlap {key} {fold_label(a)}-{fold_label(b)}: {n}")

    # 2. crop_key -> one fold
    crop_folds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        crop_folds[row["crop_key"]].add(int(row["fold"]))
    multi = [k for k, v in crop_folds.items() if len(v) > 1]
    if multi:
        problems.append(f"crop_key in >1 fold: {len(multi)}")

    # 3. list files and yaml
    list_check = {}
    for fold in FOLDS:
        txt = DATASET / ("test_images.txt" if fold == -1 else f"fold{fold}_images.txt")
        listed = read_list(txt)
        manifest_set = {r["file_name"] for r in by_fold[fold]}
        same = listed == manifest_set
        list_check[fold_label(fold)] = {"txt": len(listed), "manifest": len(manifest_set), "equal": same}
        if not same:
            problems.append(f"{txt.name} != manifest fold {fold}")
    yaml_check = {}
    for fold in range(5):
        data = yaml.safe_load((DATASET / f"fold{fold}.yaml").read_text(encoding="utf-8"))
        train_lists = data["train"] if isinstance(data["train"], list) else [data["train"]]
        expected = sorted(f"fold{k}_images.txt" for k in range(5) if k != fold)
        train_files: set[str] = set()
        for name in train_lists:
            train_files |= read_list(DATASET / name)
        val_files = read_list(DATASET / data["val"])
        ok = sorted(train_lists) == expected and data["val"] == f"fold{fold}_images.txt" and not (train_files & val_files)
        yaml_check[f"fold{fold}"] = {"train_lists": train_lists, "val": data["val"],
                                     "train_images": len(train_files), "val_images": len(val_files),
                                     "train_val_overlap": len(train_files & val_files), "ok": ok}
        if not ok:
            problems.append(f"fold{fold}.yaml mismatch")
        if data.get("names") != NAMES:
            problems.append(f"fold{fold}.yaml names {data.get('names')}")

    # 4. counts
    counts = {}
    for fold in FOLDS:
        frows = by_fold[fold]
        crops: dict[str, dict[str, str]] = {}
        for row in sorted(frows, key=lambda r: GAIN_PRIORITY[r["gain"]]):
            crops.setdefault(row["crop_key"], row)
        gain_pick = Counter(r["gain"] for r in crops.values())
        counts[fold_label(fold)] = {
            "images": len(frows),
            "crops": len(crops),
            "sessions": len({r["session"] for r in frows}),
            "split_groups": len({r["split_group"] for r in frows}),
            "crop_level_gain_pick": dict(sorted(gain_pick.items())),
            "instances_image_level": class_counts([r["file_name"] for r in frows]),
            "instances_crop_level": class_counts([r["file_name"] for r in crops.values()]),
            "primary_class_images": dict(sorted(Counter(r["primary_class"] for r in frows).items())),
        }
    seeds = sorted({r["seed"] for r in rows})

    result = {
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "rows": len(rows),
        "manifest_seed": seeds,
        "pairs_checked": len(list(itertools.combinations(FOLDS, 2))),
        "keys": KEYS,
        "overlap_nonzero": {k: v for k, v in overlaps.items() if v},
        "crop_key_multi_fold": len(multi),
        "list_files": list_check,
        "fold_yaml": yaml_check,
        "counts": counts,
        "problems": problems,
        "verdict": "pass" if not problems else "fail",
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "problems": len(problems),
                      "pairs": result["pairs_checked"], "out": str(OUT)}))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
