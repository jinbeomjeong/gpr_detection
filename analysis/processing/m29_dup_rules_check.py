"""M2-9 spec v1.7 support: duplicate-radargram rules and zero-flag counts (no depth).

Reads only metadata-level inputs; no ground-truth depth value is read.

* ``docs/datasets/fourtu_duplicate_radargrams.csv`` (curated pair list, 39 rows;
  ``gt_rows_a``/``gt_rows_b`` are row counts made by the curator).
* ``docs/processing/fourtu_lines_v1.csv`` (line list, metadata only).
* ``docs/processing/fourtu_apex_candidates_v1.csv`` (apex candidates, no depth truth).
* ``docs/processing/m29e_zero_survey_lines.csv`` (zero survey, no depth truth).
* The 4TU ``05_radargrams_npy`` files, read-only, for a full SHA-256 scan
  (checks the curated list is complete and measures the hashing cost).

Checks / outputs (``docs/processing/m29_dup_rules_check.json``):

1. Candidate rows of both files of every pair are identical (so dropping one
   copy loses no candidate).
2. The ``keep_one`` rule of spec v1.7 (strict line-set superset, else the lower
   activity number) and the ``drop_all`` activity set.
3. Ground-truth row counts removed by each handling (curator counts only).
4. Null spread of the pass-rate change caused by removing k of N ground-truth
   items under exchangeability (threshold proposal for the caveat rule).
5. ``zero_h_implausible`` line counts per site with and without the
   within-activity copies.

Run (single process, about 10 s):
    conda run -n gpr-dev python -I analysis/processing/m29_dup_rules_check.py
"""

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from gpr_detection.depth.config import DEFAULT_DEPTH_CONFIG
from gpr_detection.io.paths import load_dataset_paths

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "processing"
DUP_CSV = ROOT / "docs" / "datasets" / "fourtu_duplicate_radargrams.csv"
LINES_CSV = DOCS / "fourtu_lines_v1.csv"
CANDIDATES_CSV = DOCS / "fourtu_apex_candidates_v1.csv"
SURVEY_CSV = DOCS / "m29e_zero_survey_lines.csv"
OUT_JSON = DOCS / "m29_dup_rules_check.json"
CANDIDATE_KEYS = (
    "trace",
    "sample",
    "t_ns",
    "tau_ns",
    "snr",
    "n_left",
    "n_right",
    "v_straight_qc",
    "source",
)
N_GT_MAIN = 388  # spec 1: exclude_x ground-truth rows (count only)
P_GRID = (0.2, 0.3, 0.4, 0.5)
H_MAX_M = DEFAULT_DEPTH_CONFIG.h_max_m  # zero search limit, spec_depth 4-1 (no new constant)


def activity_key(activity: str) -> tuple[int, int]:
    """Numeric order of an activity id such as ``010.11`` -> (10, 11)."""
    site, sub = activity.split(".")
    return int(site), int(sub)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_identity(dups: pd.DataFrame, candidates: pd.DataFrame) -> list[dict]:
    rows = []
    for rec in dups.itertuples(index=False):
        a = candidates[
            (candidates["activity_id"] == rec.activity_a)
            & (candidates["path"] == rec.path_a)
        ]
        b = candidates[
            (candidates["activity_id"] == rec.activity_b)
            & (candidates["path"] == rec.path_b)
        ]
        cols = [c for c in CANDIDATE_KEYS if c in candidates.columns]
        left = a[cols].reset_index(drop=True)
        right = b[cols].reset_index(drop=True)
        same = left.shape == right.shape and bool(
            (left.fillna(-1.0) == right.fillna(-1.0)).all().all()
        )
        rows.append(
            {
                "pair_id": int(rec.pair_id),
                "a": f"{rec.activity_a}/{rec.path_a}",
                "b": f"{rec.activity_b}/{rec.path_b}",
                "n_a": len(a),
                "n_b": len(b),
                "identical": same,
            }
        )
    return rows


def keep_one_rule(
    dups: pd.DataFrame, lines_per_activity: dict[str, int]
) -> list[dict]:
    across = dups[dups["scope"] == "across_activities"]
    shared: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for rec in across.itertuples(index=False):
        pair = tuple(sorted((rec.activity_a, rec.activity_b), key=activity_key))
        paths = (
            (rec.path_a, rec.path_b)
            if pair[0] == rec.activity_a
            else (rec.path_b, rec.path_a)
        )
        shared[pair].append(paths)
    decisions = []
    for (low, high), paths in sorted(shared.items(), key=lambda kv: activity_key(kv[0][0])):
        n_shared = len(paths)
        n_low, n_high = lines_per_activity[low], lines_per_activity[high]
        if n_shared == n_low < n_high:
            kept, dropped, reason = high, low, "strict_superset"
        elif n_shared == n_high < n_low:
            kept, dropped, reason = low, high, "strict_superset"
        else:
            kept, dropped = low, high
            reason = "equal_sets" if n_shared == n_low == n_high else "lower_number"
        dropped_paths = sorted(
            p[0] if dropped == low else p[1] for p in paths
        )
        remaining = lines_per_activity[dropped] - n_shared
        decisions.append(
            {
                "activities": [low, high],
                "n_shared": n_shared,
                "n_lines": [n_low, n_high],
                "kept": kept,
                "dropped": dropped,
                "reason": reason,
                "dropped_paths": dropped_paths,
                "dropped_remaining_lines": remaining,
                "dropped_activity_removed": remaining == 0,
            }
        )
    return decisions


def null_sd_pass_change(n_total: int, k_removed: int, p: float) -> float:
    """SD of rate(subset) - rate(full) when k of n items are removed at random.

    rate(full) = (S + R) / n, rate(subset) = S / (n - k), so the difference is
    (k / n) * (p_sub - p_rem) with Var(p_sub - p_rem) ~ p (1 - p) (1/(n-k) + 1/k)
    (binomial approximation, items independent; activity clustering makes it
    larger).
    """
    var = p * (1 - p) * (1 / (n_total - k_removed) + 1 / k_removed)
    return (k_removed / n_total) * math.sqrt(var)


def main() -> int:
    started = time.perf_counter()
    dups = pd.read_csv(DUP_CSV, dtype=str, encoding="utf-8-sig")
    lines = pd.read_csv(LINES_CSV, dtype={"activity_id": str, "path": str})
    candidates = pd.read_csv(CANDIDATES_CSV, dtype={"activity_id": str, "path": str})
    survey = pd.read_csv(SURVEY_CSV, dtype={"activity_id": str, "path": str})

    # 1. full hash scan (read-only) and agreement with the curated list
    root = load_dataset_paths().fourtu
    hash_started = time.perf_counter()
    hashes = {}
    total_bytes = 0
    for rec in lines.itertuples(index=False):
        path = root / "05_radargrams_npy" / rec.activity_id / f"{rec.path}.npy"
        total_bytes += path.stat().st_size
        hashes[(rec.activity_id, rec.path)] = sha256_file(path)
    hash_s = time.perf_counter() - hash_started
    groups = defaultdict(list)
    for key, digest in hashes.items():
        groups[digest].append(key)
    scanned_pairs = {
        frozenset(f"{a}/{p}" for a, p in members)
        for members in groups.values()
        if len(members) > 1
    }
    listed_pairs = {
        frozenset((f"{r.activity_a}/{r.path_a}", f"{r.activity_b}/{r.path_b}"))
        for r in dups.itertuples(index=False)
    }
    listed_hash_ok = all(
        hashes[(r.activity_a, r.path_a)] == r.sha256_npy == hashes[(r.activity_b, r.path_b)]
        for r in dups.itertuples(index=False)
    )

    # 2. candidate identity
    identity = candidate_identity(dups, candidates)

    # 3. line counts after the within-activity rule
    within = dups[dups["scope"] == "within_activity"]
    within_dropped = {(r.activity_b, r.path_b) for r in within.itertuples(index=False)}
    lines_meta = lines.groupby("activity_id")["path"].nunique().to_dict()
    lines_unique = dict(lines_meta)
    for activity, _ in within_dropped:
        lines_unique[activity] -= 1
    decisions = keep_one_rule(dups, lines_unique)

    gt_rows = {}
    for r in dups.itertuples(index=False):
        gt_rows[r.activity_a] = int(r.gt_rows_a)
        gt_rows[r.activity_b] = int(r.gt_rows_b)
    across_acts = sorted(
        {a for r in dups[dups["scope"] == "across_activities"].itertuples(index=False)
         for a in (r.activity_a, r.activity_b)},
        key=activity_key,
    )
    keep_one_removed = [d["dropped"] for d in decisions if d["dropped_activity_removed"]]
    keep_one_gt_removed = sum(gt_rows[a] for a in keep_one_removed)
    drop_all_gt_removed = sum(gt_rows[a] for a in across_acts)

    # 4. null spread of the pass-rate change
    null_table = {
        handling: {
            str(p): round(100 * null_sd_pass_change(N_GT_MAIN, k, p), 3) for p in P_GRID
        }
        for handling, k in (
            ("keep_one", keep_one_gt_removed),
            ("drop_all", drop_all_gt_removed),
        )
    }

    # 5. zero flag counts
    flagged = survey[
        (survey["zero_rule"] == "air_surface_late") & (survey["h_est_m"] > H_MAX_M)
    ]
    flagged_unique = flagged[
        ~flagged.apply(lambda r: (r["activity_id"], r["path"]) in within_dropped, axis=1)
    ]
    per_site = (
        pd.DataFrame(
            {
                "all": flagged.groupby("site").size(),
                "unique": flagged_unique.groupby("site").size(),
                "candidates_unique": flagged_unique.groupby("site")["n_candidates"].sum(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    flagged_across = sorted(
        f"{r.activity_id}/{r.path}"
        for r in flagged.itertuples(index=False)
        if any(
            (r.activity_id, r.path) in {(d.activity_a, d.path_a), (d.activity_b, d.path_b)}
            for d in dups[dups["scope"] == "across_activities"].itertuples(index=False)
        )
    )

    summary = {
        "hash_scan": {
            "files": len(hashes),
            "bytes": total_bytes,
            "seconds": round(hash_s, 2),
            "duplicate_groups": len(scanned_pairs),
            "max_group_size": max(len(m) for m in groups.values()),
            "scan_equals_curated_list": scanned_pairs == listed_pairs,
            "curated_hashes_match_files": listed_hash_ok,
        },
        "candidate_identity": {
            "pairs": len(identity),
            "identical": sum(r["identical"] for r in identity),
            "candidates_on_copies": sum(r["n_b"] for r in identity),
            "not_identical": [r for r in identity if not r["identical"]],
        },
        "lines_meta": {a: lines_meta[a] for a in sorted({a for a, _ in within_dropped} | set(across_acts), key=activity_key)},
        "lines_after_within_rule": {a: lines_unique[a] for a in sorted({a for a, _ in within_dropped}, key=activity_key)},
        "within_dropped": sorted(f"{a}/{p}" for a, p in within_dropped),
        "within_dropped_candidates": int(
            sum(
                ((candidates["activity_id"] == a) & (candidates["path"] == p)).sum()
                for a, p in within_dropped
            )
        ),
        "keep_one": decisions,
        "keep_one_removed_activities": keep_one_removed,
        "keep_one_gt_rows_removed": keep_one_gt_removed,
        "drop_all_activities": across_acts,
        "drop_all_gt_rows_removed": drop_all_gt_removed,
        "gt_rows_by_activity_curator": {a: gt_rows[a] for a in across_acts},
        "null_sd_pass_change_pct_points": null_table,
        "null_note": "binomial, items independent, N_gt 388; activity clustering widens it",
        "zero_flag": {
            "lines_all": len(flagged),
            "lines_unique": len(flagged_unique),
            "per_site": {int(s): row.to_dict() for s, row in per_site.iterrows()},
            "flagged_lines_in_across_duplicates": flagged_across,
            "flagged_unique_with_gt_activity": int(flagged_unique["activity_has_gt"].sum()),
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
        "gt_use": "curator row counts only (no depth value read)",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("hash_scan", "candidate_identity", "keep_one_removed_activities", "keep_one_gt_rows_removed", "drop_all_gt_rows_removed", "null_sd_pass_change_pct_points")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
