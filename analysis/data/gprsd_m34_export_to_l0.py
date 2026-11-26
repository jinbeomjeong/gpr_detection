"""Convert a GPR-SD M3-4 kit Label Studio JSON export to L0 label rows + QA report.

Usage (env gpr-dev; pure stdlib + numpy):
  conda run -n gpr-dev python -I analysis/data/gprsd_m34_export_to_l0.py
      --export <ls_export.json> [--bundle main|relabel] [--period 20.0]
  conda run -n gpr-dev python -I analysis/data/gprsd_m34_export_to_l0.py --selftest

Geometry (render_config.json):
  trace_f = window_trace0 + x_px/scale_x ; sample_f = y_px/scale_y ;
L0 GridBox (closed, outward rounding, ADR-004 decision 1/8):
  floor(t1+1e-6) .. ceil(t2-1e-6)-1, zero width -> 1 cell.
Hard stop: any task with a non-empty `predictions` list (plan 5-3: no pre-annotation).
QA (cv convention docs/experiments/m34_gprsd_box_convention.md 3-6): warnings,
not rejections, except the ERROR rows. `truncated` hyperbola boxes get no
apex-centre or top/bottom margin WARNs (convention 3-4: they are not redrawn);
convert_summary counts them as `truncated_rows_skipped_margin_qa`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "data" / "labels" / "gprsd_m34"
TOL = 1e-6
CLASSES = ("pipe_hyperbola", "pipe_band", "ignore")
# samples; M3-6 closed: reflected carrier 19.3-22.3, M3-3b main-path fixed P
# (m36_gprsd_scale.md s13; 14.6/51 are time-fixed equipment components)
PERIOD_FIXED = 20.0
# Report-only sensitivity check: upper end of the M3-6 reflected carrier range.
PERIOD_REFLECTED_MAX = 22.3  # samples
TOP_MARGIN_MIN_P = 0.6  # lower edge of the 0.6-1.5 P top-margin band (convention 3-6)


# ---------------------------------------------------------------- geometry
def px_to_cont(
    x_px0: float,
    x_px1: float,
    y_px0: float,
    y_px1: float,
    t0: int,
    sx: float,
    sy: float,
    s0: int = 0,
):
    return t0 + x_px0 / sx, t0 + x_px1 / sx, s0 + y_px0 / sy, s0 + y_px1 / sy


def cont_to_grid(a: float, b: float) -> tuple[int, int]:
    """Continuous interval [a, b] (edge coordinates) -> closed cell indices.

    Rounds outward.
    """
    lo = math.floor(a + TOL)
    hi = math.ceil(b - TOL) - 1
    if hi < lo:  # zero width -> 1 cell
        hi = lo
    return lo, hi


def grid_to_px(
    i0: int, i1: int, k0: int, k1: int, t0: int, sx: float, sy: float, s0: int = 0
):
    return sx * (i0 - t0), sx * (i1 + 1 - t0), sy * (k0 - s0), sy * (k1 + 1 - s0)


def ls_box_to_px(v: dict, w: float, h: float):
    # snap to 1e-3 px so that percent values rounded to 6 decimals still land on
    # grid lines
    x0 = round(v["x"] / 100.0 * w, 3)
    y0 = round(v["y"] / 100.0 * h, 3)
    return (
        x0,
        round(x0 + v["width"] / 100.0 * w, 3),
        y0,
        round(y0 + v["height"] / 100.0 * h, 3),
    )


def px_to_ls_box(x0, x1, y0, y1, w, h):
    return {
        "x": x0 / w * 100.0,
        "y": y0 / h * 100.0,
        "width": (x1 - x0) / w * 100.0,
        "height": (y1 - y0) / h * 100.0,
        "rotation": 0,
    }


# ---------------------------------------------------------------- kit io
def load_kit(kit: Path):
    cfg_path = kit / "render_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    with (kit / "windows.csv").open(encoding="utf-8", newline="") as f:
        wins = {r["window_id"]: r for r in csv.DictReader(f)}
    return cfg, cfg_sha, wins


# ---------------------------------------------------------------- conversion
def pick_annotation(task: dict, issues: list[dict], wid: str):
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    if len(anns) > 1:
        issues.append(
            {
                "window_id": wid,
                "level": "ERROR",
                "code": "multiple_annotations",
                "detail": f"{len(anns)}; using latest",
            }
        )
    anns.sort(key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    return anns[-1]


def user_tag(a: dict) -> str:
    c = a.get("completed_by")
    if isinstance(c, dict):
        c = c.get("id")
    return f"user{c}" if c is not None else ""


def convert(
    tasks: list[dict], cfg: dict, cfg_sha: str, wins: dict, bundle: str, period: float
):
    sx, sy = cfg["scale_x_px_per_trace"], cfg["scale_y_px_per_sample"]
    rows: list[dict] = []
    status: list[dict] = []
    issues: list[dict] = []
    seen = set()
    for task in tasks:
        if task.get("predictions"):
            raise SystemExit(
                f"STOP: task {task.get('id')} has {len(task['predictions'])} "
                "predictions (must be 0)"
            )
        d = task["data"]
        wid = d["window_id"]
        if d.get("bundle", bundle) != bundle:
            issues.append(
                {
                    "window_id": wid,
                    "level": "ERROR",
                    "code": "bundle_mismatch",
                    "detail": str(d.get("bundle")),
                }
            )
        w = wins[wid]
        seen.add(wid)
        t0, t1 = int(w["window_trace0"]), int(w["window_trace1"])
        W, H = float(w["view_w_px"]), float(w["view_h_px"])
        ann = pick_annotation(task, issues, wid)
        if ann is None:
            status.append(
                {
                    "window_id": wid,
                    "bundle": bundle,
                    "labeled": "no",
                    "window_complete": "",
                    "n_boxes": 0,
                    "note": "",
                }
            )
            continue
        res = ann["result"]
        regions: dict[str, dict] = {}
        extra: dict[str, dict] = defaultdict(dict)
        wc, note = "", ""
        for r in res:
            typ, fn, rid = r["type"], r["from_name"], r.get("id")
            if r.get("original_width") not in (None, int(W)) or r.get(
                "original_height"
            ) not in (None, int(H)):
                issues.append(
                    {
                        "window_id": wid,
                        "level": "ERROR",
                        "code": "image_size_mismatch",
                        "detail": (
                            f"{r.get('original_width')}x{r.get('original_height')}"
                            f" vs {int(W)}x{int(H)}"
                        ),
                    }
                )
            if typ == "rectanglelabels":
                v = r["value"]
                if abs(v.get("rotation", 0)) > 1e-9:
                    issues.append(
                        {
                            "window_id": wid,
                            "level": "ERROR",
                            "code": "rotated_box",
                            "detail": rid,
                        }
                    )
                regions[rid] = {
                    "kind": "box",
                    "cls": v["rectanglelabels"][0],
                    "px": ls_box_to_px(v, W, H),
                }
            elif typ == "keypointlabels":
                v = r["value"]
                regions[rid] = {
                    "kind": "apex",
                    "px": (v["x"] / 100.0 * W, v["y"] / 100.0 * H),
                }
            elif typ == "choices" and fn in ("certainty", "flag"):
                extra[rid][fn] = r["value"]["choices"]
            elif typ == "choices" and fn == "window_complete":
                wc = (r["value"]["choices"] or [""])[0]
            elif typ == "textarea" and fn == "note":
                note = " | ".join(r["value"].get("text", []))
        boxes = {rid: g for rid, g in regions.items() if g["kind"] == "box"}
        apexes = {rid: g for rid, g in regions.items() if g["kind"] == "apex"}
        if not wc:
            issues.append(
                {
                    "window_id": wid,
                    "level": "ERROR",
                    "code": "window_complete_missing",
                    "detail": "",
                }
            )
        # associate apex -> hyperbola box
        hyp = {rid: g for rid, g in boxes.items() if g["cls"] == "pipe_hyperbola"}
        assigned: dict[str, list[str]] = defaultdict(list)
        for arid, a in apexes.items():
            ax, ay = a["px"]
            inside = [
                rid
                for rid, g in hyp.items()
                if g["px"][0] <= ax <= g["px"][1] and g["px"][2] <= ay <= g["px"][3]
            ]
            if inside:
                best = min(
                    inside,
                    key=lambda rid: abs(
                        ax - 0.5 * (hyp[rid]["px"][0] + hyp[rid]["px"][1])
                    ),
                )
                if len(inside) > 1:
                    issues.append(
                        {
                            "window_id": wid,
                            "level": "WARN",
                            "code": "apex_in_several_boxes",
                            "detail": f"{len(inside)} boxes; nearest centre used",
                        }
                    )
            elif hyp:
                best = min(
                    hyp,
                    key=lambda rid: (
                        abs(ax - 0.5 * (hyp[rid]["px"][0] + hyp[rid]["px"][1]))
                        + abs(ay - hyp[rid]["px"][2]) * 0.0
                    ),
                )
                issues.append(
                    {
                        "window_id": wid,
                        "level": "ERROR",
                        "code": "apex_outside_box",
                        "detail": f"apex at trace_f {t0 + ax / sx:.2f}",
                    }
                )
            else:
                best = None
                issues.append(
                    {
                        "window_id": wid,
                        "level": "ERROR",
                        "code": "apex_without_hyperbola_box",
                        "detail": f"trace_f {t0 + ax / sx:.2f}",
                    }
                )
            if best is not None:
                assigned[best].append(arid)
        n_box = 0
        for rid, g in boxes.items():
            x0, x1, y0, y1 = g["px"]
            tmin, tmax, smin, smax = px_to_cont(x0, x1, y0, y1, t0, sx, sy)
            i0, i1 = cont_to_grid(tmin, tmax)
            k0, k1 = cont_to_grid(smin, smax)
            ex = extra.get(rid, {})
            certainty = (ex.get("certainty") or [""])[0]
            flags = ";".join(ex.get("flag", []))
            row = {
                "bundle": bundle,
                "file_id": w["file_id"],
                "road": w["road"],
                "line_no": w["line_no"],
                "window_id": wid,
                "window_trace0": t0,
                "window_trace1": t1,
                "mark_xlsx_rows": w["mark_xlsx_rows"],
                "class": g["cls"],
                "trace_min_f": round(tmin, 4),
                "trace_max_f": round(tmax, 4),
                "sample_min_f": round(smin, 4),
                "sample_max_f": round(smax, 4),
                "i0": i0,
                "i1": i1,
                "k0": k0,
                "k1": k1,
                "apex_trace": "",
                "apex_sample": "",
                "apex_trace_f": "",
                "apex_sample_f": "",
                "certainty": certainty,
                "flags": flags,
                "labeler": user_tag(ann),
                "labeled_at": ann.get("updated_at") or ann.get("created_at") or "",
                "render_config_sha256": cfg_sha,
                "annotation_id": ann.get("id", ""),
                "qa": "",
            }
            qa: list[str] = []

            # loop variables bound as defaults (B023); flag() is only called in
            # this iteration, so behaviour is unchanged
            def flag(
                level: str,
                code: str,
                detail: str = "",
                *,
                qa: list[str] = qa,
                wid: str = wid,
                rid: str | None = rid,
            ) -> None:
                qa.append(code)
                issues.append(
                    {
                        "window_id": wid,
                        "level": level,
                        "code": code,
                        "detail": detail,
                        "box": rid,
                    }
                )

            # inside window / minimum size
            if (
                tmin < t0 - TOL
                or tmax > t1 + 1 + TOL
                or smin < -TOL
                or smax > float(w["n_samples"]) + TOL
            ):
                flag("ERROR", "box_outside_window")
            if x1 - x0 < sx or y1 - y0 < sy:
                flag("WARN", "box_smaller_than_one_cell")
            at_edge = (
                tmin <= t0 + TOL
                or tmax >= t1 + 1 - TOL
                or smax >= float(w["n_samples"]) - TOL
            )
            trunc = "truncated" in flags
            if g["cls"] in ("pipe_hyperbola", "pipe_band") and not certainty:
                flag("ERROR", "certainty_missing")
            if g["cls"] == "pipe_hyperbola":
                if at_edge and not trunc:
                    flag("WARN", "touches_window_edge_without_truncated_flag")
                if trunc and not at_edge:
                    flag("WARN", "truncated_flag_but_not_at_window_edge")
                aps = assigned.get(rid, [])
                if not aps:
                    flag("ERROR", "no_apex")
                else:
                    if len(aps) > 1:
                        flag("ERROR", "multiple_apex")
                    ax, ay = apexes[aps[0]]["px"]
                    atf, asf = t0 + ax / sx, ay / sy
                    row["apex_trace_f"], row["apex_sample_f"] = (
                        round(atf, 4),
                        round(asf, 4),
                    )
                    row["apex_trace"], row["apex_sample"] = (
                        math.floor(atf),
                        math.floor(asf),
                    )
                    if not (x0 <= ax <= x1 and y0 <= ay <= y1):
                        flag("ERROR", "apex_outside_own_box")
                    width_tr = tmax - tmin
                    if not trunc and abs(atf - 0.5 * (tmin + tmax)) > 0.15 * width_tr:
                        flag(
                            "WARN",
                            "apex_off_centre_gt_0.15_width",
                            f"{(atf - 0.5 * (tmin + tmax)) / width_tr:+.2f}",
                        )
                    # truncated boxes are not redrawn (convention 3-4/3-6), so
                    # margin WARNs would only pollute the redraw list; the
                    # skipped rows are counted in convert_summary instead
                    top = (asf - smin) / period
                    bot = (smax - asf) / period
                    if not trunc and (top < 0.6 or top > 1.5):
                        flag(
                            "WARN",
                            "top_margin_outside_0.6-1.5P",
                            f"{top:.2f}P (P={period})",
                        )
                    if not trunc and (bot < 3.0 or bot > 9.0):
                        flag("WARN", "bottom_outside_3-9P", f"{bot:.2f}P (P={period})")
            elif assigned.get(rid):
                flag("WARN", "apex_on_non_hyperbola")
            row["qa"] = ";".join(qa)
            rows.append(row)
            n_box += 1
        if wc == "yes" and n_box == 0:
            issues.append(
                {
                    "window_id": wid,
                    "level": "WARN",
                    "code": "complete_yes_but_no_boxes",
                    "detail": "",
                }
            )
        status.append(
            {
                "window_id": wid,
                "bundle": bundle,
                "labeled": "yes",
                "window_complete": wc,
                "n_boxes": n_box,
                "note": note,
            }
        )
    missing = sorted(set(wins) - seen) if bundle == "main" else []
    for wid in missing:
        issues.append(
            {
                "window_id": wid,
                "level": "ERROR",
                "code": "window_missing_in_export",
                "detail": "",
            }
        )
    rows.sort(
        key=lambda r: (
            r["file_id"],
            r["window_trace0"],
            r["trace_min_f"],
            r["sample_min_f"],
        )
    )
    for n, r in enumerate(rows, start=1):
        r["label_id"] = n
    return rows, status, issues


LABEL_COLS = [
    "label_id",
    "bundle",
    "file_id",
    "road",
    "line_no",
    "window_id",
    "window_trace0",
    "window_trace1",
    "mark_xlsx_rows",
    "class",
    "trace_min_f",
    "trace_max_f",
    "sample_min_f",
    "sample_max_f",
    "i0",
    "i1",
    "k0",
    "k1",
    "apex_trace",
    "apex_sample",
    "apex_trace_f",
    "apex_sample_f",
    "certainty",
    "flags",
    "labeler",
    "labeled_at",
    "render_config_sha256",
    "annotation_id",
    "qa",
]


def count_shallow_top_margin(
    rows: list[dict],
    period_samples: float = PERIOD_REFLECTED_MAX,
    min_ratio: float = TOP_MARGIN_MIN_P,
) -> int:
    """Count apex-bearing, non-truncated hyperbola boxes whose top margin < min_ratio P.

    Report-only: shows how many boxes would leave the top-margin band if P were the
    upper end of the reflected carrier range instead of the fixed P. No WARN rows.
    Truncation uses the same substring test as the QA loop in ``convert``.
    """
    count = 0
    for row in rows:
        if row["class"] != "pipe_hyperbola" or "truncated" in row["flags"]:
            continue
        if row["apex_sample_f"] == "":  # no apex -> already an ERROR row
            continue
        top_margin_p = (row["apex_sample_f"] - row["sample_min_f"]) / period_samples
        if top_margin_p < min_ratio:
            count += 1
    return count


def count_truncated_margin_qa_skips(rows: list[dict]) -> int:
    """Count apex-bearing, truncated hyperbola boxes that skipped the margin WARNs.

    These are exactly the rows for which ``convert`` would otherwise have run the
    top-margin and bottom-length checks (an apex is assigned) but did not, because
    truncated boxes are not redrawn (convention 3-4/3-6). Information only.
    """
    count = 0
    for row in rows:
        if row["class"] != "pipe_hyperbola" or "truncated" not in row["flags"]:
            continue
        if row["apex_sample_f"] == "":  # no apex -> margin checks never ran
            continue
        count += 1
    return count


def margin_report_counts(rows: list[dict]) -> dict[str, int]:
    """Return the report-only margin counts written to ``convert_summary``."""
    return {
        "top_margin_lt_0.6P_at_P22.3": count_shallow_top_margin(rows),
        "truncated_rows_skipped_margin_qa": count_truncated_margin_qa_skips(rows),
    }


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=cols, extrasaction="ignore", lineterminator="\n"
        )
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- self test
def selftest(kit: Path, period: float) -> int:
    import random

    cfg, cfg_sha, wins = load_kit(kit)
    sx, sy = cfg["scale_x_px_per_trace"], cfg["scale_y_px_per_sample"]
    rnd = random.Random(20261126)
    wl = sorted(wins.values(), key=lambda w: w["window_id"])
    results: dict[str, object] = {}

    # (1) 200 synthetic grid boxes: L0 -> px -> LS percent -> px -> L0
    #     (exact double percent)
    bad = bad4 = bad6 = 0
    for _ in range(200):
        w = rnd.choice(wl)
        t0, n_tr, n_s = (
            int(w["window_trace0"]),
            int(w["n_traces_window"]),
            int(w["n_samples"]),
        )
        i0 = t0 + rnd.randrange(0, n_tr)
        i1 = min(t0 + n_tr - 1, i0 + rnd.randrange(0, 120))
        k0 = rnd.randrange(0, n_s)
        k1 = min(n_s - 1, k0 + rnd.randrange(0, 150))
        W, H = float(w["view_w_px"]), float(w["view_h_px"])
        px = grid_to_px(i0, i1, k0, k1, t0, sx, sy)
        v = px_to_ls_box(*px, W, H)
        back = ls_box_to_px(v, W, H)
        c = px_to_cont(*back, t0, sx, sy)
        got = (*cont_to_grid(c[0], c[1]), *cont_to_grid(c[2], c[3]))
        if got != (i0, i1, k0, k1):
            bad += 1
        for nd in (4, 6):  # information only: coarse percent rounding
            vr = {k: (round(val, nd) if k != "rotation" else 0) for k, val in v.items()}
            br = ls_box_to_px(vr, W, H)
            cr = px_to_cont(*br, t0, sx, sy)
            gr = (*cont_to_grid(cr[0], cr[1]), *cont_to_grid(cr[2], cr[3]))
            if gr != (i0, i1, k0, k1):
                if nd == 4:
                    bad4 += 1
                else:
                    bad6 += 1
    results["roundtrip_200_exact_mismatch"] = bad
    results["roundtrip_200_percent_rounded_4dec_mismatch_info"] = bad4
    results["roundtrip_200_percent_rounded_6dec_mismatch_info"] = bad6

    # (2) synthetic LS export -> convert -> compare with expected L0
    w = wl[0]
    t0, n_tr, n_s = (
        int(w["window_trace0"]),
        int(w["n_traces_window"]),
        int(w["n_samples"]),
    )
    W, H = int(w["view_w_px"]), int(w["view_h_px"])
    exp = {"i0": t0 + 100, "i1": t0 + 160, "k0": 40, "k1": 130}
    px = grid_to_px(exp["i0"], exp["i1"], exp["k0"], exp["k1"], t0, sx, sy)
    box = px_to_ls_box(*px, W, H)
    # 20 samples below top (P=20.0 -> 1.0 P, inside the 0.6-1.5 P band)
    apex_px = (0.5 * (px[0] + px[1]), px[2] + 20 * sy)

    def mk(rid, typ, fn, value, **kw):
        return {
            "id": rid,
            "type": typ,
            "from_name": fn,
            "to_name": "image",
            "original_width": W,
            "original_height": H,
            "image_rotation": 0,
            "value": value,
            **kw,
        }

    result = [
        mk(
            "a1",
            "rectanglelabels",
            "label",
            {**box, "rectanglelabels": ["pipe_hyperbola"]},
        ),
        mk(
            "k1",
            "keypointlabels",
            "apex",
            {
                "x": apex_px[0] / W * 100,
                "y": apex_px[1] / H * 100,
                "width": 0.3,
                "keypointlabels": ["apex"],
            },
        ),
        mk("a1", "choices", "certainty", {"choices": ["certain"]}),
        mk("w1", "choices", "window_complete", {"choices": ["yes"]}),
        mk("n1", "textarea", "note", {"text": ["selftest"]}),
    ]
    task = {
        "id": 1,
        "data": {"window_id": w["window_id"], "bundle": "main"},
        "annotations": [
            {
                "id": 9,
                "completed_by": 3,
                "result": result,
                "was_cancelled": False,
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ],
        "predictions": [],
    }
    rows, status, issues = convert(
        [task], cfg, cfg_sha, {w["window_id"]: w}, "main", period
    )
    r0 = rows[0]
    results["synthetic_export_l0_match"] = all(r0[k] == v for k, v in exp.items())
    results["synthetic_export_apex_sample"] = r0["apex_sample"]
    results["synthetic_export_qa"] = [i["code"] for i in issues]
    results["synthetic_window_complete"] = status[0]["window_complete"]
    # predictions assertion must stop
    task_bad = json.loads(json.dumps(task))
    task_bad["predictions"] = [{"result": []}]
    try:
        convert([task_bad], cfg, cfg_sha, {w["window_id"]: w}, "main", period)
        results["predictions_assertion"] = "NOT RAISED"
    except SystemExit:
        results["predictions_assertion"] = "raised"
    # apex outside box + missing certainty -> ERROR rows
    task_q = json.loads(json.dumps(task))
    task_q["annotations"][0]["result"][1]["value"]["y"] = 99.0
    task_q["annotations"][0]["result"] = [
        r for r in task_q["annotations"][0]["result"] if r["from_name"] != "certainty"
    ]
    _, _, iss2 = convert([task_q], cfg, cfg_sha, {w["window_id"]: w}, "main", period)
    results["qa_error_cases"] = sorted(
        {i["code"] for i in iss2 if i["level"] == "ERROR"}
    )
    # report-only P=22.3 count: apex 20 samples below top -> 0.90 P (not counted);
    # 13 samples -> 0.58 P (counted) while still 0.65 P at P=20.0 (no WARN row)
    task_shallow = json.loads(json.dumps(task))
    shallow_y_px = px[2] + 13 * sy
    task_shallow["annotations"][0]["result"][1]["value"]["y"] = shallow_y_px / H * 100
    rows_shallow, _, iss3 = convert(
        [task_shallow], cfg, cfg_sha, {w["window_id"]: w}, "main", period
    )
    results["top_margin_p22_3_count_20_vs_13_samples"] = [
        count_shallow_top_margin(rows),
        count_shallow_top_margin(rows_shallow),
    ]
    results["top_margin_p22_3_shallow_qa"] = [i["code"] for i in iss3]
    # apex 2 samples above the box bottom: top 4.45 P and bottom 0.10 P, so both
    # margin WARNs fire unless the box carries the truncated flag
    margin_codes = {"top_margin_outside_0.6-1.5P", "bottom_outside_3-9P"}
    task_deep = json.loads(json.dumps(task))
    deep_y_px = px[3] - 2 * sy
    task_deep["annotations"][0]["result"][1]["value"]["y"] = deep_y_px / H * 100
    task_deep_trunc = json.loads(json.dumps(task_deep))
    task_deep_trunc["annotations"][0]["result"].append(
        mk("a1", "choices", "flag", {"choices": ["truncated"]})
    )
    rows_deep, _, iss_deep = convert(
        [task_deep], cfg, cfg_sha, {w["window_id"]: w}, "main", period
    )
    rows_trunc, _, iss_trunc = convert(
        [task_deep_trunc], cfg, cfg_sha, {w["window_id"]: w}, "main", period
    )
    results["margin_warn_untruncated_qa"] = sorted(
        {i["code"] for i in iss_deep} & margin_codes
    )
    results["margin_warn_truncated_qa"] = sorted(
        {i["code"] for i in iss_trunc} & margin_codes
    )
    results["margin_warn_truncated_row_qa"] = rows_trunc[0]["qa"]
    results["truncated_rows_skipped_margin_qa_untruncated_vs_truncated"] = [
        margin_report_counts(rows_deep)["truncated_rows_skipped_margin_qa"],
        margin_report_counts(rows_trunc)["truncated_rows_skipped_margin_qa"],
    ]
    ok = (
        bad == 0
        and results["synthetic_export_l0_match"]
        and results["predictions_assertion"] == "raised"
        and "apex_outside_box" in results["qa_error_cases"]
        and "certainty_missing" in results["qa_error_cases"]
        and results["top_margin_p22_3_count_20_vs_13_samples"] == [0, 1]
        and results["top_margin_p22_3_shallow_qa"] == []
        and results["margin_warn_untruncated_qa"] == sorted(margin_codes)
        and results["margin_warn_truncated_qa"] == []
        and not margin_codes & set(rows_trunc[0]["qa"].split(";"))
        and results["truncated_rows_skipped_margin_qa_untruncated_vs_truncated"]
        == [0, 1]
    )
    print(json.dumps({"status": "ok" if ok else "fail", **results}, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export", type=Path, help="Label Studio JSON export (full, not JSON-MIN)"
    )
    ap.add_argument("--bundle", choices=("main", "relabel"), default="main")
    ap.add_argument("--kit", type=Path, default=KIT)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--period",
        type=float,
        default=PERIOD_FIXED,
        help="carrier period P in samples (default 20.0 = M3-6 fixed value)",
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest(args.kit, args.period)
    if not args.export:
        ap.error("--export required")
    cfg, cfg_sha, wins = load_kit(args.kit)
    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    wins_b = {
        k: v for k, v in wins.items() if args.bundle == "main" or v["relabel"] == "yes"
    }
    rows, status, issues = convert(
        tasks, cfg, cfg_sha, wins_b, args.bundle, args.period
    )
    out = args.out or (args.kit / f"export_{args.bundle}")
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.bundle == "main" else "_relabel"
    write_csv(out / f"labels_l0{suffix}.csv", rows, LABEL_COLS)
    write_csv(
        out / f"windows_status{suffix}.csv",
        status,
        ["window_id", "bundle", "labeled", "window_complete", "n_boxes", "note"],
    )
    write_csv(
        out / f"qa_report{suffix}.csv",
        issues,
        ["window_id", "level", "code", "detail", "box"],
    )
    summ = {
        "export": str(args.export),
        "export_sha256": hashlib.sha256(args.export.read_bytes()).hexdigest(),
        "bundle": args.bundle,
        "tasks": len(tasks),
        "predictions_total": 0,
        "labels": len(rows),
        "by_class": {c: sum(r["class"] == c for r in rows) for c in CLASSES},
        "errors": sum(i["level"] == "ERROR" for i in issues),
        "warnings": sum(i["level"] == "WARN" for i in issues),
        "period_samples": args.period,
        "period_is_m36_fixed": args.period == PERIOD_FIXED,
        **margin_report_counts(rows),
        "render_config_sha256": cfg_sha,
    }
    (out / f"convert_summary{suffix}.json").write_text(
        json.dumps(summ, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": "ok" if summ["errors"] == 0 else "errors",
                **{k: summ[k] for k in ("labels", "errors", "warnings")},
                "out": str(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
