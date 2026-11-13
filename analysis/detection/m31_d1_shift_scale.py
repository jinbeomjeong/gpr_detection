"""M3-1 D1: position (dx) and scale (g) sensitivity of the M1-10 YOLO11m models.

Design and pre-registered decision rules: docs/experiments/m31_detection_input_tiling.md §8-3.

For every fold k the model ``runs/detect/<prefix><k>/weights/last.pt`` is run on
its own validation crops (crop unit, ``eval/last_crop/eval_images.txt``, the
list M1-10 scored). Each 256 x 256 crop is rescaled by ``g`` (isotropic, area
interpolation when shrinking, linear when enlarging), placed on a 256 x 256
canvas filled with ``--fill`` (top row at y = 0 so the surface stays on top,
crop centre column at x = 128 + dx) and cut to the canvas. Labels follow the
same map. A box that is not fully inside the canvas (or touches a fill strip of
the fill-control conditions) becomes an *ignore* box: predictions overlapping it
(IoU >= 0.5 or IoA >= 0.5) are dropped before scoring.

Inference goes through the project backend ``gpr_detection.detect.yolo``
(conf 0.001, NMS IoU 0.7, input 256, one image per call). Because every canvas
is 256 x 256 the backend letterbox is the identity.

Metrics per fold and condition: per-class AP50 (Ultralytics-style 101-point
interpolation, greedy score-ordered matching at IoU >= 0.5) and recall at
conf > 0.25 (same matching). The validation set is never test: lists whose path
contains 'test' are refused.

Outputs:
  <run>/eval/m31_d1/metrics.json and objects.csv per fold (runs/ is git-ignored),
  --summary JSON (fold table, paired differences, decision inputs).
Last stdout line is a JSON summary.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPO / "runs" / "detect"
DEFAULT_RUN_PREFIX = "yolo11m_aihubv1_20261101_fold"
LABEL_DIR = REPO / "data" / "aihub_yolo_v1" / "labels"
CLASS_NAMES = ("pipe_hyperbola", "pipe_band", "cavity", "manhole")
CANVAS = 256
IOU_MATCH = 0.5
CONF_RECALL = 0.25
CONF_AP = 0.001
NMS_IOU = 0.7
IGNORE_OVERLAP = 0.5
FILL_CONTROL_PX = 64
P_STAR_RATIO = 6.0 / 4.06

# name -> (g, dx, fill_strip) ; fill_strip: None, "L", "R"
SCALES_DX0 = (0.5, 0.74, 0.87, 1.0, 1.13, 1.28, P_STAR_RATIO, 1.67)
SHIFTS_G1 = (-96, -64, -32, 32, 64, 96)
SHIFTS_G6 = (-64, -32, 0, 32, 64)


def _cond_name(g: float, dx: int, strip: str | None = None) -> str:
    gtxt = "p6" if abs(g - P_STAR_RATIO) < 1e-9 else f"{g:.2f}"
    base = f"g{gtxt}_dx{dx:+d}"
    return base if strip is None else f"{base}_fill{strip}{FILL_CONTROL_PX}"


def default_conditions() -> dict[str, tuple[float, int, str | None]]:
    conds: dict[str, tuple[float, int, str | None]] = {}
    for g in SCALES_DX0:
        conds[_cond_name(g, 0)] = (g, 0, None)
    for dx in SHIFTS_G1:
        conds[_cond_name(1.0, dx)] = (1.0, dx, None)
    for dx in SHIFTS_G6:
        conds[_cond_name(P_STAR_RATIO, dx)] = (P_STAR_RATIO, dx, None)
    conds[_cond_name(1.0, 0, "L")] = (1.0, 0, "L")
    conds[_cond_name(1.0, 0, "R")] = (1.0, 0, "R")
    return conds


@dataclass
class Sample:
    stem: str
    image: np.ndarray  # uint8 [256, 256]
    boxes: np.ndarray  # [n, 4] xyxy px
    classes: np.ndarray  # [n]


def read_samples(eval_list: Path, limit: int | None) -> list[Sample]:
    if "test" in str(eval_list).lower():
        raise SystemExit(f"refusing a list that looks like test: {eval_list}")
    samples = []
    for line in eval_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (CANVAS, CANVAS):
            raise SystemExit(f"bad image {path}")
        rows = []
        label = LABEL_DIR / f"{path.stem}.txt"
        if label.is_file():
            for row in label.read_text(encoding="utf-8").splitlines():
                parts = row.split()
                if len(parts) == 5:
                    c, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
                    rows.append((c, (xc - w / 2) * CANVAS, (yc - h / 2) * CANVAS,
                                 (xc + w / 2) * CANVAS, (yc + h / 2) * CANVAS))
        arr = np.asarray(rows, dtype=float).reshape(-1, 5)
        samples.append(Sample(path.stem, image, arr[:, 1:5], arr[:, 0].astype(int)))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def transform(sample: Sample, g: float, dx: int, strip: str | None, fill: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (canvas, kept_boxes, kept_classes, ignore_boxes)."""
    size = max(1, round(CANVAS * g))
    interp = cv2.INTER_AREA if g < 1 else cv2.INTER_LINEAR
    resized = sample.image if size == CANVAS else cv2.resize(sample.image, (size, size), interpolation=interp)
    scale = size / CANVAS
    off_x = int(round(CANVAS / 2 + dx - size / 2))
    off_y = 0
    canvas = np.full((CANVAS, CANVAS), fill, dtype=np.uint8)
    x0, x1 = max(0, off_x), min(CANVAS, off_x + size)
    y0, y1 = max(0, off_y), min(CANVAS, off_y + size)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = resized[y0 - off_y:y1 - off_y, x0 - off_x:x1 - off_x]
    valid = np.array([x0, y0, x1, y1], dtype=float)
    if strip == "L":
        canvas[:, :FILL_CONTROL_PX] = fill
        valid[0] = max(valid[0], FILL_CONTROL_PX)
    elif strip == "R":
        canvas[:, CANVAS - FILL_CONTROL_PX:] = fill
        valid[2] = min(valid[2], CANVAS - FILL_CONTROL_PX)
    boxes = sample.boxes * scale + np.array([off_x, off_y, off_x, off_y], dtype=float)
    inside = ((boxes[:, 0] >= valid[0] - 1e-6) & (boxes[:, 1] >= valid[1] - 1e-6)
              & (boxes[:, 2] <= valid[2] + 1e-6) & (boxes[:, 3] <= valid[3] + 1e-6))
    ignore = boxes[~inside].copy()
    if ignore.size:
        ignore[:, [0, 2]] = ignore[:, [0, 2]].clip(valid[0], valid[2])
        ignore[:, [1, 3]] = ignore[:, [1, 3]].clip(valid[1], valid[3])
        ignore = ignore[(ignore[:, 2] > ignore[:, 0]) & (ignore[:, 3] > ignore[:, 1])]
    return canvas, boxes[inside], sample.classes[inside], ignore.reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def ioa_pred(pred: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Intersection over prediction area, [n_pred, n_region]."""
    if len(pred) == 0 or len(region) == 0:
        return np.zeros((len(pred), len(region)))
    ix = np.clip(np.minimum(pred[:, None, 2], region[None, :, 2]) - np.maximum(pred[:, None, 0], region[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(pred[:, None, 3], region[None, :, 3]) - np.maximum(pred[:, None, 1], region[None, :, 1]), 0, None)
    area = np.maximum((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]), 1e-9)
    return ix * iy / area[:, None]


def match(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching. Returns (pred_tp bool, gt_best_score or -1)."""
    order = np.argsort(-pred_scores, kind="stable")
    ious = iou_matrix(pred_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    gt_score = np.full(len(gt_boxes), -1.0)
    for i in order:
        if len(gt_boxes) == 0:
            break
        cand = np.where(~used & (ious[i] >= IOU_MATCH))[0]
        if cand.size:
            j = cand[np.argmax(ious[i, cand])]
            used[j] = True
            tp[i] = True
            gt_score[j] = pred_scores[i]
    return tp, gt_score


def ap_101(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(tps[order])
    fp = np.cumsum(~tps[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate_condition(detector, samples: list[Sample], cond: tuple, fill: int,
                       fold: int, name: str, object_rows: list, save_dir: Path | None) -> dict:
    g, dx, strip = cond
    per_class = {c: {"scores": [], "tps": [], "n_gt": 0, "hit": 0} for c in range(len(CLASS_NAMES))}
    n_ignored = {c: 0 for c in range(len(CLASS_NAMES))}
    for idx, sample in enumerate(samples):
        canvas, gt, gt_cls, ignore = transform(sample, g, dx, strip, fill)
        lost = (np.bincount(sample.classes, minlength=len(CLASS_NAMES))
                - np.bincount(gt_cls, minlength=len(CLASS_NAMES)))
        for c in range(len(CLASS_NAMES)):
            n_ignored[c] += int(lost[c])
        out = detector.predict(canvas)
        t = out.transform
        if not (t.scale_x == 1 and t.scale_y == 1 and t.pad_x == 0 and t.pad_y == 0):
            raise RuntimeError(f"letterbox not identity: {t}")
        if out.detections:
            pb = np.array([d.box.xyxy for d in out.detections], dtype=float)
            ps = np.array([d.score for d in out.detections], dtype=float)
            pc = np.array([d.class_id for d in out.detections], dtype=int)
        else:
            pb, ps, pc = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        if len(ignore) and len(pb):
            drop = (iou_matrix(pb, ignore).max(axis=1) >= IGNORE_OVERLAP) | (ioa_pred(pb, ignore).max(axis=1) >= IGNORE_OVERLAP)
            pb, ps, pc = pb[~drop], ps[~drop], pc[~drop]
        for c in range(len(CLASS_NAMES)):
            g_sel = gt_cls == c
            p_sel = pc == c
            tp, gt_score = match(pb[p_sel], ps[p_sel], gt[g_sel])
            per_class[c]["scores"].append(ps[p_sel])
            per_class[c]["tps"].append(tp)
            per_class[c]["n_gt"] += int(g_sel.sum())
            # recall at conf > 0.25: re-match with only confident predictions
            conf_sel = ps[p_sel] > CONF_RECALL
            _, gt_score_conf = match(pb[p_sel][conf_sel], ps[p_sel][conf_sel], gt[g_sel])
            per_class[c]["hit"] += int((gt_score_conf >= 0).sum())
            for b, s in zip(gt[g_sel], gt_score_conf):
                object_rows.append([fold, name, sample.stem, CLASS_NAMES[c],
                                    round((b[0] + b[2]) / 2, 2), round(b[2] - b[0], 2),
                                    round(b[3] - b[1], 2), int(s >= 0), round(float(s), 4)])
        if save_dir is not None and idx < 4:
            vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            for b in gt:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(math.ceil(b[2])) - 1, int(math.ceil(b[3])) - 1), (0, 255, 0), 1)
            for b, s in zip(pb, ps):
                if s > CONF_RECALL:
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(math.ceil(b[2])) - 1, int(math.ceil(b[3])) - 1), (0, 0, 255), 1)
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"{name}_{idx}.png"), vis)
    result = {"g": g, "dx": dx, "fill_strip": strip, "classes": {}}
    aps = []
    for c, acc in per_class.items():
        scores = np.concatenate(acc["scores"]) if acc["scores"] else np.zeros(0)
        tps = np.concatenate(acc["tps"]) if acc["tps"] else np.zeros(0, dtype=bool)
        ap = ap_101(scores, tps, acc["n_gt"])
        rec = acc["hit"] / acc["n_gt"] if acc["n_gt"] else None
        result["classes"][CLASS_NAMES[c]] = {"n_gt": acc["n_gt"], "n_ignored": n_ignored[c],
                                             "ap50": ap, "recall_conf025": rec}
        if ap is not None:
            aps.append(ap)
    result["map50"] = float(np.mean(aps)) if aps else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="0", help="'0' (GPU) or 'cpu'")
    parser.add_argument("--fill", type=int, default=128, help="assumed fill grey (Q3 deferred to M3-2)")
    parser.add_argument("--conditions", default="all", help="comma list of names or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="images per fold (smoke only)")
    parser.add_argument("--out-subdir", default="m31_d1")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--save-examples", action="store_true")
    parser.add_argument("--list-conditions", action="store_true")
    args = parser.parse_args(argv)

    conds = default_conditions()
    if args.list_conditions:
        print(json.dumps(list(conds)))
        return 0
    if args.conditions != "all":
        wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
        missing = [c for c in wanted if c not in conds]
        if missing:
            print(json.dumps({"error": f"unknown conditions {missing}"}))
            return 2
        conds = {c: conds[c] for c in wanted}

    from gpr_detection.detect.yolo import load_yolo_detector  # lazy (AGPL inside detect/)

    folds = [int(f) for f in args.folds.split(",")]
    summary = {"schema": "m31_d1/1", "fill": args.fill, "limit": args.limit,
               "device": args.device, "conditions": {k: list(v) for k, v in conds.items()},
               "folds": {}}
    started = time.time()
    for fold in folds:
        run_dir = args.run_root / f"{args.run_prefix}{fold}"
        weights = run_dir / "weights" / "last.pt"
        eval_list = run_dir / "eval" / "last_crop" / "eval_images.txt"
        if not weights.is_file() or not eval_list.is_file():
            print(json.dumps({"error": f"missing weights or eval list for fold {fold}"}))
            return 2
        detector = load_yolo_detector(weights, input_size=CANVAS, confidence_threshold=CONF_AP,
                                      iou_threshold=NMS_IOU, device=args.device)
        samples = read_samples(eval_list, args.limit)
        out_dir = run_dir / "eval" / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        object_rows: list = []
        fold_result = {"weights": str(weights), "eval_list": str(eval_list), "n_images": len(samples),
                       "conditions": {}}
        for name, cond in conds.items():
            t0 = time.time()
            res = evaluate_condition(detector, samples, cond, args.fill, fold, name, object_rows,
                                     out_dir / "examples" if args.save_examples else None)
            res["seconds"] = round(time.time() - t0, 1)
            fold_result["conditions"][name] = res
        m110 = json.loads((run_dir / "eval" / "last_crop" / "metrics.json").read_text(encoding="utf-8"))
        fold_result["m110_ap50"] = {pc["name"]: pc["ap50"] for pc in m110["per_class"]}
        (out_dir / "metrics.json").write_text(json.dumps(fold_result, indent=1), encoding="utf-8")
        with (out_dir / "objects.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fold", "condition", "image", "class", "x_center", "w", "h", "hit_conf025", "score"])
            writer.writerows(object_rows)
        summary["folds"][str(fold)] = fold_result
    summary["seconds_total"] = round(time.time() - started, 1)
    if args.summary is not None:
        args.summary.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    brief = {f: {n: c["classes"]["pipe_hyperbola"]["recall_conf025"] for n, c in r["conditions"].items()}
             for f, r in summary["folds"].items()}
    print(json.dumps({"seconds": summary["seconds_total"], "hyperbola_recall": brief}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
