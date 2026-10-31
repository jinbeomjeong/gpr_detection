"""M1-7 pipe_band background false-positive gallery (figure 3) and FP list (GPU inference).

Validates runs/detect/m17_aug0/weights/last.pt with YOLO.val on eval/last_crop/eval_data.yaml (fold 0 crop
list, 1,298 images, imgsz 256, conf 0.25, NMS iou 0.7, channels 1) - the same call path as
scripts/eval_detector.py --confusion-conf 0.25 - and records, by wrapping ConfusionMatrix.process_batch in
this process, the exact per-image detections and GT boxes behind the matrix (val plots go to
runs/analysis/m17_pipe_band_fp_val). Predictions are matched to ground truth the way the
Ultralytics 8.4.192 ConfusionMatrix does (class-agnostic, IoU > 0.45, greedy in descending confidence,
one-to-one); an unmatched pipe_band prediction is a background FP. The rebuilt 5x5 matrix and the val
matrix are both compared with eval/last_crop/confusion_matrix_conf0.25.csv (exit 1 on any difference).

FP categories (pipe_band predictions left unmatched):
  duplicate   - IoU >= 0.45 with a pipe_band GT that another prediction already took
  fragment    - overlaps a pipe_band GT (IoU > 0) but IoU < 0.45 (split / partial band suspected)
  other_class - no pipe_band GT overlap, overlaps a GT of another class
  no_gt       - overlaps no GT at all

Usage: python -I analysis/detection/m17_pipe_band_fp_gallery.py [--root ...] [--batch 64] [--device 0]
Writes analysis/detection/m17_pipe_band_fp_list.csv and
docs/experiments/figures/m17_pipe_band_fp_gallery_{duplicate,fragment,no_gt}.png; last stdout line is JSON.
Imports ultralytics (AGPL-3.0) as an analysis-only script, like analysis/detection/m110_epoch_budget.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

RUN = "m17_aug0"
CLASSES = ["pipe_hyperbola", "pipe_band", "cavity", "manhole"]
SHORT = {0: "hyp", 1: "band", 2: "cav", 3: "manh"}
BAND = 1
MATCH_IOU = 0.45
CONF = 0.25
SEED = 0
GT_COLOR, FP_COLOR = "#1baf7a", "#e34948"  # dataviz aqua(green role) / red
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def inter_over_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """intersection(a_i, b_j) / area(b_j)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    return inter / ((b[:, 2:] - b[:, :2]).prod(1)[None, :] + 1e-9)


def read_gt(label: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    cls, boxes = [], []
    if label.exists():
        for line in label.read_text(encoding="utf-8").split("\n"):
            p = line.split()
            if len(p) < 5:
                continue
            c, x, y, bw, bh = int(p[0]), *map(float, p[1:5])
            cls.append(c)
            boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(cls, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


def match(gt_b: np.ndarray, det_b: np.ndarray, det_s: np.ndarray) -> list[tuple[int, int]]:
    """Ultralytics 8.4.192 ConfusionMatrix.process_batch matching (class-agnostic, greedy one-to-one):
    detections with any IoU > thr, in descending confidence (stable), each claims its best unclaimed GT."""
    if len(gt_b) == 0 or len(det_b) == 0:
        return []
    iou = iou_matrix(gt_b, det_b)
    gt_match = np.full(len(gt_b), -1)
    cand = np.flatnonzero((iou > MATCH_IOU).any(0))
    for d in cand[np.argsort(-det_s[cand], kind="stable")]:
        g = iou[:, d].argmax()
        if iou[g, d] > MATCH_IOU:
            gt_match[g] = d
            iou[g] = 0
    return [(int(g), int(d)) for g, d in enumerate(gt_match) if d >= 0]


def read_cm_csv(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=int)


def draw_panel(ax, rec: dict, highlight: int, fp_cat: str) -> None:
    img = np.asarray(Image.open(rec["path"]).convert("L"))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for c, b in zip(rec["gt_cls"], rec["gt_box"]):
        ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=GT_COLOR,
                                        linewidth=1.3))
        ax.text(b[0] + 1, b[3] - 2, SHORT[c], color=GT_COLOR, fontsize=6, va="bottom",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.5, "linewidth": 0})
    for k, fp in enumerate(rec["fps"]):
        if fp["category"] != fp_cat:
            continue
        b = fp["box"]
        lw = 1.8 if k == highlight else 1.0
        ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=FP_COLOR,
                                        linewidth=lw, linestyle="-" if k == highlight else "--"))
        ax.text(b[0] + 1, b[1] + 1, f"{fp['conf']:.2f}", color="white", fontsize=6, va="top",
                bbox={"facecolor": FP_COLOR, "alpha": 0.85, "pad": 0.5, "linewidth": 0})
    fp = rec["fps"][highlight]
    stem = Path(rec["path"]).stem
    session, _, rest = stem.partition(")_")
    ax.set_title(f"{session})\n{rest}\nIoU(band)={fp['max_iou_band']:.2f}", fontsize=5.5, color=INK2, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def gallery(records: dict, fps: list[dict], cat: str, n_each: int, out: Path, title: str) -> dict:
    cand = [f for f in fps if f["category"] == cat]
    # one panel per image: the highest-conf FP of this category in the image
    best: dict[str, dict] = {}
    for f in cand:
        if f["image"] not in best or f["conf"] > best[f["image"]]["conf"]:
            best[f["image"]] = f
    per_img = sorted(best.values(), key=lambda f: -f["conf"])
    top = per_img[:n_each]
    rest = per_img[n_each:]
    rng = random.Random(SEED)
    rnd = rng.sample(rest, min(n_each, len(rest)))
    ncol = 8
    rows_top = -(-len(top) // ncol)
    rows_rnd = -(-len(rnd) // ncol)
    nrows = rows_top + rows_rnd
    head_in, cell_in = 0.75, 2.15
    fig_h = nrows * cell_in + head_in
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 1.75, fig_h), facecolor=SURFACE, squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - (head_in + 0.35) / fig_h, wspace=0.06,
                        hspace=0.42)
    for ax in axes.ravel():
        ax.axis("off")
    for i, f in enumerate(top):
        ax = axes[i // ncol, i % ncol]
        ax.axis("on")
        draw_panel(ax, records[f["image"]], f["idx_in_image"], cat)
    for i, f in enumerate(rnd):
        ax = axes[rows_top + i // ncol, i % ncol]
        ax.axis("on")
        draw_panel(ax, records[f["image"]], f["idx_in_image"], cat)
    fig.suptitle(title + f"\n위 {rows_top}줄: 신뢰도 상위 {len(top)}장(이미지당 최고 conf FP 1개 기준) · "
                 f"아래 {rows_rnd}줄: 나머지 {len(rest)}장 중 무작위 {len(rnd)}장(seed {SEED})\n"
                 "초록 = 정답 박스(hyp/band/cav/manh), 빨강 = 이 범주의 pipe_band 오검출(숫자 conf, 실선 = 패널 대표)",
                 fontsize=9, color=INK, x=0.01, y=1 - 0.08 / fig_h, ha="left", va="top")
    if rows_top and rows_rnd:
        # separator midway between the bottom of the last top row and the title area of the first random row
        y_low = axes[rows_top - 1, 0].get_position().y0
        y_high = axes[rows_top, 0].get_position().y1
        y = y_low - (y_low - y_high) * 0.12
        fig.add_artist(plt.Line2D([0.01, 0.99], [y, y], color="#898781", linewidth=1.2, linestyle="--",
                                  transform=fig.transFigure))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return {"images_in_category": len(per_img), "top": len(top), "random": len(rnd), "out": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    root = Path(a.root)
    run_dir = root / "runs" / "detect" / RUN
    ev = run_dir / "eval" / "last_crop"
    fig_dir = root / "docs" / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_csv = root / "analysis" / "detection" / "m17_pipe_band_fp_list.csv"
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        images = [p.strip() for p in (ev / "eval_images.txt").read_text(encoding="utf-8").splitlines() if p.strip()]
        cm_ref = read_cm_csv(ev / "confusion_matrix_conf0.25.csv")
        weights = run_dir / "weights" / "last.pt"
        assert weights.exists(), weights
    except (OSError, AssertionError, ValueError) as e:
        print(json.dumps({"status": "input_error", "error": repr(e)}))
        return 2

    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix

    # Same path as scripts/eval_detector.py --confusion-conf 0.25 (YOLO.val: rect batches, multi-label NMS,
    # FP16 on GPU). ConfusionMatrix.process_batch is wrapped (in this process only) to record the exact
    # detections and GT boxes it receives, so the FP list is built from the numbers behind the matrix.
    captured: dict[str, dict] = {}
    orig_pb = ConfusionMatrix.process_batch

    def recording_pb(self, detections, batch, conf=0.25, iou_thres=0.45):
        captured[str(Path(batch["im_file"]).resolve())] = {
            "det_b": detections["bboxes"].detach().float().cpu().numpy().copy(),
            "det_s": detections["conf"].detach().float().cpu().numpy().copy(),
            "det_c": detections["cls"].detach().cpu().numpy().astype(int).copy(),
            "gt_b": batch["bboxes"].detach().float().cpu().numpy().copy(),
            "gt_c": batch["cls"].detach().cpu().numpy().astype(int).reshape(-1).copy(),
            "conf": conf, "iou_thres": iou_thres,
            "ori_shape": tuple(int(v) for v in batch["ori_shape"]),
            "imgsz": tuple(int(v) for v in batch["imgsz"]),
            "ratio_pad": batch["ratio_pad"],
        }
        return orig_pb(self, detections, batch, conf=conf, iou_thres=iou_thres)

    ConfusionMatrix.process_batch = recording_pb
    model = YOLO(str(weights))
    val_dir = root / "runs" / "analysis" / "m17_pipe_band_fp_val"
    try:
        metrics = model.val(data=str(ev / "eval_data.yaml"), split="val", imgsz=256, batch=a.batch, conf=CONF,
                            iou=0.7, device=a.device, plots=True, project=str(val_dir.parent), name=val_dir.name,
                            exist_ok=True, verbose=False)
    finally:
        ConfusionMatrix.process_batch = orig_pb
    cm_val = np.asarray(metrics.confusion_matrix.matrix, dtype=int)
    assert len(captured) == len(images), (len(captured), len(images))
    records: dict[str, dict] = {}
    fps: list[dict] = []
    cm = np.zeros((5, 5), dtype=int)  # rows predicted (4 classes + background), cols true
    for path in images:
        cap = captured[str(Path(path).resolve())]
        # Matching runs in the validator's input (letterboxed rect-batch) space, as process_batch does; boxes
        # are mapped back to original pixels (gain, pad from ratio_pad) only for the CSV, IoU columns and plots.
        assert abs(cap["iou_thres"] - MATCH_IOU) < 1e-9 and abs(cap["conf"] - CONF) < 1e-9
        (gain, *_), (padw, padh) = cap["ratio_pad"][0], cap["ratio_pad"][1]
        gain, padw, padh = float(gain), float(padw), float(padh)
        to_orig = np.array([padw, padh, padw, padh])
        keep = cap["det_s"] > cap["conf"]  # process_batch filters conf > threshold
        det_bi, det_s, det_c = cap["det_b"][keep], cap["det_s"][keep], cap["det_c"][keep]
        gt_c, gt_bi = cap["gt_c"], cap["gt_b"].reshape(-1, 4)
        pairs = match(gt_bi, det_bi, det_s)
        det_b = (det_bi - to_orig) / gain
        gt_b = (gt_bi - to_orig) / gain
        f_c, f_b = read_gt(Path(path).parent.parent / "labels" / (Path(path).stem + ".txt"), *cap["ori_shape"][::-1])
        assert np.array_equal(np.sort(f_c), np.sort(gt_c)) and np.allclose(
            np.sort(f_b, 0), np.sort(gt_b, 0), atol=0.05), path  # label file == boxes seen by the validator
        mg = {g for g, _ in pairs}
        md = {d for _, d in pairs}
        for g, d in pairs:
            cm[det_c[d], gt_c[g]] += 1
        for g in range(len(gt_c)):
            if g not in mg:
                cm[4, gt_c[g]] += 1
        iou = iou_matrix(det_b, gt_b)
        cover = inter_over_b(gt_b, det_b).T if len(gt_b) else np.zeros((len(det_b), 0))  # det area inside GT
        rec = {"path": path, "gt_cls": gt_c.tolist(), "gt_box": gt_b.tolist(), "fps": []}
        for d in range(len(det_c)):
            if d in md:
                continue
            cm[det_c[d], 4] += 1
            if det_c[d] != BAND:
                continue
            band_idx = np.where(gt_c == BAND)[0]
            max_iou_band = float(iou[d, band_idx].max()) if len(band_idx) else 0.0
            cover_band = float(cover[d, band_idx].max()) if len(band_idx) else 0.0
            any_iou = float(iou[d].max()) if len(gt_c) else 0.0
            if len(band_idx) and max_iou_band >= MATCH_IOU:
                cat = "duplicate"
            elif max_iou_band > 0:
                cat = "fragment"
            elif any_iou > 0:
                cat = "other_class"
            else:
                cat = "no_gt"
            if len(gt_c):
                ctr_d = (det_b[d, :2] + det_b[d, 2:]) / 2
                ctr_g = (gt_b[:, :2] + gt_b[:, 2:]) / 2
                dist = np.linalg.norm(ctr_g - ctr_d, axis=1)
                near = int(np.lexsort((dist, -iou[d]))[0])
                near_cls, near_iou, near_dist = CLASSES[gt_c[near]], float(iou[d, near]), float(dist[near])
            else:
                near_cls, near_iou, near_dist = "", 0.0, float("nan")
            fp = {"image": path, "idx_in_image": len(rec["fps"]), "box": det_b[d].tolist(),
                  "conf": float(det_s[d]), "category": cat, "nearest_gt_class": near_cls,
                  "nearest_gt_iou": near_iou, "nearest_gt_center_dist_px": near_dist,
                  "max_iou_band": max_iou_band, "pred_area_in_band_gt": cover_band,
                  "n_gt_band": int(len(band_idx)), "n_gt_total": int(len(gt_c))}
            rec["fps"].append(fp)
            fps.append(fp)
        records[path] = rec

    diff = cm - cm_ref
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "x1", "y1", "x2", "y2", "conf", "category", "nearest_gt_class", "nearest_gt_iou",
                     "nearest_gt_center_dist_px", "max_iou_pipe_band_gt", "pred_area_in_pipe_band_gt",
                     "n_gt_pipe_band", "n_gt_total"])
        for fp in sorted(fps, key=lambda x: -x["conf"]):
            wr.writerow([fp["image"], *(f"{v:.1f}" for v in fp["box"]), f"{fp['conf']:.4f}", fp["category"],
                         fp["nearest_gt_class"], f"{fp['nearest_gt_iou']:.4f}",
                         f"{fp['nearest_gt_center_dist_px']:.1f}", f"{fp['max_iou_band']:.4f}",
                         f"{fp['pred_area_in_band_gt']:.4f}", fp["n_gt_band"], fp["n_gt_total"]])

    counts = {c: sum(1 for f in fps if f["category"] == c) for c in ("duplicate", "fragment", "other_class", "no_gt")}
    imgs_with_fp = len({f["image"] for f in fps})
    imgs_with_band_gt = sum(1 for r in records.values() if BAND in r["gt_cls"])
    other_near = {}
    for f in fps:
        if f["category"] == "other_class":
            other_near[f["nearest_gt_class"]] = other_near.get(f["nearest_gt_class"], 0) + 1
    no_gt_imgs_without_band = sum(1 for f in fps if f["category"] == "no_gt" and f["n_gt_band"] == 0)
    conf_med = {c: (float(np.median([f["conf"] for f in fps if f["category"] == c])) if counts[c] else None)
                for c in counts}
    frag_cover_med = (float(np.median([f["pred_area_in_band_gt"] for f in fps if f["category"] == "fragment"]))
                      if counts["fragment"] else None)

    # session concentration (AIHub session = stem up to the first ")"); GT pipe_band per session for scale
    def session_of(p: str) -> str:
        return Path(p).stem.partition(")_")[0] + ")"

    fp_by_sess: dict[str, int] = {}
    for f in fps:
        fp_by_sess[session_of(f["image"])] = fp_by_sess.get(session_of(f["image"]), 0) + 1
    gt_by_sess: dict[str, int] = {}
    img_by_sess: dict[str, int] = {}
    for r in records.values():
        s = session_of(r["path"])
        gt_by_sess[s] = gt_by_sess.get(s, 0) + sum(1 for c in r["gt_cls"] if c == BAND)
        img_by_sess[s] = img_by_sess.get(s, 0) + 1
    # only sessions with a pipe_band GT or FP; the rest are summarised as a count
    sessions = {s: {"fp_band": fp_by_sess.get(s, 0), "gt_band": gt_by_sess[s], "images": img_by_sess[s]}
                for s in sorted(img_by_sess, key=lambda s: -fp_by_sess.get(s, 0))
                if fp_by_sess.get(s, 0) or gt_by_sess[s]}
    sessions["(other sessions)"] = {"fp_band": 0, "gt_band": 0, "count": len(img_by_sess) - len(sessions),
                                    "images": sum(v for s, v in img_by_sess.items() if s not in sessions)}

    # no_gt FPs lying at the same depth rows as a pipe_band GT in the image (vertical overlap > 0):
    # a band continuing outside the labelled box would look like this
    def same_rows(f: dict) -> bool:
        rec = records[f["image"]]
        y1, y2 = f["box"][1], f["box"][3]
        return any(c == BAND and min(y2, b[3]) - max(y1, b[1]) > 0 for c, b in zip(rec["gt_cls"], rec["gt_box"]))

    no_gt_same_rows = sum(1 for f in fps if f["category"] == "no_gt" and same_rows(f))

    n_total = len(images)
    g0 = gallery(records, fps, "duplicate", 16, fig_dir / "m17_pipe_band_fp_gallery_duplicate.png",
                 f"pipe_band 배경 오검출 — 중복(이미 다른 예측이 가져간 정답 pipe_band와 IoU≥0.45), "
                 f"{counts['duplicate']}개 · m17_aug0 last.pt, fold 0 검증 crop {n_total:,}장, conf 0.25")
    g1 = gallery(records, fps, "fragment", 16, fig_dir / "m17_pipe_band_fp_gallery_fragment.png",
                 f"pipe_band 배경 오검출 — 조각 띠 의심(정답 pipe_band와 겹치나 IoU<0.45), "
                 f"{counts['fragment']}개 · m17_aug0 last.pt, fold 0 검증 crop {n_total:,}장, conf 0.25")
    g2 = gallery(records, fps, "no_gt", 16, fig_dir / "m17_pipe_band_fp_gallery_no_gt.png",
                 f"pipe_band 배경 오검출 — 정답 없는 곳(어떤 정답 박스와도 겹치지 않음), "
                 f"{counts['no_gt']}개 · m17_aug0 last.pt, fold 0 검증 crop {n_total:,}장, conf 0.25")

    summary = {"status": "ok" if not diff.any() else "cm_mismatch", "n_images": n_total,
               "weights": str(run_dir / "weights" / "last.pt"), "bg_fp_band": int(cm[BAND, 4]),
               "bg_fp_band_ref": int(cm_ref[BAND, 4]), "cm": cm.tolist(), "cm_minus_ref": diff.tolist(),
               "val_cm_minus_ref": (cm_val - cm_ref).tolist(), "val_save_dir": str(metrics.save_dir),
               "categories": counts, "conf_median": conf_med, "fragment_pred_area_in_band_gt_median": frag_cover_med,
               "other_class_nearest": other_near, "no_gt_in_images_without_band_gt": no_gt_imgs_without_band,
               "images_with_band_fp": imgs_with_fp, "images_with_band_gt": imgs_with_band_gt,
               "no_gt_same_rows_as_band_gt": no_gt_same_rows, "sessions": sessions,
               "galleries": [g0, g1, g2], "csv": str(out_csv)}
    (root / "analysis" / "detection" / "m17_pipe_band_fp_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not diff.any() else 1


if __name__ == "__main__":
    sys.exit(main())
