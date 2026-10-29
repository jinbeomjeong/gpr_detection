"""AIHub 71327 (3D GPR) label audit. Read-only: reads zips with zipfile, no extraction.

Outputs go to data/audit/aihub_71327/ (git-ignored): tables (csv/json) and montage PNGs.
Single process. Seed fixed for montage sampling.
"""
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

SEED = 20261029
ROOT = Path(r"C:\workspace\GPR_Data\261.3D 지표투과레이더 탐사 데이터\01-1.정식개방데이터")
OUT = Path(r"C:\workspace\gpr_detection\data\audit\aihub_71327")
OUT.mkdir(parents=True, exist_ok=True)
MAX_PER_ZIP = None  # None = all (label zips are small); else uniform sample size

FIELDS = ["equipment", "channels", "gainvalue", "permittivity", "scanclass",
          "channellayer", "roadtype"]


def parse_xml(raw: bytes, split: str, zname: str):
    r = ET.fromstring(raw)
    g = lambda t: (r.findtext(t) or "").strip()
    rec = {"split": split, "zip": zname, "filename": g("filename"),
           "source": g("source"), "w": float(g("size/width") or 0),
           "h": float(g("size/height") or 0)}
    for f in FIELDS:
        rec[f] = g(f)
    objs = []
    for o in r.findall("object"):
        bb = o.find("bndbox")
        objs.append({"name": (o.findtext("name") or "").strip(),
                     "xmin": float(bb.findtext("xmin")), "ymin": float(bb.findtext("ymin")),
                     "xmax": float(bb.findtext("xmax")), "ymax": float(bb.findtext("ymax"))})
    return rec, objs


def main():
    rng = random.Random(SEED)
    files, objects, zip_log = [], [], []
    for split, prefix in (("Training", "TL_"), ("Validation", "VL_")):
        for zp in sorted((ROOT / split / "02.라벨링데이터").glob(f"{prefix}B scan*.zip")):
            with zipfile.ZipFile(zp) as z:
                xmls = sorted(n for n in z.namelist() if n.lower().endswith(".xml"))
                total = len(xmls)
                if MAX_PER_ZIP and total > MAX_PER_ZIP:
                    idx = np.linspace(0, total - 1, MAX_PER_ZIP).astype(int)
                    xmls = [xmls[i] for i in idx]
                for n in xmls:
                    rec, objs = parse_xml(z.read(n), split, zp.name)
                    rec["nobj"] = len(objs)
                    files.append(rec)
                    for o in objs:
                        o.update(split=split, zip=zp.name, filename=rec["filename"],
                                 source=rec["source"])
                        objects.append(o)
            zip_log.append({"split": split, "zip": zp.name, "xml_total": total,
                            "xml_used": len(xmls)})
            print(split, zp.name, total, len(xmls), flush=True)

    F = pd.DataFrame(files)
    O = pd.DataFrame(objects)
    Z = pd.DataFrame(zip_log)
    F.to_csv(OUT / "files.csv", index=False, encoding="utf-8-sig")
    O.to_csv(OUT / "objects.csv", index=False, encoding="utf-8-sig")
    Z.to_csv(OUT / "zips.csv", index=False, encoding="utf-8-sig")

    # distributions (file level) per split
    dist = {}
    for f in FIELDS:
        dist[f] = F.groupby(["split", f]).size().unstack(0, fill_value=0).reset_index()
        dist[f].to_csv(OUT / f"dist_{f}.csv", index=False, encoding="utf-8-sig")
    # class stats
    O["bw"] = O.xmax - O.xmin
    O["bh"] = O.ymax - O.ymin
    O["ar"] = O.bw / O.bh.clip(lower=1e-6)
    cls = O.groupby(["name", "split"]).agg(objects=("name", "size"), sessions=("source", "nunique"),
                                           files=("filename", "nunique"),
                                           bw_med=("bw", "median"), bh_med=("bh", "median"),
                                           ar_med=("ar", "median"),
                                           ar_p10=("ar", lambda s: s.quantile(.1)),
                                           ar_p90=("ar", lambda s: s.quantile(.9))).reset_index()
    cls_all = O.groupby("name").agg(objects=("name", "size"), sessions=("source", "nunique"),
                                    files=("filename", "nunique")).reset_index()
    cls.to_csv(OUT / "class_by_split.csv", index=False, encoding="utf-8-sig")
    cls_all.to_csv(OUT / "class_all.csv", index=False, encoding="utf-8-sig")

    # session overlap
    s_tr = set(F[F.split == "Training"].source)
    s_va = set(F[F.split == "Validation"].source)
    overlap = s_tr & s_va
    ov = {"train_sessions": len(s_tr), "val_sessions": len(s_va), "overlap": len(overlap),
          "val_sessions_in_train_pct": round(100 * len(overlap) / max(1, len(s_va)), 1)}
    ov_cls = {}
    for c, sub in O.groupby("name"):
        a = set(sub[sub.split == "Training"].source)
        b = set(sub[sub.split == "Validation"].source)
        ov_cls[c] = {"train": len(a), "val": len(b), "overlap": len(a & b)}
    # files in overlapping sessions
    ov["val_files_in_overlap_sessions"] = int(F[(F.split == "Validation") & F.source.isin(overlap)].shape[0])
    ov["val_files"] = int((F.split == "Validation").sum())
    ov["train_files_in_overlap_sessions"] = int(F[(F.split == "Training") & F.source.isin(overlap)].shape[0])
    ov["train_files"] = int((F.split == "Training").sum())
    # filename overlap (exact duplicates across split)
    ov["filename_overlap"] = len(set(F[F.split == "Training"].filename) & set(F[F.split == "Validation"].filename))
    # crop (gain-stripped) overlap
    crop = lambda s: re.sub(r"_g[0-9.]+_", "_", s)
    F["crop"] = F.filename.map(crop)
    ov["crop_overlap"] = len(set(F[F.split == "Training"].crop) & set(F[F.split == "Validation"].crop))
    ov["unique_crops"] = int(F.crop.nunique())
    json.dump({"overlap": ov, "overlap_by_class": ov_cls}, open(OUT / "overlap.json", "w"), indent=1)

    # image-label pairing + montage
    pair = []
    montage_src = {}
    for _, r in Z.iterrows():
        sz = ROOT / r.split / "01.원천데이터" / r.zip.replace("TL_", "TS_").replace("VL_", "VS_")
        with zipfile.ZipFile(sz) as z:
            imgs = {Path(n).stem for n in z.namelist() if n.lower().endswith(".jpg")}
        labs = set(F[F.zip == r.zip].filename)
        pair.append({"zip": r.zip, "images": len(imgs), "labels_used": len(labs),
                     "labels_without_image": len(labs - imgs),
                     "images_without_label": len(imgs - labs) if r.xml_used == r.xml_total else None})
    pd.DataFrame(pair).to_csv(OUT / "pairing.csv", index=False, encoding="utf-8-sig")

    for cname, zkey in (("lpipe", "종배관"), ("tpipe", "횡배관")):
        sub = O[(O.name == cname)]
        sub = sub[sub.zip.str.contains(zkey)]
        # one object per file, distinct sessions where possible
        fl = sub.drop_duplicates("filename")
        # spread over splits: sample 12 files preferring distinct sessions
        by_src = defaultdict(list)
        for i, r in fl.iterrows():
            by_src[r.source].append(i)
        srcs = sorted(by_src)
        rng.shuffle(srcs)
        pick = []
        k = 0
        while len(pick) < 12 and srcs:
            s = srcs[k % len(srcs)]
            if by_src[s]:
                pick.append(by_src[s].pop(rng.randrange(len(by_src[s]))))
            k += 1
            if k > 5000:
                break
        tiles = []
        for i in pick:
            r = fl.loc[i]
            sz = ROOT / r.split / "01.원천데이터" / r.zip.replace("TL_", "TS_").replace("VL_", "VS_")
            with zipfile.ZipFile(sz) as z:
                data = z.read("/" + r.filename + ".jpg")
            im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            for _, o in sub[sub.filename == r.filename].iterrows():
                cv2.rectangle(im, (int(o.xmin), int(o.ymin)), (int(o.xmax), int(o.ymax)), (0, 0, 255), 1)
            im = cv2.resize(im, (384, 384), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, f"{r.split[0]} {r.filename[:2]} {r.filename.split('_')[1]}", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(im)
        rows = [np.hstack(tiles[j:j + 4]) for j in range(0, 12, 4)]
        cv2.imwrite(str(OUT / f"montage_{cname}.png"), np.vstack(rows))
        # also the raw (no box) side-by-side for first 4 for visual comparison
        montage_src[cname] = [(fl.loc[i].filename, fl.loc[i].source, fl.loc[i].split) for i in pick]
        print(cname, "picked", len(pick), flush=True)
    json.dump(montage_src, open(OUT / "montage_picks.json", "w"), ensure_ascii=False, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
