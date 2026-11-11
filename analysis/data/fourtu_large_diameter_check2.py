"""Supplement: ratio distribution and unverifiable (order-mismatch) activities. Read-only."""
import pandas as pd

M = r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015\01_메타데이터"
gt = pd.read_csv(M + r"\utilities_groundtruth.csv", dtype=str, encoding="utf-8-sig")
ml = pd.read_csv(M + r"\utilities_metadata_list.csv", dtype=str, encoding="utf-8-sig")
flag = "정답표와 순서 일치"
ok = ml[ml[flag] == "예"].rename(columns={"seq_meta": "seq"})
m = gt.merge(ok[["activity_id", "seq", "diameter_mm"]], on=["activity_id", "seq"], how="inner")
m["depth"] = pd.to_numeric(m.depth_m, errors="coerce")
m["dia"] = pd.to_numeric(m.diameter_mm, errors="coerce")
m = m.dropna(subset=["depth", "dia"])
m["r"] = m.dia / 2000 / m.depth
print("n pairs", len(m))
print(m.r.describe().to_string())
print(m.sort_values("r", ascending=False).head(8)[["activity_id", "seq", "depth_m", "dia", "r"]].to_string())
print("dia>=800:", m[m.dia >= 800][["activity_id", "seq", "depth_m", "dia", "r"]].to_string())
# unverifiable activities
bad = ml[ml[flag] != "예"].activity_id.unique()
mm = ml[ml.activity_id.isin(bad)].copy()
mm["dia"] = pd.to_numeric(mm.diameter_mm, errors="coerce")
g = gt[gt.activity_id.isin(bad)].copy()
g["depth"] = pd.to_numeric(g.depth_m, errors="coerce")
rows = []
for a in bad:
    md = mm[mm.activity_id == a].dia.max()
    dg = g[g.activity_id == a].depth.min()
    if pd.notna(md) and pd.notna(dg) and md / 2000 >= dg:
        rows.append((a, md, dg))
print("unverifiable activities where max meta dia/2 >= min gt depth:", rows)
