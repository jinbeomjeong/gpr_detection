"""Read-only check: 4TU ground-truth rows where diameter/2 >= depth (centre assumption <= 0).

Pairing rule: only activities whose metadata flag 'order matches' == yes; seq == seq_meta.
Inputs are never modified. Prints tables to stdout.
"""
import pandas as pd

M = r"C:\workspace\GPR_Data\4TU_네덜란드_매설관GPR_가공본_261015\01_메타데이터"
gt = pd.read_csv(M + r"\utilities_groundtruth.csv", dtype=str, encoding="utf-8-sig")
ml = pd.read_csv(M + r"\utilities_metadata_list.csv", dtype=str, encoding="utf-8-sig")
flag = "정답표와 순서 일치"
print("gt rows", len(gt), "meta rows", len(ml))
print("flag values", ml[flag].value_counts().to_dict())
ok_acts = set(ml.loc[ml[flag] == "예", "activity_id"])
# an activity counts as order-matched if all its meta rows are yes
bad = ml[ml[flag] != "예"].activity_id.unique()
print("activities with non-yes flag:", len(bad))
ok = ml[ml[flag] == "예"].rename(columns={"seq_meta": "seq"})
m = gt.merge(ok[["activity_id", "seq", "material", "재질", "diameter_mm", "note"]],
             on=["activity_id", "seq"], how="left", suffixes=("", "_meta"))
m["depth"] = pd.to_numeric(m["depth_m"], errors="coerce")
m["dia"] = pd.to_numeric(m["diameter_mm"], errors="coerce")
print("gt rows with depth:", m.depth.notna().sum(), " with dia:", m.dia.notna().sum(),
      " both:", (m.depth.notna() & m.dia.notna()).sum())
m["center"] = m.depth - m.dia / 2000
hit = m[m.center <= 0]
cols = ["activity_id", "seq", "distance_m", "depth_m", "code", "관종", "material", "dia", "note", "note_meta", "center"]
print("\nROWS center<=0:\n", hit[cols].to_string())
print("\nnear (dia/2 >= 0.8*depth):\n", m[(m.dia / 2000 >= 0.8 * m.depth)][cols].to_string())
for a in ["08.4", "08.5"]:
    print("\nGT", a, "\n", m[m.activity_id == a][cols].to_string())
    print("META (all rows)", a, "\n", ml[ml.activity_id == a].to_string())
# rows with dia but not matched due to flag
nm = gt[gt.activity_id.isin(bad)]
print("\nrows in non-order-matched activities:", len(nm))
print(ml[ml[flag] != "예"][["activity_id", flag]].drop_duplicates().to_string())
# any 08.4/08.5 note text
print(gt[gt.note.notna()][["activity_id", "seq", "note"]].head(30).to_string())
print(ml[ml.note.notna()][["activity_id", "seq_meta", "note"]].to_string())
