"""M3-2 stage C pre-check (plan v0.2.3 §3-2, addendum 8): ad-hoc, not a judgement.

Current recipe r0, fold 0, k 1.3, in-distribution positions: recall at each SNR
step (0/6/12/18 dB), pooled over both polarities, plus the C-1 cell-weighted
form (IDS/A0 1/2, 4TU/A0 1/4, 4TU/A1 1/4, polarity-averaged, unstratified).
Pass: recall(12 dB) >= 50 % and no later SNR step more than 1 %p below an
earlier one. Writes JSON to the path given as the only argument; stdout
carries only the status line (blind rule).
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATCHES = REPO / "runs" / "eval" / "m32_recipe" / "injected" / "r0" / "fold0" / "inj_matches.csv"
SNRS = (0.0, 6.0, 12.0, 18.0)
CELL_WEIGHTS = {("gz_ids", "A0"): 0.5, ("4tu", "A0"): 0.25, ("4tu", "A1"): 0.25}


def main(out_path: str) -> int:
    pooled = defaultdict(lambda: [0, 0])
    cells = defaultdict(lambda: [0, 0])
    with MATCHES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["recipe"] != "r0" or row["fold"] != "0" or float(row["k"]) != 1.3:
                continue
            if row["in_distribution"] != "True":
                continue
            snr = float(row["snr_db"])
            hit = row["hit"] == "True"
            pooled[snr][0] += hit
            pooled[snr][1] += 1
            key = (snr, row["dataset"], row["model_group"], int(row["polarity"]))
            cells[key][0] += hit
            cells[key][1] += 1
    recall = {snr: 100.0 * pooled[snr][0] / pooled[snr][1] for snr in SNRS}
    weighted = {}
    for snr in SNRS:
        total = 0.0
        for (dataset, group), weight in CELL_WEIGHTS.items():
            pol = [cells[(snr, dataset, group, p)] for p in (1, -1)]
            total += weight * sum(100.0 * h / n for h, n in pol) / 2
        weighted[snr] = total
    def monotone(values):
        seq = [values[s] for s in SNRS]
        return all(seq[j] >= seq[i] - 1.0 for i in range(len(seq)) for j in range(i + 1, len(seq)))
    result = {
        "source": str(MATCHES.relative_to(REPO)),
        "n_per_snr": {str(s): pooled[s][1] for s in SNRS},
        "recall_pooled_pct": {str(s): round(recall[s], 2) for s in SNRS},
        "recall_cell_weighted_pct": {str(s): round(weighted[s], 2) for s in SNRS},
        "cell_n": {f"{k[0]}|{k[1]}/{k[2]}|{k[3]:+d}": v[1] for k, v in sorted(cells.items())},
        "pass_recall12_pooled": recall[12.0] >= 50.0,
        "pass_recall12_weighted": weighted[12.0] >= 50.0,
        "pass_monotone_pooled": monotone(recall),
        "pass_monotone_weighted": monotone(weighted),
    }
    result["pass"] = all(result[k] for k in result if k.startswith("pass_"))
    Path(out_path).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": out_path}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
