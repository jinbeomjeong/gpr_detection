"""Validate labeling/m34_gprsd/label_config.xml against Label Studio SDK parsing (run in env gpr-label).

conda run -n gpr-label python -I analysis/data/gprsd_m34_check_label_config.py
Checks: XML parses, tag names/classes as expected, every task json has the fields the header uses and no predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    cfg_path = REPO / "labeling" / "m34_gprsd" / "label_config.xml"
    xml = cfg_path.read_text(encoding="utf-8")
    problems: list[str] = []
    try:
        from label_studio_sdk.label_interface import LabelInterface

        li = LabelInterface(xml)
        names = sorted(li.controls and [c.name for c in li.controls])
        print("controls:", names)
        expected = {"label", "apex", "certainty", "flag", "window_complete", "note"}
        if set(names) != expected:
            problems.append(f"controls {set(names)} != {expected}")
        lab = li.get_control("label")
        labels = list(lab.labels) if hasattr(lab, "labels") else []
        print("label values:", labels)
        if labels != ["pipe_hyperbola", "pipe_band", "ignore"]:
            problems.append(f"label order {labels}")
        task_dir = REPO / "data" / "labels" / "gprsd_m34"
        for fn in ("tasks_main.json", "tasks_relabel.json"):
            tasks = json.loads((task_dir / fn).read_text(encoding="utf-8"))
            for t in tasks:
                if "predictions" in t:
                    problems.append(f"{fn}: predictions present")
                for k in ("order", "total", "file_id", "image", "window_id"):
                    if k not in t["data"]:
                        problems.append(f"{fn}: missing {k}")
            print(fn, len(tasks), "tasks")
    except Exception as e:  # noqa: BLE001
        problems.append(f"{type(e).__name__}: {e}")
    print(json.dumps({"status": "ok" if not problems else "fail", "problems": problems}, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
