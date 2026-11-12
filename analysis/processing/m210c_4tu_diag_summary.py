# ruff: noqa: RUF001
# The md text is Korean prose with the spec's symbols (Greek sigma, en dash).
"""Summary of the M2-10 (c) 4TU line diagnostics (m210_c_qc.md v1.2 section 7).

Reads the ``eval_depth_4tu.py --band-shift-qc --flag-header-eps --diag-out``
csv and writes ``docs/processing/m210_c_4tu_diag.md``: per site the P3 flag
counts, the ``sigma_s`` distribution, the overlap with ``zero_window_drift``,
the P5 flag counts (lines and activities) and the reference depth std
``(v / 2) 2.5 sigma_s``. Already-seen data, exploration only, no verdict; the
factor 2.5 is the 2-D synthetic reference of section 1-1 and is never used for a
depth, a rule or a judgment.

Run:
    conda run -n gpr-dev python -I analysis/processing/m210c_4tu_diag_summary.py
"""

import argparse
import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    PROJECT_ROOT / "data" / "processing" / "m210c_4tu_diag" / "m210c_4tu_line_diag.csv"
)
DEFAULT_MD = PROJECT_ROOT / "docs" / "processing" / "m210_c_4tu_diag.md"
SPEED_OF_LIGHT_M_PER_NS = 0.299792458
P3_FLAGS = ("zero_shift_large", "zero_tracks_surface", "zero_shift_not_surface")
P5_FLAG = "eps_meta_equals_header_default"
P5_PREDICTED_ACTIVITIES = 51
"""Design-scan prediction (section 3): activities with metadata eps_r 9.0."""
QUANTILES = (0.1, 0.5, 0.9)
DEPTH_REFERENCE_COLUMN = "2.5·σ_s 환산(해석 불가)"
"""The former reference depth std column; not interpretable (spec v1.2 1-1)."""
TWO_EVENT_RULE = "air_first_event"
SINGLE_BAND_RULE = "air_single_event"


def _is_true(value: str) -> bool:
    return value == "True"


def _quantiles(values: Sequence[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    picks = []
    for q in QUANTILES:
        position = q * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        value = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
        picks.append(f"{value:.3f}")
    return " / ".join(picks)


def _site_key(site: str) -> tuple[int, str]:
    return (int(site), site) if site.isdigit() else (10**6, site)


def site_rows(lines: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """One summary row per site, then the total."""
    by_site: dict[str, list[Mapping[str, str]]] = {}
    for line in lines:
        by_site.setdefault(line["site"], []).append(line)
    groups = [(site, by_site[site]) for site in sorted(by_site, key=_site_key)]
    groups.append(("전체", list(lines)))
    rows = []
    for site, members in groups:
        sigma = [
            float(m["band_shift_std_ns"]) for m in members if m["band_shift_std_ns"]
        ]
        depth_std = [
            SPEED_OF_LIGHT_M_PER_NS / math.sqrt(float(m["eps_r_meta"]))
            / 2.0 * float(m["pipe_time_std_ref_ns"]) * 100.0
            for m in members
            if m["pipe_time_std_ref_ns"] and m["eps_r_meta"]
        ]  # fmt: skip
        drift = [m for m in members if _is_true(m["zero_window_drift"])]
        p5_lines = [m for m in members if _is_true(m[P5_FLAG])]
        rows.append(
            {
                "현장": site,
                "측선": str(len(members)),
                **{
                    flag: str(sum(_is_true(m[flag]) for m in members))
                    for flag in P3_FLAGS
                },
                "sigma_s ns (10/50/90 %)": _quantiles(sigma),
                "zero_window_drift": str(len(drift)),
                "drift ∩ shift_large": str(
                    sum(_is_true(m["zero_shift_large"]) for m in drift)
                ),
                "P5 측선": str(len(p5_lines)),
                "P5 활동": str(len({m["activity_id"] for m in p5_lines})),
                DEPTH_REFERENCE_COLUMN: _quantiles(depth_std),
            }
        )
    return rows


def _median_sigma(lines: Sequence[Mapping[str, str]]) -> tuple[float | None, int]:
    values = sorted(
        float(m["band_shift_std_ns"]) for m in lines if m["band_shift_std_ns"]
    )
    if not values:
        return None, 0
    middle = len(values) // 2
    median = (
        values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    )
    return median, len(values)


def zero_rule_lines(lines: Sequence[Mapping[str, str]]) -> list[str]:
    """``sigma_s`` median of the two-event and single-band lines, all and site 9."""
    text = []
    for label, scope in (
        ("전체", lines),
        ("현장 9", [m for m in lines if m["site"] == "9"]),
    ):
        parts = []
        for rule, name in ((TWO_EVENT_RULE, "두 사건"), (SINGLE_BAND_RULE, "단일 띠")):
            median, count = _median_sigma([m for m in scope if m["zero_rule"] == rule])
            value = "-" if median is None else f"{median:.2f} ns"
            parts.append(f"{name}(`{rule}`) {value}({count}측선)")
        text.append(f"- {label}: sigma_s 중앙 " + " 대 ".join(parts))
    return text


def _table(rows: Sequence[Mapping[str, str]]) -> list[str]:
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(row[c] for c in columns) + " |" for row in rows]
    return lines


def write_markdown(csv_path: Path, md_path: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        lines = list(csv.DictReader(handle))
    rows = site_rows(lines)
    total = rows[-1]
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    relative = csv_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    text = [
        "# M2-10 (c) 4TU 영점 진단 (P3·P5 플래그)",
        "",
        "> **이미 본 자료·탐색 보고, 판정 아님.** 판정률·MAE는 다시 계산하지 않았다"
        "(깊이 불변). 이 결과로 P3·P5 규칙이나 문턱을 바꾸지 않는다.",
        "",
        f"- 입력: `{relative}` (SHA-256 `{digest[:12]}…`, 측선 {len(lines)}개)",
        "- 생성: `scripts/eval_depth_4tu.py --band-shift-qc --flag-header-eps "
        "--diag-out data/processing/m210c_4tu_diag --prefix m210c_4tu` "
        "(`MKL_CBWR=COMPATIBLE`) → `analysis/processing/m210c_4tu_diag_summary.py`",
        "- 명세: [`m210_c_qc.md`](m210_c_qc.md) v1.2 §7",
        f"- 참고 깊이 std 열(`{DEPTH_REFERENCE_COLUMN}`)은 해석에 쓰지 않는다. "
        "2.5는 띠 포락선 피크 이동(합성에서 기하의 36–41 %) 기준 배율인데 "
        "σ_s는 띠 상관 지연(합성에서 기하의 11–12 %)이라 정의가 다르다. "
        "이 열은 관 시각 변화의 크기를 나타내지 않는다(명세 v1.2 §1-1).",
        "",
        "## 현장별",
        "",
        *_table(rows),
        "",
        f"- `zero_tracks_surface`는 C1 > C0 기준이 정의상 거의 늘 참이라 "
        f"`zero_shift_large`와 같다({total['zero_tracks_surface']} = "
        f"{total['zero_shift_large']}, not_surface "
        f"{total['zero_shift_not_surface']}). '띠가 흔들림(σ_s > 0.2 ns)'만 뜻하고 "
        "지표 추종의 확인이 아니다(명세 v1.2 §1-3).",
        "",
        "## 영점 규칙별 σ_s",
        "",
        *zero_rule_lines(lines),
        "",
        "## P5 예측 대조",
        "",
        f"- `{P5_FLAG}` 활동 {total['P5 활동']}개(측선 {total['P5 측선']}개). "
        f"설계 스캔 예측은 메타 εr 9.0 활동 {P5_PREDICTED_ACTIVITIES}개 근처다(§3).",
        "",
    ]
    md_path.write_text("\n".join(text), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_markdown(args.csv, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
