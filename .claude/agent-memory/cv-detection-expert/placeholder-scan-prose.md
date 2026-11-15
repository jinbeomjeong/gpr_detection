---
name: placeholder-scan-prose
description: verif placeholder_scan matches brace-wrapped uppercase names even in explanatory prose/backticks; write slot names without braces when describing them
metadata:
  type: feedback
---

`placeholder_scan.py` (pattern `\{[A-Z][A-Z0-9_]*\}`) flags every brace-wrapped name, including ones inside backticks in sentences that merely *describe* a slot (e.g. J1 prereg :5-6 "`{C0B}` 자리는 ... 채운다"). After the real slots are filled, the descriptive mention still fails the scan.

**Why:** found 2026-11-15 while checking M3-3 J1 C1 readiness — the plan doc deliberately writes "자리 C2·R2_J1_POST_SHA" without braces, J1 did not.

**How to apply:** in prereg/spec prose, name slots without braces (e.g. "자리 C0B"); keep braces only at the actual fill sites. When auditing sealing readiness, run the scan and check that each hit is a fill site, not prose.
