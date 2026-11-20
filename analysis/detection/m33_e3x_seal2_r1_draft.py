"""Seal 2 r1 record of M3-3 E3x (plan v0.5 A-1): write or check the draft.

Spec: ``docs/processing/m33_e3x_plan.md`` v0.5 appendix A-1 and
``docs/processing/m33_e3x_texture_defect_review.md`` §10 H4.

The record ``analysis/detection/m33_e3x_seal2_r1.json`` keeps every key of
``m33_e3x_seal2.json`` (seal 2, commit 729d19c, left unchanged) with the new
code blobs, and adds:

* ``supersedes``: path, sha256 (LF) and commit of the seal 2 record;
* ``plan_code_blobs``: the seal 2 ``code_blobs`` (the code that made the
  plan tables, which seal 2 r1 reuses without a new plan run, v5-5);
* ``plan_reuse``: every top-level definition of ``m33_e3x.py`` and
  ``m33_e3x_place.py`` whose source differs from the seal 2 blob, where each
  is referenced (line, enclosing definition), and whether it is reachable
  from the ``--stage plan`` path in a static name-reference graph. A
  reachable change needs a written note (``REACHABLE_NOTES``); without one
  the tool stops with exit 1.

The pool, plan-table and plan-report sha, ``plan_head``, ``plan_run_id``,
``seeds`` and ``src_tree_id`` are copied from seal 2 after asserting that
the files and the ``src`` tree still have those values (the plan tables
are read, never written; no result folder is opened).

Exit codes

====  ==========  =========================================================
exit  status      meaning
====  ==========  =========================================================
0     ``ok``      draft written (or ``--check``: the file equals a rebuild)
1     ``differs`` an input differs from seal 2, a reachable change has no
                  note, or ``--check`` found another record
2     ``input``   output exists (never overwritten), seal 2 record missing
3     ``error``   internal error (git failure included)
====  ==========  =========================================================

Stdout: one JSON line ``{"status", "out", "sha256_lf", "n_changed",
"n_reachable_changed"}``.
"""

import argparse
import ast
import json
import subprocess
import sys
from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m32_recipe_inputs as inputs
import m33_e3x as e3x
import m33_inputs

REPO = e3x.REPO
SEAL2 = REPO / "analysis" / "detection" / "m33_e3x_seal2.json"
SEAL2_COMMIT = "729d19ced8674039efac0cfe906fcb41f45c31e3"
DEFAULT_OUT = e3x.SEAL2_RECORD
SCHEMA = "m33_e3x_seal2/2"
GIT_TIMEOUT_S = 60
MODULES = {
    "m33_e3x": "analysis/detection/m33_e3x.py",
    "m33_e3x_place": "analysis/detection/m33_e3x_place.py",
}
MODULE_ALIASES = {"place": "m33_e3x_place"}
"""``import m33_e3x_place as place`` in the runner."""
PLAN_ROOTS = (
    "m33_e3x:main",
    "m33_e3x:parse_args",
    "m33_e3x:run",
    "m33_e3x:start_context",
    "m33_e3x:run_plan",
    "m33_e3x:print_final_line",
)
"""``python m33_e3x.py --stage plan``: main -> parse_args -> run -> start_context
-> ``STAGE_RUNNERS[Stage.PLAN]`` (= run_plan) -> print_final_line."""
DISPATCH_EDGES = {"m33_e3x:run": ("m33_e3x:start_context", "m33_e3x:run_plan")}
"""``run`` dispatches by stage; for the plan stage only these two are called."""
REACHABLE_NOTES = {
    "m33_e3x:DEFAULT_PRECHECK_DIR": (
        "argparse default of --precheck-dir only (parse_args); the plan stage "
        "never reads settings.precheck_dir: Settings.stage_dir returns "
        "out_root/plan when precheck is False"
    ),
    "m33_e3x:SEAL2_RECORD": (
        "argparse default of --seal-record only (parse_args); the plan stage "
        "never reads settings.seal_record: tracked_inputs adds it only when "
        "settings.precheck or the stage is not plan"
    ),
}
"""Changed names the plan path can reach, with why the plan tables do not change."""

Definitions = dict[str, ast.AST]


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=GIT_TIMEOUT_S,
        check=True,
    )
    return completed.stdout


def top_level_definitions(source: str) -> dict[str, tuple[ast.AST, str]]:
    """Top-level function, class and assignment nodes with their source text."""
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        names = []
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        segment = ast.get_source_segment(source, node) or ""
        for name in names:
            found[name] = (node, segment)
    return found


def changed_names(old: Mapping, new: Mapping) -> list[str]:
    """Names added, removed or whose source segment differs."""
    names = set(old) | set(new)
    return sorted(
        name
        for name in names
        if name not in old or name not in new or old[name][1] != new[name][1]
    )


def _referenced(node: ast.AST, module: str, defined: Mapping[str, set]) -> set:
    """Graph nodes a definition refers to (own-module names, ``place.x``)."""
    found = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in defined[module]:
            found.add(f"{module}:{child.id}")
        if (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id in MODULE_ALIASES
        ):
            target = MODULE_ALIASES[child.value.id]
            if child.attr in defined[target]:
                found.add(f"{target}:{child.attr}")
    return found


def reference_graph(definitions: Mapping[str, Mapping]) -> dict[str, set[str]]:
    """``module:name -> referenced module:name`` over both modules."""
    defined = {module: set(names) for module, names in definitions.items()}
    graph = {}
    for module, names in definitions.items():
        for name, (node, _segment) in names.items():
            key = f"{module}:{name}"
            graph[key] = _referenced(node, module, defined) - {key}
    for key, targets in DISPATCH_EDGES.items():
        graph[key] = set(targets)
    return graph


def reachable(graph: Mapping[str, set[str]], roots: Iterable[str]) -> set[str]:
    seen, queue = set(roots), deque(roots)
    while queue:
        for target in graph.get(queue.popleft(), ()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def references(source: str, name: str, module_alias: str | None) -> list[dict]:
    """Lines (and enclosing top-level definition) that refer to ``name``."""
    tree = ast.parse(source)
    found = []
    for node in tree.body:
        owner = getattr(node, "name", None)
        for child in ast.walk(node):
            is_name = isinstance(child, ast.Name) and child.id == name
            is_attribute = (
                module_alias is not None
                and isinstance(child, ast.Attribute)
                and child.attr == name
                and isinstance(child.value, ast.Name)
                and child.value.id == module_alias
            )
            if (is_name or is_attribute) and owner != name:
                found.append({"line": child.lineno, "in": owner})
    return sorted(found, key=lambda item: item["line"])


def plan_reuse(old_blobs: Mapping[str, str]) -> dict:
    """Changed definitions since seal 2 and their reach from the plan path."""
    new_sources = {
        module: (REPO / path).read_text(encoding="utf-8")
        for module, path in MODULES.items()
    }
    old_sources = {
        module: _git("cat-file", "blob", old_blobs[path])
        for module, path in MODULES.items()
    }
    new_defs = {m: top_level_definitions(s) for m, s in new_sources.items()}
    old_defs = {m: top_level_definitions(s) for m, s in old_sources.items()}
    reach = reachable(reference_graph(new_defs), PLAN_ROOTS)
    changed = []
    for module in MODULES:
        for name in changed_names(old_defs[module], new_defs[module]):
            key = f"{module}:{name}"
            entry = {
                "name": key,
                "status": _change_status(name, old_defs[module], new_defs[module]),
                "reachable_from_plan": key in reach,
                "references": _module_references(new_sources, module, name),
            }
            if key in REACHABLE_NOTES:
                entry["note"] = REACHABLE_NOTES[key]
            changed.append(entry)
    return {
        "method": (
            "top-level definitions whose ast source segment differs from the seal 2 "
            "blob; reachability over a static graph of the names each definition "
            "refers to (own module, place.<name>), annotations included"
        ),
        "plan_roots": list(PLAN_ROOTS),
        "dispatch_edges": {k: list(v) for k, v in DISPATCH_EDGES.items()},
        "old_blobs": {path: old_blobs[path] for path in MODULES.values()},
        "changed": changed,
        "reachable_changed": [c["name"] for c in changed if c["reachable_from_plan"]],
        "n_plan_path_definitions": len(reach),
    }


def _change_status(name: str, old: Mapping, new: Mapping) -> str:
    if name not in old:
        return "added"
    if name not in new:
        return "removed"
    return "changed"


def _module_references(sources: Mapping[str, str], module: str, name: str) -> dict:
    found = {MODULES[module]: references(sources[module], name, None)}
    if module == "m33_e3x_place":
        found[MODULES["m33_e3x"]] = references(sources["m33_e3x"], name, "place")
    return found


def _assert_seal2_inputs(seal2: Mapping) -> list[str]:
    """Inputs seal 2 r1 copies must still have their seal 2 values."""
    problems = []
    if inputs.sha256_file_lf(e3x.pool.DEFAULT_OUT) != seal2["pool_csv_sha256_lf"]:
        problems.append("pool_csv")
    if e3x.plan_file_hashes(e3x.SEALED_PLAN_DIR) != seal2["plan_files_sha256_lf"]:
        problems.append("plan_files")
    report = e3x.SEALED_PLAN_DIR / "plan_report.json"
    if inputs.sha256_file_lf(report) != seal2["plan_report_sha256_lf"]:
        problems.append("plan_report")
    if _git("rev-parse", "HEAD:src").strip() != seal2["src_tree_id"]:
        problems.append("src_tree_id")
    return problems


def build_record() -> tuple[dict, list[str]]:
    """The seal 2 r1 record and the problems that stop it."""
    seal2 = json.loads(SEAL2.read_text(encoding="utf-8"))
    problems = _assert_seal2_inputs(seal2)
    reuse = plan_reuse(seal2["code_blobs"])
    problems += [
        f"reachable change without a note: {name}"
        for name in reuse["reachable_changed"]
        if name not in REACHABLE_NOTES
    ]
    blobs = m33_inputs.git_blobs(e3x.code_paths())
    if None in blobs.values():
        raise RuntimeError("git hash-object failed")
    record = {
        "schema": SCHEMA,
        "pool_csv_sha256_lf": seal2["pool_csv_sha256_lf"],
        "plan_files_sha256_lf": seal2["plan_files_sha256_lf"],
        "plan_report_sha256_lf": seal2["plan_report_sha256_lf"],
        "code_blobs": blobs,
        "src_tree_id": seal2["src_tree_id"],
        "plan_head": seal2["plan_head"],
        "plan_run_id": seal2["plan_run_id"],
        "seeds": seal2["seeds"],
        "supersedes": {
            "path": SEAL2.relative_to(REPO).as_posix(),
            "sha256_lf": inputs.sha256_file_lf(SEAL2),
            "commit": SEAL2_COMMIT,
        },
        "plan_code_blobs": seal2["code_blobs"],
        "plan_reuse": reuse,
    }
    return record, problems


def _record_text(record: Mapping) -> str:
    return json.dumps(record, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns the exit code (module docstring table)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="compare, never write")
    args = parser.parse_args(argv)
    status, digest, n_changed, n_reachable = "error", None, 0, 0
    try:
        if not SEAL2.is_file():
            raise inputs.InputError(f"seal 2 record missing: {SEAL2}")
        if not args.check and args.out.exists():
            raise inputs.InputError(f"output exists: {args.out}")
        record, problems = build_record()
        n_changed = len(record["plan_reuse"]["changed"])
        n_reachable = len(record["plan_reuse"]["reachable_changed"])
        text = _record_text(record)
        if args.check:
            same = args.out.is_file() and args.out.read_text(encoding="utf-8") == text
            status = "ok" if same and not problems else "differs"
        elif problems:
            status = "differs"
        else:
            with args.out.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            status = "ok"
        for problem in problems:
            print(f"problem: {problem}", file=sys.stderr)
        if args.out.is_file():
            digest = inputs.sha256_file_lf(args.out)
    except inputs.InputError as error:
        print(f"input error: {error}", file=sys.stderr)
        status = "input"
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"internal error: {error!r}", file=sys.stderr)
    final = {
        "status": status,
        "out": str(args.out),
        "sha256_lf": digest,
        "n_changed": n_changed,
        "n_reachable_changed": n_reachable,
    }
    print(json.dumps(final), flush=True)
    return {"ok": 0, "differs": 1, "input": 2}.get(status, 3)


if __name__ == "__main__":
    sys.exit(main())
