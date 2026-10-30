"""Provenance of training and evaluation runs: git state, versions, file hashes.

Written into ``gpr_train_settings.yaml`` and the evaluation ``metrics.json`` so a
number can be traced to the code, libraries and weights that produced it.
Package versions are read from the installed metadata, so asking for
``torch`` or ``ultralytics`` never imports them.
"""

import hashlib
import subprocess
from collections.abc import Iterable
from datetime import datetime
from importlib import metadata
from pathlib import Path

_HASH_CHUNK_BYTES = 1 << 20
_GIT_TIMEOUT_S = 30


def _run_git(repo_dir: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state(repo_dir: Path) -> dict[str, str | bool | None]:
    """Return the checked-out commit and whether tracked files have local edits.

    Args:
        repo_dir: Any folder inside the repository.

    Returns:
        ``{"commit": <hash or None>, "dirty": <bool or None>}``; ``None`` when git
        is unavailable or ``repo_dir`` is not in a repository. Untracked files do
        not count as dirty.
    """
    commit = _run_git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    changes = _run_git(repo_dir, "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if changes is None else changes != ""}


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    """Return the installed version of each distribution (``None`` if missing)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def local_timestamp() -> str:
    """Return the current local time as ISO 8601 with offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
