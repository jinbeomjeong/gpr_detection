"""Leak-free grouped k-fold splits and the assertion that guards them (ADR-003).

A split group is the connected component of every key that must not cross a
fold boundary. For AIHub those keys are the XML ``<source>`` session, the
session derived from the member name and the gain-free crop key: the first two
disagree for 224 files, so grouping on ``<source>`` alone could still put one
name-derived session (or one crop's gain copies) into two folds.
"""

from collections.abc import Hashable, Iterable, Mapping, Sequence

from sklearn.model_selection import StratifiedGroupKFold

FOLD_COLUMN = "fold"
AIHUB_LEAK_KEY_COLUMNS = ("session", "session_from_name", "crop_key")
CROP_KEY_COLUMN = "crop_key"
GAIN_COLUMN = "gain"
FILE_NAME_COLUMN = "file_name"
DEFAULT_GAIN_PRIORITY = ("1.0", "0.5", "0.25")
"""Crop-level evaluation keeps the gain 1.0 copy, else 0.5, else 0.25 (M1-7)."""
_MAX_REPORTED_LEAKS = 5


class SplitLeakageError(ValueError):
    """Raised when a key that must stay in one fold appears in several folds."""


class _UnionFind:
    """Disjoint sets over hashable keys (path halving, union by size)."""

    def __init__(self) -> None:
        self._parent: dict[Hashable, Hashable] = {}
        self._size: dict[Hashable, int] = {}

    def find(self, key: Hashable) -> Hashable:
        if key not in self._parent:
            self._parent[key] = key
            self._size[key] = 1
            return key
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, first: Hashable, second: Hashable) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first == root_second:
            return
        if self._size[root_first] < self._size[root_second]:
            root_first, root_second = root_second, root_first
        self._parent[root_second] = root_first
        self._size[root_first] += self._size[root_second]


def connected_group_ids(keys_per_item: Sequence[Sequence[Hashable]]) -> list[int]:
    """Give items that share any key (directly or through a chain) one group id.

    Keys are compared by value, so prefix them by kind (e.g. ``("session", s)``)
    when different key kinds could hold equal strings.

    Args:
        keys_per_item: For each item, the keys that must stay in one fold.
            Every item needs at least one key.

    Returns:
        Group id per item, numbered 0, 1, ... in order of first appearance, so
        the result depends only on the input order.

    Raises:
        ValueError: If an item has no keys.
    """
    union_find = _UnionFind()
    for index, keys in enumerate(keys_per_item):
        if not keys:
            raise ValueError(f"Item {index} has no grouping keys")
        for key in keys[1:]:
            union_find.union(keys[0], key)

    group_id_by_root: dict[Hashable, int] = {}
    group_ids = []
    for keys in keys_per_item:
        root = union_find.find(keys[0])
        group_ids.append(group_id_by_root.setdefault(root, len(group_id_by_root)))
    return group_ids


def assign_group_folds(
    strata: Sequence[Hashable], groups: Sequence[Hashable], k: int, seed: int
) -> list[int]:
    """Assign each item a fold in ``0..k-1`` with ``StratifiedGroupKFold``.

    Every group lands in exactly one fold; the strata are balanced across folds
    as far as the groups allow.

    Args:
        strata: Stratification label per item.
        groups: Group id per item.
        k: Number of folds (at least 2).
        seed: ``random_state`` of the shuffled split, recorded with the result.

    Returns:
        Fold index per item, in input order.

    Raises:
        ValueError: If the lengths differ or there are fewer groups than folds.
    """
    if len(strata) != len(groups):
        raise ValueError(f"{len(strata)} strata but {len(groups)} groups")
    group_count = len(set(groups))
    if group_count < k:
        raise ValueError(f"Need at least k={k} groups for {k} folds, got {group_count}")

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [-1] * len(strata)
    # Only the group structure matters to the splitter; X is a placeholder.
    placeholder_features = [[0]] * len(strata)
    for fold, (_, eval_indices) in enumerate(
        splitter.split(placeholder_features, strata, groups)
    ):
        for index in eval_indices:
            folds[index] = fold
    return folds


def assert_no_fold_leakage(
    rows: Iterable[Mapping[str, object]],
    key_columns: Sequence[str] = AIHUB_LEAK_KEY_COLUMNS,
    fold_column: str = FOLD_COLUMN,
) -> None:
    """Check that no key value appears in more than one fold.

    Args:
        rows: Manifest rows (e.g. ``csv.DictReader`` rows or dicts).
        key_columns: Columns whose values must each stay in a single fold.
        fold_column: Column holding the fold index.

    Raises:
        SplitLeakageError: If any key value is found in two or more folds; the
            message lists up to five offending values per column.
        KeyError: If a row lacks one of the columns.
    """
    folds_by_value: dict[str, dict[object, set[object]]] = {
        column: {} for column in key_columns
    }
    for row in rows:
        fold = row[fold_column]
        for column in key_columns:
            folds_by_value[column].setdefault(row[column], set()).add(fold)

    problems = []
    for column, folds_of in folds_by_value.items():
        leaks = {value: folds for value, folds in folds_of.items() if len(folds) > 1}
        if not leaks:
            continue
        examples = sorted(leaks.items(), key=lambda item: str(item[0]))
        shown = [
            f"{value!r} in folds {sorted(folds, key=str)}"
            for value, folds in examples[:_MAX_REPORTED_LEAKS]
        ]
        problems.append(f"{column}: {len(leaks)} values span folds, e.g. {shown}")
    if problems:
        raise SplitLeakageError("Fold leakage found -> " + "; ".join(problems))


def select_crop_representatives(
    rows: Iterable[Mapping[str, object]],
    gain_priority: Sequence[str] = DEFAULT_GAIN_PRIORITY,
) -> list[Mapping[str, object]]:
    """Pick one gain copy per crop for crop-level evaluation.

    The gain copies of a crop are the same B-scan at different display gains,
    not independent samples, so crop-level metrics count each crop once
    (dataset_spec_m1 section 4 item 3).

    Args:
        rows: Manifest rows with ``crop_key``, ``gain`` and ``file_name``.
        gain_priority: Gains in order of preference. Gains are compared as
            numbers (``"1"`` matches ``"1.0"``). Rows with a gain outside this
            list are never chosen, so a crop with only such rows is left out.

    Returns:
        One row per crop: the copy with the most preferred gain, ties broken
        by the smaller ``file_name``. Sorted by ``file_name``, so the result
        does not depend on the input order.

    Raises:
        ValueError: If ``gain_priority`` is empty.
        KeyError: If a row lacks one of the columns.
    """
    if not gain_priority:
        raise ValueError("gain_priority must list at least one gain")
    rank_by_gain = {float(gain): rank for rank, gain in enumerate(gain_priority)}

    best_by_crop: dict[object, tuple[int, str, Mapping[str, object]]] = {}
    for row in rows:
        rank = rank_by_gain.get(float(str(row[GAIN_COLUMN])))
        if rank is None:
            continue
        candidate = (rank, str(row[FILE_NAME_COLUMN]), row)
        current = best_by_crop.get(row[CROP_KEY_COLUMN])
        if current is None or candidate[:2] < current[:2]:
            best_by_crop[row[CROP_KEY_COLUMN]] = candidate

    chosen = sorted(best_by_crop.values(), key=lambda candidate: candidate[1])
    return [row for _, _, row in chosen]
