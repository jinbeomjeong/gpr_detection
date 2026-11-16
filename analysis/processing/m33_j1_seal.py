"""M3-3 J1 sealed values that ``m33_j1_judge.py`` asserts (C2 commit).

Spec: ``docs/processing/m33_j1_preregistration.md`` v0.3.4 §1-2 (fixed code
and inputs) and §5 J1-2 assertions 1-3.

The values live in this separate module so that the judge can assert its own
blob and the candidate module's blob: a file cannot hold its own blob. The C2
commit fills every ``None`` below with ``m33_j1_judge.py --measure-seal``; the
judge refuses to run (exit 2) while any of them is still ``None``. The commit
that last changed this file is taken as the C2 commit (assertion 1: HEAD is
C2 or a later docs-only commit).

Values that are already sealed elsewhere (C1, [U3]) are filled here and
never re-measured.
"""

C1_SEAL_COMMIT = "68f4719db43d2d9c2ec45e10d8232ddd39597f44"
"""C1 seal commit (2026-11-16); the R2 run must start at it or after it."""

FOLD0_WEIGHTS_SHA256 = (
    "7ead71c7fa62d85edb11f5614e78b11619062b455dca62f5cc66c93b2dcf8d92"
)
"""N2-a' fold 0 ``last.pt`` sha256 (J1-2 assertion 3)."""

SIGNAL_CANDIDATES_SHA256_LF = (
    "b348a2ffe9fe11f139be4fa8512c68a6eba449c37c99ba284f2154144968d43f"
)
"""LF sha of ``fourtu_apex_candidates_v1.csv`` ([U3] §1-2), report ``signal``."""

SEALED_BLOBS: dict[str, str | None] = {
    "scripts/eval_depth_4tu.py": "9289f6df5a96645305e469ebd18e1ddf9138bd2a",
    "analysis/processing/m33_j1_candidates.py": (
        "8c9b4c2f201c9389452dbdf1840ade53728f47c8"
    ),
    "analysis/processing/m33_j1_judge.py": "e71a2f943777a552ba3618eaba2dd87149442a9a",
    "docs/processing/fourtu_lines_v1.csv": "fbc6da72ab89e48b7463dd0becdfe3e38a7b40d7",
    "docs/datasets/fourtu_duplicate_radargrams.csv": (
        "b6516bbcf8f98b33256067528d4222d8efd5a0a6"
    ),
    "docs/processing/m29c_qc_constants.json": (
        "13f2fd3ac76d6996ac9c590a2cadcd988d34a376"
    ),
}
"""``git rev-parse HEAD:<path>`` at C2 (§1-2 git blob column).

Measured with ``--measure-seal`` (``git hash-object``) on 2026-11-16 before
the C2 commit.
"""

SEALED_SHA256_LF: dict[str, str | None] = {
    "scripts/eval_depth_4tu.py": (
        "e09efd48c0cb2fc5ac063fc8f380b7f3382a728f86ee38152dc538e246e4a07a"
    ),
    "analysis/processing/m33_j1_candidates.py": (
        "a4c1786bed2a0d38e8b73fbe66659862049565e0c2d6fedf5f7c68cbb64f30f9"
    ),
    "analysis/processing/m33_j1_judge.py": (
        "b43eb758f459beed99b9760137d43da6c973bbbcd41c67fb2e8d927fbe8660d1"
    ),
    "docs/processing/fourtu_lines_v1.csv": (
        "475f0a71fdd79fcca3fde035bce0876778f4771b61e4dc50195868e085104cc6"
    ),
    "docs/datasets/fourtu_duplicate_radargrams.csv": (
        "7602d94fbd5824f3536d463e2b248e4ea474770f852519db3ea30097312c488a"
    ),
    "docs/processing/m29c_qc_constants.json": (
        "6a88bd8d2501132b0db43eeec2dec9ef347de6e260a9f98aef9a2322ec56ba8c"
    ),
}
"""sha256 of the CRLF -> LF content of the same files (§1-2 SHA-256(LF))."""

SRC_TREE_ID: str | None = "84b02beda30544960d4d2ac02cf461fe66556c4f"
"""``git rev-parse HEAD:src/gpr_detection`` at C2 (src unchanged since 1d99bf7)."""

FOURTU_META_SHA256: dict[str, str | None] = {
    "utilities_groundtruth.csv": (
        "18e217b1fa890151423e10aa08dc1104c1240d51c4e2828b7b104b5b2777a41e"
    ),
    "utilities_metadata_list.csv": (
        "d7c8ec9e3898caecbc09774d28b08b48c5320074338f29ef475b55ac06a0dc7c"
    ),
    "radargrams_index.csv": (
        "28a550e190cbba31dc8d504d3a29a80779eec1b544ee73df07223e03fcc6b106"
    ),
    "activities_metadata_ko.csv": (
        "91bca3447ed861655ca996b18c42719c81196fb118576d0d20fe628aca3899a9"
    ),
}
"""Raw sha256 of the four 4TU metadata csv files (read-only dataset; the same
values as [U3] §1-2)."""

R2_J1_POST_SHA256: str | None = (
    "f6a63e667cf10f7e5c8f0f55bca08c6804622ed1bf1416f039fb487c4a8b8137"
)
"""R2 ``j1_inputs.folds["0"].post_sha256`` (placeholder R2_J1_POST_SHA)."""

R2_J1_PRE_SHA256: str | None = (
    "d8f409dd0c6290d53b82153c210ed64c45442ecb6ae5ebc978064b4ed85e3d07"
)
"""R2 ``j1_inputs.folds["0"].pre_sha256`` (placeholder R2_J1_PRE_SHA)."""


def unfilled_names() -> list[str]:
    """Names of the C2 values that are still ``None`` (empty after C2)."""
    missing = [f"SEALED_BLOBS[{k}]" for k, v in SEALED_BLOBS.items() if v is None]
    missing += [
        f"SEALED_SHA256_LF[{k}]" for k, v in SEALED_SHA256_LF.items() if v is None
    ]
    missing += [
        f"FOURTU_META_SHA256[{k}]" for k, v in FOURTU_META_SHA256.items() if v is None
    ]
    scalars = {
        "SRC_TREE_ID": SRC_TREE_ID,
        "R2_J1_POST_SHA256": R2_J1_POST_SHA256,
        "R2_J1_PRE_SHA256": R2_J1_PRE_SHA256,
    }
    missing += [name for name, value in scalars.items() if value is None]
    return missing
