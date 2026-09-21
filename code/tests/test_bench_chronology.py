"""Epic 10.4.3 — Chronology-tamper harness (spec for roadmap task 4.5.3).

This is a spec harness only: it hashes a local ``dataset_hash()``, not
production ingest. It is not wired to ``pipeline/benchmark_lock.py`` and does
not mean CI chronology-tamper detection is live.

A chronological dataset must be tamper-evident: reordering rows (replaying
the future before the past) or reordering columns (schema tampering) must
change the dataset fingerprint, while identical re-generation must not.

Hash construction (documented choice per task): per-row
``pd.util.hash_pandas_object(frame, index=False)`` uint64 hashes, then
SHA-256 over the raw hash-array bytes in row order.

Why this construction:

- ``index=False`` keeps the fingerprint a function of *content and order*
  only — the pandas index is not part of the dataset contract.
- Row order is preserved in the byte serialization, so any permutation of
  distinguishable rows changes the digest (chronology tampering).
- ``hash_pandas_object`` combines per-column hashes in *schema order*, so
  column reordering also changes the digest (column-order tampering).
  Columns are deliberately NOT name-sorted before hashing: canonicalizing
  column order would blind the harness to exactly that attack.
- Dict-valued object columns (``home_usage`` / ``away_usage``) are
  canonicalized via ``json.dumps(..., sort_keys=True)`` first, because
  pandas factorization cannot hash raw dicts.

Adversarial note (identical-row swap): a permutation that exchanges two
*identical* rows preserves the hash. That is acceptable — and unavoidable
for any content-based fingerprint — because swapping indistinguishable
rows destroys no chronological information: every downstream consumer
observes the same sequence of records either way. The harness therefore
also asserts the bench dataset has all-distinct rows, so every
non-identity permutation of it is detectable (see the tamper test).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tests.synth.factories import make_stint, make_stint_frame

SEED = 20260919
N_ROWS = 24


def _bench_dataset(seed: int = SEED, n: int = N_ROWS) -> pd.DataFrame:
    """Small synthetic stint dataset with strictly increasing timestamps.

    Built via ``tests.synth.factories`` so the schema mirrors what
    ``pipeline/stints.py`` consumers see. Rows are all distinct (unique
    ``stint_id``/``game_date``/scores), which the tamper test relies on.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-11-01")
    stints = []
    for i in range(n):
        stints.append(
            make_stint(
                game_id=f"00225{i // 4:05d}",
                stint_id=i,
                period=1 + (i % 4),
                possessions=float(rng.uniform(0.5, 3.0)),
                home_xpts=float(rng.uniform(0.5, 2.5)),
                away_xpts=float(rng.uniform(0.5, 2.5)),
                game_date=base + pd.Timedelta(hours=3 * i),
            )
        )
    return make_stint_frame(stints=stints)


def dataset_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of per-row ``hash_pandas_object`` digests, in row order.

    See module docstring for the construction rationale. Dict-valued
    object cells are JSON-canonicalized before hashing.
    """
    df = frame.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: json.dumps(v, sort_keys=True, default=str)
                if isinstance(v, dict)
                else v
            )
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


@pytest.mark.bench
def test_bench_chronology_row_permutation_changes_hash():
    """Roadmap 4.5.3 acceptance: permuting row order must change the hash."""
    df = _bench_dataset()

    # Precondition: every row is distinguishable, so *any* non-identity
    # permutation is genuine chronology tampering (see adversarial test).
    # Distinctness is checked on the scalar schema: home_usage/away_usage
    # are unhashable empty dicts in this dataset and carry no content.
    scalar = df.drop(columns=["home_usage", "away_usage"])
    assert len(scalar.drop_duplicates()) == len(df), "bench dataset must have all-distinct rows"
    assert df["game_date"].is_monotonic_increasing, "dataset must be in chronological order"

    base = dataset_hash(df)

    # Full reversal (maximal chronology violation).
    reversed_df = df.iloc[::-1].reset_index(drop=True)
    assert dataset_hash(reversed_df) != base, "row reversal must change the dataset hash"

    # Several deterministic non-identity permutations — every one must be caught.
    rng = np.random.default_rng(SEED + 1)
    for _ in range(5):
        perm = rng.permutation(len(df))
        assert not np.array_equal(perm, np.arange(len(df)))
        permuted = df.iloc[perm].reset_index(drop=True)
        assert dataset_hash(permuted) != base, f"row permutation {perm.tolist()} undetected"


@pytest.mark.bench
def test_bench_chronology_regeneration_is_hash_stable():
    """Identical re-generation with the same seed must produce the same hash."""
    first = _bench_dataset(seed=SEED)
    second = _bench_dataset(seed=SEED)
    pd.testing.assert_frame_equal(first, second)
    assert dataset_hash(first) == dataset_hash(second)

    # Sanity that the fingerprint carries content: a different seed (different
    # data) must not collide with the canonical dataset's hash.
    other = _bench_dataset(seed=SEED + 777)
    assert dataset_hash(other) != dataset_hash(first)


@pytest.mark.bench
def test_bench_chronology_column_reorder_changes_hash():
    """Column-order tampering must change the hash (schema order is hashed)."""
    df = _bench_dataset()
    base = dataset_hash(df)

    reversed_cols = df[list(reversed(df.columns))]
    assert list(reversed_cols.columns) != list(df.columns)
    assert dataset_hash(reversed_cols) != base, "column reversal must change the dataset hash"

    # A milder rotation (not a symmetric reversal) must also be caught.
    rotated = df[list(df.columns[1:]) + [df.columns[0]]]
    assert dataset_hash(rotated) != base, "column rotation must change the dataset hash"

    # The tampered frames still carry identical values — only order changed.
    assert sorted(reversed_cols.columns) == sorted(df.columns)
    pd.testing.assert_frame_equal(
        reversed_cols[sorted(df.columns)].reset_index(drop=True),
        df[sorted(df.columns)].reset_index(drop=True),
    )


@pytest.mark.bench
def test_bench_chronology_identical_row_swap_is_hash_invariant():
    """Adversarial: swapping two *identical* rows preserves the hash.

    This is the one permutation a content fingerprint cannot detect, and it
    is acceptable: exchanging indistinguishable records removes no
    chronological information — the observed record sequence is unchanged
    (the swapped frame is ``equals``-identical to the original). The harness
    tolerates this because the tamper test above asserts the real bench
    dataset has all-distinct rows, where every non-identity permutation is
    detectable.
    """
    df = _bench_dataset()
    dup = df.copy()
    dup.iloc[3] = df.iloc[5]  # rows 3 and 5 are now identical records
    assert dup.iloc[3].equals(dup.iloc[5])

    swapped = dup.copy()
    swapped.iloc[3], swapped.iloc[5] = dup.iloc[5].copy(), dup.iloc[3].copy()

    # The swap is a no-op on content: frames are identical, so any
    # content-based hash MUST be invariant (a hash that changed here would
    # be flagging order noise, not tampering).
    assert swapped.equals(dup)
    assert dataset_hash(swapped) == dataset_hash(dup)

    # Tightening: on the all-distinct bench dataset, even the smallest
    # possible tamper — swapping two *adjacent* rows — is detected.
    adjacent = df.iloc[[0, 1, 3, 2] + list(range(4, len(df)))].reset_index(drop=True)
    assert dataset_hash(adjacent) != dataset_hash(df)
