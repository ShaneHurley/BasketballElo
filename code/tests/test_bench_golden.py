"""Epic 10.5.1 — Golden-master season snapshot (drift gate).

The golden file is ``tests/synth/golden_snapshots/golden_season.json``
(committed; inside ``tests/synth/`` next to the factories — see the
``tests/synth/golden.py`` module docstring for the location rationale and
the float-precision policy). It pins the outputs of the synthetic-feasible
pipeline stages — ``pipeline.ratings``, ``pipeline.lineup_elo``,
``pipeline.chemistry``, and the ``pipeline.market`` devig/cover-prob math —
on a fixed 40-game synthetic season.

Tests:

1. ``test_generate_golden_season_deterministic`` — same seed, identical frame.
2. ``test_snapshot_byte_identical_same_seed`` — two snapshots, byte-identical
   canonical JSON.
3. ``test_golden_file_matches_regeneration`` — **the drift gate**: committed
   file matches regeneration byte-for-byte, with the embedded
   ``FEATURE_SCHEMA_VERSION`` asserted first for a clear error on version
   bump. ``--regenerate-golden`` rewrites the file instead of comparing.
4. ``test_snapshot_drifts_on_margin_flip`` — flipping one game's margin must
   change the snapshot.
5. ``test_snapshot_drifts_on_one_percent_possessions_perturbation`` —
   adversarial: a 1% perturbation of a single stint's possessions must
   change the snapshot (sub-threshold formula sensitivity).
6. ``test_canonical_json_immune_to_order_and_subprecision_noise`` —
   adversarial: dict insertion order and sub-10-sig-digit float noise must
   NOT change the canonical form (no false drift), while super-precision
   differences MUST.
7. ``test_snapshot_strictly_json_native`` — the snapshot serializes with no
   ``default=str`` fallback and ``allow_nan=False``, so nothing is silently
   stringified and no bare ``NaN`` tokens enter the file.
8. ``test_regenerate_golden_option_registered`` — the CLI flag exists.
"""
from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from pipeline.config import FEATURE_SCHEMA_VERSION
from tests.synth import golden as G
from tests.synth.golden import (
    GOLDEN_SEASON_NAME,
    canonical_snapshot_json,
    derive_game_stints,
    generate_golden_season,
    golden_path,
    load_golden,
    regenerate_golden_season,
    snapshot_stage_outputs,
)


@pytest.fixture
def regenerate_golden(request: pytest.FixtureRequest) -> bool:
    """True when pytest was invoked with ``--regenerate-golden``."""
    return bool(request.config.getoption("--regenerate-golden"))


# ---------------------------------------------------------------------------
# 1. Season generation determinism
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_generate_golden_season_deterministic():
    """Two calls with the default seed produce the identical 40-game frame."""
    a = generate_golden_season()
    b = generate_golden_season()
    assert len(a) == 40
    pd.testing.assert_frame_equal(a, b)
    # Non-vacuous: a different seed produces a different season.
    c = generate_golden_season(seed=G.DEFAULT_SEED + 1)
    assert not a.equals(c)


# ---------------------------------------------------------------------------
# 2. Snapshot determinism (byte-identical canonical JSON)
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_snapshot_byte_identical_same_seed():
    """Two snapshots of the golden season are byte-identical canonical JSON."""
    s1 = snapshot_stage_outputs(generate_golden_season())
    s2 = snapshot_stage_outputs(generate_golden_season())
    j1 = canonical_snapshot_json(s1)
    j2 = canonical_snapshot_json(s2)
    assert isinstance(j1, str)
    assert j1 == j2
    # Non-vacuous: the snapshot actually contains stage outputs.
    assert len(s1["per_game"]) == 40
    assert s1["final_state"]["player_ratings"], "no player ratings snapshotted"
    assert s1["final_state"]["lineup_elo"], "no lineup Elo state snapshotted"
    assert s1["final_state"]["chemistry"]["duo"], "no chemistry duos snapshotted"


# ---------------------------------------------------------------------------
# 3. Drift gate (+ --regenerate-golden rewrite path)
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_golden_file_matches_regeneration(regenerate_golden: bool):
    """Drift gate: the committed golden file matches regeneration byte-for-byte.

    With ``--regenerate-golden`` the file is rewritten instead (the reviewed
    regeneration path; pair with a FEATURE_SCHEMA_VERSION bump).
    """
    if regenerate_golden:
        path = regenerate_golden_season()
        assert path == golden_path(GOLDEN_SEASON_NAME)
        return

    path = golden_path(GOLDEN_SEASON_NAME)
    assert path.exists(), (
        f"golden file missing: {path}. Bootstrap it with: "
        f"pytest tests/test_bench_golden.py --regenerate-golden"
    )

    committed = load_golden(GOLDEN_SEASON_NAME)
    assert committed["FEATURE_SCHEMA_VERSION"] == int(FEATURE_SCHEMA_VERSION), (
        f"golden file was written under FEATURE_SCHEMA_VERSION="
        f"{committed['FEATURE_SCHEMA_VERSION']} but config has "
        f"{FEATURE_SCHEMA_VERSION}. Bump/regenerate with: "
        f"pytest tests/test_bench_golden.py --regenerate-golden"
    )

    snapshot = snapshot_stage_outputs(generate_golden_season())

    # Semantic layer: canonical payload equality (order/noise-immune).
    assert canonical_snapshot_json(committed["payload"]) == canonical_snapshot_json(snapshot), (
        "golden payload drifted — a snapshotted stage's formula changed. "
        "If intentional: bump FEATURE_SCHEMA_VERSION and regenerate with "
        "--regenerate-golden."
    )

    # Byte layer: full-file text equality through the shared serializer.
    expected_text = G._serialize_golden_body(snapshot, FEATURE_SCHEMA_VERSION)
    assert path.read_text() == expected_text, "golden file bytes differ from regeneration"


# ---------------------------------------------------------------------------
# 4. Sensitivity: formula-relevant input change must drift the snapshot
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_snapshot_drifts_on_margin_flip():
    """Flipping one game's margin changes the snapshot (drift gate fires)."""
    season = generate_golden_season()
    base = canonical_snapshot_json(snapshot_stage_outputs(season))

    flipped = season.copy()
    i = 20  # mid-season game: later per-game rows AND final state must move
    flipped.loc[flipped.index[i], "ACTUAL_HOME"] = flipped.loc[flipped.index[i], "ACTUAL_HOME"] + 4.0
    flipped.loc[flipped.index[i], "ACTUAL_MARGIN"] = (
        flipped.loc[flipped.index[i], "ACTUAL_HOME"] - flipped.loc[flipped.index[i], "ACTUAL_AWAY"]
    )
    assert not flipped.equals(season)  # the flip is real

    perturbed = canonical_snapshot_json(snapshot_stage_outputs(flipped))
    assert perturbed != base, "margin flip did not drift the snapshot — gate is blind"


@pytest.mark.bench
def test_snapshot_drifts_on_one_percent_possessions_perturbation():
    """ADVERSARIAL: a 1% perturbation of ONE stint's possessions must drift.

    Falsification attempt against the drift gate's sensitivity: if a subtle
    formula-relevant change this small slipped through, the gate would be
    blind to real formula bugs of comparable magnitude. Possessions feed
    stint weights, K-factors, chemistry shrinkage, and possession counters,
    so the effect must survive the 10-sig-digit rounding.
    """
    season = generate_golden_season()
    stints = derive_game_stints(season)
    base = canonical_snapshot_json(G._run_stages(season, stints))

    stints_pert = copy.deepcopy(stints)
    gid = str(season["GAME_ID"].iloc[10])
    target = stints_pert[gid][2]["row"]
    target["possessions"] = target["possessions"] * 1.01
    assert target["possessions"] != stints[gid][2]["row"]["possessions"]

    perturbed = canonical_snapshot_json(G._run_stages(season, stints_pert))
    assert perturbed != base, "1% possessions perturbation did not drift the snapshot"


# ---------------------------------------------------------------------------
# 5. Adversarial: canonicalization cannot produce false drift
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_canonical_json_immune_to_order_and_subprecision_noise():
    """Dict insertion order and sub-precision float noise must not drift.

    Caveat found by the adversarial protocol: rounding to 10 significant
    digits absorbs sub-quantum noise *unless the value straddles a rounding
    boundary* — a boundary-straddling value flips its last retained digit no
    matter how small the noise. Empirically, a 1e-11 relative perturbation
    of one stint's possessions flips 16 of ~50k snapshot leaves by exactly
    one last-digit unit. This is acceptable: the rounding exists to absorb
    cross-platform 1-ulp noise (~1e-16 relative), whose expected flip count
    over the whole snapshot is ~1e-4 — nine orders of magnitude below the
    unrounded false-drift rate — while identical code+inputs always produce
    byte-identical output.
    """
    a = {"x": 1.0, "y": [1.0, 2.0], "z": {"p": 3.0, "q": 4.0}}
    b = {"z": {"q": 4.0, "p": 3.0}, "y": [1.0, 2.0], "x": 1.0}
    assert canonical_snapshot_json(a) == canonical_snapshot_json(b)

    # Sub-precision noise (relative 1e-12, away from a rounding boundary) is
    # absorbed…
    x = 1.23456789012345
    noisy = x * (1.0 + 1e-12)
    assert canonical_snapshot_json({"v": x}) == canonical_snapshot_json({"v": noisy})

    # …while a super-precision difference (relative 1e-8) must drift.
    moved = x * (1.0 + 1e-8)
    assert canonical_snapshot_json({"v": x}) != canonical_snapshot_json({"v": moved})

    # -0.0 and 0.0 canonicalize identically (sign-of-zero cannot false-drift).
    assert canonical_snapshot_json({"v": -0.0}) == canonical_snapshot_json({"v": 0.0})

    # Canonical form is idempotent under a JSON round-trip.
    s = canonical_snapshot_json(a)
    assert canonical_snapshot_json(json.loads(s)) == s


# ---------------------------------------------------------------------------
# 6. Snapshot is strictly JSON-native (nothing silently stringified)
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_snapshot_strictly_json_native():
    """The snapshot serializes with no default=str fallback and no NaN tokens.

    ``write_golden`` uses ``json.dumps(default=str)``; if a non-native object
    (numpy scalar, Timestamp, tuple key) ever leaked into the snapshot it
    would be silently stringified and could mask a schema drift. Serializing
    with ``allow_nan=False`` and no ``default`` proves the payload needs
    neither crutch.
    """
    snapshot = snapshot_stage_outputs(generate_golden_season())
    text = json.dumps(snapshot, allow_nan=False)  # no default= on purpose
    assert json.loads(text) == snapshot
    assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# 7. CLI flag registration
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_regenerate_golden_option_registered(pytestconfig: pytest.Config):
    """The --regenerate-golden option is registered and boolean."""
    assert isinstance(pytestconfig.getoption("--regenerate-golden"), bool)
