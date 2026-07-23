"""Task 032: external EPM/RAPM priors must be versioned by observation date.
A future/retrospective snapshot must never join to an earlier game, and an
unavailable metric must be an explicit missing flag, never a silent zero."""
import pandas as pd
import pytest

from pipeline.epm_priors import EpmPriorSchemaError, EpmPriorTracker


def _csv(tmp_path, rows):
    path = tmp_path / "epm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_requires_observation_date_column(tmp_path):
    path = tmp_path / "epm_unversioned.csv"
    pd.DataFrame([{"player_id": "1", "epm": 3.0}]).to_csv(path, index=False)
    with pytest.raises(EpmPriorSchemaError):
        EpmPriorTracker(csv_path=path)


def test_future_snapshot_cannot_join_to_past_game(tmp_path):
    path = _csv(tmp_path, [
        {"player_id": "1", "epm": 1.0, "observation_date": "2025-11-01"},
        {"player_id": "1", "epm": 9.0, "observation_date": "2026-03-01"},  # future, retrospective update
    ])
    tracker = EpmPriorTracker(csv_path=path)
    past_game_date = pd.Timestamp("2025-12-01")
    val = tracker.get_as_of("1", as_of=past_game_date)
    assert val == 1.0, "future EPM snapshot leaked into a past game's as-of query"
    assert tracker.scaled_impact("1", as_of=past_game_date) == 1.0 * 50.0

    later_game_date = pd.Timestamp("2026-04-01")
    val2 = tracker.get_as_of("1", as_of=later_game_date)
    assert val2 == 9.0


def test_missing_snapshot_is_flagged_not_zero(tmp_path):
    path = _csv(tmp_path, [
        {"player_id": "1", "epm": 1.0, "observation_date": "2026-01-01"},
    ])
    tracker = EpmPriorTracker(csv_path=path)
    # Cutoff before player 1's only snapshot, and player "2" never appears.
    before_any_snapshot = pd.Timestamp("2025-10-01")
    assert tracker.missing("1", as_of=before_any_snapshot) is True
    assert tracker.missing("2", as_of=pd.Timestamp("2026-06-01")) is True
    # scaled_impact still returns a neutral 0.0 for blending, but callers
    # must use `missing()` (or the *_missing_frac feature) to distinguish
    # "unavailable" from "known zero impact".
    assert tracker.scaled_impact("1", as_of=before_any_snapshot) == 0.0


def test_feature_dict_reports_missing_fraction(tmp_path):
    path = _csv(tmp_path, [
        {"player_id": "1", "epm": 2.0, "observation_date": "2025-11-01"},
        {"player_id": "2", "epm": -1.0, "observation_date": "2025-11-01"},
    ])
    tracker = EpmPriorTracker(csv_path=path)
    cutoff = pd.Timestamp("2025-12-01")
    feats = tracker.feature_dict(["1", "2", "3"], ["4"], ho_off=1500.0, ao_off=1500.0, as_of=cutoff)
    # players 3 and 4 have no snapshot at all -> missing.
    assert feats["epm_missing_frac_home"] == pytest.approx(1 / 3)
    assert feats["epm_missing_frac_away"] == 1.0


def test_latest_eligible_snapshot_selected_when_multiple_qualify(tmp_path):
    path = _csv(tmp_path, [
        {"player_id": "1", "epm": 1.0, "observation_date": "2025-10-01"},
        {"player_id": "1", "epm": 2.0, "observation_date": "2025-11-01"},
        {"player_id": "1", "epm": 3.0, "observation_date": "2025-12-15"},
    ])
    tracker = EpmPriorTracker(csv_path=path)
    cutoff = pd.Timestamp("2025-12-01")  # between the 11-01 and 12-15 snapshots
    assert tracker.get_as_of("1", as_of=cutoff) == 2.0


def test_no_cutoff_falls_back_to_latest_known(tmp_path):
    path = _csv(tmp_path, [
        {"player_id": "1", "epm": 1.0, "observation_date": "2025-10-01"},
        {"player_id": "1", "epm": 5.0, "observation_date": "2026-05-01"},
    ])
    tracker = EpmPriorTracker(csv_path=path)
    assert tracker.get_as_of("1", as_of=None) == 5.0
