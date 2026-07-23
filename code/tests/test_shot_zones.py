"""Tests for coordinate shot zones and zone calibration."""
import pandas as pd
import numpy as np

from pipeline.shot_zones import (
    assign_shot_zone_scalar,
    legacy_zone_alias,
    ZoneCalibration,
    compute_zone_calibration,
)
from pipeline.lineup_composite import composite_lineup_rating, DEFAULT_WEIGHTS


def test_corner3_from_coordinates():
    z = assign_shot_zone_scalar(23.0, x_ft=22.0, y_ft=10.0, is_3pt_text=True)
    assert z == "corner3"


def test_abovebreak3():
    z = assign_shot_zone_scalar(24.0, x_ft=20.0, y_ft=20.0, is_3pt_text=True)
    assert z == "abovebreak3"


def test_legacy_alias():
    assert legacy_zone_alias("ab3") == "abovebreak3"
    assert legacy_zone_alias("rim") == "restricted"


def test_zone_calibration_xpoints():
    calib = ZoneCalibration(
        fg_pct={"restricted": 0.60},
        point_value={"restricted": 2.0},
        counts={"restricted": 5000},
    )
    assert abs(calib.xpoints("restricted") - 1.2) < 0.01


def test_lineup5_weight_minimum():
    assert DEFAULT_WEIGHTS["lineup5"] >= 0.50
    val = composite_lineup_rating(
        player_off_delta=100,
        player_def_delta=0,
        lineup5_net=50,
        chem_duo_net=10,
        chem_trio_net=5,
    )
    assert val != 0.0


def test_compute_zone_calibration_from_df():
    df = pd.DataFrame({
        "EVENTMSGTYPE": [1, 2, 1, 2],
        "shot_zone": ["restricted", "restricted", "abovebreak3", "abovebreak3"],
        "is_fg_make": [1, 0, 1, 0],
    })
    calib = compute_zone_calibration(df)
    assert "restricted" in calib.fg_pct
    assert calib.pps["restricted"] > 0
