"""Task 048 — one row-role per outer fold; overlap/chronology must raise."""
import numpy as np
import pandas as pd
import pytest

from pipeline.dataset_roles import (
    RoleAssignmentError,
    assert_role_isolated,
    assign_fold_roles,
    build_role_frame,
    rows_for_role,
    validate_roles,
)


def _days(n):
    return pd.date_range("2025-10-01", periods=n, freq="D").values


class TestAssignFoldRoles:
    def test_basic_partition_into_four_roles(self):
        ts = _days(100)
        roles = assign_fold_roles(
            ts,
            train_end=ts[60],
            calibration_end=ts[80],
            policy_tuning_end=ts[90],
        )
        assert set(roles) == {"train", "calibration", "policy_tuning", "test"}
        assert (roles[:60] == "train").all()
        assert (roles[60:80] == "calibration").all()
        assert (roles[80:90] == "policy_tuning").all()
        assert (roles[90:] == "test").all()

    def test_rejects_inverted_boundaries(self):
        ts = _days(10)
        with pytest.raises(RoleAssignmentError):
            assign_fold_roles(ts, train_end=ts[8], calibration_end=ts[2], policy_tuning_end=ts[9])

    def test_test_end_bounds_leave_remainder_unused(self):
        ts = _days(20)
        roles = assign_fold_roles(
            ts, train_end=ts[10], calibration_end=ts[13], policy_tuning_end=ts[15], test_end=ts[18],
        )
        assert (roles[18:] == "unused").all()
        assert (roles[15:18] == "test").all()


class TestValidateRoles:
    def test_valid_chronological_roles_pass(self):
        ts = _days(40)
        roles = assign_fold_roles(ts, train_end=ts[20], calibration_end=ts[30], policy_tuning_end=ts[35])
        validate_roles(ts, roles)  # should not raise

    def test_overlap_raises(self):
        ts = _days(10)
        roles = np.array(["train"] * 5 + ["calibration"] * 5, dtype=object)
        # Manually corrupt: swap one row so calibration min < train max.
        roles[4] = "calibration"
        roles[5] = "train"
        with pytest.raises(RoleAssignmentError):
            validate_roles(ts, roles)

    def test_unknown_role_label_raises(self):
        ts = _days(3)
        roles = np.array(["train", "bogus_role", "test"], dtype=object)
        with pytest.raises(RoleAssignmentError):
            validate_roles(ts, roles)

    def test_tie_at_boundary_raises(self):
        ts = np.array([1, 2, 2, 3])
        roles = np.array(["train", "train", "calibration", "test"], dtype=object)
        # max(train)=2 == min(calibration)=2 -> must raise (no shared instant).
        with pytest.raises(RoleAssignmentError):
            validate_roles(ts, roles)


class TestRoleFrameHelpers:
    def test_build_role_frame_and_rows_for_role(self):
        ts = _days(30)
        df = pd.DataFrame({"game_date": ts, "x": np.arange(30)})
        roles = assign_fold_roles(ts, train_end=ts[15], calibration_end=ts[20], policy_tuning_end=ts[25])
        role_df = build_role_frame(df, "game_date", roles)
        assert len(rows_for_role(role_df, "train")) == 15
        assert len(rows_for_role(role_df, "test")) == 5

    def test_assert_role_isolated_passes_when_pure(self):
        df = pd.DataFrame({"role": ["policy_tuning"] * 5})
        assert_role_isolated(df, "policy_tuning", context="test")

    def test_assert_role_isolated_raises_when_mixed(self):
        df = pd.DataFrame({"role": ["policy_tuning", "test"]})
        with pytest.raises(RoleAssignmentError):
            assert_role_isolated(df, "policy_tuning", context="test")
