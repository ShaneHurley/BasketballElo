"""Task 056 — tune the betting policy (edge/confidence/abstention/Kelly)
only on policy_tuning-role rows, and assert that final-performance
reporting never happens on those same rows (they must be disjoint from the
test role).

Staking/Kelly/edge-gate *design* is out of this phase's testing scope
(owned by a parallel/later agent per the task brief); this test only
enforces the row-role isolation contract those modules must respect,
using pipeline.dataset_roles (Task 048) as the shared mechanism.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.dataset_roles import (
    RoleAssignmentError,
    assert_role_isolated,
    assign_fold_roles,
    build_role_frame,
    rows_for_role,
)


def _policy_and_test_frames(n=100):
    ts = pd.date_range("2025-10-01", periods=n, freq="D")
    df = pd.DataFrame({
        "game_date": ts,
        "edge": np.random.RandomState(0).normal(size=n),
        "actual_margin": np.random.RandomState(1).normal(size=n),
    })
    roles = assign_fold_roles(
        ts.values, train_end=ts.values[60], calibration_end=ts.values[75],
        policy_tuning_end=ts.values[90],
    )
    return build_role_frame(df, "game_date", roles)


def _tune_policy(policy_rows: pd.DataFrame) -> dict:
    """Stand-in for the real edge/confidence/Kelly tuner (owned elsewhere).
    Only asserts it received policy_tuning-role rows before doing anything
    — this is the isolation contract Task 056 requires regardless of the
    specific policy-tuning algorithm."""
    assert_role_isolated(policy_rows, "policy_tuning", context="_tune_policy")
    return {"edge_threshold": float(np.median(np.abs(policy_rows["edge"])))}


def _report_final_performance(test_rows: pd.DataFrame) -> float:
    assert_role_isolated(test_rows, "test", context="_report_final_performance")
    return float(test_rows["actual_margin"].mean())


class TestPolicyTuningRoleIsolation:
    def test_policy_tuning_only_sees_policy_tuning_rows(self):
        role_df = _policy_and_test_frames()
        policy_rows = rows_for_role(role_df, "policy_tuning")
        knobs = _tune_policy(policy_rows)
        assert "edge_threshold" in knobs

    def test_final_reporting_never_touches_policy_tuning_rows(self):
        role_df = _policy_and_test_frames()
        test_rows = rows_for_role(role_df, "test")
        # This is the explicit Task 056 pass condition: the rows used to
        # choose policy knobs and the rows used to report final
        # performance must be disjoint.
        policy_rows = rows_for_role(role_df, "policy_tuning")
        assert set(policy_rows.index).isdisjoint(set(test_rows.index))
        _report_final_performance(test_rows)

    def test_passing_test_rows_to_policy_tuner_is_rejected(self):
        role_df = _policy_and_test_frames()
        mixed = pd.concat([
            rows_for_role(role_df, "policy_tuning"),
            rows_for_role(role_df, "test").iloc[:5],
        ])
        with pytest.raises(RoleAssignmentError):
            _tune_policy(mixed)

    def test_passing_policy_tuning_rows_to_final_reporting_is_rejected(self):
        """Regression test for the exact defect Task 056 forbids: reporting
        final performance on rows that were also used to choose the
        policy."""
        role_df = _policy_and_test_frames()
        with pytest.raises(RoleAssignmentError):
            _report_final_performance(rows_for_role(role_df, "policy_tuning"))
