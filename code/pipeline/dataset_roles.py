"""Dataset role assignment for leak-free validation (Task 048).

Every row used anywhere in the Stage-5 validation/calibration/tuning
machinery must have exactly one declared role *per outer fold*:

- ``train``: fit base learners / engine features / preprocessing.
- ``calibration``: fit a single calibrator for one target (Task 053).
- ``policy_tuning``: choose betting-policy knobs (edge/confidence/
  abstention/Kelly) — never used to report final performance (Task 056).
- ``test``: untouched, reported-on holdout. Never used to fit or tune
  anything.

This module only assigns/validates roles; it does not fit any model.
Rule 6/11 of the plan requires raising instead of silently tolerating any
overlap or chronology violation between roles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_ROLES = ("train", "calibration", "policy_tuning", "test")

# Declared chronological order: train must end before calibration begins,
# calibration before policy_tuning, policy_tuning before test. Roles may be
# used more than once, but their *time ranges* per outer fold must not
# overlap or invert.
_ROLE_ORDER = {"train": 0, "calibration": 1, "policy_tuning": 2, "test": 3}


class RoleAssignmentError(ValueError):
    """Raised when role ranges overlap or violate chronology (Task 048)."""


def assign_fold_roles(
    timestamps,
    train_end,
    calibration_end,
    policy_tuning_end,
    *,
    test_end=None,
) -> np.ndarray:
    """Assign one role per row for a single outer fold.

    Parameters
    ----------
    timestamps : array-like of orderable values (dates/ints/floats).
    train_end : rows with timestamp < train_end -> "train".
    calibration_end : rows with train_end <= timestamp < calibration_end -> "calibration".
    policy_tuning_end : rows with calibration_end <= timestamp < policy_tuning_end -> "policy_tuning".
    test_end : optional upper bound; rows with timestamp >= policy_tuning_end
        (and < test_end if given) -> "test". Rows at/after test_end (if given)
        get role "unused" (not yet in this fold's declared window at all).

    Returns
    -------
    np.ndarray of dtype object with one role string (or "unused") per row.
    """
    ts = np.asarray(timestamps)
    if not (train_end <= calibration_end <= policy_tuning_end):
        raise RoleAssignmentError(
            "assign_fold_roles: boundaries must satisfy "
            "train_end <= calibration_end <= policy_tuning_end, got "
            f"({train_end}, {calibration_end}, {policy_tuning_end})"
        )
    if test_end is not None and test_end < policy_tuning_end:
        raise RoleAssignmentError(
            "assign_fold_roles: test_end must be >= policy_tuning_end"
        )

    roles = np.full(ts.shape, "unused", dtype=object)
    roles[ts < train_end] = "train"
    roles[(ts >= train_end) & (ts < calibration_end)] = "calibration"
    roles[(ts >= calibration_end) & (ts < policy_tuning_end)] = "policy_tuning"
    if test_end is not None:
        roles[(ts >= policy_tuning_end) & (ts < test_end)] = "test"
    else:
        roles[ts >= policy_tuning_end] = "test"
    return roles


def validate_roles(timestamps, roles) -> None:
    """Raise ``RoleAssignmentError`` if role time-ranges overlap or invert.

    A role assignment is valid only if, for every pair of roles that both
    appear in the data, every timestamp with the earlier-declared role is
    strictly less than every timestamp with the later-declared role (ties
    at a shared boundary point are rejected — a row cannot be simultaneously
    "before" and "at" a cutoff for two different roles).
    """
    ts = np.asarray(timestamps)
    roles = np.asarray(roles, dtype=object)
    if len(ts) != len(roles):
        raise RoleAssignmentError("timestamps and roles must be the same length")

    present = [r for r in VALID_ROLES if (roles == r).any()]
    for r in np.unique(roles):
        if r not in VALID_ROLES and r != "unused":
            raise RoleAssignmentError(f"Unknown role label: {r!r}")

    ordered = sorted(present, key=lambda r: _ROLE_ORDER[r])
    for earlier, later in zip(ordered, ordered[1:]):
        earlier_max = ts[roles == earlier].max()
        later_min = ts[roles == later].min()
        if earlier_max >= later_min:
            raise RoleAssignmentError(
                f"Role chronology violated: max({earlier})={earlier_max!r} "
                f">= min({later})={later_min!r}. Roles must not overlap."
            )

    # Every row must have exactly one role (including "unused" — never two).
    # This is automatically true for a 1-D array assignment, but guard
    # against callers who hand-build overlapping boolean masks elsewhere.
    if roles.ndim != 1:
        raise RoleAssignmentError("roles must be a 1-D array (one role per row)")


def build_role_frame(df: pd.DataFrame, timestamp_col: str, roles: np.ndarray) -> pd.DataFrame:
    """Attach a validated ``role`` column to ``df`` (copy)."""
    validate_roles(df[timestamp_col].values, roles)
    out = df.copy()
    out["role"] = roles
    return out


def rows_for_role(df: pd.DataFrame, role: str) -> pd.DataFrame:
    if "role" not in df.columns:
        raise RoleAssignmentError("DataFrame has no 'role' column; call build_role_frame first")
    return df.loc[df["role"] == role]


def assert_role_isolated(df: pd.DataFrame, expected_role: str, context: str = "") -> None:
    """Fail closed if any row lacks ``role == expected_role`` (Task 056)."""
    if "role" not in df.columns:
        raise RoleAssignmentError(
            f"assert_role_isolated({context}): DataFrame has no 'role' column"
        )
    bad = df.loc[df["role"] != expected_role]
    if len(bad):
        raise RoleAssignmentError(
            f"assert_role_isolated({context}): expected only role="
            f"{expected_role!r}, found {sorted(bad['role'].unique())} "
            f"on {len(bad)} row(s)"
        )
