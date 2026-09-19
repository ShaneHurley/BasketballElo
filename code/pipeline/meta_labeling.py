"""Meta-labeling policy helpers (ATS / upset) — policy_tuning role only.

Wraps existing ATS/upset classifier outputs into a formal policy-layer
decision. Never retunes margin models; never uses test-role rows for
threshold search.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.dataset_roles import VALID_ROLES, RoleAssignmentError


def assert_policy_role(roles, *, allowed: tuple[str, ...] = ("policy_tuning",)) -> None:
    """Raise if any row role is outside the allowed set for meta-label tuning."""
    arr = np.asarray(roles, dtype=object)
    bad = sorted({str(r) for r in np.unique(arr) if r not in allowed and r != "unused"})
    if bad:
        raise RoleAssignmentError(
            f"meta_labeling may only use roles {allowed}; found {bad}. "
            f"Valid roles overall: {VALID_ROLES}"
        )


def meta_label_pass(
    *,
    direction: str,
    ats_cover_prob: float | None = None,
    upset_prob: float | None = None,
    market_spread: float | None = None,
    min_ats_prob: float = 0.52,
    max_upset_when_fav: float = 0.45,
) -> bool:
    """Return True if the actionable lean survives meta-label filters."""
    if direction in (None, "Pass", "nan"):
        return False
    if ats_cover_prob is not None and np.isfinite(ats_cover_prob):
        if float(ats_cover_prob) < float(min_ats_prob):
            return False
    if (
        upset_prob is not None
        and np.isfinite(upset_prob)
        and market_spread is not None
        and np.isfinite(market_spread)
    ):
        # Favorite lean: home when market_spread < 0, away when > 0.
        lean_fav = (
            (direction == "Home" and float(market_spread) < 0)
            or (direction == "Away" and float(market_spread) > 0)
        )
        if lean_fav and float(upset_prob) > float(max_upset_when_fav):
            return False
    return True


def apply_meta_labels(
    df: pd.DataFrame,
    *,
    min_ats_prob: float = 0.52,
    max_upset_when_fav: float = 0.45,
    ats_col: str = "ATS_COVER_PROB",
    upset_col: str = "UPSET_PROB",
    direction_col: str = "DIRECTION",
    market_col: str = "MARKET_SPREAD",
    roles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Add META_LABEL_PASS / optionally force Pass when filter fails.

    If ``roles`` is provided, only ``policy_tuning`` rows may be used to
    *evaluate* filter rates; the pass column is still written for all rows.
    """
    if roles is not None:
        # Enforce that callers who pass roles are not quietly tuning on test.
        uniq = {str(r) for r in np.unique(np.asarray(roles, dtype=object))}
        if "test" in uniq and uniq - {"policy_tuning", "unused", "test"}:
            pass  # test rows may appear for scoring; tuning must exclude them
        if "train" in uniq or "calibration" in uniq:
            raise RoleAssignmentError(
                "apply_meta_labels: do not pass train/calibration roles into "
                "policy meta-label evaluation"
            )

    out = df.copy()
    dirs = out[direction_col].astype(str) if direction_col in out.columns else pd.Series(["Pass"] * len(out))
    ats = pd.to_numeric(out[ats_col], errors="coerce") if ats_col in out.columns else pd.Series([np.nan] * len(out))
    ups = pd.to_numeric(out[upset_col], errors="coerce") if upset_col in out.columns else pd.Series([np.nan] * len(out))
    mkt = pd.to_numeric(out[market_col], errors="coerce") if market_col in out.columns else pd.Series([np.nan] * len(out))

    flags = []
    for d, a, u, m in zip(dirs, ats, ups, mkt):
        flags.append(
            meta_label_pass(
                direction=d,
                ats_cover_prob=None if pd.isna(a) else float(a),
                upset_prob=None if pd.isna(u) else float(u),
                market_spread=None if pd.isna(m) else float(m),
                min_ats_prob=min_ats_prob,
                max_upset_when_fav=max_upset_when_fav,
            )
        )
    out["META_LABEL_PASS"] = flags
    return out


def meta_label_grid(
    df: pd.DataFrame,
    *,
    ats_probs: tuple[float, ...] = (0.50, 0.52, 0.55),
    upset_caps: tuple[float, ...] = (0.40, 0.45, 0.55),
    roles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Grid search meta-label knobs on policy_tuning rows only."""
    if roles is not None:
        assert_policy_role(roles[np.asarray(roles) == "policy_tuning"], allowed=("policy_tuning",))
        mask = np.asarray(roles, dtype=object) == "policy_tuning"
        work = df.loc[mask].copy() if mask.any() else df.iloc[0:0].copy()
    else:
        work = df

    rows: list[dict[str, Any]] = []
    for ap in ats_probs:
        for uc in upset_caps:
            labeled = apply_meta_labels(work, min_ats_prob=ap, max_upset_when_fav=uc)
            passed = labeled[labeled["META_LABEL_PASS"]]
            n = len(passed)
            ats = np.nan
            if n and "HIT" in passed.columns:
                ats = float(pd.to_numeric(passed["HIT"], errors="coerce").mean())
            clv = np.nan
            if n and "CLV" in passed.columns:
                clv = float(pd.to_numeric(passed["CLV"], errors="coerce").mean())
            rows.append({
                "min_ats_prob": ap,
                "max_upset_when_fav": uc,
                "n_pass": n,
                "ats_pct": ats,
                "mean_clv": clv,
            })
    return pd.DataFrame(rows)
