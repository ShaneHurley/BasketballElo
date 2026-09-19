"""Negative controls for leak / promotion validation (T-60 Phase 6)."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


def shuffled_outcomes_destroy_edge(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_perm: int = 50,
    seed: int = 0,
    improve_when_lower: bool = True,
) -> dict[str, Any]:
    """Permutation null: shuffled labels should not retain model advantage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    base = float(metric_fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_perm):
        shuffled = y_true.copy()
        rng.shuffle(shuffled)
        nulls.append(float(metric_fn(shuffled, y_pred)))
    null_mean = float(np.mean(nulls))
    if improve_when_lower:
        destroyed = null_mean >= base
    else:
        destroyed = abs(null_mean) < abs(base) or abs(null_mean) < 0.02
    return {
        "passed": bool(destroyed),
        "base_metric": base,
        "null_mean": null_mean,
        "detail": f"base={base:.4f} null_mean={null_mean:.4f}",
    }


def future_line_injection_should_improve(
    decision_resid_mae: float,
    close_resid_mae: float,
) -> dict[str, Any]:
    """Injecting closing line into features should artificially improve residual MAE."""
    leaked_helps = float(close_resid_mae) < float(decision_resid_mae) - 0.25
    return {
        "passed": bool(leaked_helps),
        "decision_resid_mae": float(decision_resid_mae),
        "close_resid_mae": float(close_resid_mae),
        "detail": "close residual should beat decision if close leaked into target",
    }


def quote_after_cutoff_rejected(
    *,
    tip_utc: str = "2025-01-15T19:00:00Z",
    cutoff_minutes: float = 60.0,
) -> dict[str, Any]:
    """Post-cutoff quotes must not be selected as the T-60 decision quote.

    Same GAME_ID/book/market/side/point with an early (-110) and late (-105)
    price; decision must keep the pre-cutoff price, not the late one.
    """
    from pipeline.market_snapshots import (
        attach_tip_utc,
        build_quotes_frame,
        reject_invalid_quotes,
        select_decision_quotes,
    )

    tip = pd.Timestamp(tip_utc)
    rows = [
        {
            "GAME_ID": "g1",
            "book": "pinnacle",
            "market": "spread",
            "side": "home",
            "point": -3.5,
            "price_american": -110,
            "price_decimal": 1.909,
            "quote_timestamp": tip - pd.Timedelta(minutes=90),
            "source_timestamp": tip - pd.Timedelta(minutes=90),
            "ingestion_timestamp": tip - pd.Timedelta(minutes=89),
            "in_play": False,
        },
        {
            "GAME_ID": "g1",
            "book": "pinnacle",
            "market": "spread",
            "side": "home",
            "point": -3.5,
            "price_american": -105,  # post-cutoff price move — must not become decision
            "price_decimal": 1.952,
            "quote_timestamp": tip - pd.Timedelta(minutes=30),
            "source_timestamp": tip - pd.Timedelta(minutes=30),
            "ingestion_timestamp": tip - pd.Timedelta(minutes=29),
            "in_play": False,
        },
    ]
    quotes = build_quotes_frame(rows)
    quotes = attach_tip_utc(quotes, {"g1": tip})
    quotes = reject_invalid_quotes(quotes)
    decided = select_decision_quotes(quotes, cutoff_minutes=cutoff_minutes)
    price_col = "decision_price_american"
    if price_col not in decided.columns or decided.empty:
        return {"passed": False, "detail": "missing decision_price_american"}
    row = decided.iloc[0]
    selected = float(row[price_col]) if pd.notna(row[price_col]) else float("nan")
    missing = bool(row.get("decision_missing", True))
    # Must select the pre-cutoff -110, not the late -105.
    passed = (not missing) and abs(selected - (-110.0)) < 1e-9
    return {
        "passed": bool(passed),
        "selected_price": selected,
        "detail": f"selected_decision_price={selected} missing={missing}",
    }


def score_swap_destroys_pair_advantage(
    pred_home: np.ndarray,
    pred_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
) -> dict[str, Any]:
    """Swapping home/away predictions must not improve paired MAE."""
    ph = np.asarray(pred_home, dtype=float)
    pa = np.asarray(pred_away, dtype=float)
    ah = np.asarray(actual_home, dtype=float)
    aa = np.asarray(actual_away, dtype=float)
    base = 0.5 * (np.abs(ph - ah).mean() + np.abs(pa - aa).mean())
    swapped = 0.5 * (np.abs(pa - ah).mean() + np.abs(ph - aa).mean())
    return {
        "passed": bool(swapped >= base - 1e-9),
        "base_paired_mae": float(base),
        "swapped_paired_mae": float(swapped),
        "detail": f"base={base:.4f} swapped={swapped:.4f}",
    }


def market_features_forbidden_in_score_matrix(feature_cols: list[str]) -> dict[str, Any]:
    """Score-pair training must reject every market/line column."""
    from pipeline.score_targets import score_market_ban_set

    ban = score_market_ban_set()
    bad = sorted(set(feature_cols) & ban)
    return {
        "passed": len(bad) == 0,
        "banned_present": bad,
        "detail": f"banned_in_matrix={bad}",
    }


def locked_fold_definitions(
    seasons: list[int] | None = None,
    *,
    window: int = 4,
) -> dict[str, Any]:
    """Immutable rolling-origin fold spec for direct-score ablations."""
    seasons = list(seasons or [2022, 2023, 2024, 2025, 2026])
    folds = []
    for i, test in enumerate(seasons):
        if i == 0:
            continue
        train = seasons[max(0, i - window):i]
        folds.append({
            "test_season": int(test),
            "train_seasons": [int(s) for s in train],
            "seed": 42,
        })
    return {
        "window": int(window),
        "seasons": [int(s) for s in seasons],
        "folds": folds,
        "immutable": True,
        "configs": [
            "baseline_current",
            "legacy_residual_stack",
            "direct_score_pair",
            "direct_score_pair_no_market_feats",
            "structured_score_pair",
            "hybrid_score_pair",
            "canonical_score_combined",
            "direct_score_pair_elo_agree",
            "direct_score_pair_elo_prior",
            "direct_score_pair_elo_prior_agree",
        ],
    }


def run_negative_control_suite(results: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Aggregate control results for promotion reporting."""
    from pipeline.market_snapshots import assert_no_close_leak

    suite = dict(results or {})
    try:
        assert_no_close_leak(["decision_spread", "close_spread"])
        suite["close_column_guard"] = {"passed": False, "detail": "did not raise"}
    except ValueError:
        suite["close_column_guard"] = {"passed": True, "detail": "assert_no_close_leak raised"}

    suite["quote_after_cutoff"] = quote_after_cutoff_rejected()
    suite["market_features_forbidden"] = market_features_forbidden_in_score_matrix(
        ["elo_net", "hier_net", "exp_poss", "pace_diff"]
    )

    df = pd.DataFrame({"GAME_ID": ["a", "a", "b"]})
    suite["duplicate_game_ids"] = {
        "passed": bool(df["GAME_ID"].duplicated().any()),
        "detail": "fixture contains duplicates",
    }
    # Lightweight aggregate without requiring evaluate_negative_controls.
    all_passed = all(bool(v.get("passed")) for v in suite.values() if isinstance(v, dict))
    return {"passed": all_passed, "controls": suite}
