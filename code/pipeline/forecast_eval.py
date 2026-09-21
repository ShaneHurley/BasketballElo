"""Actuals-first evaluation hygiene (blueprint step 1).

Decouples predictive loss (error vs scoreboard) from bet-grading / tip_proxy.
Includes Diebold–Mariano for locked-baseline comparisons and pace×efficiency
totals decomposition diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.calibration_metrics import (
    compute_brier,
    compute_ece,
    compute_log_loss,
    murphy_brier_decomposition,
)

# Locked research baseline from baseline_kalman_v6 (MAE / ECE / Brier).
KALMAN_V6_ACTUALS = {
    "label": "baseline_kalman_v6",
    "spread_mae": 11.245676663758646,
    "ece": 0.06668157336253726,
    "brier": 0.2161,
    "market_mae_footnote": 10.43,
    "artifact": "output/baselines/kalman_v6_actuals_baseline.json",
}


def absolute_errors(y_true, y_pred) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return np.abs(p[mask] - y[mask])


def mae(y_true, y_pred) -> float:
    e = absolute_errors(y_true, y_pred)
    return float(e.mean()) if e.size else float("nan")


def compute_spiegelhalter_z(y_true, y_pred_prob) -> dict[str, float]:
    """Global Spiegelhalter Z calibration test.

    ``Z = sum(y - p) / sqrt(sum p(1-p))`` under H0 of perfect calibration.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n = int(len(y))
    if n < 2:
        return {"z": float("nan"), "p_value": float("nan"), "n": n}
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    num = float(np.sum(y - p))
    den = float(np.sqrt(np.sum(p * (1.0 - p))))
    if den <= 0.0 or not np.isfinite(den):
        return {"z": float("nan"), "p_value": float("nan"), "n": n}
    z = num / den
    try:
        from scipy.stats import norm

        p_value = float(norm.sf(abs(z)) * 2.0)
    except Exception:
        from math import erfc, sqrt

        p_value = float(erfc(abs(z) / sqrt(2.0)))
    return {"z": float(z), "p_value": p_value, "n": n}


def compute_decile_z(y_true, y_pred_prob, *, n_bins: int = 10) -> dict[str, Any]:
    """Equal-mass (``pd.qcut``) Spiegelhalter Z per probability decile.

    Never gate bins by a production confidence threshold (selection bias).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n = int(len(y))
    empty = {"bins": {}, "max_abs_z": float("nan"), "n": n, "n_bins": 0}
    if n < n_bins:
        return empty
    try:
        labels = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return empty
    bins_out: dict[str, float] = {}
    max_abs = 0.0
    for b in sorted(pd.Series(labels).dropna().unique()):
        m = labels == b
        z_info = compute_spiegelhalter_z(y[m], p[m])
        z_val = float(z_info["z"]) if np.isfinite(z_info["z"]) else float("nan")
        bins_out[f"bin_{int(b)}"] = z_val
        if np.isfinite(z_val):
            max_abs = max(max_abs, abs(z_val))
    return {
        "bins": bins_out,
        "max_abs_z": float(max_abs) if bins_out else float("nan"),
        "n": n,
        "n_bins": int(len(bins_out)),
    }


def _newey_west_mean_var(d: np.ndarray, lags: int) -> tuple[float, float]:
    """Return (d_bar, HAC variance of the mean) for a 1-d loss-diff series."""
    n = int(len(d))
    d_bar = float(d.mean())
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    lags = max(0, int(lags))
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        var += 2.0 * w * cov
    # variance of the sample mean
    v_mean = max(var, 0.0) / max(n, 1)
    return d_bar, float(v_mean)


def diebold_mariano(
    y_true,
    pred_a,
    pred_b,
    *,
    loss: str = "abs",
    h: int = 1,
    season_series=None,
    min_nba_lags: int = 7,
) -> dict[str, Any]:
    """Diebold–Mariano test: H0 equal predictive accuracy of A vs B.

    Positive ``dm_stat`` means A has *higher* loss than B (A worse).

    When ``season_series`` is provided, Newey–West HAC is computed **per season**
    with ``lags = max(ceil(n_s**(1/3)), min_nba_lags)``, then pooled:

        d_bar = sum(n_s * d_bar_s) / N
        v = sum((n_s / N)**2 * v_s)

    so June→October gaps never enter the lag structure. Without seasons, uses
    the same NBA lag rule on the full series (``h`` retained for back-compat
    only when season_series is None and ``h`` > 1 overrides min lag).
    """
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if season_series is not None:
        seasons = np.asarray(season_series)
        if len(seasons) != len(y):
            raise ValueError("season_series length must match predictions")
        seasons = seasons[mask]
    else:
        seasons = None
    y, a, b = y[mask], a[mask], b[mask]
    n = int(len(y))
    if n < 8:
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_loss_diff": float("nan"),
            "n": n,
            "seasons": [],
        }
    if loss == "abs":
        d = np.abs(a - y) - np.abs(b - y)
    elif loss == "sq":
        d = (a - y) ** 2 - (b - y) ** 2
    else:
        raise ValueError(f"unsupported loss={loss!r}")

    season_rows: list[dict[str, Any]] = []
    if seasons is not None:
        uniq = pd.unique(seasons)
        N = float(n)
        d_bar_pool = 0.0
        v_pool = 0.0
        for s in uniq:
            m = seasons == s
            d_s = d[m]
            n_s = int(len(d_s))
            if n_s < 4:
                continue
            lags = max(int(np.ceil(n_s ** (1.0 / 3.0))), int(min_nba_lags))
            d_bar_s, v_s = _newey_west_mean_var(d_s, lags)
            w = n_s / N
            d_bar_pool += w * d_bar_s
            v_pool += (w ** 2) * v_s
            season_rows.append({
                "season": str(s),
                "n": n_s,
                "lags": lags,
                "d_bar": d_bar_s,
                "v_mean": v_s,
            })
        d_bar = float(d_bar_pool)
        se = float(np.sqrt(max(v_pool, 0.0)))
    else:
        lags = max(int(np.ceil(n ** (1.0 / 3.0))), int(min_nba_lags))
        if int(h) > 1:
            lags = max(lags, int(h) - 1)
        d_bar, v_mean = _newey_west_mean_var(d, lags)
        se = float(np.sqrt(max(v_mean, 0.0)))
        season_rows.append({"season": "all", "n": n, "lags": lags, "d_bar": d_bar, "v_mean": v_mean})

    if se <= 0.0 or not np.isfinite(se):
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "mean_loss_diff": d_bar,
            "n": n,
            "seasons": season_rows,
        }
    dm = d_bar / se
    from math import erfc, sqrt

    p_value = float(erfc(abs(dm) / sqrt(2.0)))
    return {
        "dm_stat": float(dm),
        "p_value": p_value,
        "mean_loss_diff": float(d_bar),
        "n": n,
        "seasons": season_rows,
    }


def pace_efficiency_mae(
    *,
    actual_total: Any,
    pred_pace: Any,
    pred_home_ppp: Any,
    pred_away_ppp: Any,
    actual_pace: Any | None = None,
    actual_home_ppp: Any | None = None,
    actual_away_ppp: Any | None = None,
) -> dict[str, float]:
    """Decompose totals error into pace MAE and efficiency MAE when actuals exist.

    Predicted total ≈ pace × (home_ppp + away_ppp). When actual pace/PPP are
    missing, only ``total_mae`` from the reconstructed prediction is returned.
    """
    at = np.asarray(actual_total, dtype=float)
    pace = np.asarray(pred_pace, dtype=float)
    hppp = np.asarray(pred_home_ppp, dtype=float)
    appp = np.asarray(pred_away_ppp, dtype=float)
    pred_total = pace * (hppp + appp)
    out: dict[str, float] = {
        "total_mae": mae(at, pred_total),
        "pace_mae": float("nan"),
        "eff_mae": float("nan"),
        "n": int(np.isfinite(at).sum()),
    }
    if actual_pace is not None:
        out["pace_mae"] = mae(actual_pace, pace)
    if actual_home_ppp is not None and actual_away_ppp is not None:
        act_eff = np.asarray(actual_home_ppp, dtype=float) + np.asarray(
            actual_away_ppp, dtype=float
        )
        pred_eff = hppp + appp
        out["eff_mae"] = mae(act_eff, pred_eff)
    return out


def actuals_scorecard(
    results: pd.DataFrame,
    *,
    pred_spread_col: str = "PRED_SPREAD",
    actual_margin_col: str = "ACTUAL_MARGIN",
    win_prob_col: str = "HOME_WIN_PROB",
    home_win_col: str = "HOME_WIN",
) -> dict[str, Any]:
    """Primary actuals metrics for a walk-forward results frame."""
    if results is None or results.empty:
        return {
            "spread_mae": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "ece": float("nan"),
            "murphy": {},
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
            "spiegelhalter": {},
            "decile_z": {},
            "n": 0,
        }
    # Column aliases used by simulate.py / suite exports
    if win_prob_col not in results.columns and "WIN_PROB" in results.columns:
        win_prob_col = "WIN_PROB"
    pred = results.get(pred_spread_col)
    act = results.get(actual_margin_col)
    out: dict[str, Any] = {
        "spread_mae": mae(act, pred) if pred is not None and act is not None else float("nan"),
        "n": int(len(results)),
    }
    y = None
    p = None
    if win_prob_col in results.columns:
        p = pd.to_numeric(results[win_prob_col], errors="coerce")
        if home_win_col in results.columns:
            y = pd.to_numeric(results[home_win_col], errors="coerce")
        elif actual_margin_col in results.columns:
            y = (pd.to_numeric(results[actual_margin_col], errors="coerce") > 0).astype(float)
    if y is not None and p is not None:
        out["brier"] = compute_brier(y, p)
        out["log_loss"] = compute_log_loss(y, p)
        out["ece"] = compute_ece(y, p)
        murphy = murphy_brier_decomposition(y, p)
        out["murphy"] = murphy
        out["reliability"] = murphy.get("reliability", float("nan"))
        out["resolution"] = murphy.get("resolution", float("nan"))
        out["uncertainty"] = murphy.get("uncertainty", float("nan"))
        out["spiegelhalter"] = compute_spiegelhalter_z(y, p)
        out["decile_z"] = compute_decile_z(y, p)
    else:
        out["brier"] = float("nan")
        out["log_loss"] = float("nan")
        out["ece"] = float("nan")
        out["murphy"] = {}
        out["reliability"] = float("nan")
        out["resolution"] = float("nan")
        out["uncertainty"] = float("nan")
        out["spiegelhalter"] = {}
        out["decile_z"] = {}
    return out


def compare_actuals_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    *,
    mae_tol: float = 0.25,
    ece_tol: float = 0.02,
    brier_tol: float = 0.01,
) -> tuple[bool, str]:
    """Gate on actuals metrics only (never ATS/ROI/CLV)."""
    base = baseline or KALMAN_V6_ACTUALS
    cur_mae = current.get("spread_mae", current.get("spread_mae_mean"))
    base_mae = base.get("spread_mae", (base.get("stability") or {}).get("spread_mae_mean"))
    cur_ece = current.get("ece", current.get("ece_mean"))
    base_ece = base.get("ece", (base.get("stability") or {}).get("ece_mean"))
    cur_brier = current.get("brier")
    base_brier = base.get("brier")
    msgs: list[str] = []
    ok = True
    if (
        cur_mae is not None and base_mae is not None
        and np.isfinite(float(cur_mae)) and np.isfinite(float(base_mae))
        and float(cur_mae) > float(base_mae) + mae_tol
    ):
        ok = False
        msgs.append(f"spread_mae {cur_mae:.3f} > baseline {base_mae:.3f}+{mae_tol}")
    if (
        cur_ece is not None and base_ece is not None
        and np.isfinite(float(cur_ece)) and np.isfinite(float(base_ece))
        and float(cur_ece) > float(base_ece) + ece_tol
    ):
        ok = False
        msgs.append(f"ece {cur_ece:.4f} > baseline {base_ece:.4f}+{ece_tol}")
    if (
        cur_brier is not None and base_brier is not None
        and np.isfinite(float(cur_brier)) and np.isfinite(float(base_brier))
        and float(cur_brier) > float(base_brier) + brier_tol
    ):
        ok = False
        msgs.append(f"brier {cur_brier:.4f} > baseline {base_brier:.4f}+{brier_tol}")
    if ok:
        return True, "actuals vs locked baseline OK (MAE/ECE/Brier)"
    return False, "; ".join(msgs)


def load_locked_actuals_baseline(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load ``output/baselines/kalman_v6_actuals_baseline.json`` if present."""
    if path is None:
        # code/pipeline -> repo root
        root = Path(__file__).resolve().parents[2]
        path = root / "output" / "baselines" / "kalman_v6_actuals_baseline.json"
    p = Path(path)
    if not p.exists():
        return dict(KALMAN_V6_ACTUALS)
    data = json.loads(p.read_text())
    stab = data.get("stability") or {}
    return {
        "label": "baseline_kalman_v6",
        "spread_mae": stab.get("spread_mae_mean", KALMAN_V6_ACTUALS["spread_mae"]),
        "ece": stab.get("ece_mean", KALMAN_V6_ACTUALS["ece"]),
        "brier": KALMAN_V6_ACTUALS["brier"],
        "market_mae_footnote": KALMAN_V6_ACTUALS["market_mae_footnote"],
        "artifact": str(p),
        "raw": data,
    }
