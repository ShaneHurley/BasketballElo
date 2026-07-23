"""Moneyline diagnostics: calibration, ROI, underdog/favorite value tracking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.diagnostics import _norm_results, BREAKEVEN
from pipeline.config import (
    ML_MAX_FAVORITE_DECIMAL_DEFAULT,
    ML_MIN_EV,
    MIN_ML_WIN_PCT,
)
from pipeline.market import (
    american_to_decimal,
    implied_probability,
    ml_prob_for_side,
    select_ml_bet,
    UNDERDOG_DECIMAL,
)
from pipeline.metrics import walkforward_favorite_decimal


def _ml_outcome(row, side: str) -> int | float:
    if pd.isna(row.get("ACTUAL_HOME")) or pd.isna(row.get("ACTUAL_AWAY")):
        return np.nan
    home_win = float(row["ACTUAL_HOME"]) > float(row["ACTUAL_AWAY"])
    if side == "Home":
        return int(home_win)
    if side == "Away":
        return int(not home_win)
    return np.nan


def enrich_ml_columns(
    df: pd.DataFrame,
    min_ev: float = ML_MIN_EV,
    max_fav_dec: float | None = None,
    min_win_pct: float | None = None,
) -> pd.DataFrame:
    """Add per-game ML EV, best side, dog/fav tag, and flat ROI."""
    out = _norm_results(df)
    if "WIN_PROB" not in out.columns or "MARKET_ML" not in out.columns:
        return out

    if max_fav_dec is None:
        max_fav_dec = ML_MAX_FAVORITE_DECIMAL_DEFAULT
    if min_win_pct is None:
        min_win_pct = float(MIN_ML_WIN_PCT)

    rows = []
    for r in out.itertuples(index=False):
        rd = r._asdict()
        ml = rd.get("MARKET_ML")
        wp = rd.get("WIN_PROB")
        if pd.isna(ml) or pd.isna(wp):
            rows.append({
                "ml_ev_home": np.nan, "ml_ev_away": np.nan,
                "ml_best_ev": np.nan, "ml_best_side": "Pass",
                "ml_predicted_winner": "Pass",
                "ml_bet_decimal": np.nan, "ml_implied_prob": np.nan,
                "ml_value_type": "none", "ml_flat_profit": 0.0,
            })
            continue
        dec_h = american_to_decimal(float(ml))
        dec_a = american_to_decimal(float(-ml))
        p_h = ml_prob_for_side(float(wp), float(ml), "Home")
        p_a = ml_prob_for_side(float(wp), float(ml), "Away")
        ev_h = (p_h * dec_h) - 1.0
        ev_a = (p_a * dec_a) - 1.0
        side, ev, dec = select_ml_bet(
            float(wp), float(ml),
            min_ev=min_ev,
            max_favorite_decimal=max_fav_dec,
            min_win_pct=min_win_pct,
        )
        if side == "Pass":
            vtype = "none"
        elif dec >= UNDERDOG_DECIMAL:
            vtype = "underdog_value"
        else:
            vtype = "favorite_value"
        won = _ml_outcome(rd, side) if side != "Pass" else np.nan
        profit = 0.0
        if side != "Pass" and pd.notna(won):
            profit = (dec - 1.0) if won else -1.0
        bet_model_p = p_h if side == "Home" else (p_a if side == "Away" else np.nan)
        from pipeline.market import ml_predicted_winner_side
        predicted_winner = ml_predicted_winner_side(float(wp))
        rows.append({
            "ml_ev_home": ev_h,
            "ml_ev_away": ev_a,
            "ml_best_ev": ev if side != "Pass" else max(ev_h, ev_a),
            "ml_best_side": side,
            "ml_predicted_winner": predicted_winner,
            "ml_bet_decimal": dec if side != "Pass" else np.nan,
            "ml_implied_prob": (
                implied_probability(float(ml) if side == "Home" else float(-ml))
                if side != "Pass" else np.nan
            ),
            "ml_bet_model_prob": bet_model_p,
            "ml_value_type": vtype,
            "ml_flat_profit": profit,
            "ml_won": won,
        })
    extra = pd.DataFrame(rows, index=out.index)
    return pd.concat([out, extra], axis=1)


def _prob_metrics(g: pd.DataFrame, prob_col: str) -> tuple[float, float]:
    from pipeline.calibration_metrics import compute_brier, compute_ece

    if prob_col not in g.columns:
        return np.nan, np.nan
    hw = (g["ACTUAL_MARGIN"] > 0).astype(int)
    wp_col = g[prob_col].dropna()
    if wp_col.empty:
        return np.nan, np.nan
    hw_aligned = hw.loc[wp_col.index]
    return (
        float(compute_brier(hw_aligned.values, wp_col.values)),
        float(compute_ece(hw_aligned.values, wp_col.values)),
    )


def _reliability_plot(ax, g: pd.DataFrame, prob_col: str, label: str, color: str):
    sub = g.dropna(subset=[prob_col]).copy()
    if sub.empty:
        return
    sub["home_win"] = (sub["ACTUAL_MARGIN"] > 0).astype(int)
    bins = np.linspace(0.2, 0.85, 14)
    sub["pb"] = pd.cut(sub[prob_col], bins=bins)
    cal = sub.groupby("pb", observed=True)["home_win"].mean()
    centers = [iv.mid for iv in cal.index]
    ax.scatter(centers, cal.values, s=50, color=color, label=label, zorder=3)


def run_ml_diagnostics(
    results_df: pd.DataFrame,
    *,
    min_ev: float = ML_MIN_EV,
    max_fav_dec: float | None = None,
    min_win_pct: float | None = None,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
) -> pd.DataFrame:
    """Win-prob calibration, ML ROI by season, dog vs fav split."""
    if results_df is None or results_df.empty:
        print("No results for ML diagnostics.")
        return pd.DataFrame()

    if max_fav_dec is None:
        max_fav_dec = walkforward_favorite_decimal(results_df)
    if min_win_pct is None:
        min_win_pct = float(MIN_ML_WIN_PCT)

    df = enrich_ml_columns(
        results_df, min_ev=min_ev, max_fav_dec=max_fav_dec, min_win_pct=min_win_pct,
    )
    save_path = Path(save_dir) if save_dir else None
    has_raw = "WIN_PROB_RAW" in df.columns

    print("\n" + "=" * 72)
    print("MONEYLINE DIAGNOSTICS (MetaWin / WIN_PROB vs market)")
    print("=" * 72)
    from pipeline.config import (
        ML_BET_PREDICTED_WINNER_ONLY,
        ML_COIN_FLIP_BAND,
        ML_COIN_FLIP_UNDERDOG_MIN_EV,
        ML_COIN_FLIP_UNDERDOG_ONLY,
        ML_MIN_EV,
        ML_CALIB_METHOD,
    )
    if ML_BET_PREDICTED_WINNER_ONLY:
        winner_mode = (
            f"predicted winner if EV>{ML_MIN_EV:.0%} and win%>={min_win_pct:.0f}; "
            f"coin-flip underdog={'on' if ML_COIN_FLIP_UNDERDOG_ONLY else 'off'}"
        )
    else:
        winner_mode = f"best calibrated-EV side if EV>{ML_MIN_EV:.0%} and win%>={min_win_pct:.0f}"
    print(f"  bet policy={winner_mode}  min_ev={min_ev:.1%}  min_win%={min_win_pct:.0f}")
    print(f"  max_favorite_decimal={max_fav_dec:.2f}  ml_calib={ML_CALIB_METHOD}")
    from pipeline.config import ML_MARKET_SHRINK, ML_UNDERDOG_EXTRA_SHRINK
    print(f"  market_shrink={ML_MARKET_SHRINK:.2f}  underdog_extra_shrink={ML_UNDERDOG_EXTRA_SHRINK:.2f}")

    seasons = sorted(df["_season"].unique())
    summary_rows = []

    for label, g in list(zip(seasons, [df[df["_season"] == s] for s in seasons])) + [("ALL", df)]:
        brier, ece = _prob_metrics(g, "WIN_PROB")
        raw_brier, raw_ece = _prob_metrics(g, "WIN_PROB_RAW") if has_raw else (np.nan, np.nan)
        ml_acc = float(g["MODEL_ML_CORRECT"].mean()) if "MODEL_ML_CORRECT" in g.columns else np.nan
        if has_raw and "WIN_PROB_RAW" in g.columns:
            raw_acc = float(
                ((g["WIN_PROB_RAW"] > 0.5) == (g["ACTUAL_MARGIN"] > 0)).mean()
            )
        else:
            raw_acc = np.nan
        bets = g[g["ml_best_side"] != "Pass"]
        n_ml = len(bets)
        n_games = len(g)
        wp = bets["ml_won"].mean() if n_ml else np.nan
        roi = bets["ml_flat_profit"].mean() if n_ml else np.nan
        dog = bets[bets["ml_value_type"] == "underdog_value"]
        fav = bets[bets["ml_value_type"] == "favorite_value"]
        dog_pred = float(dog["ml_bet_model_prob"].mean()) if len(dog) else np.nan
        dog_obs = float(dog["ml_won"].mean()) if len(dog) else np.nan
        fav_pred = float(fav["ml_bet_model_prob"].mean()) if len(fav) else np.nan
        fav_obs = float(fav["ml_won"].mean()) if len(fav) else np.nan
        summary_rows.append({
            "season": label,
            "games": n_games,
            "raw_brier": raw_brier,
            "brier": brier,
            "raw_ece": raw_ece,
            "ece": ece,
            "raw_winner_acc": raw_acc,
            "ml_winner_acc": ml_acc,
            "ml_bets": n_ml,
            "ml_bet_rate": n_ml / n_games if n_games else np.nan,
            "ml_win_pct": wp,
            "ml_roi": roi,
            "underdog_bets": len(dog),
            "underdog_share": len(dog) / n_ml if n_ml else np.nan,
            "underdog_pred_win": dog_pred,
            "underdog_win_pct": dog_obs,
            "underdog_cal_gap": (dog_pred - dog_obs) if pd.notna(dog_pred) and pd.notna(dog_obs) else np.nan,
            "underdog_roi": dog["ml_flat_profit"].mean() if len(dog) else np.nan,
            "favorite_bets": len(fav),
            "favorite_pred_win": fav_pred,
            "favorite_win_pct": fav_obs,
            "favorite_cal_gap": (fav_pred - fav_obs) if pd.notna(fav_pred) and pd.notna(fav_obs) else np.nan,
            "favorite_roi": fav["ml_flat_profit"].mean() if len(fav) else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    print(summary.round(3).to_string(index=False))
    print(
        "\n  Note: promote models on Brier/ECE/ROI — raw_winner_acc is reference only."
    )

    # Both-sides EV: predicted winner vs actual bet side
    if "ml_best_side" in df.columns and "WIN_PROB" in df.columns:
        from pipeline.market import ml_predicted_winner_side
        tmp = df.copy()
        tmp["_pred_win"] = tmp["WIN_PROB"].map(ml_predicted_winner_side)
        bets = tmp[tmp["ml_best_side"] != "Pass"]
        if len(bets):
            mismatch = bets[bets["ml_best_side"] != bets["_pred_win"]]
            fav_pass = tmp[
                (tmp["_pred_win"].isin(["Home", "Away"]))
                & (tmp["ml_best_side"] == "Pass")
            ]
            print(
                f"\n  Both-sides EV: {len(mismatch)}/{len(bets)} bets on non-predicted winner "
                f"(dog/value); {len(fav_pass)} predicted-winner Pass (ev too low / short price)."
            )

    if show_plots and "WIN_PROB" in df.columns:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        g = df.dropna(subset=["WIN_PROB"])
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        if has_raw:
            _reliability_plot(ax, g, "WIN_PROB_RAW", "Raw WIN_PROB", "#C44E52")
        _reliability_plot(ax, g, "WIN_PROB", "Calibrated WIN_PROB", "#4C72B0")
        ax.set_xlabel("Model P(home win)")
        ax.set_ylabel("Actual home win rate")
        ax.set_title("ML Win-Probability Reliability", fontweight="bold")
        ax.legend()
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "13_ml_winprob_reliability.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {save_path / '13_ml_winprob_reliability.png'}")
        else:
            plt.show()

    if save_path:
        summary.to_csv(save_path / "ml_diagnostics_summary.csv", index=False)

    return summary


def ml_value_plays(
    results_df: pd.DataFrame,
    *,
    min_ev: float = ML_MIN_EV,
    max_fav_dec: float | None = None,
    min_win_pct: float | None = None,
    value_types: tuple[str, ...] = ("underdog_value", "favorite_value"),
) -> pd.DataFrame:
    """Games flagged as underdog value or undervalued favorite."""
    if max_fav_dec is None:
        max_fav_dec = ML_MAX_FAVORITE_DECIMAL_DEFAULT
    if "ml_value_type" in results_df.columns:
        df = _norm_results(results_df)
    else:
        df = enrich_ml_columns(
            results_df, min_ev=min_ev, max_fav_dec=max_fav_dec, min_win_pct=min_win_pct,
        )
    plays = df[
        df["ml_value_type"].isin(value_types) & (df["ml_best_side"] != "Pass")
    ].copy()
    cols = [
        "DATE", "date", "GAME_ID", "HOME", "AWAY", "_season",
        "WIN_PROB", "WIN_PROB_RAW", "MARKET_ML", "ml_best_side", "ml_best_ev",
        "ml_bet_decimal", "ml_implied_prob", "ml_bet_model_prob", "ml_value_type",
        "ml_won", "ml_flat_profit",
        "ML_DIRECTION", "ML_EV", "CONFIDENCE", "CONFIDENCE_TIER",
        "ACTUAL_HOME", "ACTUAL_AWAY",
    ]
    keep = [c for c in cols if c in plays.columns]
    sort_col = "DATE" if "DATE" in plays.columns else ("date" if "date" in plays.columns else None)
    out = plays[keep]
    if sort_col:
        out = out.sort_values([sort_col, "ml_best_ev"], ascending=[True, False])
    return out


def export_ml_tracking(
    results_df: pd.DataFrame,
    path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """Write underdog/favorite value plays to CSV for tracking."""
    plays = ml_value_plays(results_df, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plays.to_csv(path, index=False)
    n_dog = int((plays["ml_value_type"] == "underdog_value").sum()) if not plays.empty else 0
    n_fav = int((plays["ml_value_type"] == "favorite_value").sum()) if not plays.empty else 0
    print(f"ML tracking export → {path}  ({len(plays)} plays: {n_dog} dogs, {n_fav} favorites)")
    return plays
