"""Generate analysis graphs for walk-forward backtest results.

Uses mode-aware spread frames (ACTIONABLE in confidence_only, DIRECTION in edge modes).
Prefer ``pipeline.diagnostics.run_backtest_diagnostics`` for the canonical report;
this module adds legacy comparison plots and a thin wrapper around diagnostics.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.bet_selection import actionable_spread_frame, lean_spread_frame, uses_edge_gates, win_pct_column
from pipeline.config import BET_SELECTION_MODE, ROOT as REPO_ROOT

plt.rcParams.update({
    "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})

BREAKEVEN = 0.524


def _resolve_root(root: Path | None = None) -> Path:
    return Path(root) if root is not None else REPO_ROOT


def ats_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Actionable spread bets with ATS win flags."""
    m = actionable_spread_frame(d)
    if m.empty:
        return m
    side = m.get("DIRECTION")
    if side is None or side.isna().all():
        from pipeline.bet_selection import spread_side_series
        side = spread_side_series(m)
    cover = m["ACTUAL_MARGIN"] + m["MARKET_SPREAD"]
    m = m.copy()
    m["ats_win"] = (
        ((side == "Home") & (cover > 0))
        | ((side == "Away") & (cover < 0))
    ).astype(int)
    m["ats_push"] = (cover == 0).astype(int)
    return m


def lean_ats_frame(d: pd.DataFrame) -> pd.DataFrame:
    """All model leans (includes sub-threshold edges)."""
    from pipeline.diagnostics import _ats_lean_frame
    return _ats_lean_frame(d)


def ml_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Moneyline bets using ML_DIRECTION with profit at the line."""
    m = d[d["MARKET_ML"].notna() & (d["ML_DIRECTION"] != "Pass")].copy()
    if m.empty:
        return m

    def dec(o):
        o = float(o)
        return 1 + o / 100.0 if o > 0 else 1 + 100.0 / abs(o)

    profits, win = [], []
    for _, r in m.iterrows():
        ml = r["MARKET_ML"]
        home_win = r["ACTUAL_HOME"] > r["ACTUAL_AWAY"]
        if r["ML_DIRECTION"] == "Home":
            d_odds = dec(ml)
            won = home_win
        else:
            d_odds = dec(-ml)
            won = not home_win
        profits.append((d_odds - 1) if won else -1.0)
        win.append(int(won))
    m["ml_profit"] = profits
    m["ml_win"] = win
    return m


def run_analysis(
    df: pd.DataFrame | None = None,
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Build core ATS / confidence / O/U plots from a results dataframe."""
    root = _resolve_root(root)
    out = out_dir or (root / "analysis_plots")
    out.mkdir(parents=True, exist_ok=True)

    if df is None:
        csv = root / "backtest_results.csv"
        if not csv.exists():
            raise FileNotFoundError(f"No backtest_results.csv at {csv}")
        df = pd.read_csv(csv)
    df = df.copy()
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["season"] = df.get("simulated_season_window", df.get("season", ""))
    seasons = sorted(df["season"].dropna().unique())
    palette = {s: c for s, c in zip(seasons, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])}

    ats = ats_frame(df)
    lean = lean_ats_frame(df)
    bet_label = "Actionable ATS" if not uses_edge_gates() else "Active spread bets"

    # 1. ATS win% by season
    fig, ax = plt.subplots(figsize=(9, 5.5))
    rows = []
    for s in seasons:
        g = ats[(ats["season"] == s) & (ats["ats_push"] == 0)]
        if len(g):
            rows.append((s, g["ats_win"].mean(), len(g)))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ns = [r[2] for r in rows]
    bars = ax.bar(labels, vals, color="#55A868", edgecolor="black", alpha=0.85)
    ax.axhline(BREAKEVEN, ls="--", color="black", lw=1.3, label=f"Break-even = {BREAKEVEN:.1%}")
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.1%}\n(n={n})", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.45, 0.82)
    ax.set_ylabel(f"{bet_label} win rate")
    ax.set_title(f"{bet_label} by Season ({BET_SELECTION_MODE})", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_ats_winrate_by_season.png", dpi=150)
    plt.close(fig)

    # 2. Lean vs actionable counts
    if not uses_edge_gates() and "ACTIONABLE" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        counts = []
        for s in seasons:
            g = df[df["season"] == s]
            counts.append((s, int((g["EDGE"].abs() > 0).sum()), int((g["ACTIONABLE"] == 1).sum())))
        x = np.arange(len(counts))
        w = 0.35
        ax.bar(x - w / 2, [c[1] for c in counts], w, label="All leans", color="#4C72B0")
        ax.bar(x + w / 2, [c[2] for c in counts], w, label="Actionable", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels([c[0] for c in counts], rotation=15)
        ax.set_title("Lean vs Actionable Games by Season", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "02_lean_vs_actionable.png", dpi=150)
        plt.close(fig)

    # 3. WIN_PCT deciles (confidence_only)
    score_col = win_pct_column(df)
    if score_col in df.columns and not lean.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        lean_scored = lean[lean["ats_push"] == 0].copy()
        if score_col in lean_scored.columns:
            lean_scored["_score"] = pd.to_numeric(lean_scored[score_col], errors="coerce")
            lean_scored = lean_scored[lean_scored["_score"].notna()]
            lean_scored["decile"] = pd.qcut(lean_scored["_score"], 10, duplicates="drop")
            g = lean_scored.groupby("decile", observed=True)["ats_win"].agg(["mean", "count"])
            ax.bar(range(len(g)), g["mean"], color="#4C72B0", edgecolor="black")
            ax.axhline(BREAKEVEN, ls="--", color="black")
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels([f"{iv.left:.0f}-{iv.right:.0f}" for iv in g.index], rotation=45)
            ax.set_title(f"ATS Win% by {score_col} Decile (all leans)", fontweight="bold")
            fig.tight_layout()
            fig.savefig(out / "03_winpct_deciles.png", dpi=150)
            plt.close(fig)

    # 4. O/U by total-edge bucket
    from pipeline.diagnostics import _ou_edge_bucket_table
    ou_tbl = _ou_edge_bucket_table(df)
    if not ou_tbl.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(ou_tbl["bucket"].astype(str), ou_tbl["ou_win_pct"], color="#DD8452", edgecolor="black")
        ax.axhline(BREAKEVEN, ls="--", color="black")
        ax.set_title("O/U Win% by |TOTAL_EDGE| Bucket", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "04_ou_by_edge_bucket.png", dpi=150)
        plt.close(fig)

    print("Saved plots to", out)
    for p in sorted(out.glob("*.png")):
        print("  ", p.name)
    return out


def generate_betting_plots(df, OUT=None, save_dir=None):
    """Bankroll curves, ROI by confidence tier — delegates to diagnostics."""
    from pipeline.diagnostics import generate_betting_plots as _gen
    target = save_dir or OUT
    return _gen(df, save_dir=target)


if __name__ == "__main__":
    run_analysis()
