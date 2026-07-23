"""Empirical leak diagnostics on backtest_results.csv.

The walk-forward code looks leak-free, but a 62.9% ATS rate at a 1.5pt edge is
statistically implausible for a pure-ratings model. This script tries to
distinguish a real edge from leakage / data-quality artifacts.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np
import pandas as pd

CSV = "/Users/shurley/Documents/basketball/backtest_results.csv"
df = pd.read_csv(CSV)
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df["season"] = df["simulated_season_window"]

print("=" * 70)
print("ROWS:", len(df), "| seasons:", df["season"].unique().tolist())
print("Columns:", list(df.columns))

mkt = df[df["MARKET_SPREAD"].notna()].copy()
print(f"\nGames with a market line: {len(mkt)} / {len(df)}")

# ----------------------------------------------------------------------
# 1. Is the MARKET itself well-calibrated? (sanity check the odds source)
#    A real Vegas line: actual_margin ~ -market_spread, home cover ~50%.
# ----------------------------------------------------------------------
mkt["home_cover"] = (mkt["ACTUAL_MARGIN"] + mkt["MARKET_SPREAD"]) > 0
print("\n[1] MARKET sanity")
print(f"   Home ATS cover rate (all mkt games): {mkt['home_cover'].mean():.3f}  (expect ~0.50)")
print(f"   mean ACTUAL_MARGIN : {mkt['ACTUAL_MARGIN'].mean():.2f}")
print(f"   mean -MARKET_SPREAD: {(-mkt['MARKET_SPREAD']).mean():.2f}  (expect close to actual)")
# correlation of market implied margin with actual
corr_mkt = np.corrcoef(-mkt["MARKET_SPREAD"], mkt["ACTUAL_MARGIN"])[0, 1]
print(f"   corr(-MARKET_SPREAD, ACTUAL_MARGIN): {corr_mkt:.3f}")
# market MAE
mkt_mae = ((-mkt["MARKET_SPREAD"]) - mkt["ACTUAL_MARGIN"]).abs().mean()
print(f"   MARKET spread MAE vs actual: {mkt_mae:.2f}  (real Vegas ~ 9-10)")

# ----------------------------------------------------------------------
# 2. Model accuracy
# ----------------------------------------------------------------------
print("\n[2] MODEL accuracy")
model_mae = (mkt["PRED_SPREAD"] - mkt["ACTUAL_MARGIN"]).abs().mean()
corr_model = np.corrcoef(mkt["PRED_SPREAD"], mkt["ACTUAL_MARGIN"])[0, 1]
print(f"   MODEL spread MAE vs actual: {model_mae:.2f}")
print(f"   corr(PRED_SPREAD, ACTUAL_MARGIN): {corr_model:.3f}")

# ----------------------------------------------------------------------
# 3. THE KEY TEST.
#    Edge = PRED_SPREAD - (-MARKET_SPREAD) = how much model disagrees w/ mkt.
#    If NO leak: knowing the model's disagreement should NOT predict which
#    side beats the *closing-ish* market line better than ~52-53%.
#    If model truly beats market, then market is "soft": model_resid should be
#    much smaller than market_resid on the games we bet.
# ----------------------------------------------------------------------
print("\n[3] EDGE diagnostics")
mkt["model_implied"] = mkt["PRED_SPREAD"]
mkt["mkt_implied"] = -mkt["MARKET_SPREAD"]
mkt["edge_signed"] = mkt["model_implied"] - mkt["mkt_implied"]   # >0 -> model likes home
print(f"   corr(EDGE col, edge_signed recomputed): "
      f"{np.corrcoef(mkt['EDGE'].fillna(0), mkt['edge_signed'])[0,1]:.3f}")

active = mkt[mkt["DIRECTION"] != "Pass"].copy()
active["ats_win"] = (
    ((active["DIRECTION"] == "Home") & ((active["ACTUAL_MARGIN"] + active["MARKET_SPREAD"]) > 0))
    | ((active["DIRECTION"] == "Away") & ((active["ACTUAL_MARGIN"] + active["MARKET_SPREAD"]) < 0))
)
print(f"   Active bets: {len(active)} | ATS win%: {active['ats_win'].mean():.3f}")

# On bet games, compare model error vs market error
active["model_err"] = (active["PRED_SPREAD"] - active["ACTUAL_MARGIN"]).abs()
active["mkt_err"] = ((-active["MARKET_SPREAD"]) - active["ACTUAL_MARGIN"]).abs()
print(f"   On BET games  -> MODEL MAE {active['model_err'].mean():.2f} | MARKET MAE {active['mkt_err'].mean():.2f}")
print(f"   Model beats market on {(active['model_err'] < active['mkt_err']).mean():.3f} of bet games")

# ----------------------------------------------------------------------
# 4. The smoking gun for LEAKAGE: does the SIGN of the model's residual
#    relative to the market correlate with actual outcome impossibly well
#    even though absolute MAE is normal?
#    Bucket by |edge| and look at ATS win%.
# ----------------------------------------------------------------------
print("\n[4] ATS win% by |edge| bucket (active bets)")
for lo, hi in [(0, 1.5), (1.5, 3), (3, 5), (5, 8), (8, 100)]:
    b = active[(active["EDGE"].abs() >= lo) & (active["EDGE"].abs() < hi)]
    if len(b):
        print(f"   edge [{lo:>4},{hi:>4}): n={len(b):4d}  ATS={b['ats_win'].mean():.3f}")

# ----------------------------------------------------------------------
# 5. Compare market line to a "no-vig fair" check: is MARKET_SPREAD possibly
#    contaminated by the result? If market is *too* accurate (MAE << 9),
#    odds data may be post-hoc / closing+. If market MAE is normal but model
#    still wins big, suspect leak in PRED.
# ----------------------------------------------------------------------
print("\n[5] Per-season market vs model MAE and ATS")
for s, g in mkt.groupby("season"):
    a = g[g["DIRECTION"] != "Pass"]
    aw = (
        ((a["DIRECTION"] == "Home") & ((a["ACTUAL_MARGIN"] + a["MARKET_SPREAD"]) > 0))
        | ((a["DIRECTION"] == "Away") & ((a["ACTUAL_MARGIN"] + a["MARKET_SPREAD"]) < 0))
    ).mean() if len(a) else np.nan
    mm = (g["PRED_SPREAD"] - g["ACTUAL_MARGIN"]).abs().mean()
    km = ((-g["MARKET_SPREAD"]) - g["ACTUAL_MARGIN"]).abs().mean()
    print(f"   {s}: n_mkt={len(g):4d} | model_MAE={mm:5.2f} | mkt_MAE={km:5.2f} | "
          f"bets={len(a):4d} | ATS={aw:.3f}")

# ----------------------------------------------------------------------
# 6. Check if PRED_SPREAD ~ ACTUAL too tightly conditional on direction
#    (a leak would show pred correlating w/ actual *beyond* market info).
#    Regress actual on [market_implied, model_implied]; if model coef is large
#    & significant, model adds real (or leaked) info beyond market.
# ----------------------------------------------------------------------
print("\n[6] Incremental info of model beyond market (OLS)")
import numpy as np
X = np.column_stack([
    np.ones(len(mkt)),
    mkt["mkt_implied"].values,
    mkt["model_implied"].values,
])
y = mkt["ACTUAL_MARGIN"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"   intercept={beta[0]:.3f}  market_coef={beta[1]:.3f}  model_coef={beta[2]:.3f}")
print("   (If market is sharp & model leak-free, model_coef ~ 0. "
      "Large positive model_coef => model has real or leaked signal.)")

# residual diagnostic: does model_implied predict (actual - market_implied)?
resid = mkt["ACTUAL_MARGIN"] - mkt["mkt_implied"]
c = np.corrcoef(mkt["edge_signed"], resid)[0, 1]
print(f"   corr(model_edge, actual-market): {c:.3f}  "
      "(real ~0.05-0.15; >0.3 strongly suggests leak)")
