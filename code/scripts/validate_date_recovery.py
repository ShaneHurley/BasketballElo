"""Validate the GAME_ID->date matcher on the 2025-26 file (which HAS real dates).

We strip the true dates, recover them from all_odds.csv via the matcher, then
compare recovered vs truth. If accurate here, it will work for the V3 seasons.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shurley/Documents/basketball")
sys.path.insert(0, str(ROOT))
from pipeline.dates import load_odds_schedule, build_gameid_date_map, to_tricode

COMBINED = ROOT / "[10-21-2025]-[05-18-2026]-combined-stats.csv"

# 1. Truth: one row per game (game_id, home, away, true_date)
print("Loading 2025-26 combined stats (game-level columns only)...")
cols = pd.read_csv(COMBINED, nrows=1).columns.tolist()
use = [c for c in ["game_id", "date", "home_team", "away_team"] if c in cols]
raw = pd.read_csv(COMBINED, usecols=use, low_memory=False)
print("  sample teams:", raw["home_team"].dropna().unique()[:6].tolist())
truth = raw.drop_duplicates("game_id").copy()
truth["true_date"] = pd.to_datetime(truth["date"], errors="coerce").dt.normalize()
truth["home"] = truth["home_team"].apply(to_tricode)
truth["away"] = truth["away_team"].apply(to_tricode)
truth = truth.rename(columns={"game_id": "GAME_ID"})
print(f"  unique games: {len(truth)} | true date range "
      f"{truth.true_date.min().date()} -> {truth.true_date.max().date()}")

# 2. Recover dates with the matcher (season start year 2025 -> 2025-26)
sched = load_odds_schedule(str(ROOT / "all_odds.csv"))
print(f"  odds schedule rows for 2025-26: {(sched.season == 2026).sum()}")
date_map = build_gameid_date_map(
    truth[["GAME_ID", "home", "away"]], sched, season_start_year=2025)

truth["rec_date"] = truth["GAME_ID"].map(date_map)

# 3. Evaluate
ev = truth.dropna(subset=["true_date", "rec_date"]).copy()
ev["err_days"] = (ev["rec_date"].dt.normalize() - ev["true_date"]).dt.days.abs()
print("\n================ DATE RECOVERY ACCURACY ================")
print(f"  games evaluated      : {len(ev)}")
print(f"  exact date match     : {(ev['err_days'] == 0).mean():.1%}")
print(f"  within 1 day         : {(ev['err_days'] <= 1).mean():.1%}")
print(f"  within 3 days        : {(ev['err_days'] <= 3).mean():.1%}")
print(f"  median |err| days    : {ev['err_days'].median():.1f}")
print(f"  mean   |err| days    : {ev['err_days'].mean():.2f}")

# Note: odds only cover through ~2026-02-12, so later games are interpolated.
covered = ev[ev["true_date"] <= sched[sched.season == 2026]["date"].max()]
print(f"\n  Games in odds-covered window (<= {sched[sched.season==2026]['date'].max().date()}): {len(covered)}")
print(f"    exact match in covered window: {(covered['err_days'] == 0).mean():.1%}")
print(f"    within 1 day  in covered window: {(covered['err_days'] <= 1).mean():.1%}")

# Chronological ordering correctness (Spearman-like: does rec order == true order?)
ev2 = ev.sort_values("GAME_ID")
order_ok = (ev2["rec_date"].rank() .corr(ev2["true_date"].rank()))
print(f"\n  ordering corr (recovered vs true): {order_ok:.4f}  (1.0 = perfect)")
