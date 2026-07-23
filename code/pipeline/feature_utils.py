"""Shared feature helpers (no circular imports)."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd


def schedule_density(date_history, gdate):
    """Games in last 7 days and 3-in-4 nights flag."""
    from pipeline.dates import is_valid_timestamp

    if not is_valid_timestamp(gdate) or not date_history:
        return 0.0, 0
    gdate = pd.Timestamp(gdate)
    last7 = 0
    last3 = 0
    for d in date_history:
        if not is_valid_timestamp(d):
            continue
        diff = (gdate - pd.Timestamp(d)).days
        if 0 < diff <= 7:
            last7 += 1
        if 0 < diff <= 3:
            last3 += 1
    return float(last7), int(last3 >= 2)


def schedule_density_extended(date_history, gdate):
    """Extended fatigue schedule: last7, 3-in-4, 4-in-6 flags."""
    from pipeline.dates import is_valid_timestamp

    if not is_valid_timestamp(gdate) or not date_history:
        return 0.0, 0, 0
    gdate = pd.Timestamp(gdate)
    last7 = 0
    last3 = 0
    last5 = 0
    for d in date_history:
        if not is_valid_timestamp(d):
            continue
        diff = (gdate - pd.Timestamp(d)).days
        if 0 < diff <= 7:
            last7 += 1
        if 0 < diff <= 3:
            last3 += 1
        if 0 < diff <= 5:
            last5 += 1
    three_in_four = int(last3 >= 2)
    four_in_six = int(last5 >= 3)
    return float(last7), three_in_four, four_in_six


def rest_bucket_flags(rest_days: int) -> dict:
    """One-hot rest buckets: 0 / 1 / 2 / 3+ days."""
    r = int(rest_days) if rest_days is not None else 3
    return {
        "rest_0": int(r <= 0),
        "rest_1": int(r == 1),
        "rest_2": int(r == 2),
        "rest_3plus": int(r >= 3),
    }


def engine_implied_margins(elo_tracker, hier_engine,
                           ho_off, ho_def, ao_off, ao_def,
                           home_starters, away_starters, exp_poss):
    cfg = getattr(elo_tracker, "cfg", {}) or {}
    scaling = cfg.get("ELO_SCALING_FACTOR", 1000) or 1000
    hb = cfg.get("HOME_PPP_BOOST", 0.024)
    lx = getattr(elo_tracker, "league_xppp", 1.10)
    exp_ppp_h = lx + hb + (ho_off - ao_def) / scaling
    exp_ppp_a = lx - hb + (ao_off - ho_def) / scaling
    elo_margin = (exp_ppp_h - exp_ppp_a) * float(exp_poss or 0.0)
    try:
        hph, hpa, _, _ = hier_engine.predict_pts(home_starters, away_starters, exp_poss)
        hier_margin = float(hph - hpa)
    except Exception:
        hier_margin = 0.0
    return float(elo_margin), float(hier_margin)


def _top_lineup(weighted_ids, n=5):
    if not weighted_ids:
        return []
    ranked = sorted(weighted_ids, key=lambda x: -x[1])[:n]
    return [p for p, _ in ranked]
