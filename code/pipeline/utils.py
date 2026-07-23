"""Shared helpers."""
import math

import numpy as np
import pandas as pd

from pipeline.config import name_to_id


def _coerce_int(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _parse_player_string(s):
    """Parse lineup string entries (numeric IDs or player names)."""
    if pd.isna(s) or str(s).strip() == "" or str(s).lower() == "nan":
        return []

    s_str = str(s).strip()

    if any(c.isalpha() for c in s_str) or "," in s_str:
        tokens = [t.strip() for t in s_str.replace("-", ",").split(",") if t.strip()]
        processed_ids = []
        for token in tokens:
            if token.lower() == "nan":
                continue
            if token in name_to_id:
                processed_ids.append(int(name_to_id[token]))
            else:
                processed_ids.append(abs(hash(token)) % 1000000)
        return processed_ids

    clean_str = s_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    for d in ["-", ",", " "]:
        if d in clean_str:
            return [int(float(t.strip())) for t in clean_str.split(d) if t.strip() and t.lower() != "nan"]
    try:
        return [int(float(clean_str))]
    except ValueError:
        return []


def _map_player_name(raw, mapping=None):
    """Resolve 'id/Name' or 'Full Name' to integer player-id."""
    mapping = mapping if mapping is not None else name_to_id
    if pd.isna(raw) or not str(raw).strip():
        return None
    s = str(raw).strip()
    if "/" in s:
        try:
            return int(s.split("/")[0])
        except ValueError:
            s = s.split("/")[-1]
    pid = mapping.get(s)
    return int(pid) if pid is not None else (hash(s) % 10**9)


def map_elo_params(optuna_params: dict) -> dict:
    """Map Optuna Elo tuner keys to PlayerRatingTracker config keys."""
    return {
        "K_OFF": optuna_params["k_off"],
        "K_DEF": optuna_params["k_def"],
        "ELO_SCALING_FACTOR": optuna_params["elo_scaling"],
        "HOME_PPP_BOOST": optuna_params["home_boost"],
        "OFFSEASON_REVERSION": optuna_params["offseason_reversion"],
        "USAGE_FLOOR": optuna_params["usage_floor"],
        "assist_split": optuna_params["assist_split"],
        "k_mult_half_life": optuna_params.get("k_mult_half_life", 15.0),
        "rd_floor": optuna_params.get("rd_floor", 30.0),
        "garbage_time_weight": optuna_params.get("garbage_time_weight", 0.5),
        "clutch_boost": optuna_params.get("clutch_boost", 1.3),
        "tov_penalty": optuna_params.get("tov_penalty", 0.85),
        "foul_draw_boost": optuna_params.get("foul_draw_boost", 1.1),
        "variance_dampen": optuna_params.get("variance_dampen", 0.9),
        "xppp_actual_blend": optuna_params.get("xppp_actual_blend", 0.05),
        "k_def_events": optuna_params.get("k_def_events", 0.15),
        "tov_rate_threshold": optuna_params.get("tov_rate_threshold", 0.15),
        "three_pa_rate_threshold": optuna_params.get("three_pa_rate_threshold", 0.45),
    }
