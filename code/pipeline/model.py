"""Meta-model and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

# ═══════════════════════════════════════════════════════════════════════════
#   ADAPTIVE SHRINKING TUNER — with plateau and exploration jitter
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveParameterBounds:
    """
    Tracks the history of best Optuna values across seasons and produces
    progressively narrower search ranges centered on the running best.
    Includes plateau (stops shrinking after N seasons) and occasional jitter
    to force exploration and avoid premature convergence.
    """
    DECAY_RATE = 0.85              # slower decay (was 0.75)
    MIN_RANGE_FRACTION = 0.25      # keep at least 25% of original range (was 10%)
    MIN_RANGE_FRACTION_TIGHT = 0.35  # when a param hits bounds repeatedly
    PLATEAU_AFTER = 5              # stop shrinking after this many seasons
    INITIAL_SHRINK = 0.50          # after first season, use ±50% of original range
    EXPLORE_JITTER = 0.05          # random ±5% jitter around the center (applied 30% of the time)

    # Optuna search names → stored config keys (Elo/Hier rename on return).
    PARAM_ALIASES = {
        "k_off": ("K_OFF",),
        "k_def": ("K_DEF",),
        "elo_scaling": ("ELO_SCALING_FACTOR",),
        "home_boost": ("HOME_PPP_BOOST",),
        "offseason_reversion": ("OFFSEASON_REVERSION",),
        "usage_floor": ("USAGE_FLOOR",),
        "k_mult_half_life": ("k_mult_half_life",),
        "rd_floor": ("rd_floor",),
        "garbage_time_weight": ("garbage_time_weight",),
        "clutch_boost": ("clutch_boost",),
        "tov_penalty": ("tov_penalty",),
        "foul_draw_boost": ("foul_draw_boost",),
        "variance_dampen": ("variance_dampen",),
        "xppp_actual_blend": ("xppp_actual_blend",),
        "k_def_events": ("k_def_events",),
        "league_rtg": ("league_avg_rtg",),
        "elo_blend_alpha": ("elo_blend_alpha",),
        "elo_ridge_alpha": ("elo_ridge_alpha",),
    }
    # Categorical / non-numeric params — never used as shrink centers.
    NON_ADAPTIVE_PARAMS = frozenset({"margin_cap", "total_elo_beta"})

    def __init__(self):
        self.history = {}          # {season_int: {"param_name": best_value, ...}}
        self.seasons_seen = 0
        self._bound_hits: dict[str, int] = {}

    @classmethod
    def _sanitize_params(cls, best_params: dict) -> dict:
        cleaned = {}
        for key, value in best_params.items():
            if key in cls.NON_ADAPTIVE_PARAMS or value is None:
                continue
            if isinstance(value, (int, float)) and not np.isfinite(value):
                continue
            cleaned[key] = value
        return cleaned

    def record_best(self, season: int, best_params: dict, param_bounds: dict | None = None):
        cleaned = self._sanitize_params(best_params)
        if season in self.history:
            self.history[season].update(cleaned)
        else:
            self.history[season] = cleaned
        self.seasons_seen = len(self.history)
        if param_bounds:
            for key, val in cleaned.items():
                if key not in param_bounds or not isinstance(val, (int, float)):
                    continue
                lo, hi = param_bounds[key]
                span = max(float(hi) - float(lo), 1e-9)
                edge = min(abs(float(val) - float(lo)), abs(float(hi) - float(val))) / span
                if edge <= 0.02:
                    self._bound_hits[key] = self._bound_hits.get(key, 0) + 1
        print(f"✅ AdaptiveParameterBounds: Recorded best params for season {season}. "
              f"Total seasons: {self.seasons_seen}.")

    def _shrink_factor(self) -> float:
        if self.seasons_seen == 0:
            return 1.0
        min_frac = self.MIN_RANGE_FRACTION
        if max(self._bound_hits.values(), default=0) >= 2:
            min_frac = max(min_frac, self.MIN_RANGE_FRACTION_TIGHT)
        if self.seasons_seen >= self.PLATEAU_AFTER:
            return min_frac
        raw = self.INITIAL_SHRINK * (self.DECAY_RATE ** (self.seasons_seen - 1))
        return max(raw, min_frac)

    def _running_best(self) -> dict:
        if not self.history:
            return {}
        latest_season = max(self.history.keys())
        return self.history[latest_season]

    def _lookup_center(self, param_name: str, best: dict,
                       original_low: float, original_high: float):
        def _as_center(value):
            if value is None:
                return None
            try:
                center = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(center):
                return None
            return float(np.clip(center, original_low, original_high))

        if param_name in best:
            center = _as_center(best[param_name])
            if center is not None:
                return center
        for alt in self.PARAM_ALIASES.get(param_name, ()):
            if alt in best:
                center = _as_center(best[alt])
                if center is not None:
                    return center
        return None

    @staticmethod
    def _ensure_valid_float_range(low: float, high: float,
                                  original_low: float, original_high: float) -> tuple:
        """Guarantee low < high; fall back to full original range if needed."""
        if high > low:
            return low, high
        span = max(original_high - original_low, 1e-9)
        eps = max(span * 0.05, 1e-6)
        mid = float(np.clip((low + high) / 2.0, original_low, original_high))
        low = max(original_low, mid - eps)
        high = min(original_high, mid + eps)
        if high > low:
            return low, high
        return original_low, original_high

    def suggest_bounds_float(self, param_name: str, original_low: float, original_high: float,
                             log: bool = False) -> tuple:
        best = self._running_best()
        shrink = self._shrink_factor()
        original_span = original_high - original_low

        if self.seasons_seen == 0:
            return original_low, original_high

        center = self._lookup_center(param_name, best, original_low, original_high)
        if center is None:
            return original_low, original_high
        # Apply jitter with 30% probability to force exploration
        if self.seasons_seen > 2 and np.random.random() < 0.3:
            center = center * (1 + np.random.uniform(-self.EXPLORE_JITTER, self.EXPLORE_JITTER))
            center = float(np.clip(center, original_low, original_high))

        half_width = (original_span * shrink) / 2.0

        if log:
            import math
            center = float(np.clip(center, original_low, original_high))
            log_center = math.log(max(center, original_low + 1e-12))
            log_low = math.log(original_low)
            log_high = math.log(original_high)
            log_span = log_high - log_low
            half_log = (log_span * shrink) / 2.0
            new_low = math.exp(max(log_low, log_center - half_log))
            new_high = math.exp(min(log_high, log_center + half_log))
        else:
            new_low  = max(original_low,  center - half_width)
            new_high = min(original_high, center + half_width)

        return self._ensure_valid_float_range(
            new_low, new_high, original_low, original_high,
        )

    def suggest_bounds_int(self, param_name: str, original_low: int, original_high: int) -> tuple:
        best = self._running_best()
        shrink = self._shrink_factor()
        original_span = original_high - original_low

        if self.seasons_seen == 0:
            return original_low, original_high

        center_f = self._lookup_center(param_name, best, float(original_low), float(original_high))
        if center_f is None:
            return original_low, original_high
        center = int(round(center_f))
        # Apply jitter with 30% probability
        if self.seasons_seen > 2 and np.random.random() < 0.3:
            center = int(round(center * (1 + np.random.uniform(-self.EXPLORE_JITTER, self.EXPLORE_JITTER))))
            center = int(np.clip(center, original_low, original_high))

        half_width = max(1, int(round((original_span * shrink) / 2.0)))

        new_low  = max(original_low,  center - half_width)
        new_high = min(original_high, center + half_width)

        new_low, new_high = self._ensure_valid_float_range(
            float(new_low), float(new_high), float(original_low), float(original_high),
        )
        low_i = int(round(new_low))
        high_i = int(round(new_high))
        if high_i <= low_i:
            low_i = max(original_low, center - 1)
            high_i = min(original_high, center + 1)
        if high_i <= low_i:
            return original_low, original_high
        return low_i, high_i

    def report(self):
        print(f"\n{'═'*60}")
        print(f"  📊 AdaptiveParameterBounds Report")
        print(f"  Seasons recorded: {self.seasons_seen}")
        print(f"  Current range shrink factor: {self._shrink_factor():.3f}x of original")
        if self.history:
            latest = self._running_best()
            print("  Current best params (running anchor):")
            for k, v in latest.items():
                print(f"    {k:30s}: {v:.5f}" if isinstance(v, float) else f"    {k:30s}: {v}")
        print(f"{'═'*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# SAFE FEATURE SET – no look‑ahead, no future data
# ──────────────────────────────────────────────────────────────────────────────

def engineer_interaction_features(df):
    """
    Adds explicit non‑linear interaction features to the feature DataFrame.
    This makes them available to both Ridge and XGBoost.
    """
    df_eng = df.copy()

    # Quality × Pace
    df_eng['elo_pace_interaction'] = df_eng['elo_net'] * df_eng['exp_poss']

    # Quality × Rest differential
    rest_diff = df_eng['h_rest'] - df_eng['a_rest']
    df_eng['elo_rest_interaction'] = df_eng['elo_net'] * rest_diff

    # Form (rolling xPPP net) × Pace
    df_eng['form_pace_interaction'] = df_eng['roll_net_xppp'] * df_eng['exp_poss']

    # Uncertainty‑adjusted quality
    total_uncertainty = df_eng['h_rating_uncertainty'] + df_eng['a_rating_uncertainty']
    df_eng['elo_uncertainty_adj'] = df_eng['elo_net'] / (total_uncertainty + 1e-6)

    # Team net rating (pts/100) × pace — blends box-score form with tempo
    if 'off_rtg_net' in df_eng.columns:
        df_eng['rtg_pace_interaction'] = df_eng['off_rtg_net'] * df_eng['exp_poss']

    # Extended engine margin features (Phase 3)
    if 'elo_margin' in df_eng.columns and 'exp_poss' in df_eng.columns:
        df_eng['elo_margin_per100'] = df_eng['elo_margin'] / (df_eng['exp_poss'] + 1e-6) * 100.0
    if 'elo_margin' in df_eng.columns and 'hier_margin' in df_eng.columns:
        df_eng['engine_margin_spread'] = df_eng['elo_margin'] - df_eng['hier_margin']
    if 'elo_margin' in df_eng.columns and 'pace_diff' in df_eng.columns:
        df_eng['elo_margin_pace'] = df_eng['elo_margin'] * df_eng['pace_diff']

    if 'elo_margin_calibrated' in df_eng.columns and 'market_spread' in df_eng.columns:
        df_eng['elo_edge_pts'] = df_eng['elo_margin_calibrated'] + df_eng['market_spread'].fillna(0)
        meta_col = next(
            (c for c in ('pred_margin', 'elo_hier_blend', 'hier_margin', 'elo_margin')
             if c in df_eng.columns),
            None,
        )
        if meta_col is not None:
            meta_edge = df_eng[meta_col] + df_eng['market_spread'].fillna(0)
            df_eng['elo_meta_agreement'] = (
                np.sign(df_eng['elo_edge_pts']) == np.sign(meta_edge)
            ).astype(float)

    return df_eng

# In Cell 10, update SAFE_FEATURE_COLS

from pipeline.teamstats import FORM_FEATURE_COLS, FORM_FEATURE_COLS_BASE, FOUR_FACTOR_COLS, MULTIWINDOW_FORM_COLS

# Already-computed game-context features (previously built but never fed to the model).
CONTEXT_FEATURE_COLS = [
    "h_experience", "a_experience", "days_since_season_start",
    "h_new_starters", "a_new_starters",
]

# Direct engine-implied margins so the stack can weight the engines directly.
ENGINE_MARGIN_COLS = [
    "elo_margin", "hier_margin", "elo_margin_per100", "engine_margin_spread", "elo_margin_pace",
    "elo_margin_calibrated", "elo_hier_blend", "elo_vs_hier_spread", "elo_vs_market",
    "elo_luck_adj_net", "elo_def_event_rate", "elo_tov_rate", "elo_matchup_asym",
    "elo_margin_z", "elo_consistency", "elo_edge_pts",
]

ELO_INTERACTION_COLS = [
    "elo_pace_interaction", "elo_rest_interaction", "elo_uncertainty_adj", "elo_margin_pace",
    "elo_meta_agreement",
]

# Market context features (available pre-tip; not used for spread edge directly).
MARKET_TOTAL_COLS = ["market_total", "market_total_minus_league"]
MARKET_LINE_COLS = ["spread_move", "public_home_pct", "market_fair_win_prob"]

# Elo-only inputs for the two-stage stack ridge head (expanded for calibration).
ELO_STACK_FEATURES = [
    "elo_margin", "elo_margin_calibrated", "elo_hier_blend", "elo_vs_hier_spread",
    "elo_net", "elo_diff_off", "elo_diff_def", "exp_poss", "elo_pace_interaction",
    "lineup5_net", "team_elo_spread", "team_elo_net",
    "uncertainty_diff", "elo_uncertainty_adj", "h_rating_uncertainty",
    "elo_vs_market", "elo_luck_adj_net", "elo_def_event_rate", "elo_tov_rate",
    "elo_matchup_asym", "elo_margin_z", "elo_consistency",
]

ELO_WIN_FEATURES = [
    "elo_net", "uncertainty_diff", "elo_margin_calibrated", "elo_vs_market",
    "elo_matchup_asym", "elo_consistency", "elo_margin_z", "h_rating_uncertainty",
    "a_rating_uncertainty",
]

# Schedule-density / fatigue features derived from game dates.
SCHEDULE_FEATURE_COLS = [
    "h_games_last7", "a_games_last7", "games_last7_diff", "h_3in4", "a_3in4",
    "h_4in6", "a_4in6",
    "h_rest_0", "h_rest_1", "h_rest_2", "h_rest_3plus",
    "a_rest_0", "a_rest_1", "a_rest_2", "a_rest_3plus",
    "h_travel_rest_interaction", "a_travel_rest_interaction", "travel_rest_interaction_diff",
]

AVAILABILITY_IMPACT_COLS = [
    "h_expected_off_drop", "a_expected_off_drop",
    "h_expected_def_drop", "a_expected_def_drop",
    "h_expected_pace_delta", "a_expected_pace_delta",
    "expected_off_drop_diff", "expected_def_drop_diff", "expected_pace_delta_diff",
]

# Team-specific home-court edge (rolling venue-split margins).
HCA_FEATURE_COLS = ["h_home_edge", "a_road_edge", "hca_net"]

# Core (engine + tempo + interaction + SOS) features, excluding the toggleable
# groups below.  Kept as its own list so the ablation harness can rebuild subsets.
CORE_FEATURE_COLS = [
    "elo_net", "hier_net",
    "h_elo_off", "h_elo_def", "a_elo_off", "a_elo_def",
    "elo_diff_off", "elo_diff_def",
    "h_hier_off", "h_hier_def", "a_hier_off", "a_hier_def",
    "exp_poss", "h_rest", "a_rest", "h_b2b", "a_b2b",
    "is_altitude", "h_season_phase", "a_season_phase",
    "h_rating_uncertainty", "a_rating_uncertainty", "uncertainty_diff",
    "h_roll_off_xppp", "h_roll_def_xppp", "a_roll_off_xppp", "a_roll_def_xppp",
    "roll_net_xppp",
    "elo_pace_interaction", "elo_rest_interaction", "form_pace_interaction", "elo_uncertainty_adj",
    "rtg_pace_interaction",
    "h_recent_net", "a_recent_net", "recent_diff",
    "pace_diff", "pace_abs_diff", "pace_interaction",
    "h_sos", "a_sos", "sos_diff",
    "elo_margin_per100", "engine_margin_spread", "elo_margin_pace",
    "elo_margin_calibrated", "elo_hier_blend", "elo_vs_hier_spread", "elo_vs_market",
]

# Extended features from roadmap modules (chemistry, lineup Elo, travel, team Elo, market micro).
SHOT_QUALITY_COLS = [
    "h_xefg", "a_xefg", "shot_quality_edge",
    "h_rim_rate", "a_rim_rate", "rim_rate_diff",
    "h_three_rate", "a_three_rate", "three_rate_diff",
    "avg_shot_distance_diff",
    "h_elo_def_rim", "a_elo_def_rim", "h_elo_def_peri", "a_elo_def_peri",
    "rim_def_diff", "peri_def_diff",
]

LINEUP_COMPOSITE_COLS = [
    "h_lineup_composite", "a_lineup_composite", "lineup_composite_diff",
    "h_chem_duo_net", "a_chem_duo_net", "h_chem_trio_net", "a_chem_trio_net",
]

HAPM_COLS = ["h_hapm_net", "a_hapm_net", "hapm_net_diff"]

EXTENDED_FEATURE_COLS = [
    "h_lineup5_off", "h_lineup5_def", "h_lineup5_chem", "a_lineup5_off", "a_lineup5_def", "a_lineup5_chem",
    "lineup5_net", "lineup5_chem_diff", "lineup5_sample_min",
    "h_chem_net", "a_chem_net", "chem_diff", "h_onoff_net", "a_onoff_net", "chem_uncertainty", "usage_conflict",
    *LINEUP_COMPOSITE_COLS,
    *SHOT_QUALITY_COLS,
    *HAPM_COLS,
    "team_elo_spread", "team_elo_total_adj", "team_elo_net",
    "h_travel_miles_7d", "a_travel_miles_7d", "travel_miles_diff", "h_tz_shift", "a_tz_shift",
    "h_road_trip", "a_road_trip",
    "h_fatigue_index", "a_fatigue_index", "fatigue_diff",
    "reverse_line_movement", "steam_flag", "fair_spread_vigfree", "public_away_pct",
    "h_star_out", "a_star_out", "epm_prior_diff", "epm_blend_off_diff",
    "ref_pace_bias", "ref_foul_bias",
]

MARKET_MICRO_COLS = ["reverse_line_movement", "steam_flag", "fair_spread_vigfree", "public_away_pct"]

SAFE_FEATURE_COLS = [
    *CORE_FEATURE_COLS,
    *FORM_FEATURE_COLS,
    *MULTIWINDOW_FORM_COLS,
    *CONTEXT_FEATURE_COLS,
    *SCHEDULE_FEATURE_COLS,
    *HCA_FEATURE_COLS,
    *ENGINE_MARGIN_COLS,
    *MARKET_TOTAL_COLS,
    *MARKET_LINE_COLS,
    *EXTENDED_FEATURE_COLS,
    *AVAILABILITY_IMPACT_COLS,
    "elo_stack_pred",
]

# Moneyline: strength, Elo, efficiency, HCA, availability, rest, market movement.
# Do NOT share identical matrix with totals.
WIN_FEATURE_COLS = [
    "elo_net", "hier_net", "team_elo_spread", "team_elo_net",
    "elo_margin", "elo_margin_calibrated", "elo_hier_blend", "elo_vs_market",
    "elo_diff_off", "elo_diff_def", "elo_matchup_asym", "elo_consistency", "elo_margin_z",
    "h_elo_off", "h_elo_def", "a_elo_off", "a_elo_def",
    "h_off_rtg", "h_def_rtg", "a_off_rtg", "a_def_rtg", "off_rtg_net",
    "h_off_rtg_adj", "a_off_rtg_adj", "h_def_rtg_adj", "a_def_rtg_adj", "off_rtg_adj_net",
    "h_home_edge", "a_road_edge", "hca_net", "is_altitude",
    "h_star_out", "a_star_out", "lineup5_net", "lineup_composite_diff",
    "h_hapm_net", "a_hapm_net", "hapm_net_diff", "epm_prior_diff",
    *AVAILABILITY_IMPACT_COLS,
    "h_rest", "a_rest", "h_b2b", "a_b2b",
    "h_rest_0", "h_rest_1", "h_rest_2", "h_rest_3plus",
    "a_rest_0", "a_rest_1", "a_rest_2", "a_rest_3plus",
    "h_3in4", "a_3in4", "travel_miles_diff", "h_fatigue_index", "a_fatigue_index", "fatigue_diff",
    "h_travel_rest_interaction", "a_travel_rest_interaction",
    "spread_move", "public_home_pct", "market_fair_win_prob",
    "fair_spread_vigfree", "reverse_line_movement", "steam_flag",
    "uncertainty_diff", "h_rating_uncertainty", "a_rating_uncertainty",
    "h_recent_net", "a_recent_net", "recent_diff",
    "sos_diff",
]

# Totals / scoring: pace, efficiency pairs, shot quality, fatigue — de-emphasize Elo.
TOTAL_FEATURE_COLS = [
    "exp_poss", "pace_diff", "pace_abs_diff", "pace_interaction",
    "h_rest", "a_rest", "h_b2b", "a_b2b",
    "h_rest_0", "h_rest_1", "h_rest_2", "h_rest_3plus",
    "a_rest_0", "a_rest_1", "a_rest_2", "a_rest_3plus",
    "h_3in4", "a_3in4", "h_4in6", "a_4in6",
    "h_travel_rest_interaction", "a_travel_rest_interaction", "travel_rest_interaction_diff",
    "market_total", "market_total_minus_league",
    "ref_pace_bias", "ref_foul_bias",
    "rtg_pace_interaction", "form_pace_interaction",
    "h_fatigue_index", "a_fatigue_index", "fatigue_diff",
    "h_roll_off_xppp", "a_roll_off_xppp", "h_roll_def_xppp", "a_roll_def_xppp",
    "roll_net_xppp",
    "h_off_rtg", "a_off_rtg", "h_def_rtg", "a_def_rtg",
    "h_off_rtg_adj", "a_off_rtg_adj", "h_def_rtg_adj", "a_def_rtg_adj",
    "h_efg", "a_efg", "efg_diff",
    "h_ftr", "a_ftr", "ftr_diff",
    "h_tov_pct", "a_tov_pct", "tov_pct_diff",
    "h_oreb_pct", "a_oreb_pct", "h_dreb_pct", "a_dreb_pct",
    "h_xefg", "a_xefg", "shot_quality_edge",
    "h_rim_rate", "a_rim_rate", "rim_rate_diff",
    "h_three_rate", "a_three_rate", "three_rate_diff",
    "h_sos", "a_sos", "sos_diff",
    "h_pace", "a_pace",
    "h_expected_off_drop", "a_expected_off_drop",
    "h_expected_pace_delta", "a_expected_pace_delta",
    "expected_off_drop_diff", "expected_pace_delta_diff",
    # Light multi-window pace/efficiency (totals care about short-term scoring form)
    "h_off_rtg_roll_3", "a_off_rtg_roll_3", "off_rtg_diff_roll_3",
    "h_off_rtg_roll_5", "a_off_rtg_roll_5", "off_rtg_diff_roll_5",
    "h_off_rtg_roll_10", "a_off_rtg_roll_10",
    "h_def_rtg_roll_5", "a_def_rtg_roll_5",
    "h_efg_roll_5", "a_efg_roll_5",
    "h_pts_std_roll_5", "a_pts_std_roll_5",
    "h_pts_std_roll_10", "a_pts_std_roll_10",
]

WIN_SPREAD_DERIVED_COLS = ["meta_pred_margin", "spread_quantile_width", "ats_classifier_prob"]


def total_feature_cols(df: pd.DataFrame | None = None) -> list[str]:
    """Pace/total-focused feature subset (columns present in df when provided)."""
    base = list(dict.fromkeys(TOTAL_FEATURE_COLS + MARKET_TOTAL_COLS))
    if df is None:
        return base
    return [c for c in base if c in df.columns]


def win_feature_cols(df: pd.DataFrame | None = None, use_spread_features: bool | None = None) -> list[str]:
    """Feature columns for MetaWinModel — market-specific, not full SAFE set."""
    from pipeline.config import ML_USE_SPREAD_FEATURES
    use = ML_USE_SPREAD_FEATURES if use_spread_features is None else use_spread_features
    cols = list(WIN_FEATURE_COLS)
    if use:
        extra = WIN_SPREAD_DERIVED_COLS
        if df is not None:
            extra = [c for c in extra if c in df.columns]
        cols.extend([c for c in extra if c not in cols])
    if df is None:
        return cols
    present = [c for c in cols if c in df.columns]
    # Fall back to SAFE intersection if curated list too sparse on older frames
    if len(present) < 15:
        return [c for c in SAFE_FEATURE_COLS if c in df.columns]
    return present

# Default values (kept for compatibility)
DEFAULT_XGB_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'min_child_weight': 100,
    'colsample_bytree': 0.4,
    'subsample': 0.5,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'gamma': 5.0,
    'objective': 'reg:pseudohubererror',   # Huber loss for robustness
    'huber_slope': 1.0,
    'eval_metric': 'mae',
    'early_stopping_rounds': 50,
    'n_jobs': -1,
    'random_state': 42
}
DEFAULT_RIDGE_ALPHA = 50.0
DEFAULT_BLEND_WEIGHT = 0.70   # kept only for signature compatibility

# ──────────────────────────────────────────────────────────────────────────────
# STACKED META MODEL – Ridge + XGBoost Residuals with Early Stopping
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor, LogisticRegression
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from pipeline.cv import PurgedGroupTimeSeriesSplit, ChronologicalPartitionCV, PastOnlyGroupCV, bind_cv_groups
from pipeline.oof import ManualOOFStacker, cross_fit_derived_feature
from pipeline.market_targets import (
    closing_spread_series,
    margin_close_residual,
    residual_to_margin,
    market_total_series,
    total_market_residual,
    residual_to_total,
    valid_closing_fraction,
    valid_market_total_fraction,
)
from pipeline.config import (
    ELO_BLEND_ALPHA,
    ELO_RIDGE_ALPHA,
    ELO_UNCERTAINTY_BLEND_BOOST,
    ELO_WIN_BLEND,
    TOTAL_ELO_BETA,
    DEFAULT_LEAGUE_XPPP,
    META_LOSS_ATS_ROI_WEIGHT,
    META_LOSS_BRIER_WEIGHT,
    META_LOSS_CLV_ROI_WEIGHT,
    META_LOSS_MAE_WEIGHT,
    TUNING_EMBARGO_GAMES,
    TUNING_INVALID_SCORE,
    BET_SELECTION_MODE,
    META_LOSS_ECE_WEIGHT,
    APPLY_SPREAD_CALIB_IN_CV,
    META_WIN_LOSS_BRIER_WEIGHT,
    META_WIN_LOSS_ECE_WEIGHT,
    META_WIN_LOSS_ML_ROI_WEIGHT,
    TOTAL_LOSS_MAE_WEIGHT,
    TOTAL_LOSS_OU_ROI_WEIGHT,
    TOTAL_TRAIN_TARGET,
    ML_USE_SPREAD_FEATURES,
)

MARGIN_CAP_CHOICES = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, None]


class _ScaledEstimator:
    """Bundles a fitted scaler + estimator as one object so cross-fit
    helpers (``pipeline.oof.cross_fit_derived_feature``) can attach
    ``fit_max_timestamp`` to a single artifact (Task 052)."""

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


def elo_implied_total_from_row(row) -> float | None:
    """Pace-adjusted total from team offensive/defensive Elo ratings."""
    if isinstance(row, dict):
        exp_poss = row.get("exp_poss")
        ho = float(row.get("h_elo_off") or 1500)
        hd = float(row.get("h_elo_def") or 1500)
        ao = float(row.get("a_elo_off") or 1500)
        ad = float(row.get("a_elo_def") or 1500)
    else:
        exp_poss = row.get("exp_poss")
        ho = float(row.get("h_elo_off") or 1500)
        hd = float(row.get("h_elo_def") or 1500)
        ao = float(row.get("a_elo_off") or 1500)
        ad = float(row.get("a_elo_def") or 1500)
    if exp_poss is None or pd.isna(exp_poss):
        return None
    scaling = 1000.0
    hb = 0.002
    ppp_h = DEFAULT_LEAGUE_XPPP + hb + (ho - ad) / scaling
    ppp_a = DEFAULT_LEAGUE_XPPP - hb + (ao - hd) / scaling
    return float((ppp_h + ppp_a) * float(exp_poss))


def _stack_cv_splitter(fast_mode=False, use_purged_cv=True, n_splits=None):
    """Internal CV for ManualOOFStacker OOF meta-features (Task 049).

    ``use_purged_cv=True`` (default, production) returns ``PastOnlyGroupCV``
    — every fold satisfies ``max(train_time) < min(validation_time)``.
    ``ChronologicalPartitionCV`` (leaky: later folds train on future rows,
    see ``LEAK_REGISTRY.md::stack_cv_future_leakage``) is no longer used by
    any production path; it is retained in ``pipeline/cv.py`` only as the
    subject of its own regression test.
    """
    if n_splits is None:
        n_splits = 3 if fast_mode else 5
    if use_purged_cv:
        return PastOnlyGroupCV(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def _stack_fit_groups(X_df):
    """Chronological group labels for purged stack CV (game dates)."""
    if X_df is not None and "game_date" in X_df.columns:
        return X_df["game_date"].values
    return None


def _outer_game_cv_splits(df, n_splits, embargo=None):
    """Chronological outer CV grouped by game date (no cross-game leakage)."""
    embargo = TUNING_EMBARGO_GAMES if embargo is None else embargo
    groups = _stack_fit_groups(df)
    if groups is None and "GAME_ID" in df.columns:
        groups = df["GAME_ID"].values
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo=embargo)
    yield from cv.split(df, groups=groups)


# ------------------------------------------------------------
# MetaScoreModel – Parallel Stacking with Huber meta‑learner
# ------------------------------------------------------------
class MetaScoreModel:
    def __init__(self, ridge_alpha=10.0, cb_params=None, huber_epsilon=1.0,
                 use_isotonic_calibration=False, cv_splitter=None,
                 margin_cap=30.0, total_mode="model", league_avg_total=225.0,
                 feature_cols=None, use_elo_stack=True,
                 prediction_mode="absolute", residual_alpha=0.5,
                 use_purged_cv=True, segment=None, ou_min_edge=3.0,
                 elo_blend_alpha=0.35, elo_ridge_alpha=3.0,
                 dynamic_elo_blend=True, total_elo_beta=None,
                 elo_win_blend=None, train_target="close_residual",
                 use_quantile_heads=True, quantile_alphas=(0.1, 0.9),
                 use_lightgbm_base=False):
        """
        Parameters
        ----------
        ridge_alpha : float
            Regularisation strength for Ridge base estimator.
        cb_params : dict or None
            Parameters for CatBoostRegressor (depth, iterations, etc.).
        huber_epsilon : float
            Epsilon parameter for the HuberRegressor meta‑learner.
            Controls the transition point from quadratic to linear loss.
        use_isotonic_calibration : bool
            If True, use IsotonicRegression for win‑prob calibration;
            otherwise, use LogisticRegression (Platt scaling).
        margin_cap : float or None
            Symmetric cap applied to the training margin target. None disables
            capping (rely on the Huber meta-learner for robustness). Widened
            from the legacy 20.0 so big favourites aren't shrunk toward the mean.
        total_mode : str
            "model" trains a second head to predict game total (pace/scoring
            aware); "fixed" reverts to the legacy constant ``league_avg_total``;
            "external" skips the embedded total head (use ``MetaTotalModel``).
        league_avg_total : float
            Fallback total used when total_mode == "fixed" or the total head is
            unavailable.
        feature_cols : list or None
            Feature columns the model consumes. Defaults to SAFE_FEATURE_COLS;
            the ablation harness passes subsets here.
        """
        self.ridge_alpha = ridge_alpha
        self.huber_epsilon = huber_epsilon
        self.use_isotonic = use_isotonic_calibration
        self.margin_cap = margin_cap
        self.total_mode = total_mode
        self.league_avg_total = league_avg_total
        self.use_elo_stack = use_elo_stack
        self.prediction_mode = prediction_mode  # absolute | residual | blend
        self.residual_alpha = residual_alpha
        self.train_target = train_target  # close_residual | absolute
        self.use_quantile_heads = use_quantile_heads
        self.quantile_alphas = tuple(quantile_alphas)
        self.use_lightgbm_base = use_lightgbm_base
        self.segment = segment
        self.ou_min_edge = ou_min_edge
        self.elo_blend_alpha = elo_blend_alpha
        self.elo_ridge_alpha = elo_ridge_alpha
        self.dynamic_elo_blend = dynamic_elo_blend
        self.total_elo_beta = TOTAL_ELO_BETA if total_elo_beta is None else total_elo_beta
        self.quantile_models = {}
        self.elo_ridge = None
        self.elo_scaler = None
        self._elo_cols = []

        # Default CatBoost params (safe, low depth to avoid overfitting)
        default_cb = {
            'depth': 4,
            'iterations': 300,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'loss_function': 'Huber:delta=1.5',
            'verbose': 0,
            'random_seed': 42,
            'thread_count': 1,
        }
        if cb_params is not None:
            default_cb.update(cb_params)
        self.cb_params = default_cb

        # Base estimators
        base_models = [
            ('ridge', Ridge(alpha=self.ridge_alpha)),
            ('catboost', CatBoostRegressor(**self.cb_params))
        ]
        if self.use_lightgbm_base:
            try:
                from lightgbm import LGBMRegressor
                base_models.append((
                    'lightgbm',
                    LGBMRegressor(
                        n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
                    ),
                ))
            except ImportError:
                pass

        # Final meta‑learner with Huber loss
        final_estimator = HuberRegressor(epsilon=self.huber_epsilon)

        # Manual past-only OOF stacking (Task 049/050/051). sklearn's
        # StackingRegressor requires its internal CV to be a *partition*
        # (every sample in exactly one held-out fold), which a strictly
        # past-only splitter cannot satisfy for its earliest block — so we
        # never hand a past-only splitter to StackingRegressor. Instead
        # ManualOOFStacker generates OOF base predictions manually, dropping
        # burn-in rows with no valid past-only fold from meta-training
        # (Task 051) and stamping fit_max_timestamp on every fitted
        # component (Task 050/058).
        self.use_purged_cv = use_purged_cv
        if cv_splitter is None:
            cv_splitter = _stack_cv_splitter(use_purged_cv=use_purged_cv)
        self.stack = ManualOOFStacker(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=cv_splitter,
        )

        # Second head for game TOTAL (separate estimator instances so the two
        # stacks never share fitted state).
        total_base = [
            ('ridge', Ridge(alpha=self.ridge_alpha)),
            ('catboost', CatBoostRegressor(**self.cb_params)),
        ]
        self.total_stack = ManualOOFStacker(
            estimators=total_base,
            final_estimator=HuberRegressor(epsilon=self.huber_epsilon),
            cv=_stack_cv_splitter(use_purged_cv=use_purged_cv),
        )
        self._total_head_ok = False

        # In tune_meta_model (when building the stack for tuning), do the same.

        self.scaler = StandardScaler()
        self.calibrator = None
        self.fitted = False
        base_feats = list(feature_cols) if feature_cols else SAFE_FEATURE_COLS.copy()
        if use_elo_stack and "elo_stack_pred" not in base_feats:
            base_feats.append("elo_stack_pred")
        self.features = base_feats

    def _elo_matrix(self, X_df):
        cols = [c for c in ELO_STACK_FEATURES if c in X_df.columns]
        self._elo_cols = cols
        if not cols:
            return np.zeros((len(X_df), 1))
        return X_df[cols].fillna(0).to_numpy(dtype=float)

    def _augment_elo_stack(self, X_df):
        """Add elo_stack_pred column from the Elo-only ridge head."""
        out = X_df.copy()
        if self.use_elo_stack and self.elo_ridge is not None and self._elo_cols:
            Xe = self._elo_matrix(out)
            out["elo_stack_pred"] = self.elo_ridge.predict(self.elo_scaler.transform(Xe))
        else:
            out["elo_stack_pred"] = 0.0
        return out

    def _get_features(self, X_df):
        """Extract and clean the safe feature set."""
        missing = [c for c in self.features if c not in X_df.columns]
        if missing:
            X_df = X_df.copy()
            for col in missing:
                X_df[col] = 0.0
        return X_df[self.features].fillna(0)

    def _effective_train_target(self, X_df) -> str:
        if self.train_target != "close_residual":
            return "absolute"
        close = closing_spread_series(X_df)
        if close is None or valid_closing_fraction(X_df) < 0.5:
            return "absolute"
        return "close_residual"

    def _build_stack_target(self, y_margin, X_df):
        mode = self._effective_train_target(X_df)
        if mode == "close_residual":
            close = closing_spread_series(X_df)
            return margin_close_residual(y_margin, close.values), mode
        return np.asarray(y_margin, dtype=float), mode

    def _stack_output_to_margin(self, stack_pred, X_df, mode=None):
        if mode is None:
            mode = self._effective_train_target(X_df)
        pred = np.asarray(stack_pred, dtype=float)
        if mode != "close_residual":
            return pred
        close = closing_spread_series(X_df)
        if close is None:
            return pred
        return residual_to_margin(pred, close.values)

    def _closing_from_feat(self, feat_dict):
        for key in ("closing_spread", "market_spread"):
            val = feat_dict.get(key)
            if val is not None and not pd.isna(val):
                return float(val)
        return np.nan

    def _fit_quantile_heads(self, X_scaled, y_target):
        self.quantile_models = {}
        if not self.use_quantile_heads:
            return
        for alpha in self.quantile_alphas:
            key = f"q{int(alpha * 100)}"
            qparams = dict(self.cb_params)
            qparams.update({
                "loss_function": f"Quantile:alpha={alpha}",
                "iterations": min(int(qparams.get("iterations", 300)), 250),
                "depth": min(int(qparams.get("depth", 4)), 4),
                "allow_writing_files": False,
            })
            model = CatBoostRegressor(**qparams)
            try:
                model.fit(X_scaled, y_target)
                self.quantile_models[key] = model
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ quantile head {key} fit failed ({e})")

    def fit(self, X_df, y_home, y_away, calib_df=None):
        """
        Fit the stacking ensemble on the base set, then optionally calibrate
        win probabilities on a held‑out calibration set.
        """
        y_margin = np.asarray(y_home - y_away, dtype=float)
        y_stack, train_mode = self._build_stack_target(y_margin, X_df)
        self._active_train_mode = train_mode
        if self.margin_cap is not None:
            y_margin_target = np.clip(y_stack, -self.margin_cap, self.margin_cap)
        else:
            y_margin_target = y_stack

        X_work = X_df.copy()
        stack_groups = _stack_fit_groups(X_df)
        elo_groups = stack_groups if stack_groups is not None else np.arange(len(X_df))
        if self.use_elo_stack:
            Xe = self._elo_matrix(X_work)
            if Xe.shape[1] > 0:
                # Cross-fit elo_stack_pred past-only (Task 052): the Elo-only
                # ridge head is itself target-fitted, so feeding an in-sample
                # (fit-on-everything) prediction into the margin stack's
                # training rows would leak the target into a feature. Each
                # training row's elo_stack_pred instead comes from a ridge
                # fit only on strictly earlier rows; the ridge refit on all
                # rows (self.elo_scaler/self.elo_ridge) is used only at
                # predict() time, when every row being scored is later than
                # every row it was fit on.
                def _elo_fit(Xtr, ytr):
                    sc = StandardScaler()
                    Xs = sc.fit_transform(Xtr)
                    m = Ridge(alpha=self.elo_ridge_alpha)
                    m.fit(Xs, ytr)
                    return _ScaledEstimator(sc, m)

                def _elo_predict(artifact, Xap):
                    return artifact.predict(Xap)

                elo_oof, elo_covered, elo_artifact_final = cross_fit_derived_feature(
                    _elo_fit, _elo_predict, Xe, y_margin_target, elo_groups,
                    cv=_stack_cv_splitter(use_purged_cv=self.use_purged_cv),
                )
                self.elo_scaler, self.elo_ridge = elo_artifact_final.scaler, elo_artifact_final.model
                self.elo_ridge.fit_max_timestamp = elo_artifact_final.fit_max_timestamp
                self.elo_scaler.fit_max_timestamp = elo_artifact_final.fit_max_timestamp
                self._elo_stack_covered = elo_covered
                X_work = X_work.copy()
                X_work["elo_stack_pred"] = np.where(elo_covered, elo_oof, 0.0)
            else:
                X_work["elo_stack_pred"] = 0.0
                self._elo_stack_covered = np.ones(len(X_work), dtype=bool)
        else:
            self._elo_stack_covered = np.ones(len(X_work), dtype=bool)

        X = self._get_features(X_work)
        X_scaled = self.scaler.fit_transform(X)
        self.stack.fit(X_scaled, y_margin_target, groups=stack_groups)

        self._fit_quantile_heads(X_scaled, y_margin_target)

        # Fit the TOTAL head (pace/scoring aware) on the same scaled features.
        if self.total_mode == "model":
            y_total = np.asarray(y_home + y_away, dtype=float)
            try:
                self.total_stack.fit(X_scaled, y_total, groups=stack_groups)
                self._total_head_ok = True
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ total head fit failed ({e}); falling back to fixed total.")
                self._total_head_ok = False

        # --- Calibration (if calibration set provided) ---
        if calib_df is not None:
            raw_preds = self._predict_raw(calib_df)['pred_margin']
            raw_preds = raw_preds.reshape(-1, 1)
            y_win_cal = (calib_df['actual_home'] > calib_df['actual_away']).astype(int)

            if self.use_isotonic:
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(raw_preds.ravel(), y_win_cal)
                print(f"✅ Isotonic calibration fitted on {len(raw_preds)} samples.")
            else:
                # Platt scaling (logistic) – use strong L2 to avoid overfitting
                self.calibrator = LogisticRegression(C=0.1, solver='lbfgs', max_iter=100)
                self.calibrator.fit(raw_preds, y_win_cal)
                print(f"✅ Platt calibration (logistic) fitted on {len(raw_preds)} samples.")

        self.fitted = True
        print(f"✅ Stacked ensemble fitted with {len(self.stack.estimators_)} base models.")

    def _predict_raw(self, X_df):
        """Return raw predicted margin (before calibration)."""
        X_work = self._augment_elo_stack(X_df)
        X = self._get_features(X_work)
        X_scaled = self.scaler.transform(X)
        stack_out = self.stack.predict(X_scaled)
        mode = getattr(self, "_active_train_mode", self._effective_train_target(X_df))
        margin = self._stack_output_to_margin(stack_out, X_df, mode=mode)
        out = {"pred_margin": margin, "pred_stack": stack_out}
        if self.quantile_models:
            q_margins = {}
            for key, qm in self.quantile_models.items():
                q_stack = qm.predict(X_scaled)
                q_margins[key] = self._stack_output_to_margin(q_stack, X_df, mode=mode)
            if "q10" in q_margins and "q90" in q_margins:
                out["spread_q10"] = q_margins["q10"]
                out["spread_q90"] = q_margins["q90"]
                out["spread_quantile_width"] = q_margins["q90"] - q_margins["q10"]
        return out

    def _apply_prediction_mode(self, raw_margin, feat_dict):
        """Blend absolute prediction with market-residual mode (predict-time only)."""
        if getattr(self, "_active_train_mode", "absolute") == "close_residual":
            return raw_margin
        market_spread = feat_dict.get("market_spread") or feat_dict.get("closing_spread")
        if self.prediction_mode == "absolute" or market_spread is None or pd.isna(market_spread):
            return raw_margin
        market_prior = -float(market_spread)
        if self.prediction_mode == "residual":
            return market_prior + self.residual_alpha * (raw_margin - market_prior)
        # blend
        return (1 - self.residual_alpha) * market_prior + self.residual_alpha * raw_margin

    def _dynamic_elo_blend_alpha(self, row: dict) -> float:
        """Increase Elo anchor weight when lineup uncertainty is high."""
        base = float(self.elo_blend_alpha)
        if not self.dynamic_elo_blend:
            return base
        unc = float(row.get("uncertainty_diff", 0) or 0)
        boost = ELO_UNCERTAINTY_BLEND_BOOST * max(0.0, (unc - 50.0) / 300.0)
        return float(np.clip(base + boost, 0.0, 0.65))

    def _elo_implied_total(self, row: dict) -> float | None:
        return elo_implied_total_from_row(row)

    def predict(self, feat_dict):
        """
        Predict for a single game given a feature dictionary.
        Returns: dict with pred_home, pred_away, pred_margin, pred_total, win_prob.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        row = dict(feat_dict)
        X_work = self._augment_elo_stack(pd.DataFrame([row]))
        X_scaled = self.scaler.transform(self._get_features(X_work))
        raw_out = self._predict_raw(pd.DataFrame([row]))
        raw_margin = float(raw_out["pred_margin"][0] if hasattr(raw_out["pred_margin"], "__len__") else raw_out["pred_margin"])
        raw_margin = self._apply_prediction_mode(raw_margin, row)

        spread_q10 = spread_q90 = spread_q_width = None
        if "spread_q10" in raw_out:
            spread_q10 = float(raw_out["spread_q10"][0])
            spread_q90 = float(raw_out["spread_q90"][0])
            spread_q_width = float(raw_out["spread_quantile_width"][0])

        # Blend meta prediction toward calibrated ELO anchor (dynamic weight)
        elo_anchor = row.get("elo_margin_calibrated") or row.get("elo_stack_pred") or row.get("elo_margin")
        blend_a = self._dynamic_elo_blend_alpha(row)
        if elo_anchor is not None and not pd.isna(elo_anchor) and blend_a > 0:
            raw_margin = (1.0 - blend_a) * raw_margin + blend_a * float(elo_anchor)

        # Calibrate win probability
        if self.calibrator is not None:
            if self.use_isotonic:
                win_prob = self.calibrator.predict([raw_margin])[0]
            else:
                win_prob = self.calibrator.predict_proba(np.array([[raw_margin]]))[0, 1]
        else:
            # Fallback sigmoid (no calibration)
            win_prob = 1.0 / (1.0 + np.exp(-raw_margin / 12.0))
            win_prob = np.clip(win_prob, 0.01, 0.99)

        # Predicted total: learned head when available, else legacy fixed total.
        if self.total_mode == "model" and self._total_head_ok and getattr(self, "total_stack", None) is not None:
            try:
                pred_total = float(self.total_stack.predict(X_scaled)[0])
            except Exception:
                pred_total = self.league_avg_total
        else:
            pred_total = self.league_avg_total
        elo_total = self._elo_implied_total(row)
        if elo_total is not None and self.total_elo_beta > 0:
            pred_total = (1.0 - self.total_elo_beta) * pred_total + self.total_elo_beta * elo_total
        pred_home = (pred_total + raw_margin) / 2.0
        pred_away = (pred_total - raw_margin) / 2.0

        return {
            'pred_home': float(pred_home),
            'pred_away': float(pred_away),
            'pred_margin': float(raw_margin),
            'pred_total': float(pred_home + pred_away),
            'win_prob': float(win_prob),
            'ou_direction': self._ou_signal(feat_dict, pred_total),
            'spread_q10': spread_q10,
            'spread_q90': spread_q90,
            'spread_quantile_width': spread_q_width,
        }

    def _ou_signal(self, feat_dict, pred_total):
        """Over/Under bet direction when edge exceeds threshold."""
        mkt = feat_dict.get("market_total")
        if mkt is None or pd.isna(mkt) or mkt == 0:
            return "Pass"
        diff = pred_total - float(mkt)
        if abs(diff) >= self.ou_min_edge:
            return "Over" if diff > 0 else "Under"
        return "Pass"

    def calibrate_prob(self, raw_margin):
        """Calibrated win probability from a raw margin using the trained
        calibrator. Used as the cold-start for the in-season rolling calibrator
        so a single coherent calibration path is used throughout."""
        if self.calibrator is not None:
            try:
                if self.use_isotonic:
                    p = self.calibrator.predict([raw_margin])[0]
                else:
                    p = self.calibrator.predict_proba(np.array([[raw_margin]]))[0, 1]
                return float(np.clip(p, 0.01, 0.99))
            except Exception:
                pass
        return float(np.clip(1.0 / (1.0 + np.exp(-raw_margin / 12.0)), 0.01, 0.99))

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------
# Hyperparameter Tuning for MetaScoreModel
# ------------------------------------------------------------
def _fit_elo_stack_cv(X_tr, y_tr, elo_cols, alpha=5.0):
    """Fit elo-only ridge on a train fold; returns (scaler, ridge, avail_cols)."""
    avail = [c for c in elo_cols if c in X_tr.columns]
    if not avail:
        return None, None, avail
    Xe = X_tr[avail].fillna(0).to_numpy(dtype=float)
    y_arr = np.asarray(y_tr, dtype=float)
    if len(y_arr) != len(Xe):
        raise ValueError(
            f"ELO stack CV: X/y length mismatch ({len(Xe)} vs {len(y_arr)})"
        )
    scaler_e = StandardScaler()
    Xe_s = scaler_e.fit_transform(Xe)
    elo_ridge = Ridge(alpha=alpha)
    elo_ridge.fit(Xe_s, y_arr)
    return scaler_e, elo_ridge, avail


def _apply_elo_stack_cv(X_df, scaler_e, elo_ridge, avail):
    """Apply a train-fitted elo ridge head (no refit on val/test)."""
    X_work = X_df.copy()
    if not avail or scaler_e is None or elo_ridge is None:
        X_work["elo_stack_pred"] = 0.0
        return X_work
    Xe = X_work[avail].fillna(0).to_numpy(dtype=float)
    Xe_s = scaler_e.transform(Xe)
    X_work["elo_stack_pred"] = elo_ridge.predict(Xe_s)
    return X_work


def _augment_elo_stack_cv(X_df, y_margin_target, elo_cols, alpha=5.0):
    """Fit elo ridge on X_df and add elo_stack_pred (single-split helper)."""
    scaler_e, elo_ridge, avail = _fit_elo_stack_cv(
        X_df, y_margin_target, elo_cols, alpha=alpha,
    )
    return _apply_elo_stack_cv(X_df, scaler_e, elo_ridge, avail)


def _margin_tuning_objective(
    df, X_base, trial, b, EMBARGO, use_elo_stack=True, fast_mode=False,
    train_target="close_residual",
):
    """Shared margin-model Optuna objective — margin/probability-first
    (Task 055: no ROI/edge-grid/CLV terms; overfit penalty penalizes
    val_mae exceeding train_mae)."""
    close_all = closing_spread_series(df)
    use_close_residual = (
        train_target == "close_residual"
        and close_all is not None
        and valid_closing_fraction(df) >= 0.5
    )

    cb_iter_lo, cb_iter_hi = (200, 400) if fast_mode else (300, 500)
    outer_splits = 2 if fast_mode else 3
    inner_splits = 3 if fast_mode else 5
    stack_jobs = 1 if fast_mode else -1

    if b is not None:
        lo, hi = b.suggest_bounds_float('ridge_alpha', 0.01, 200.0, log=True)
        ridge_alpha = trial.suggest_float('ridge_alpha', lo, hi, log=True)
        lo, hi = b.suggest_bounds_int('cb_depth', 2, 7)
        cb_depth = trial.suggest_int('cb_depth', lo, hi)
        lo, hi = b.suggest_bounds_int('cb_iter', cb_iter_lo, cb_iter_hi)
        lo = max(cb_iter_lo, (lo // 100) * 100)
        hi = min(cb_iter_hi, max(lo + 100, ((hi + 99) // 100) * 100))
        if hi <= lo:
            lo, hi = cb_iter_lo, min(cb_iter_lo + 100, cb_iter_hi)
        cb_iter = trial.suggest_int('cb_iter', lo, hi, step=100)
        lo, hi = b.suggest_bounds_float('cb_lr', 0.01, 0.1, log=True)
        cb_lr = trial.suggest_float('cb_lr', lo, hi, log=True)
        lo, hi = b.suggest_bounds_float('cb_l2', 1.0, 10.0, log=True)
        cb_l2 = trial.suggest_float('cb_l2', lo, hi, log=True)
        lo, hi = b.suggest_bounds_float('huber_epsilon', 1.01, 4.0)
        huber_epsilon = trial.suggest_float('huber_epsilon', lo, hi)
        lo, hi = b.suggest_bounds_float('elo_blend_alpha', 0.0, 0.65)
        elo_blend_alpha = trial.suggest_float('elo_blend_alpha', lo, hi)
        lo, hi = b.suggest_bounds_float('elo_ridge_alpha', 1.0, 12.0, log=True)
        elo_ridge_alpha = trial.suggest_float('elo_ridge_alpha', lo, hi, log=True)
    else:
        ridge_alpha = trial.suggest_float('ridge_alpha', 0.1, 100.0, log=True)
        cb_depth = trial.suggest_int('cb_depth', 2, 7)
        cb_iter = trial.suggest_int('cb_iter', cb_iter_lo, cb_iter_hi, step=100)
        cb_lr = trial.suggest_float('cb_lr', 0.01, 0.1, log=True)
        cb_l2 = trial.suggest_float('cb_l2', 1.0, 10.0, log=True)
        huber_epsilon = trial.suggest_float('huber_epsilon', 1.01, 4.0)
        elo_blend_alpha = trial.suggest_float('elo_blend_alpha', 0.0, 0.65)
        elo_ridge_alpha = trial.suggest_float('elo_ridge_alpha', 1.0, 12.0, log=True)

    margin_cap_choice = trial.suggest_categorical('margin_cap', MARGIN_CAP_CHOICES)

    base_models = [
        ('ridge', Ridge(alpha=ridge_alpha)),
        ('catboost', CatBoostRegressor(
            depth=cb_depth, iterations=cb_iter, learning_rate=cb_lr,
            l2_leaf_reg=cb_l2, loss_function='Huber:delta=1.5',
            verbose=0, random_seed=42, thread_count=1,
            allow_writing_files=False,
        ))
    ]
    stack = ManualOOFStacker(
        estimators=base_models,
        final_estimator=HuberRegressor(epsilon=huber_epsilon),
        cv=_stack_cv_splitter(fast_mode=fast_mode, n_splits=inner_splits),
    )

    y_margin = df['actual_margin']
    y_home = df['actual_home']
    y_away = df['actual_away']
    scores = []
    fold_idx = 0
    outer_splits_list = list(_outer_game_cv_splits(df, outer_splits, EMBARGO))
    n_outer = len(outer_splits_list)

    for train_idx, val_idx in outer_splits_list:
        if len(val_idx) < 20:
            continue

        ym_tr = y_margin.iloc[train_idx]
        ym_val = y_margin.iloc[val_idx]
        if use_close_residual:
            ym_tr_fit = margin_close_residual(ym_tr.values, close_all.iloc[train_idx].values)
            if margin_cap_choice is not None:
                ym_tr_fit = np.clip(ym_tr_fit, -margin_cap_choice, margin_cap_choice)
        elif margin_cap_choice is not None:
            ym_tr_fit = ym_tr.clip(-margin_cap_choice, margin_cap_choice)
        else:
            ym_tr_fit = ym_tr

        X_tr = X_base.iloc[train_idx].copy()
        X_val = X_base.iloc[val_idx].copy()
        if use_elo_stack:
            elo_sc, elo_ridge, elo_avail = _fit_elo_stack_cv(
                X_tr, np.asarray(ym_tr_fit, dtype=float), ELO_STACK_FEATURES, alpha=elo_ridge_alpha,
            )
            X_tr = _apply_elo_stack_cv(X_tr, elo_sc, elo_ridge, elo_avail)
            X_val = _apply_elo_stack_cv(X_val, elo_sc, elo_ridge, elo_avail)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        print(
            f"    trial {trial.number + 1}: outer fold {fold_idx + 1}/{n_outer} "
            f"(train={len(train_idx)}, val={len(val_idx)}, cb_iter={cb_iter})...",
            flush=True,
        )
        groups_tr = _stack_fit_groups(df.iloc[train_idx])
        stack.fit(X_tr_scaled, ym_tr_fit, groups=groups_tr)

        val_stack = stack.predict(X_val_scaled)
        if use_close_residual:
            val_preds = residual_to_margin(val_stack, close_all.iloc[val_idx].values)
        else:
            val_preds = val_stack
        if APPLY_SPREAD_CALIB_IN_CV:
            from pipeline.market import SpreadCalibrator
            sc = SpreadCalibrator(window=max(50, len(train_idx)), min_samples=min(30, len(train_idx) // 3))
            train_raw = stack.predict(X_tr_scaled)
            if use_close_residual:
                train_raw = residual_to_margin(train_raw, close_all.iloc[train_idx].values)
            for p, a in zip(train_raw, ym_tr.values):
                if np.isfinite(p) and np.isfinite(a):
                    sc.update(float(p), float(a))
            val_preds = np.array([sc.correct(float(p)) for p in val_preds], dtype=float)
        anchor = None
        for col in ("elo_margin_calibrated", "elo_stack_pred", "elo_margin"):
            if col in df.columns:
                anchor = df[col].iloc[val_idx].values.astype(float)
                break
            if col in X_val.columns:
                anchor = X_val[col].values.astype(float)
                break
        if anchor is not None and elo_blend_alpha > 0:
            mask = np.isfinite(anchor)
            val_preds = val_preds.copy()
            val_preds[mask] = (
                (1.0 - elo_blend_alpha) * val_preds[mask]
                + elo_blend_alpha * anchor[mask]
            )
        val_probs = np.clip(1.0 / (1.0 + np.exp(-val_preds / 12.0)), 0.01, 0.99)
        val_mae = mean_absolute_error(ym_val, val_preds)
        brier = brier_score_loss((ym_val > 0).astype(int), val_probs)
        from pipeline.calibration_metrics import compute_ece
        val_ece = compute_ece((ym_val > 0).astype(int), val_probs)
        train_stack = stack.predict(X_tr_scaled)
        if use_close_residual:
            train_preds = residual_to_margin(train_stack, close_all.iloc[train_idx].values)
        else:
            train_preds = train_stack
        train_mae = mean_absolute_error(ym_tr, train_preds)
        # Task 055: penalize validation error *exceeding* training error
        # (overfitting), not the reverse. The prior sign — max(0, train_mae
        # - val_mae) — rewarded configurations where validation error was
        # *lower* than training error and never penalized true overfitting
        # (val_mae >> train_mae), which is backwards.
        overfit_penalty = max(0.0, (val_mae - train_mae) - 1.0) * 2.0

        # Task 055: margin-model tuning must be a proper-scoring-rule
        # objective on margin/probability quality only. ATS ROI, the
        # best-of-edge-grid search, and the CLV term are removed — they
        # (a) require betting columns to be present (market_spread), which
        # the pass condition explicitly forbids depending on, and (b) let
        # the tuner chase a noisy, threshold-selected backtest ROI number
        # instead of the agreed margin-first objective (Phase 4A of the
        # plan). Betting-policy tuning is a separate, later step (Task 056)
        # on policy-tuning-role rows only, never mixed into this loss.
        combined = (
            META_LOSS_MAE_WEIGHT * val_mae
            + META_LOSS_BRIER_WEIGHT * brier
            + META_LOSS_ECE_WEIGHT * (val_ece if np.isfinite(val_ece) else 0.0)
            + overfit_penalty
        )
        scores.append(combined)
        trial.report(combined, step=fold_idx)
        fold_idx += 1
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores) if scores else TUNING_INVALID_SCORE


def meta_params_for_bounds(best_params: dict) -> dict:
    """Flatten cached MetaScoreModel params for AdaptiveParameterBounds."""
    out = dict(best_params)
    cb = out.pop("cb_params", None) or {}
    if cb:
        for src, dst in (
            ("depth", "cb_depth"),
            ("iterations", "cb_iter"),
            ("learning_rate", "cb_lr"),
            ("l2_leaf_reg", "cb_l2"),
        ):
            if out.get(dst) is None:
                val = cb.get(src)
                if val is not None:
                    out[dst] = val
    return AdaptiveParameterBounds._sanitize_params(out)


def tune_margin_model(train_features_df, n_trials=25, season=None,
                      bounds: 'AdaptiveParameterBounds' = None,
                      feature_cols=None, use_elo_stack=True,
                      fast_mode=None, train_target="close_residual"):
    """Tune margin stack with ROI-first objective (matches production fit).

    fast_mode: lighter CV + fewer CatBoost trees during search. Auto-enabled
    when n_trials <= 5 (smoke tests). Full backtests should use fast_mode=False.
    """
    df = train_features_df.dropna(subset=['actual_margin']).reset_index(drop=True)
    cols = list(feature_cols) if feature_cols else SAFE_FEATURE_COLS
    cols = [c for c in cols if c in df.columns and c != 'elo_stack_pred']
    X_base = df[cols].fillna(0)
    EMBARGO = TUNING_EMBARGO_GAMES
    b = bounds
    if fast_mode is None:
        fast_mode = n_trials <= 5

    if n_trials <= 0:
        print("⏭️ Skipping margin tuning (n_trials=0) — using default hyperparameters.")
        return {
            'ridge_alpha': 7.72,
            'huber_epsilon': 1.42,
            'margin_cap': 35.0,
            'elo_blend_alpha': ELO_BLEND_ALPHA,
            'elo_ridge_alpha': ELO_RIDGE_ALPHA,
            'cb_params': {
                'depth': 2,
                'iterations': 500,
                'learning_rate': 0.014,
                'l2_leaf_reg': 7.16,
                'thread_count': 1,
            },
        }

    def objective(trial):
        return _margin_tuning_objective(
            df, X_base, trial, b, EMBARGO,
            use_elo_stack=use_elo_stack, fast_mode=fast_mode,
            train_target=train_target,
        )

    mode_label = "FAST" if fast_mode else "FULL"
    outer = 2 if fast_mode else 3
    inner = 3 if fast_mode else 5
    print(
        f"⏳ Tuning margin model ({mode_label}: {n_trials} trials, "
        f"{outer}×{inner} CV, ~{n_trials * outer} stack fits)...",
        flush=True,
    )

    def _trial_callback(study, trial):
        if trial.value is not None:
            print(
                f"  ✓ trial {trial.number + 1}/{n_trials} score={trial.value:.4f}",
                flush=True,
            )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=min(5, max(1, n_trials)),
            n_warmup_steps=1,
        ),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[_trial_callback],
    )
    print(f"✅ Margin tuning complete. Best score: {study.best_value:.4f}")
    best = study.best_params
    if b is not None and season is not None:
        b.record_best(season, meta_params_for_bounds({
            'ridge_alpha': best['ridge_alpha'],
            'huber_epsilon': best['huber_epsilon'],
            'margin_cap': best.get('margin_cap'),
            'elo_blend_alpha': best.get('elo_blend_alpha'),
            'elo_ridge_alpha': best.get('elo_ridge_alpha'),
            'cb_params': {
                'depth': best['cb_depth'],
                'iterations': best['cb_iter'],
                'learning_rate': best['cb_lr'],
                'l2_leaf_reg': best['cb_l2'],
            },
        }))
        b.report()
    return {
        'ridge_alpha': best['ridge_alpha'],
        'huber_epsilon': best['huber_epsilon'],
        'margin_cap': best.get('margin_cap', 30.0),
        'elo_blend_alpha': best.get('elo_blend_alpha', 0.35),
        'elo_ridge_alpha': best.get('elo_ridge_alpha', 3.0),
        'cb_params': {
            'depth': best['cb_depth'],
            'iterations': best['cb_iter'],
            'learning_rate': best['cb_lr'],
            'l2_leaf_reg': best['cb_l2'],
            'thread_count': 1,
        }
    }


def tune_total_model(train_features_df, n_trials=20, bounds=None, feature_cols=None,
                     use_elo_stack=True, fast_mode=None, season=None,
                     train_target: str | None = None):
    """Tune standalone total model with MAE + O/U ROI objective.

    fast_mode: lighter outer/inner CV and fewer CatBoost trees during search.
    Auto-enabled when n_trials <= 5 (smoke tests).
    """
    if fast_mode is None:
        fast_mode = n_trials <= 5
    if train_target is None:
        train_target = TOTAL_TRAIN_TARGET

    df = train_features_df.dropna(subset=['actual_total']).reset_index(drop=True)
    cols = list(feature_cols) if feature_cols else total_feature_cols(df)
    cols = [c for c in cols if c in df.columns and c != 'elo_stack_pred']
    X_base = df[cols].fillna(0)
    y_total = df['actual_total']
    mkt_all = market_total_series(df)
    use_market_residual = (
        train_target == "market_residual"
        and mkt_all is not None
        and valid_market_total_fraction(df) >= 0.5
    )
    EMBARGO = TUNING_EMBARGO_GAMES
    cb_iter_lo, cb_iter_hi = (200, 400) if fast_mode else (200, 800)
    outer_splits = 2 if fast_mode else 3
    inner_splits = 3 if fast_mode else 5
    stack_jobs = 1 if fast_mode else -1
    b = bounds

    def objective(trial):
        if b is not None:
            lo, hi = b.suggest_bounds_float('ridge_alpha', 0.1, 100.0, log=True)
            ridge_alpha = trial.suggest_float('ridge_alpha', lo, hi, log=True)
            lo, hi = b.suggest_bounds_int('cb_depth', 2, 6)
            cb_depth = trial.suggest_int('cb_depth', lo, hi)
            lo, hi = b.suggest_bounds_int('cb_iter', cb_iter_lo, cb_iter_hi)
            lo = max(cb_iter_lo, (lo // 100) * 100)
            hi = min(cb_iter_hi, max(lo + 100, ((hi + 99) // 100) * 100))
            if hi <= lo:
                lo, hi = cb_iter_lo, min(cb_iter_lo + 100, cb_iter_hi)
            cb_iter = trial.suggest_int('cb_iter', lo, hi, step=100)
        else:
            ridge_alpha = trial.suggest_float('ridge_alpha', 0.1, 100.0, log=True)
            cb_depth = trial.suggest_int('cb_depth', 2, 6)
            cb_iter = trial.suggest_int('cb_iter', cb_iter_lo, cb_iter_hi, step=100)
        total_elo_beta = trial.suggest_float('total_elo_beta', 0.0, 0.5)
        huber_eps = trial.suggest_float('huber_epsilon', 1.0, 2.0)

        fold_scores = []
        fold_idx = 0
        outer_splits_list = list(_outer_game_cv_splits(df, outer_splits, EMBARGO))
        n_outer = len(outer_splits_list)
        for train_idx, val_idx in outer_splits_list:
            if len(val_idx) < 20:
                continue
            X_tr = X_base.iloc[train_idx].copy()
            X_val = X_base.iloc[val_idx].copy()
            y_tr = y_total.iloc[train_idx]
            y_val = y_total.iloc[val_idx]
            if use_market_residual:
                mkt_tr = mkt_all.iloc[train_idx].values
                mkt_val = mkt_all.iloc[val_idx].values
                y_tr_fit = total_market_residual(y_tr.values, mkt_tr)
                target_mode = "market_residual"
            else:
                y_tr_fit = y_tr.values
                mkt_val = mkt_all.iloc[val_idx].values if mkt_all is not None else None
                target_mode = "absolute"

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            print(
                f"    total trial {trial.number + 1}: fold {fold_idx + 1}/{n_outer} "
                f"(train={len(train_idx)}, val={len(val_idx)}, cb_iter={cb_iter})...",
                flush=True,
            )
            groups_tr = _stack_fit_groups(df.iloc[train_idx])
            tot_stack = ManualOOFStacker(
                estimators=[
                    ('ridge', Ridge(alpha=ridge_alpha)),
                    ('catboost', CatBoostRegressor(
                        depth=cb_depth, iterations=cb_iter, learning_rate=0.05,
                        verbose=0, random_seed=42, thread_count=1,
                        allow_writing_files=False,
                    )),
                ],
                final_estimator=HuberRegressor(epsilon=huber_eps),
                cv=_stack_cv_splitter(fast_mode=fast_mode, n_splits=inner_splits),
            )
            tot_stack.fit(X_tr_s, y_tr_fit, groups=groups_tr)
            pred = tot_stack.predict(X_val_s)
            if target_mode == "market_residual" and mkt_val is not None:
                pred = residual_to_total(pred, mkt_val)
            if total_elo_beta > 0:
                val_rows = df.iloc[val_idx]
                elo_totals = np.array([
                    elo_implied_total_from_row(r.to_dict()) or np.nan
                    for _, r in val_rows.iterrows()
                ])
                mask = ~np.isnan(elo_totals)
                if mask.any():
                    pred = pred.copy()
                    pred[mask] = (
                        (1.0 - total_elo_beta) * pred[mask]
                        + total_elo_beta * elo_totals[mask]
                    )
            val_mae = mean_absolute_error(y_val, pred)

            ou_roi = 0.0
            if mkt_val is not None and np.isfinite(mkt_val).sum() >= 15:
                mkt_v = np.asarray(mkt_val, dtype=float)
                edge = pred - mkt_v
                best_r = -1.0
                for thr in (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0):
                    active = np.abs(edge) >= thr
                    if active.sum() < 10:
                        continue
                    actual = y_val.values
                    over_win = (edge > 0) & (actual > mkt_v)
                    under_win = (edge < 0) & (actual < mkt_v)
                    win = over_win | under_win
                    wp = win[active].mean()
                    roi = wp * (100.0 / 110.0) - (1.0 - wp)
                    best_r = max(best_r, roi)
                ou_roi = best_r if best_r > -1 else 0.0

            combined = (
                TOTAL_LOSS_MAE_WEIGHT * val_mae
                - TOTAL_LOSS_OU_ROI_WEIGHT * ou_roi
            )
            fold_scores.append(combined)
            fold_idx += 1
        return np.mean(fold_scores) if fold_scores else TUNING_INVALID_SCORE

    mode_label = "FAST" if fast_mode else "FULL"
    target_label = "residual" if use_market_residual else "absolute"
    print(f"  ⏳ Total-model tuning ({mode_label}, {n_trials} trials, {target_label})...", flush=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    if b is not None and season is not None:
        b.record_best(season, meta_params_for_bounds({
            'ridge_alpha': best['ridge_alpha'],
            'total_elo_beta': best.get('total_elo_beta', TOTAL_ELO_BETA),
            'huber_epsilon': best.get('huber_epsilon', 1.35),
            'cb_params': {
                'depth': best['cb_depth'],
                'iterations': best['cb_iter'],
            },
        }))
    return {
        'ridge_alpha': best['ridge_alpha'],
        'total_elo_beta': best.get('total_elo_beta', TOTAL_ELO_BETA),
        'huber_epsilon': best.get('huber_epsilon', 1.35),
        'train_target': "market_residual" if use_market_residual else "absolute",
        'feature_cols': cols,
        'cb_params': {
            'depth': best['cb_depth'],
            'iterations': best['cb_iter'],
            'learning_rate': 0.05,
            'thread_count': 1,
        },
        '_cv_score': float(study.best_value),
    }


class MetaTotalModel:
    """Standalone game-total regressor (decoupled from MetaScoreModel margin head)."""

    def __init__(
        self,
        ridge_alpha=10.0,
        cb_params=None,
        huber_epsilon=1.35,
        feature_cols=None,
        total_elo_beta=None,
        train_target="absolute",
        use_quantile_heads=False,
        league_avg_total=225.0,
    ):
        self.ridge_alpha = ridge_alpha
        self.huber_epsilon = huber_epsilon
        self.total_elo_beta = TOTAL_ELO_BETA if total_elo_beta is None else total_elo_beta
        self.train_target = train_target
        self.use_quantile_heads = use_quantile_heads
        self.league_avg_total = league_avg_total
        default_cb = {
            'depth': 4,
            'iterations': 300,
            'learning_rate': 0.05,
            'verbose': 0,
            'random_seed': 42,
            'thread_count': 1,
            'allow_writing_files': False,
        }
        if cb_params:
            default_cb.update(cb_params)
        self.cb_params = default_cb
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.stack = None
        self.scaler = StandardScaler()
        self.quantile_models = {}
        self.fitted = False
        self._active_train_mode = "absolute"
        self._fit_cols_ = []

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        if self.feature_cols:
            return [c for c in self.feature_cols if c in df.columns]
        return total_feature_cols(df)

    def _effective_train_target(self, df: pd.DataFrame) -> str:
        if self.train_target != "market_residual":
            return "absolute"
        mkt = market_total_series(df)
        if mkt is None or valid_market_total_fraction(df) < 0.5:
            return "absolute"
        return "market_residual"

    def _build_target(self, df: pd.DataFrame):
        y_total = np.asarray(df["actual_total"], dtype=float)
        mode = self._effective_train_target(df)
        if mode == "market_residual":
            mkt = market_total_series(df)
            return total_market_residual(y_total, mkt.values), mode
        return y_total, mode

    def fit(self, X_df, y_home=None, y_away=None, calib_df=None):
        df = X_df.copy()
        if "actual_total" not in df.columns and y_home is not None and y_away is not None:
            df["actual_total"] = np.asarray(y_home, dtype=float) + np.asarray(y_away, dtype=float)
        cols = self._resolve_cols(df)
        self._fit_cols_ = cols
        y_target, mode = self._build_target(df)
        self._active_train_mode = mode

        X = df[cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.stack = ManualOOFStacker(
            estimators=[
                ('ridge', Ridge(alpha=self.ridge_alpha)),
                ('catboost', CatBoostRegressor(**self.cb_params)),
            ],
            final_estimator=HuberRegressor(epsilon=self.huber_epsilon),
            cv=_stack_cv_splitter(use_purged_cv=True),
        )
        groups = _stack_fit_groups(df)
        self.stack.fit(X_scaled, y_target, groups=groups)

        if self.use_quantile_heads:
            from sklearn.ensemble import GradientBoostingRegressor
            for alpha, key in ((0.1, "q10"), (0.9, "q90")):
                qm = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=80, max_depth=3)
                qm.fit(X_scaled, y_target)
                self.quantile_models[key] = qm

        self.fitted = True
        return self

    def _predict_raw_total(self, feat_dict: dict) -> dict:
        row = pd.DataFrame([feat_dict])
        cols = self._fit_cols_ or self._resolve_cols(row)
        X = row[cols].fillna(0)
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X_scaled = self.scaler.transform(X[cols])
        raw = float(self.stack.predict(X_scaled)[0])
        mode = self._active_train_mode
        mkt = feat_dict.get("market_total")
        if mode == "market_residual" and mkt is not None and np.isfinite(float(mkt)):
            pred_total = float(residual_to_total(np.array([raw]), np.array([float(mkt)]))[0])
        else:
            pred_total = raw
        elo_total = elo_implied_total_from_row(feat_dict)
        if elo_total is not None and self.total_elo_beta > 0:
            pred_total = (1.0 - self.total_elo_beta) * pred_total + self.total_elo_beta * elo_total
        out = {"pred_total": float(pred_total), "pred_total_raw": float(raw)}
        if self.quantile_models:
            for key, qm in self.quantile_models.items():
                q_raw = float(qm.predict(X_scaled)[0])
                if mode == "market_residual" and mkt is not None and np.isfinite(float(mkt)):
                    q_val = float(residual_to_total(np.array([q_raw]), np.array([float(mkt)]))[0])
                else:
                    q_val = q_raw
                out[f"total_{key}"] = q_val
            if "total_q10" in out and "total_q90" in out:
                out["total_quantile_width"] = out["total_q90"] - out["total_q10"]
        return out

    def predict_total(self, feat_dict: dict) -> float:
        if not self.fitted:
            raise RuntimeError("MetaTotalModel must be fitted first.")
        return self._predict_raw_total(feat_dict)["pred_total"]

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class MetaScorePairModel:
    """Independent home/away score regressors → total, margin, O/U distribution inputs."""

    def __init__(
        self,
        ridge_alpha=10.0,
        cb_params=None,
        huber_epsilon=1.35,
        feature_cols=None,
        train_target="absolute",
        use_quantile_heads=False,
        sigma_floor=8.0,
    ):
        self.ridge_alpha = ridge_alpha
        self.huber_epsilon = huber_epsilon
        self.train_target = train_target
        self.use_quantile_heads = use_quantile_heads
        self.sigma_floor = sigma_floor
        default_cb = {
            "depth": 4,
            "iterations": 300,
            "learning_rate": 0.05,
            "verbose": 0,
            "random_seed": 42,
            "thread_count": 1,
            "allow_writing_files": False,
        }
        if cb_params:
            default_cb.update(cb_params)
        self.cb_params = default_cb
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.home_stack = None
        self.away_stack = None
        self.scaler = StandardScaler()
        self.fitted = False
        self._fit_cols_ = []
        self._rmse_home = float(sigma_floor)
        self._rmse_away = float(sigma_floor)
        self._rmse_total = float(sigma_floor) * np.sqrt(2.0)

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        if self.feature_cols:
            return [c for c in self.feature_cols if c in df.columns]
        return total_feature_cols(df)

    def _make_stack(self):
        return ManualOOFStacker(
            estimators=[
                ("ridge", Ridge(alpha=self.ridge_alpha)),
                ("catboost", CatBoostRegressor(**self.cb_params)),
            ],
            final_estimator=HuberRegressor(epsilon=self.huber_epsilon),
            cv=_stack_cv_splitter(use_purged_cv=True),
        )

    def fit(self, X_df, y_home=None, y_away=None, calib_df=None):
        df = X_df.copy()
        if y_home is None:
            y_home = df["actual_home"] if "actual_home" in df.columns else None
        if y_away is None:
            y_away = df["actual_away"] if "actual_away" in df.columns else None
        if y_home is None or y_away is None:
            raise ValueError("MetaScorePairModel requires actual_home/actual_away or y_home/y_away")
        y_h = np.asarray(y_home, dtype=float)
        y_a = np.asarray(y_away, dtype=float)
        cols = self._resolve_cols(df)
        self._fit_cols_ = cols
        X = df[cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        groups = _stack_fit_groups(df)
        self.home_stack = self._make_stack()
        self.away_stack = self._make_stack()
        for stack, y in ((self.home_stack, y_h), (self.away_stack, y_a)):
            stack.fit(X_scaled, y, groups=groups)

        # In-sample residual SD as initial sigma (overwritten by walk-forward when available)
        pred_h = self.home_stack.predict(X_scaled)
        pred_a = self.away_stack.predict(X_scaled)
        self._rmse_home = float(max(self.sigma_floor, np.sqrt(np.mean((pred_h - y_h) ** 2))))
        self._rmse_away = float(max(self.sigma_floor, np.sqrt(np.mean((pred_a - y_a) ** 2))))
        pred_t = pred_h + pred_a
        y_t = y_h + y_a
        self._rmse_total = float(max(self.sigma_floor, np.sqrt(np.mean((pred_t - y_t) ** 2))))
        self.fitted = True
        return self

    def predict_scores(self, feat_dict: dict) -> dict:
        if not self.fitted:
            raise RuntimeError("MetaScorePairModel must be fitted first.")
        row = pd.DataFrame([feat_dict])
        cols = self._fit_cols_ or self._resolve_cols(row)
        for c in cols:
            if c not in row.columns:
                row[c] = 0.0
        X_scaled = self.scaler.transform(row[cols].fillna(0))
        pred_home = float(self.home_stack.predict(X_scaled)[0])
        pred_away = float(self.away_stack.predict(X_scaled)[0])
        pred_total = pred_home + pred_away
        pred_margin = pred_home - pred_away
        sigma_total = float(self._rmse_total)
        out = {
            "pred_home": pred_home,
            "pred_away": pred_away,
            "pred_home_pts": pred_home,
            "pred_away_pts": pred_away,
            "pred_total": pred_total,
            "pred_margin": pred_margin,
            "sigma_total": sigma_total,
            "sigma_home": float(self._rmse_home),
            "sigma_away": float(self._rmse_away),
        }
        mkt = feat_dict.get("market_total")
        if mkt is not None and np.isfinite(float(mkt)):
            from pipeline.market import total_over_prob_gaussian
            out["p_over"] = total_over_prob_gaussian(pred_total, float(mkt), sigma_total)
            out["p_under"] = 1.0 - out["p_over"]
        return out

    def _predict_raw_total(self, feat_dict: dict) -> dict:
        return self.predict_scores(feat_dict)

    def predict_total(self, feat_dict: dict) -> float:
        return self.predict_scores(feat_dict)["pred_total"]

    def set_walkforward_rmse(self, rmse_home=None, rmse_away=None, rmse_total=None):
        if rmse_home is not None and np.isfinite(rmse_home):
            self._rmse_home = float(max(self.sigma_floor, rmse_home))
        if rmse_away is not None and np.isfinite(rmse_away):
            self._rmse_away = float(max(self.sigma_floor, rmse_away))
        if rmse_total is not None and np.isfinite(rmse_total):
            self._rmse_total = float(max(self.sigma_floor, rmse_total))

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def _meta_win_tuning_objective(
    trial,
    df,
    X_base,
    y_win,
    pred_margin_series,
    *,
    fast_mode,
    outer_splits,
    inner_splits,
    use_spread_features,
):
    from pipeline.calibration_metrics import compute_ece

    C = trial.suggest_float("C", 0.01, 1.0, log=True)
    elo_win_blend = trial.suggest_float("elo_win_blend", 0.0, 0.6)
    calib_method = trial.suggest_categorical("calib_method", ["isotonic", "none"])

    scores = []
    EMBARGO = TUNING_EMBARGO_GAMES
    for train_idx, val_idx in _outer_game_cv_splits(df, outer_splits, EMBARGO):
        if len(val_idx) < 30:
            continue
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        cols = win_feature_cols(train_df, use_spread_features=use_spread_features)
        if pred_margin_series is not None:
            train_df["meta_pred_margin"] = pred_margin_series.iloc[train_idx].values
            val_df["meta_pred_margin"] = pred_margin_series.iloc[val_idx].values
        X_tr = train_df[cols].fillna(0)
        X_val = val_df[cols].fillna(0)
        y_tr = y_win.iloc[train_idx].astype(int)
        y_val = y_win.iloc[val_idx].astype(int)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        lr = LogisticRegression(C=C, solver="lbfgs", max_iter=200)
        lr.fit(X_tr_s, y_tr)

        elo_cols = [c for c in ELO_WIN_FEATURES if c in train_df.columns]
        p_val = lr.predict_proba(scaler.transform(X_val))[:, 1]
        if elo_cols:
            elo_sc = StandardScaler()
            Xe_tr = elo_sc.fit_transform(train_df[elo_cols].fillna(0))
            Xe_val = elo_sc.transform(val_df[elo_cols].fillna(0))
            elo_lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=200)
            elo_lr.fit(Xe_tr, y_tr)
            p_elo = elo_lr.predict_proba(Xe_val)[:, 1]
            p_val = (1.0 - elo_win_blend) * p_val + elo_win_blend * p_elo

        if calib_method == "isotonic" and len(train_idx) >= 80:
            iso = IsotonicRegression(out_of_bounds="clip")
            p_tr = lr.predict_proba(X_tr_s)[:, 1]
            iso.fit(p_tr, y_tr.values)
            p_val = iso.predict(p_val)

        brier = brier_score_loss(y_val, p_val)
        ece = compute_ece(y_val.values, p_val)

        ml_roi = 0.0
        if "market_ml" in val_df.columns:
            from pipeline.market import fair_home_win_prob
            best_r = -1.0
            for ev_thr in (0.03, 0.05, 0.08, 0.10):
                rois = []
                for idx, row in val_df.iterrows():
                    mkt_ml = row.get("market_ml")
                    if mkt_ml is None or pd.isna(mkt_ml):
                        continue
                    i_loc = val_df.index.get_loc(idx)
                    p = float(p_val[i_loc])
                    fair = fair_home_win_prob(float(mkt_ml))
                    if fair is None:
                        continue
                    ev_home = p - fair
                    ev_away = (1 - p) - (1 - fair)
                    if ev_home >= ev_thr and p >= 0.5:
                        rois.append(float(y_val.loc[idx]))
                    elif ev_away >= ev_thr and p < 0.5:
                        rois.append(float(1 - y_val.loc[idx]))
                if len(rois) >= 10:
                    wp = np.mean(rois)
                    roi = wp * (100.0 / 110.0) - (1.0 - wp)
                    best_r = max(best_r, roi)
            ml_roi = best_r if best_r > -1 else 0.0

        combined = (
            META_WIN_LOSS_BRIER_WEIGHT * brier
            + META_WIN_LOSS_ECE_WEIGHT * (ece if np.isfinite(ece) else 0.0)
            - META_WIN_LOSS_ML_ROI_WEIGHT * ml_roi
        )
        scores.append(combined)
    return np.mean(scores) if scores else TUNING_INVALID_SCORE


def tune_meta_win_model(
    train_features_df,
    n_trials=25,
    season=None,
    bounds=None,
    feature_cols=None,
    pred_margin_col="pred_margin",
    fast_mode=None,
    use_spread_features: bool | None = None,
):
    """Tune MetaWinModel hyperparameters with Brier + ECE + ML ROI objective."""
    if fast_mode is None:
        fast_mode = n_trials <= 5

    df = train_features_df.dropna(subset=["actual_home", "actual_away"]).reset_index(drop=True)
    y_win = (df["actual_home"] > df["actual_away"]).astype(int)
    pred_margin_series = df[pred_margin_col] if pred_margin_col in df.columns else None
    outer_splits = 2 if fast_mode else 3
    inner_splits = 3 if fast_mode else 5

    if n_trials <= 0:
        return {
            "C": 0.1,
            "elo_win_blend": ELO_WIN_BLEND,
            "calib_method": "isotonic",
            "feature_cols": win_feature_cols(df, use_spread_features=use_spread_features),
        }

    cols_placeholder = win_feature_cols(df, use_spread_features=use_spread_features)
    X_base = df[cols_placeholder].fillna(0)

    def objective(trial):
        return _meta_win_tuning_objective(
            trial, df, X_base, y_win, pred_margin_series,
            fast_mode=fast_mode,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            use_spread_features=use_spread_features,
        )

    mode_label = "FAST" if fast_mode else "FULL"
    print(f"  ⏳ MetaWin tuning ({mode_label}, {n_trials} trials)...", flush=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    return {
        "C": best["C"],
        "elo_win_blend": best.get("elo_win_blend", ELO_WIN_BLEND),
        "calib_method": best.get("calib_method", "isotonic"),
        "feature_cols": win_feature_cols(df, use_spread_features=use_spread_features),
        "_cv_score": float(study.best_value),
    }


def tune_meta_model(train_features_df, n_trials=25, season=None,
                    bounds: 'AdaptiveParameterBounds' = None,
                    use_isotonic_calib=False, feature_cols=None):
    """Alias for tune_margin_model (backward compatible)."""
    return tune_margin_model(
        train_features_df, n_trials=n_trials, season=season,
        bounds=bounds, feature_cols=feature_cols, use_elo_stack=True,
    )


class MetaWinModel:
    """Separate ML winner classifier with optional Elo-only head blend."""

    def __init__(self, C=0.1, feature_cols=None, elo_win_blend=None, calib_method="isotonic"):
        self.C = C
        self.feature_cols = list(feature_cols) if feature_cols else win_feature_cols()
        self.elo_win_blend = ELO_WIN_BLEND if elo_win_blend is None else elo_win_blend
        self.calib_method = calib_method
        self.model = LogisticRegression(C=C, solver='lbfgs', max_iter=200)
        self.scaler = StandardScaler()
        self.elo_model = LogisticRegression(C=0.1, solver='lbfgs', max_iter=200)
        self.elo_scaler = StandardScaler()
        self._elo_cols_ = []
        self.elo_fitted = False
        self.calibrator = None
        self.fitted = False

    def _elo_matrix(self, X_df):
        cols = [c for c in ELO_WIN_FEATURES if c in X_df.columns]
        self._elo_cols_ = cols
        if not cols:
            return None
        return X_df[cols].fillna(0)

    def fit(self, X_df, y_win, calib_df=None, pred_margin_col='pred_margin'):
        cols = [c for c in self.feature_cols if c in X_df.columns]
        self._fit_cols_ = list(cols)
        self._uses_pred_margin_ = pred_margin_col in X_df.columns
        X = X_df[cols].fillna(0)
        if self._uses_pred_margin_:
            X = X.copy()
            X['meta_pred_margin'] = X_df[pred_margin_col].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y_win.astype(int))

        Xe = self._elo_matrix(X_df)
        if Xe is not None and Xe.shape[1] > 0:
            Xe_s = self.elo_scaler.fit_transform(Xe)
            self.elo_model.fit(Xe_s, y_win.astype(int))
            self.elo_fitted = True

        if calib_df is not None and self.calib_method != "none":
            Xc = calib_df[self._fit_cols_].fillna(0)
            if self._uses_pred_margin_ and pred_margin_col in calib_df.columns:
                Xc = Xc.copy()
                Xc['meta_pred_margin'] = calib_df[pred_margin_col].fillna(0)
            raw = self._blend_raw_proba(Xc, calib_df, pred_margin_col)
            y_cal = (calib_df['actual_home'] > calib_df['actual_away']).astype(int)
            if self.calib_method == "platt":
                self.calibrator = LogisticRegression(C=0.1, solver='lbfgs', max_iter=100)
                self.calibrator.fit(raw.reshape(-1, 1), y_cal)
            else:
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(raw, y_cal)
        self.fitted = True

    def _blend_raw_proba(self, X, X_df, pred_margin_col='pred_margin'):
        p = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        if not self.elo_fitted:
            return p
        Xe = self._elo_matrix(X_df)
        if Xe is None:
            return p
        p_elo = self.elo_model.predict_proba(self.elo_scaler.transform(Xe))[:, 1]
        w = self.elo_win_blend
        return (1.0 - w) * p + w * p_elo

    def predict_proba(self, feat_dict, pred_margin=None):
        if not self.fitted:
            raise RuntimeError("MetaWinModel must be fitted first.")
        row = pd.DataFrame([feat_dict])
        cols = getattr(self, '_fit_cols_', [c for c in self.feature_cols if c in row.columns])
        X = row[cols].fillna(0)
        if getattr(self, '_uses_pred_margin_', pred_margin is not None):
            X = X.copy()
            X['meta_pred_margin'] = 0.0 if pred_margin is None else pred_margin
        p = float(self._blend_raw_proba(X, row, pred_margin_col='meta_pred_margin')[0])
        if self.calibrator is not None:
            if isinstance(self.calibrator, LogisticRegression):
                p = float(self.calibrator.predict_proba(np.array([[p]]))[0, 1])
            else:
                p = float(self.calibrator.predict([p])[0])
        return float(np.clip(p, 0.01, 0.99))

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def build_feature_row(raw_feat: dict) -> dict:
    """Apply interaction features to a single pre-game feature dict (train/serve parity)."""
    df = pd.DataFrame([raw_feat])
    return engineer_interaction_features(df).iloc[0].to_dict()
