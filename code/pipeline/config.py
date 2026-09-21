"""Pipeline configuration, paths, and constants."""
from pathlib import Path

import pandas as pd

IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/basketballData")
else:
    ROOT = _REPO_ROOT / "basketballData"

ROOT.mkdir(parents=True, exist_ok=True)

# Prefer basketballData/data/ (real on-disk location); fall back to ROOT.
_DATA_DIR = ROOT / "data" if (ROOT / "data").is_dir() else ROOT
V3_DATA_PATHS = {
    y: p
    for y, p in {
        yr: _DATA_DIR / f"events_{yr}_{str(yr + 1)[-2:]}_pbp_V3.csv"
        for yr in range(2015, 2025)
    }.items()
    if p.exists()
}

PBP_2026_PATH = _REPO_ROOT / "[10-21-2025]-[05-18-2026]-combined-stats.csv"
if not PBP_2026_PATH.exists():
    PBP_2026_PATH = ROOT / "[10-21-2025]-[05-18-2026]-combined-stats.csv"

MODERN_ODDS_PATH = ROOT / "all_odds.csv"
if not MODERN_ODDS_PATH.exists():
    MODERN_ODDS_PATH = _REPO_ROOT / "all_odds.csv"

# Pinnacle full-season 2025-26 lines (decimal odds, team1/team2, timestamped).
# Supplements all_odds.csv which only covers the early part of the 2026 season.
PINNACLE_LINES_PATH = ROOT / "nba_main_lines.csv"
if not PINNACLE_LINES_PATH.exists():
    PINNACLE_LINES_PATH = _REPO_ROOT / "nba_main_lines.csv"

PLAYER_LIST_PATH = ROOT / "nba_players_all.csv"
SPREAD_CSV_PATH = ROOT / "nba_betting_spread.csv"
ML_CSV_PATH = ROOT / "nba_betting_money_line.csv"

BASE_ELO = 1500.0
DEFAULT_LEAGUE_RTG = 110.0
DEFAULT_LEAGUE_XPPP = 1.10
LEAGUE_AVG_TOTAL = 225.0
OFFSEASON_REVERSION = 0.15
# Team xPPP Elo warm-start (empty history / high roster turnover only)
XPPP_WARM_START_GAMES = 8
XPPP_TURNOVER_THRESHOLD = 0.35
PACE_OFFSEASON_SHRINK = 0.5
ASSIST_SPLIT = 0.76
USAGE_FLOOR = 0.27
HOME_PPP_BOOST = 0.002
K_MULT_HALF_LIFE = 24.4
GARBAGE_TIME_WEIGHT = 0.30
ALTITUDE_TEAMS = {"DEN", "UTA"}
GOOD_BET_EDGE = 3.5

SOS_WINDOW = 15
# Post-hoc / live edge floor from walk-forward edge-bucket diagnostics (newest data).
OPTIMAL_BET_EDGE = 5.5
SPREAD_CALIB_WINDOW = 50
# Ablation / walk-forward candidates for rolling spread window (prior-year MAE pick).
SPREAD_CALIB_WINDOW_CANDIDATES = (30, 50, 80)
# When True, pick SPREAD_CALIB_WINDOW from candidates using prior-season corrected MAE.
SPREAD_CALIB_ADAPTIVE_WINDOW = True
# When True, SpreadCalibrator fits separate (a, b) by |pred| bins (large-favorite slope).
SPREAD_CALIB_BINNED = True
SPREAD_CALIB_ABS_BINS = (0.0, 4.0, 8.0, 12.0, 99.0)

# Multi-year walk-forward: model fit window vs full rating warm-start history.
MODEL_TRAIN_WINDOW_DEFAULT = 4
RATING_HISTORY_USE_ALL = True  # warm Elo/hier on all seasons before test
# Exponential sample weights for meta fit: half-life in seasons (~180 days each).
META_RECENCY_HALF_LIFE_SEASONS = 1.25
META_RECENCY_SEASON_DAYS = 180.0
# Only keep cover/confidence calibrator when prior-year ECE improves vs raw.
COVER_CALIB_REQUIRE_ECE_IMPROVEMENT = True
COVER_CALIB_REQUIRE_LOGLOSS_IMPROVEMENT = False  # Phase 2 optional gate
# Wire decision (T-60) vs close lines from timestamped Pinnacle into odds_dict.
USE_DECISION_LINE_FROM_SNAPSHOTS = True

# Suite data-integrity gates (calib3h blockers).
MIN_ODDS_MATCH_RATE = 0.50  # fail closed before walk-forward when odds exist
REQUIRE_CLV_ON_REVIEW = True  # fail when distinct decision/close exists but CLV all-NaN
USE_ROLLING_LEAGUE_Z = False
LEAGUE_Z_WINDOW_GAMES = 400
LOG_LOSS_IN_REVIEW = True

# Tuning objective / CV constants
TUNING_XPPP_WEIGHT = 0.7
TUNING_POINTS_WEIGHT = 0.3
TUNING_EMBARGO_GAMES = 15
TUNING_INVALID_SCORE = 9999.0
TUNING_BOUND_EDGE_FRACTION = 0.02
# Floor AdaptiveParameterBounds shrink so Optuna can still explore edges.
ADAPTIVE_BOUNDS_MIN_SHRINK = 0.45

META_LOSS_MAE_WEIGHT = 0.35
META_LOSS_BRIER_WEIGHT = 0.25
# Deprecated: ROI terms removed from Optuna (Task 055). Kept as 0 so imports
# cannot silently re-enable ROI chasing.
META_LOSS_ATS_ROI_WEIGHT = 0.0
META_LOSS_CLV_ROI_WEIGHT = 0.0
META_LOSS_ECE_WEIGHT = 0.10
META_WIN_LOSS_LOGLOSS_WEIGHT = 0.0

# ELO calibration / meta-anchor defaults (overridden by walk-forward tuning).
# Default/start alpha; Optuna explores [0, ELO_BLEND_ALPHA_MAX]. Kept ≤ max so
# cold starts do not open at a market-hugging blend.
ELO_BLEND_ALPHA = 0.15
# Cap Optuna + predict-time Elo→meta blend. Raised from 0.20 so a well-scaled
# Elo anchor can restore variance after residual-mode compression is fixed;
# dispersion penalty in margin tuning still blocks market-hugging.
ELO_BLEND_ALPHA_MAX = 0.35
ELO_RIDGE_ALPHA = 4.66
ELO_CALIB_MIN_SAMPLES = 80
ELO_CALIB_HUBER_EPS = 1.35

# Context-aware Elo update defaults (tunable via Optuna).
CLUTCH_BOOST = 1.30
TOV_PENALTY = 0.85
FOUL_DRAW_BOOST = 1.10
VARIANCE_DAMPEN = 0.90
XPPP_ACTUAL_BLEND = 0.05
K_DEF_EVENTS = 0.15
TOV_RATE_THRESHOLD = 0.15
THREE_PA_RATE_THRESHOLD = 0.45

# Betting / meta-model Elo prominence.
ELO_UNCERTAINTY_BLEND_BOOST = 0.15
ELO_AGREEMENT_EXTRA_EDGE = 1.0
TOTAL_ELO_BETA = 0.0  # de-emphasize Elo on totals unless ablation shows lift
ELO_WIN_BLEND = 0.35

TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

names_dict: dict = {}
name_to_id: dict = {}

if PLAYER_LIST_PATH.exists():
    _names_df = pd.read_csv(PLAYER_LIST_PATH, low_memory=False)
    pid_col = "person_id" if "person_id" in _names_df.columns else "PERSON_ID"
    name_col = "display_first_last" if "display_first_last" in _names_df.columns else "DISPLAY_FIRST_LAST"
    names_dict = _names_df.set_index(pid_col)[name_col].to_dict()
    name_to_id = {v: k for k, v in names_dict.items()}

STATE_DIR = _REPO_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

# --- Artifacts / backtest CSV schema ---
# Bumped for T-60 plan Phase 1: decision_residual targets, NaN totals,
# two-sided ML prices, fail-closed unresolved dates.
ARTIFACT_SCHEMA_VERSION = 3

# --- Task 004/017 versioned schema components ---
# Bump whichever of these changes whenever the corresponding logic changes so
# every downstream cache (stints, tuning, backtest CSVs, model artifacts) is
# automatically invalidated instead of silently reused with stale semantics.
# Bumped for Tasks 006-013 (canonical games, mixed-date parsing, garbage-time
# point preservation, zero-possession stint preservation, GAME_ID stint
# boundaries).
PREPROCESSING_SCHEMA_VERSION = 2
# Bumped for Epic 9.1 / P0.5 scalar Kalman RD (player Elo updates).
FEATURE_SCHEMA_VERSION = 7
# Bumped for Epic 8.2/12.5: player RAPM + informed L-RAPM walk-forward features
# (actuals-first; not market-blend).
MARKET_SNAPSHOT_SCHEMA_VERSION = 3
VALIDATION_SCHEMA_VERSION = 2
# Freeze definition: predictions may use only data with source/ingestion
# timestamps at or before (scheduled tip - this many minutes).
DECISION_CUTOFF_MINUTES_BEFORE_TIP = 60
# Minimum seasons with usable T-60 decision quotes before betting-head training.
MIN_T60_SEASONS_FOR_BETTING = 3
MIN_T60_DECISION_COVERAGE = 0.50
# When True, skip ATS classifier fit if the T-60 coverage gate fails.
ENFORCE_T60_BETTING_GATE = True
# Matchup scenario mixture: points variance → RD-ish uncertainty scale.
SCENARIO_VAR_TO_RD_SCALE = 50.0
# ATS promotion: allow CLV bootstrap lower CI slightly below zero (noise).
ATS_CLV_CI_LOWER_TOL = -0.05
REQUIRED_BACKTEST_COLUMNS = (
    "EDGE",
    "EDGE_LEAN",
    "WIN_PCT",
    "ACTIONABLE",
    "DIRECTION",
    "CONFIDENCE",
    "MIN_CONFIDENCE_SCORE",
)

# --- Bet selection & staking ---
# edge_bucket: |edge| >= MIN_EDGE_BUCKET is primary; confidence is a secondary stake/filter.
# confidence_only: |edge| >= CONFIDENCE_MIN_EDGE, then calibrated win% gates actionable bets.
BET_SELECTION_MODE = "edge_bucket"  # legacy_tiers | edge_bucket | edge_bucket_ats | confidence_only
# Floor for confidence-mode / secondary confidence gating (aligned with MIN_EDGE_BUCKET).
CONFIDENCE_MIN_EDGE = 5.5
# When True, require higher confidence for small edges (see edge_scaled_min_confidence).
CONFIDENCE_EDGE_SCALED = True
MIN_EDGE_BUCKET = 5.5
EDGE_STAKE_TIERS = ((5.5, 8.0, 1.0), (8.0, 999.0, 1.5))
# Median model conf width is ~26pts; 22 was over-abstaining (~3% bet rate / 0 bets
# on early single-season walks). 28 keeps a real uncertainty filter without silence.
MAX_QUANTILE_WIDTH = 28.0
# Actionable spread gates (confidence_only and live predict).
# 0.70: 0.85 was silencing season-1 walks when Elo/Meta disagree lightly.
MIN_DISAGREEMENT_TRUST = 0.70
SKIP_PHANTOM_INJURY = True
# False: |market|<=TIGHT_SPREAD_MAX is common; hard-rejecting it killed most edges.
# Keep the flag for ablation / research, but do not gate production bets on it.
SKIP_TIGHT_SPREAD = False
# Dead-zone filter: one (lo, hi) or a list of bands; None disables.
# Diagnostics: |edge| 2.5–4 and ~5.5–6 lose money; require Elo agreement to keep.
EDGE_AVOID_BAND = [(3.0, 4.0), (5.5, 6.0)]
EDGE_AVOID_BAND_MIN_ELO_AGREE = 0.5
USE_TIER_STAKE_GATES = False
WALKFORWARD_EDGE_MIN_FLOOR = 5.5
# Walk-forward diagnostics showed confidence tier 2 ≈ −47% ROI — zero that stake.
CONFIDENCE_TIER_2_STAKE_MULT = 0.0

# --- Calibration ---
CALIBRATION_MODE = "legacy_stack"  # legacy_stack | simplified | beta_ats
USE_VENN_ABERS_FILTER = False
VENN_ABERS_MAX_WIDTH = 0.15
USE_SKELLAM_COVER_PROB = False
ATS_CLASSIFIER_BLEND = 0.5
ATS_CLASSIFIER_MIN_PROB = 0.524
# Cap calibrated cover probs so overconfident tails cannot drive stakes.
COVER_PROB_CAP_LO = 0.45
COVER_PROB_CAP_HI = 0.62
# Soft-cap confidence_score (0-100) to dampen discrete 99 spikes from Phase 2a.
CONFIDENCE_SCORE_SOFT_MAX = 85.0
# Temperature >1 softens calibrated cover probs toward 0.5 (continuity / anti-clump).
CONFIDENCE_PROB_TEMPERATURE = 1.15
# Heteroscedastic cover σ: max(conf_width/2.5, scale * MATCHUP_VOL_SIGMA, 4).
USE_MATCHUP_VOL_FOR_COVER = True
MATCHUP_VOL_SIGMA_COVER_SCALE = 1.0
# Optional scenario-mixture std as another cover-σ floor when available.
USE_SCENARIO_MIX_SIGMA_FOR_COVER = True
# Proxy mixture σ from rotation uncertainty when full scenario mixer is unavailable.
SCENARIO_MIX_SIGMA_BASE = 10.0
SCENARIO_MIX_ROTATION_SCALE = 8.0
SCENARIO_MIX_VOL_BLEND = 0.35
# Dampen favorite-side ATS cover when P(upset) is elevated.
UPSET_ATS_COVER_DAMPEN = 0.35
UPSET_ATS_DAMPEN_THRESHOLD = 0.55

# --- ML win-probability path ---
WIN_PROB_SOURCE = "auto"  # auto | meta_win | rolling_platt | margin_isotonic
REQUIRE_META_WIN_WHEN_FITTED = True
ML_CALIB_METHOD = "isotonic"  # platt | isotonic | none — prefer isotonic for betting calib
ML_CALIB_SCOPE = "prior_year"
# Fit separate underdog vs favorite ML calibrators (fixes dog/fav cal gaps).
ML_SPLIT_CALIB_BY_FAVORITE = True
# Only apply ML calibrated probs when prior-year ECE improves vs raw.
ML_CALIB_REQUIRE_ECE_IMPROVEMENT = True
# UpsetClassifier blend into underdog ML EV when P(upset) is high.
USE_UPSET_CLASSIFIER = True
UPSET_ML_EV_BLEND = 0.25
UPSET_VOL_HIGH_SIGMA = 14.0
# Also shrink ATS cover when leaning WITH the favorite and P(upset) is high.
UPSET_ATS_COVER_BLEND = 0.35
UPSET_ATS_MIN_PROB = 0.55
# Blowout probability thresholds for MetaScore heads.
BLOWOUT_THRESHOLDS = (10, 15, 20)
USE_BLOWOUT_STAKE_ADJUST = True
BLOWOUT_FAV_STAKE_BOOST = 1.15
BLOWOUT_DOG_TRAP_MIN_P = 0.35
# Proxy scenario-mix σ from quantile width when rotation mixer is unavailable.
SCENARIO_MIX_SIGMA_FROM_QUANTILE = True
SCENARIO_MIX_QUANTILE_SCALE = 0.40
# Minimum calibrated win% (0-100) to flag an ML bet (separate from spread CONFIDENCE gate).
MIN_ML_WIN_PCT = 55
# Extra EV required when betting the underdog side (decimal >= 2.0). None = use min_ev only.
ML_UNDERDOG_MIN_EV = None
# Shrink model win prob toward market implied before ML EV (0=off). Applied per bet side.
ML_MARKET_SHRINK = 0.20
# Additional shrink when the bet side is an underdog (decimal >= 2.0).
ML_UNDERDOG_EXTRA_SHRINK = 0.10
# Extra shrink for large favorites (fav cal gaps ~15–30pp in recent walk-forwards).
ML_FAVORITE_EXTRA_SHRINK = 0.15
ML_FAV_ABS_SPREAD_MIN = 8.0
ML_FAV_WIN_PROB_MIN = 0.75
# False = pick best calibrated-EV side; True = only bet predicted winner.
ML_BET_PREDICTED_WINNER_ONLY = False
# Default ML gate: minimum expected ROI (EV) to place any moneyline bet. Pass otherwise.
ML_MIN_EV = 0.08
# |WIN_PROB - 0.5| within this band = coin flip (tight toss-up zone).
ML_COIN_FLIP_BAND = 0.03
# When True, coin-flip games may bet underdog only; False = Pass on coin flips.
ML_COIN_FLIP_UNDERDOG_ONLY = False
# In coin-flip games, underdog bets need this higher EV bar (only exceptional value).
ML_COIN_FLIP_UNDERDOG_MIN_EV = 0.10
# Default cap for walk-forward favorite decimal search (allows shorter favorites than 1.45).
ML_MAX_FAVORITE_DECIMAL_DEFAULT = 1.60

# Phase 2 backtest: logistic on raw features → isotonic for calibrated P(cover).
# Phase 2a tunes composite weights; inline backtest tuning refines each season.
CONFIDENCE_MODE = "unified"  # default | unified
# "auto" = Phase 2a picks method by prior-year ECE (then Brier); else fixed method.
CONFIDENCE_CALIB_METHOD = "auto"  # auto | isotonic | platt | logistic_features | logistic_isotonic | none
CONFIDENCE_CALIB_METHOD_CANDIDATES = (
    "logistic_features", "isotonic", "platt", "logistic_isotonic",
)
CONFIDENCE_CALIB_SCOPE = "prior_year"  # prior_year (last season only) | all_prior
TUNE_CONFIDENCE_WEIGHTS_IN_BACKTEST = True  # walk-forward weight / logistic-C search each season
# Raised from 46 after confidence-threshold grid peak at 55 (~56.6% ATS / +8% ROI).
MIN_CONFIDENCE_SCORE = 55
# Soft upper dampening via CONFIDENCE_SCORE_SOFT_MAX; hard max still optional.
MAX_CONFIDENCE_SCORE = None
CONFIDENCE_SELECTION_MODE = "min_score"  # min_score | off — only gate for actionable spread bets
CONFIDENCE_STAKE_MODE = True  # scale stake by confidence_score / 100
# Year-over-year search shrink for Phase 2a per-variable weight tuning (0–1).
CONFIDENCE_WEIGHT_SHRINK = 0.55
CONFIDENCE_WEIGHT_SEARCH_SAMPLES = 120
# Adaptive gate may RAISE min_conf above MIN_CONFIDENCE_SCORE, never lower it.
# Cap season lift so 60–65 floors cannot collapse volume (seen 2024–25 → 80 bets).
CONFIDENCE_ADAPTIVE_FLOOR = True
CONFIDENCE_ADAPTIVE_TARGET_FRAC = 0.20  # unused for lowering; kept for diagnostics
CONFIDENCE_HARD_FLOOR = 55.0  # must not undercut MIN_CONFIDENCE_SCORE
CONFIDENCE_GATE_MAX_LIFT = 8  # season gate ∈ [MIN, MIN+8]
# Absolute Phase 2a weight clamps (prevent cover_scale 35→312 explosions).
CONFIDENCE_WEIGHT_ABS_BOUNDS = {
    "edge_slope": (0.5, 8.0),
    "bucket_lift_scale": (10.0, 120.0),
    "interval_scale": (2.0, 30.0),
    "interval_cap": (18.0, 36.0),
    "unc_penalty_scale": (1.0, 40.0),
    "agree_bonus": (0.0, 40.0),
    "agree_penalty": (-20.0, 0.0),
    "cover_scale": (10.0, 80.0),
    "vol_penalty_scale": (2.0, 40.0),
    "trust_scale": (2.0, 40.0),
    "phantom_penalty": (0.0, 25.0),
    "ats_scale": (5.0, 60.0),
    "win_prob_scale": (5.0, 60.0),
    "elo_margin_scale": (0.2, 6.0),
    "elo_align_bonus": (1.0, 30.0),
}

# --- Stake sizing ---
STAKE_SIZING_MODE = "edge_scaled"  # legacy_kelly | edge_scaled | volatility_adjusted
KELLY_FRACTION_CAP = 0.25
SLATE_CORRELATION_PENALTY = 0.15
SLATE_CORRELATION_MIN_BETS = 5

# --- Feature flags ---
USE_GARBAGE_WEIGHTED_FORM = False
USE_DECAYED_SOS = False
SOS_DECAY_LAMBDA = 0.05

# --- Pythagorean expectation (rolling, pre-game only) ---
PYTHAGOREAN_EXPONENT = 13.91
PYTHAG_MIN_GAMES = 5

# --- ML market de-vig (shrink target only; does not change posted odds for EV) ---
ML_MARKET_DEVIG = True

# --- Optional CLV gate for actionable spread bets (pre-game closing line) ---
REQUIRE_NONNEGATIVE_CLV = False
MIN_CLV_POINTS = 0.0
TIGHT_SPREAD_MAX = 3.5  # diagnostics segment |market spread| <= this
# Reject total-head tuning when CV MAE exceeds this (reuse margin-only fit).
# Raised slightly after dropping poisoned early seasons; do not delete the gate.
TOTAL_HEAD_CV_SANITY_CAP = 45.0

# --- Totals (O/U) ---
# Gates: |PRED_TOTAL - MARKET_TOTAL| >= OU_MIN_EDGE and optional Gaussian P(side).
OU_BET_ENABLED = True
OU_MIN_EDGE = 4.5
OU_REQUIRE_POSITIVE_CI = True
# Use N(pred_total, sigma^2) for P(Over)/P(Under) betting gates.
OU_USE_GAUSSIAN_PROB = True
# Minimum P(chosen side) to fire an O/U bet (with edge).
OU_MIN_PROB = 0.55
# Fallback sigma (points) when walk-forward RMSE is unavailable.
OU_DEFAULT_SIGMA = 12.0
# Prefer score-pair derived total when available and validated.
# Canonical totals path (T-60 Phase 5). MetaTotalModel remains legacy fallback only.
USE_SCORE_PAIR_TOTAL = True
# When True, MetaScorePairModel (actual home/away points) is the sole production
# source of truth for scores; margin/total are derived. Margin-residual and
# market-total heads remain research ablations only.
USE_CANONICAL_SCORE_PAIR = True
# Soft blend of structured pace/efficiency prior into paired home/away scores
# (applied before deriving margin/total).
SCORE_PAIR_STRUCT_BETA = 0.15
# Soft blend of Elo/matchup prior into paired home/away (research ablation; 0 = off).
SCORE_PAIR_ELO_BETA = 0.0
# Default residual correlation between home and away score errors (shared pace).
SCORE_PAIR_DEFAULT_CORR = 0.35

# --- Model options ---
USE_LIGHTGBM_BASE = False
APPLY_SPREAD_CALIB_IN_CV = True
# Hierarchical pace / rotation / shot-decomposition / calibration upgrades.
USE_HIERARCHICAL_PACE = True
USE_ROTATION_SCENARIOS = True
USE_HIERARCHICAL_SHOT_ZONES = True
USE_STRUCTURED_SCORE_FEATURES = True
USE_HYBRID_STRUCTURED_BLEND = True
# Off until single-path ATS cover calibration is verified (avoids 4th stacked calibrator).
USE_TARGET_FAMILY_CALIBRATION = False
LINEUP_COMPOSITE_EVIDENCE_POOLING = True
# Soft blend weights when USE_HYBRID_STRUCTURED_BLEND (structured prior → heads).
HYBRID_TOTAL_BETA = 0.20
HYBRID_MARGIN_BETA = 0.12

# --- Dedicated ML / Total model training (Phase 3) ---
ML_USE_SPREAD_FEATURES = True
TOTAL_TRAIN_TARGET = "market_residual"  # market_residual | absolute
META_WIN_TRIALS_DEFAULT = 25
META_WIN_LOSS_BRIER_WEIGHT = 0.70
META_WIN_LOSS_ECE_WEIGHT = 0.30
# ROI reserved for policy_tuning gates only — not Optuna model search.
META_WIN_LOSS_ML_ROI_WEIGHT = 0.0
TOTAL_LOSS_MAE_WEIGHT = 1.0
TOTAL_LOSS_OU_ROI_WEIGHT = 0.0  # ROI reserved for policy_tuning only
OU_CALIBRATOR_MIN_PROB = 0.524
