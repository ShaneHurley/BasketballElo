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

V3_DATA_PATHS = {
    2021: ROOT / "events_2021_22_pbp_V3.csv",
    2022: ROOT / "events_2022_23_pbp_V3.csv",
    2023: ROOT / "events_2023_24_pbp_V3.csv",
    2024: ROOT / "events_2024_25_pbp_V3.csv",
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
ASSIST_SPLIT = 0.76
USAGE_FLOOR = 0.27
HOME_PPP_BOOST = 0.002
K_MULT_HALF_LIFE = 24.4
GARBAGE_TIME_WEIGHT = 0.30
ALTITUDE_TEAMS = {"DEN", "UTA"}
GOOD_BET_EDGE = 3.5

SOS_WINDOW = 15
OPTIMAL_BET_EDGE = 2.5
SPREAD_CALIB_WINDOW = 50

# Tuning objective / CV constants
TUNING_XPPP_WEIGHT = 0.7
TUNING_POINTS_WEIGHT = 0.3
TUNING_EMBARGO_GAMES = 15
TUNING_INVALID_SCORE = 9999.0
TUNING_BOUND_EDGE_FRACTION = 0.02

META_LOSS_MAE_WEIGHT = 0.35
META_LOSS_BRIER_WEIGHT = 0.25
META_LOSS_ATS_ROI_WEIGHT = 0.50
META_LOSS_CLV_ROI_WEIGHT = 0.15

# ELO calibration / meta-anchor defaults (overridden by walk-forward tuning).
ELO_BLEND_ALPHA = 0.46
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
ARTIFACT_SCHEMA_VERSION = 2

# --- Task 004/017 versioned schema components ---
# Bump whichever of these changes whenever the corresponding logic changes so
# every downstream cache (stints, tuning, backtest CSVs, model artifacts) is
# automatically invalidated instead of silently reused with stale semantics.
# Bumped for Tasks 006-013 (canonical games, mixed-date parsing, garbage-time
# point preservation, zero-possession stint preservation, GAME_ID stint
# boundaries).
PREPROCESSING_SCHEMA_VERSION = 2
# Bumped for Tasks 031-032 (T-60 lineup features no longer fall back to the
# current game's own actual/first-PBP lineup; EPM prior features now require
# a versioned observation_date and expose missing flags instead of a silent
# zero). Any cache/artifact built under FEATURE_SCHEMA_VERSION < 2 encodes
# the removed leak and must not be reused.
FEATURE_SCHEMA_VERSION = 2
# Bumped for Tasks 018-029 (quote-level market_snapshots.py schema,
# open/decision(T-60)/close selection, two-sided ML/spread/total prices,
# canonical-identity game matching, devig.py, point/price CLV separation).
MARKET_SNAPSHOT_SCHEMA_VERSION = 2
VALIDATION_SCHEMA_VERSION = 1
# Freeze definition: predictions may use only data with source/ingestion
# timestamps at or before (scheduled tip - this many minutes).
DECISION_CUTOFF_MINUTES_BEFORE_TIP = 60
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
# confidence_only: |edge| >= CONFIDENCE_MIN_EDGE, then calibrated win% gates actionable bets.
BET_SELECTION_MODE = "confidence_only"  # legacy_tiers | edge_bucket | edge_bucket_ats | confidence_only
# Minimum |model-market| edge (pts) before a game is scored or bet in confidence modes.
CONFIDENCE_MIN_EDGE = 3.0
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
# Optional dead-zone filter: (lo, hi) absolute edge pts; None disables.
# Was (7.0, 9.5) — that band had the strongest lean ATS in diagnostics; leave off.
EDGE_AVOID_BAND = None
EDGE_AVOID_BAND_MIN_ELO_AGREE = 0.5
USE_TIER_STAKE_GATES = False
WALKFORWARD_EDGE_MIN_FLOOR = 5.5  # ignored when BET_SELECTION_MODE == confidence_only

# --- Calibration ---
CALIBRATION_MODE = "legacy_stack"  # legacy_stack | simplified | beta_ats
USE_VENN_ABERS_FILTER = False
VENN_ABERS_MAX_WIDTH = 0.15
USE_SKELLAM_COVER_PROB = False
ATS_CLASSIFIER_BLEND = 0.5
ATS_CLASSIFIER_MIN_PROB = 0.524

# --- ML win-probability path ---
WIN_PROB_SOURCE = "auto"  # auto | meta_win | rolling_platt | margin_isotonic
REQUIRE_META_WIN_WHEN_FITTED = True
ML_CALIB_METHOD = "isotonic"  # platt | isotonic | none — prefer isotonic for betting calib
ML_CALIB_SCOPE = "prior_year"
# Minimum calibrated win% (0-100) to flag an ML bet (separate from spread CONFIDENCE gate).
MIN_ML_WIN_PCT = 55
# Extra EV required when betting the underdog side (decimal >= 2.0). None = use min_ev only.
ML_UNDERDOG_MIN_EV = None
# Shrink model win prob toward market implied before ML EV (0=off). Applied per bet side.
ML_MARKET_SHRINK = 0.20
# Additional shrink when the bet side is an underdog (decimal >= 2.0).
ML_UNDERDOG_EXTRA_SHRINK = 0.10
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
CONFIDENCE_CALIB_METHOD = "isotonic"  # isotonic | platt | logistic_features | logistic_isotonic | none
CONFIDENCE_CALIB_SCOPE = "prior_year"  # prior_year (last season only) | all_prior
TUNE_CONFIDENCE_WEIGHTS_IN_BACKTEST = True  # walk-forward weight / logistic-C search each season
MIN_CONFIDENCE_SCORE = 55
# No hard upper cap by default; allows high-confidence tail when ranking is calibrated.
MAX_CONFIDENCE_SCORE = None
CONFIDENCE_SELECTION_MODE = "min_score"  # min_score | off — only gate for actionable spread bets
CONFIDENCE_STAKE_MODE = True  # scale stake by confidence_score / 100
# Year-over-year search shrink for Phase 2a per-variable weight tuning (0–1).
CONFIDENCE_WEIGHT_SHRINK = 0.55
CONFIDENCE_WEIGHT_SEARCH_SAMPLES = 120

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
TOTAL_HEAD_CV_SANITY_CAP = 35.0

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
USE_SCORE_PAIR_TOTAL = True

# --- Model options ---
USE_LIGHTGBM_BASE = False
APPLY_SPREAD_CALIB_IN_CV = True
META_LOSS_ECE_WEIGHT = 0.10

# --- Dedicated ML / Total model training (Phase 3) ---
ML_USE_SPREAD_FEATURES = True
TOTAL_TRAIN_TARGET = "market_residual"  # market_residual | absolute
META_WIN_TRIALS_DEFAULT = 25
META_WIN_LOSS_BRIER_WEIGHT = 0.40
META_WIN_LOSS_ECE_WEIGHT = 0.20
META_WIN_LOSS_ML_ROI_WEIGHT = 0.40
TOTAL_LOSS_MAE_WEIGHT = 0.55
TOTAL_LOSS_OU_ROI_WEIGHT = 0.45
OU_CALIBRATOR_MIN_PROB = 0.524
