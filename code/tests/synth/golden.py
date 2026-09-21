"""Golden-master season snapshot for Epic 10.5 (Task 10.5.1).

A fixed 40-game synthetic season (8 teams x 10 games each, via
``factories.make_season``) is fed through the pipeline stages that run on
synthetic inputs without external data, and every stage output is collected
into one JSON-serializable snapshot. The committed snapshot — the *golden
file* — lives at::

    code/tests/synth/golden_snapshots/golden_season.json

i.e. inside ``code/tests/synth/`` next to the factories that generate it
(``tests/fixtures/`` is reserved for non-synth suite fixtures).

Drift gate: ``tests/test_bench_golden.py`` regenerates the snapshot and
compares it byte-for-byte with the committed file. Any formula change in a
snapshotted stage drifts the file and fails the gate. Intentional formula
changes require an explicit, reviewed regeneration::

    cd code && python3 -m pytest tests/test_bench_golden.py --regenerate-golden

plus a ``FEATURE_SCHEMA_VERSION`` bump — the version is embedded in the
golden file body by ``write_golden`` and asserted by the gate (Task 10.5.4).

Float policy: every float in the snapshot is rounded to 10 significant
digits (``f"{x:.10g}"``) before serialization. This absorbs 1-ulp platform
noise (e.g. libm ``exp`` differences) while staying far tighter than any
real formula change. Non-finite floats serialize as the strings ``"NaN"`` /
``"inf"`` / ``"-inf"`` so the file stays valid JSON (``allow_nan=False``
clean), and ``-0.0`` is normalized to ``0.0`` so sign-of-zero cannot false-
drift the byte comparison.

Stages snapshotted (per-game outputs use strictly pre-game tracker state;
end-of-season state is the raw tracker tables):

* ``pipeline.ratings.PlayerRatingTracker`` — predicted margin / uncertainty /
  implied total per game; full player rating table at season end.
* ``pipeline.lineup_elo.LineupEloTracker`` — ``feature_dict`` per game; raw
  off/def/chem/poss per 5-man key at season end.
* ``pipeline.chemistry.ChemistryTracker`` — ``feature_dict`` per game; raw
  duo/trio/on-off state at season end.
* ``pipeline.market`` — two-sided de-vig (``fair_probs_from_ml_pair``),
  Gaussian cover probs (``spread_cover_prob``), spread/ML edges, Kelly
  fraction, O/U selection, and microstructure features on the season's odds.

Stages deliberately skipped, and why: the meta feature builders
(``pipeline.game_features`` / ``pipeline.features``) need real PBP/odds
history frames; the ML models (XGBoost/CatBoost meta heads) need training
data; the calibrators (``pipeline.elo_calibration`` etc.) need fitted
historical slices. None of that exists for a synthetic season. Their math
is covered by dedicated bench files (``test_bench_calibration.py``,
``test_bench_staking_grading.py``, …) instead.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.config import FEATURE_SCHEMA_VERSION
from pipeline.chemistry import ChemistryTracker
from pipeline.lineup_elo import LineupEloTracker
from pipeline.market import (
    american_to_decimal,
    fair_probs_from_ml_pair,
    market_microstructure_features,
    moneyline_edge,
    select_ou_bet,
    spread_cover_prob,
    spread_edge,
    spread_kelly_fraction,
)
from pipeline.ratings import PlayerRatingTracker

from tests.synth.factories import DEFAULT_TEAMS, make_season, make_stint

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_snapshots"
DEFAULT_SEED = 20260919

#: Name (not filename) of the committed golden season snapshot.
GOLDEN_SEASON_NAME = "golden_season"

#: Significant digits every snapshot float is rounded to (see module docstring).
FLOAT_PRECISION_DIGITS = 10

#: Season / stint-expansion shape constants (fixed so the golden is stable).
GOLDEN_N_TEAMS_GAMES_PER_TEAM = 10  # 8 teams x 10 games / 2 = 40 games
GOLDEN_N_STINTS_PER_GAME = 6
GOLDEN_ROSTER_SIZE = 8
GOLDEN_GAME_POSSESSIONS = 100.0

#: Fixed market-math knobs (documented so a config.py change to the
#: production defaults cannot silently drift the golden — these are pinned).
COVER_CONF_WIDTH = 24.0
OU_SIGMA = 12.0


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _serialize_golden_body(payload: Mapping[str, Any], schema_version: int) -> str:
    """Single source of truth for the on-disk golden file format.

    The drift gate reconstructs the expected file text through this exact
    function so the comparison is byte-for-byte, not dict-for-dict.
    """
    body = {"FEATURE_SCHEMA_VERSION": int(schema_version), "payload": payload}
    return json.dumps(body, indent=2, sort_keys=True, default=str)


def write_golden(name: str, payload: dict[str, Any], *, schema_version: int) -> Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(name)
    path.write_text(_serialize_golden_body(payload, schema_version))
    return path


def load_golden(name: str) -> dict[str, Any]:
    return json.loads(golden_path(name).read_text())


# ---------------------------------------------------------------------------
# Canonical JSON / float rounding
# ---------------------------------------------------------------------------

def _round_json(obj: Any) -> Any:
    """Recursively coerce ``obj`` to strictly JSON-native values.

    Floats are rounded to ``FLOAT_PRECISION_DIGITS`` significant digits;
    non-finite floats become the strings ``"NaN"``/``"inf"``/``"-inf"``;
    ``-0.0`` normalizes to ``0.0``; numpy scalars become Python scalars.
    Unknown types raise ``TypeError`` — silently stringifying them (the way
    ``json.dumps(default=str)`` would) could mask a real schema drift.
    """
    if isinstance(obj, Mapping):
        return {str(k): _round_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_json(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == 0.0:
            return 0.0  # normalize -0.0 so sign-of-zero cannot false-drift
        return float(f"{x:.{FLOAT_PRECISION_DIGITS}g}")
    if isinstance(obj, str) or obj is None:
        return obj
    raise TypeError(f"snapshot value of type {type(obj).__name__} is not JSON-native")


def canonical_snapshot_json(snapshot: Mapping[str, Any]) -> str:
    """Canonical JSON form: sorted keys, compact separators, rounded floats.

    Byte-stable across dict insertion orders and across platforms (shortest
    round-trip repr of an IEEE-754 double is deterministic), so equality of
    these strings is the drift-gate comparison.
    """
    return json.dumps(
        _round_json(snapshot),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


# ---------------------------------------------------------------------------
# Golden season generation
# ---------------------------------------------------------------------------

def generate_golden_season(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """The fixed 40-game synthetic golden season (8 teams x 10 games each).

    Two calls with the same ``seed`` produce identical frames (deterministic
    via ``np.random.default_rng`` inside ``make_season``).
    """
    df = make_season(DEFAULT_TEAMS, GOLDEN_N_TEAMS_GAMES_PER_TEAM, seed)
    assert len(df) == 40, f"golden season must be exactly 40 games, got {len(df)}"
    return df


def _game_ordinal(game_id: Any) -> int:
    """Ordinal embedded in an NBA-style game id (last 5 digits)."""
    return int(str(game_id)[-5:])


def _roster(team: Any) -> list[str]:
    """Fixed 8-man roster for a team (player ids are team-scoped strings)."""
    return [f"{team}_{i}" for i in range(1, GOLDEN_ROSTER_SIZE + 1)]


def _lineup_for_stint(roster: list[str], stint_idx: int, rng: np.random.Generator) -> list[str]:
    """Even stints use the primary five (so one lineup per team warms past
    the lineup-Elo ``MIN_POSSESSIONS`` shrinkage threshold); odd stints draw
    a random five (exercising the cold/shrinkage branch and chemistry churn).
    """
    if stint_idx % 2 == 0:
        return list(roster[:5])
    idx = sorted(int(i) for i in rng.choice(len(roster), size=5, replace=False))
    return [roster[i] for i in idx]


def derive_game_stints(season_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Deterministically expand each game row into stint rows + PBP contexts.

    The per-game RNG is seeded by the ordinal embedded in ``GAME_ID``, so the
    expansion is a pure function of the season frame — no hidden global seed.
    Scoring is allocated from the game's final score, so changing a game's
    margin changes every stint's ``xpts`` (drift-gate sensitivity), and each
    stint's actual ``pts`` is jittered off its ``xpts`` so the
    ``XPPP_ACTUAL_BLEND`` path in ``ratings.py`` sees distinct actual vs
    expected scoring. Context rates (tov / foul-draw / 3pa / rim / def
    events / luck) straddle the ``_context_multiplier`` thresholds so those
    branches fire on some stints and not others.
    """
    periods = (1, 1, 2, 2, 3, 4)
    games: dict[str, list[dict[str, Any]]] = {}
    for row in season_df.to_dict("records"):
        gid = str(row["GAME_ID"])
        rng = np.random.default_rng(_game_ordinal(gid))
        home_roster = _roster(row["home_team"])
        away_roster = _roster(row["away_team"])

        poss_w = rng.uniform(0.7, 1.3, GOLDEN_N_STINTS_PER_GAME)
        poss = GOLDEN_GAME_POSSESSIONS * poss_w / poss_w.sum()
        hx_w = rng.uniform(0.8, 1.2, GOLDEN_N_STINTS_PER_GAME)
        ax_w = rng.uniform(0.8, 1.2, GOLDEN_N_STINTS_PER_GAME)
        home_x = float(row["ACTUAL_HOME"]) * hx_w / hx_w.sum()
        away_x = float(row["ACTUAL_AWAY"]) * ax_w / ax_w.sum()

        h_running = 0.0
        a_running = 0.0
        stints: list[dict[str, Any]] = []
        for s in range(GOLDEN_N_STINTS_PER_GAME):
            h_line = _lineup_for_stint(home_roster, s, rng)
            a_line = _lineup_for_stint(away_roster, s, rng)
            h_usage = {
                p: [float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0))]
                for p in h_line
            }
            a_usage = {
                p: [float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.0, 1.0))]
                for p in a_line
            }
            h_pts = float(home_x[s]) * float(rng.uniform(0.9, 1.1))
            a_pts = float(away_x[s]) * float(rng.uniform(0.9, 1.1))
            period = periods[s]
            garbage = bool(period >= 4 and abs(h_running - a_running) >= 15.0)
            stint_row = make_stint(
                game_id=gid,
                stint_id=s,
                period=period,
                home_players=h_line,
                away_players=a_line,
                possessions=float(poss[s]),
                home_xpts=float(home_x[s]),
                away_xpts=float(away_x[s]),
                home_pts=h_pts,
                away_pts=a_pts,
                home_score_start=h_running,
                away_score_start=a_running,
                home_team=row["home_team"],
                away_team=row["away_team"],
                game_date=row["game_date"],
                garbage=garbage,
                home_usage=h_usage,
                away_usage=a_usage,
            )
            stint_ctx = {
                "garbage": garbage,
                "home_pts": h_pts,
                "away_pts": a_pts,
                "home_tov_rate": float(rng.uniform(0.05, 0.25)),
                "away_tov_rate": float(rng.uniform(0.05, 0.25)),
                "home_foul_draw_rate": float(rng.uniform(0.05, 0.20)),
                "away_foul_draw_rate": float(rng.uniform(0.05, 0.20)),
                "home_3pa_rate": float(rng.uniform(0.20, 0.55)),
                "away_3pa_rate": float(rng.uniform(0.20, 0.55)),
                "home_rim_rate": float(rng.uniform(0.20, 0.45)),
                "away_rim_rate": float(rng.uniform(0.20, 0.45)),
                "home_def_events": float(rng.uniform(0.0, 0.30)),
                "away_def_events": float(rng.uniform(0.0, 0.30)),
                "home_luck_ppp": float(rng.uniform(-0.05, 0.05)),
                "away_luck_ppp": float(rng.uniform(-0.05, 0.05)),
            }
            h_running += h_pts
            a_running += a_pts
            stints.append({"row": stint_row, "ctx": stint_ctx})
        games[gid] = stints
    return games


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _player_state_snapshot(tracker: PlayerRatingTracker) -> dict[str, Any]:
    fields = (
        "O_mu", "D_mu", "D_rim_mu", "D_peri_mu",
        "O_rd", "D_rd", "D_rim_rd", "D_peri_rd",
        "O_sigma", "D_sigma", "Possessions",
        "luck_pts", "def_events", "off_tov",
    )
    out: dict[str, Any] = {}
    for pid in sorted(tracker.players):
        p = tracker.players[pid]
        state = {f: p[f] for f in fields}
        state["games"] = tracker.player_games[pid]
        last = p.get("last_date")
        state["last_date"] = last.isoformat() if last is not None else None
        out[pid] = state
    return out


def _lineup_elo_state_snapshot(tracker: LineupEloTracker) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = set(tracker.off) | set(tracker.dff) | set(tracker.poss) | set(tracker.chemistry)
    for key in sorted(keys):
        name = "/".join(key)
        out[name] = {
            "off": tracker.off.get(key, 0.0),
            "dff": tracker.dff.get(key, 0.0),
            "chem": tracker.chemistry.get(key, 0.0),
            "poss": tracker.poss.get(key, 0.0),
        }
    return out


def _chemistry_state_snapshot(tracker: ChemistryTracker) -> dict[str, Any]:
    duo = {
        "/".join(k): {"off": tracker.duo_off.get(k, 0.0), "poss": tracker.duo_poss[k]}
        for k in sorted(tracker.duo_poss.keys())
    }
    trio = {
        "/".join(k): {"off": tracker.trio_off.get(k, 0.0), "poss": tracker.trio_poss[k]}
        for k in sorted(tracker.trio_poss.keys())
    }
    on_off = {pid: dict(tracker.on_off[pid]) for pid in sorted(tracker.on_off.keys())}
    return {"duo": duo, "trio": trio, "on_off": on_off}


def _run_stages(
    season_df: pd.DataFrame,
    game_stints: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Replay the season through the snapshotted stages (pre-rounding).

    Per-game market/feature outputs use strictly pre-game tracker state
    (predict, *then* observe/update), mirroring live inference order.
    """
    prt = PlayerRatingTracker()
    lelo = LineupEloTracker()
    chem = ChemistryTracker()

    records = season_df.to_dict("records")
    n_games = len(records)
    per_game: list[dict[str, Any]] = []

    for g_idx, row in enumerate(records):
        gid = str(row["GAME_ID"])
        stints = game_stints[gid]
        home5 = _roster(row["home_team"])[:5]
        away5 = _roster(row["away_team"])[:5]
        season_progress = g_idx / max(n_games - 1, 1)

        # --- pre-game: ratings predictions --------------------------------
        margin_pred, unc = prt.predict_game_margin(home5, away5, GOLDEN_GAME_POSSESSIONS)
        total_pred = prt.implied_total(home5, away5, GOLDEN_GAME_POSSESSIONS)
        # Market convention: spread < 0 favors home, so model spread = -margin.
        model_spread = -float(margin_pred)

        # --- pre-game: market math on this game's odds ---------------------
        mkt_spread = float(row["MARKET_SPREAD"])
        ml_home = float(row["MARKET_ML_HOME"])
        ml_away = float(row["MARKET_ML_AWAY"])
        fair_home, fair_away = fair_probs_from_ml_pair(ml_home, ml_away)
        win_prob_home = spread_cover_prob(model_spread, 0.0, conf_width=COVER_CONF_WIDTH)
        cover_home = spread_cover_prob(model_spread, mkt_spread, conf_width=COVER_CONF_WIDTH, direction="Home")
        cover_away = spread_cover_prob(model_spread, mkt_spread, conf_width=COVER_CONF_WIDTH, direction="Away")
        ou_direction, ou_p_over, ou_edge = select_ou_bet(total_pred, float(row["MARKET_TOTAL"]), sigma=OU_SIGMA)
        micro = market_microstructure_features(
            float(row["CLOSING_SPREAD"]) - mkt_spread,  # 0.0 for single-snapshot factory rows
            float("nan"),  # public_home_pct is absent from make_game rows
            mkt_spread,
        )

        per_game.append({
            "game_id": gid,
            "game_date": pd.Timestamp(row["game_date"]).strftime("%Y-%m-%d"),
            "home_team": str(row["home_team"]),
            "away_team": str(row["away_team"]),
            "season_progress": season_progress,
            "elo": {
                "pred_margin": margin_pred,
                "uncertainty": unc,
                "implied_total": total_pred,
            },
            "lineup_elo": lelo.feature_dict(home5, away5, prt),
            "chemistry": chem.feature_dict(home5, away5, prt),
            "market": {
                "fair_home": fair_home,
                "fair_away": fair_away,
                "model_spread": model_spread,
                "spread_edge": spread_edge(model_spread, mkt_spread),
                "cover_prob_home": cover_home,
                "cover_prob_away": cover_away,
                "win_prob_home": win_prob_home,
                "ml_decimal_home": american_to_decimal(ml_home),
                "ml_decimal_away": american_to_decimal(ml_away),
                "ml_ev_home": moneyline_edge(win_prob_home, ml_home),
                "kelly_home": spread_kelly_fraction(cover_home, float(row["JUICE"])),
                "ou_direction": ou_direction,
                "ou_p_over": ou_p_over,
                "ou_edge": ou_edge,
                "microstructure": micro,
            },
        })

        # --- observe: inactivity decay, then stint updates -----------------
        game_player_ids = [
            pid
            for s in stints
            for pid in (str(s["row"]["HOME_players"]).split("-") + str(s["row"]["AWAY_players"]).split("-"))
            if pid
        ]
        # Production passes datetime.date (see game_features.py); a raw
        # Timestamp would silently no-op the inflation branch.
        prt.apply_inactivity_decay(game_player_ids, pd.Timestamp(row["game_date"]).date())

        for s in stints:
            r = s["row"]
            ids_a = [p for p in str(r["HOME_players"]).split("-") if p]
            ids_b = [p for p in str(r["AWAY_players"]).split("-") if p]
            poss = float(r["possessions"])
            hx = float(r["home_xpts"])
            ax = float(r["away_xpts"])
            prt.process_stint(
                ids_a,
                ids_b,
                poss,
                hx,
                ax,
                usage_A=r["home_usage"],
                usage_B=r["away_usage"],
                period=int(r["PERIOD"]),
                start_A=float(r["HOME_SCORE_START"]),
                start_B=float(r["AWAY_SCORE_START"]),
                end_A=float(r["HOME_SCORE_END"]),
                end_B=float(r["AWAY_SCORE_END"]),
                season_progress=season_progress,
                stint_ctx=dict(s["ctx"]),
            )
            lelo.update_stint(ids_a, ids_b, hx, ax, poss, player_tracker=prt)
            chem.update_stint(ids_a, ids_b, hx, ax, poss, player_tracker=prt)

    return {
        "meta": {
            "generator": "generate_golden_season",
            "n_games": n_games,
            "teams": sorted({str(r["home_team"]) for r in records} | {str(r["away_team"]) for r in records}),
            "stints_per_game": GOLDEN_N_STINTS_PER_GAME,
            "roster_size": GOLDEN_ROSTER_SIZE,
            "game_possessions": GOLDEN_GAME_POSSESSIONS,
            "float_significant_digits": FLOAT_PRECISION_DIGITS,
            "stages": ["player_ratings", "lineup_elo", "chemistry", "market"],
            "skipped_stages": {
                "game_features": "needs real PBP/odds history frames",
                "ml_models": "XGBoost/CatBoost meta heads need training data",
                "calibrators": "need fitted historical slices",
            },
        },
        "per_game": per_game,
        "final_state": {
            "player_ratings": _player_state_snapshot(prt),
            "lineup_elo": _lineup_elo_state_snapshot(lelo),
            "chemistry": _chemistry_state_snapshot(chem),
        },
    }


def snapshot_stage_outputs(season_df: pd.DataFrame) -> dict[str, Any]:
    """Run each feasible pipeline stage on ``season_df`` and collect outputs.

    Returns a strictly JSON-native dict (all floats rounded to
    ``FLOAT_PRECISION_DIGITS`` significant digits — see module docstring).
    Deterministic: two calls on equal frames produce byte-identical
    :func:`canonical_snapshot_json` output.
    """
    return _round_json(_run_stages(season_df, derive_game_stints(season_df)))


def regenerate_golden_season(seed: int = DEFAULT_SEED) -> Path:
    """Regenerate and rewrite the committed golden season snapshot file."""
    snapshot = snapshot_stage_outputs(generate_golden_season(seed))
    return write_golden(GOLDEN_SEASON_NAME, snapshot, schema_version=FEATURE_SCHEMA_VERSION)
