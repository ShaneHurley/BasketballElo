"""Epic 10.4.4 — Per-stage permutation nulls for the rating engines.

For each rating engine — player Elo/Glicko (``pipeline.ratings``),
hierarchical possession engine (``pipeline.hierarchical``), lineup Elo
(``pipeline.lineup_elo``), and chemistry tracker (``pipeline.chemistry``) —
this harness:

1. trains the engine walk-forward on a synthetic season with a strong,
   known edge (six teams, constant 5-man lineups, offensive ratings spread
   99–121 pts/100 poss, +2.5 home bonus, per-possession noise 0.75);
2. PRECONDITION (adversarial protocol): asserts the engine shows real
   predictive signal on the held-out tail — Pearson correlation well above
   the null scale AND game-margin MSE below the best-constant-predictor
   noise floor. A permutation test on an engine with no signal to begin
   with proves nothing, so this is asserted, not assumed;
3. reuses ``pipeline.negative_controls.shuffled_outcomes_destroy_edge`` on
   the walk-forward predictions (with both correlation and MSE metrics) —
   permuting outcomes must destroy the prediction/outcome alignment;
4. retrains a fresh engine on the same season with outcome labels
   (``home_xpts``/``away_xpts`` pairs) permuted across stints, and asserts
   the re-learned signal drops to noise level.

Design notes / findings (surfaced per the adversarial protocol, not hidden):

- Stint-level label permutation shrinks game-margin variance: permuted
  stints mix strong- and weak-team outcomes into every game, so shuffled
  game margins have materially lower variance than real ones. The
  shuffled-retrain MSE is therefore compared against its *own* eval noise
  floor (best constant predictor on the shuffled eval slice), never the
  real-data floor — comparing across the two would be apples-to-oranges.
- The hierarchical engine is constructed with its public-API milder
  K-factors (``k_off=1.0, k_def=0.75``). With default K its combo ratings
  random-walk hard enough that raw predicted margins are over-dispersed
  (MSE above the constant-predictor floor despite real correlation
  signal). No ``pipeline/`` source is modified — this is ordinary
  constructor configuration, and it makes the MSE-floor precondition
  meaningful for all four engines.
- All randomness flows through ``np.random.default_rng`` with fixed seeds;
  every number in this file is deterministic.

Runtime: ~10s (200 games x 24 stints, 2 walk-forwards per engine).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.chemistry import ChemistryTracker
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.lineup_elo import LineupEloTracker
from pipeline.negative_controls import shuffled_outcomes_destroy_edge
from pipeline.ratings import PlayerRatingTracker
from tests.synth.factories import make_stint, make_stint_frame

SEED = 21
N_GAMES = 200
STINTS_PER_GAME = 24
TRAIN_GAMES = 120  # eval slice = games [TRAIN_GAMES, N_GAMES), n=80

# Six teams with a strong, known edge gradient (pts per 100 possessions)
# and constant 5-man lineups so lineup-keyed engines can learn.
_TEAMS = [(f"T{i}", rtg, [str(i * 5 + j + 1) for j in range(5)])
          for i, rtg in enumerate([121.0, 115.0, 112.0, 108.0, 104.0, 99.0])]
_HOME_BONUS = 2.5
_NOISE_SD = 0.75

# Assertion thresholds. With n=80 eval games the null correlation scale is
# ~1/sqrt(80) ~= 0.11, so 0.30 is nearly 3 null-sd and 0.15 is ~1.3 null-sd.
_MIN_REAL_CORR = 0.30
_MAX_SHUFFLED_CORR = 0.15
_MIN_CORR_DESTRUCTION = 0.30  # r_real - |r_shuffled|
_NOISE_FLOOR_TOL = 0.95  # shuffled MSE must be >= 95% of its own noise floor


def _simulate_season(seed: int = SEED, *, shuffle_outcomes: bool = False) -> pd.DataFrame:
    """Synthetic season of factory-built stints with a known team-strength edge.

    With ``shuffle_outcomes=True`` the (home_xpts, away_xpts) outcome pairs
    are permuted jointly across all stints — the marginal outcome
    distribution is preserved but the lineup -> outcome mapping is
    destroyed (a proper permutation null).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(N_GAMES):
        hi, ai = rng.choice(len(_TEAMS), size=2, replace=False)
        home_team, home_rtg, home_players = _TEAMS[hi]
        away_team, away_rtg, away_players = _TEAMS[ai]
        for s in range(STINTS_PER_GAME):
            poss = float(rng.uniform(2.0, 5.0))
            xh = poss * (home_rtg + _HOME_BONUS) / 100.0 + rng.normal(0, _NOISE_SD) * np.sqrt(poss)
            xa = poss * away_rtg / 100.0 + rng.normal(0, _NOISE_SD) * np.sqrt(poss)
            rows.append(
                make_stint(
                    game_id=f"g{g:04d}",
                    stint_id=s,
                    home_players=home_players,
                    away_players=away_players,
                    possessions=poss,
                    home_xpts=max(xh, 0.0),
                    away_xpts=max(xa, 0.0),
                    home_team=home_team,
                    away_team=away_team,
                )
            )
    df = make_stint_frame(stints=rows)
    if shuffle_outcomes:
        perm = np.random.default_rng(seed + 999).permutation(len(df))
        df = df.copy()
        df[["home_xpts", "away_xpts"]] = df[["home_xpts", "away_xpts"]].to_numpy()[perm]
    return df


def _make_engine(engine: str):
    if engine == "player":
        return PlayerRatingTracker()
    if engine == "hierarchical":
        # Public-API milder K (see module docstring) — no source changes.
        return HierarchicalPossessionEngine(k_off=1.0, k_def=0.75)
    if engine == "lineup":
        return LineupEloTracker()
    if engine == "chemistry":
        return ChemistryTracker()
    raise ValueError(f"unknown engine {engine}")


def _walk_forward(df: pd.DataFrame, engine: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-game walk-forward: predict the game margin, then train on its stints.

    Returns (actual_margins, predicted_margins) per game. Predictions always
    precede the updates for that game — no within-game leakage.
    """
    eng = _make_engine(engine)
    y_true, y_pred = [], []
    for _, gdf in df.groupby("GAME_ID", sort=False):
        home_ids = gdf.iloc[0]["HOME_players"].split("-")
        away_ids = gdf.iloc[0]["AWAY_players"].split("-")
        poss_total = float(gdf["possessions"].sum())

        if engine == "player":
            margin, _ = eng.predict_game_margin(home_ids, away_ids, poss_total)
        elif engine == "hierarchical":
            xo, xd, _, _ = eng.predict_pts(home_ids, away_ids, poss_total)
            margin = xo - xd
        elif engine == "lineup":
            margin, _ = eng.expected_margin(home_ids, away_ids, poss_total)
        else:  # chemistry — chem_diff is per-100-possessions
            margin = eng.feature_dict(home_ids, away_ids)["chem_diff"] * poss_total / 100.0
        y_pred.append(float(margin))
        y_true.append(float(gdf["home_xpts"].sum() - gdf["away_xpts"].sum()))

        for r in gdf.itertuples(index=False):
            h_ids = r.HOME_players.split("-")
            a_ids = r.AWAY_players.split("-")
            if engine == "player":
                eng.process_stint(
                    h_ids, a_ids, r.possessions, r.home_xpts, r.away_xpts,
                    period=1, start_A=0.0, start_B=0.0,
                    end_A=r.home_xpts, end_B=r.away_xpts,
                )
            elif engine == "hierarchical":
                eng.update(h_ids, a_ids, r.home_xpts, r.away_xpts, r.possessions)
            else:
                eng.update_stint(h_ids, a_ids, r.home_xpts, r.away_xpts, r.possessions)
    return np.array(y_true), np.array(y_pred)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _noise_floor(y_eval: np.ndarray) -> float:
    """MSE of the best constant predictor on the eval slice (noise level)."""
    return _mse(y_eval, np.full_like(y_eval, y_eval.mean()))


def _run_permutation_null(engine: str) -> None:
    # --- Real labels: walk-forward signal -------------------------------
    df = _simulate_season(SEED)
    y_true, y_pred = _walk_forward(df, engine)
    y_eval, p_eval = y_true[TRAIN_GAMES:], y_pred[TRAIN_GAMES:]
    floor = _noise_floor(y_eval)
    mse_real = _mse(y_eval, p_eval)
    r_real = _pearson(y_eval, p_eval)

    # PRECONDITION (adversarial protocol): the engine must show real signal
    # on real labels — correlation well above the null scale AND MSE below
    # the best-constant noise floor. If this fails with a strong synthetic
    # edge, the engine has no signal to permute away: a finding, not
    # something to weaken the test around.
    assert r_real >= _MIN_REAL_CORR, (
        f"{engine}: NO SIGNAL on real labels — r={r_real:+.3f} < {_MIN_REAL_CORR} "
        f"(mse={mse_real:.1f}, floor={floor:.1f}). Permutation null is vacuous without signal."
    )
    assert mse_real < floor, (
        f"{engine}: real-label MSE {mse_real:.1f} not below noise floor {floor:.1f} "
        f"(r={r_real:+.3f})"
    )

    # --- Reused negative control: permuting outcomes must destroy the ----
    # --- alignment of the walk-forward predictions. ----------------------
    ctrl_corr = shuffled_outcomes_destroy_edge(
        y_eval, p_eval, metric_fn=_pearson, n_perm=40, seed=SEED + 7,
        improve_when_lower=False,
    )
    assert ctrl_corr["passed"], f"{engine}: correlation control failed — {ctrl_corr['detail']}"
    assert abs(ctrl_corr["null_mean"]) < abs(ctrl_corr["base_metric"])

    ctrl_mse = shuffled_outcomes_destroy_edge(
        y_eval, p_eval, metric_fn=_mse, n_perm=40, seed=SEED + 7,
        improve_when_lower=True,
    )
    assert ctrl_mse["passed"], f"{engine}: MSE control failed — {ctrl_mse['detail']}"

    # --- Engine-level null: retrain on permuted outcome labels -----------
    df_shuf = _simulate_season(SEED, shuffle_outcomes=True)
    y_shuf, p_shuf = _walk_forward(df_shuf, engine)
    y_shuf_eval, p_shuf_eval = y_shuf[TRAIN_GAMES:], p_shuf[TRAIN_GAMES:]
    r_shuf = _pearson(y_shuf_eval, p_shuf_eval)
    floor_shuf = _noise_floor(y_shuf_eval)
    mse_shuf = _mse(y_shuf_eval, p_shuf_eval)

    assert abs(r_shuf) <= _MAX_SHUFFLED_CORR, (
        f"{engine}: signal survived label permutation — r_shuffled={r_shuf:+.3f} "
        f"(real r={r_real:+.3f})"
    )
    assert r_real - abs(r_shuf) >= _MIN_CORR_DESTRUCTION, (
        f"{engine}: permutation did not destroy signal — r_real={r_real:+.3f}, "
        f"r_shuffled={r_shuf:+.3f}"
    )
    # Shuffled predictions must sit AT/ABOVE their own noise floor (a
    # signal-free model cannot beat the best constant predictor).
    assert mse_shuf >= _NOISE_FLOOR_TOL * floor_shuf, (
        f"{engine}: shuffled MSE {mse_shuf:.1f} beats its own noise floor "
        f"{floor_shuf:.1f} — residual signal survived permutation"
    )


@pytest.mark.bench
def test_bench_permutation_player_elo_glicko():
    """PlayerRatingTracker (ratings.py): signal on real labels, noise after shuffle."""
    _run_permutation_null("player")


@pytest.mark.bench
def test_bench_permutation_hierarchical():
    """HierarchicalPossessionEngine (hierarchical.py): signal, then noise."""
    _run_permutation_null("hierarchical")


@pytest.mark.bench
def test_bench_permutation_lineup_elo():
    """LineupEloTracker (lineup_elo.py): signal, then noise."""
    _run_permutation_null("lineup")


@pytest.mark.bench
def test_bench_permutation_chemistry():
    """ChemistryTracker (chemistry.py): signal, then noise."""
    _run_permutation_null("chemistry")
