# Elo → points-per-possession

Early notebooks used a classic logistic update on possession outcomes. The production engine still uses Elo-like ratings, but the **unit of prediction** is expected points per possession (PPP).

## Expected PPP

From `pipeline/ratings.py` / config:

\[
\text{xPPP}_A = \text{league\_xppp} + \text{HOME\_PPP\_BOOST} + \frac{\mu^{\text{off}}_A - \mu^{\text{def}}_B}{\text{ELO\_SCALING\_FACTOR}}
\]

Current constants (`pipeline/config.py`):

| Constant | Value |
|----------|-------|
| `BASE_ELO` | `1500.0` |
| `HOME_PPP_BOOST` | `0.002` |
| `ELO_SCALING_FACTOR` | `1000` (in tracker defaults) |

!!! note "History"
    May 2026 used `HOME_PPP_BOOST = 0.024`. The leak-free rewrite cut this to `0.002` — a large recalibration that often indicates earlier home-edge contamination.

## Team-level Elo (coarse anchor)

`pipeline/team_elo.py`:

\[
\widehat{\text{margin}} = \frac{\mu^{\text{spread}}_{\text{home}} - \mu^{\text{spread}}_{\text{away}}}{25} + \text{HOME\_ADV}
\]

with `HOME_ADV = 2.5`, `K_SPREAD = 20`, decaying as \(K / (1 + 0.02 \cdot n_{\text{games}})\).

## Why PPP instead of win/loss Elo

Training on efficiency (and later xPoints) preserves margin information that binary win/loss Elo throws away. Win probability is derived downstream (sigmoid / isotonic / cover probability), not as the sole rating target.
