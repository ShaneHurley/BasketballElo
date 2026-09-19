# Staking & bankroll

## Cover probability

Default path (`pipeline/market.py`) uses a Gaussian cover probability:

\[
z = \frac{\text{model\_spread} + \text{market\_spread}}{\sigma},\qquad
P(\text{home covers}) = \Phi(z)
\]

with \(\sigma\) floored (e.g. from quantile width / volatility, minimum ≈ 4.0). Skellam is available as an alternative (`pipeline/skellam.py`) but not the default production gate.

## Fractional Kelly

At American −110, \(b = 100/110\). Full Kelly:

\[
f^\star = \frac{b p - (1-p)}{b}
\]

Production profiles apply a **fraction** of Kelly (`stake_profiles.py`), shrink \(p\) toward market-fair under uncertainty, and cap the fraction (never full Kelly in the aggressive profile sense used for bankroll safety).

## Slate / covariance awareness

`optimize_slate_stakes` / `simulate_bankroll` shrink the outcome covariance toward independence and Monte Carlo bankroll paths so correlation cannot inflate recommended size.

## Confidence score

`bet_confidence.py` builds a hand-weighted ranking score (edge, width, RD, Elo/meta agreement, disagreement trust, phantom-injury flags, ATS classifier, …) tuned walk-forward — a ranking layer, not a probability by itself.
