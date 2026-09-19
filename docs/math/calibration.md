# Calibration stack

Prediction quality is not left to raw margins. Several calibrated layers sit between engines and bets.

## Three-stage Elo calibrator

`pipeline/elo_calibration.py` — walk-forward Huber residual chain:

1. **Engine** features → Huber → `pred1`  
2. **Uncertainty** features fit on residual → `pred2` (isotonic re-fit on a later slice)  
3. **Market** features fit on remaining residual → `pred3`  

\[
\text{elo\_margin\_calibrated} = \text{pred1} + \text{pred2} + \text{pred3}
\]

Market information enters **last**, so the model must earn residual signal before hugging the line.

## MetaScore blend

`pipeline/model.py` stacks Ridge + CatBoost via `ManualOOFStacker`, then blends toward calibrated Elo:

\[
\text{raw\_margin} = (1-\alpha)\,\text{stack} + \alpha\,\text{elo\_calibrated}
\]

with \(\alpha\) clipped by `ELO_BLEND_ALPHA_MAX` (currently `0.35` in `config.py`).

## Slice registry

`pipeline/calibration_registry.py` fingerprints row-id sets (SHA-256) so Elo / MetaScore / spread / MetaWin / ATS calibrators **cannot** silently reuse the same chronological slice.

## Venn-Abers (optional filter)

`pipeline/venn_abers.py` fits isotonic twice (label forced 0 and 1) producing \((p_0, p_1)\):

\[
p^\star = \frac{p_1}{1 - p_0 + p_1},\qquad \text{width} = p_1 - p_0
\]

Wide intervals can reject bets when the filter is enabled.

## ATS classifier

`pipeline/ats_classifier.py` estimates \(P(\text{cover})\) directly (logistic + isotonic) as an alternative to margin → Gaussian CDF.
