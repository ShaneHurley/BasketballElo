# De-vig methods

Sportsbook prices include overround ("vig"). `pipeline/devig.py` implements four transforms and selects among them using **strictly prior-fold** OOS loss — no peeking at the evaluation slate.

## Methods

### Multiplicative

\[
p_i = \frac{q_i}{\sum_j q_j}
\]

### Power (Shin–Jullien style)

Solve \(\sum_i q_i^k = 1\) for \(k\), then \(p_i = q_i^k\).

### Odds-ratio

Solve for \(c\) in \(\sum_i \frac{c q_i}{1 - q_i + c q_i} = 1\).

### Shin (insider model)

\[
p_i = \frac{\sqrt{z^2 + 4(1-z) q_i^2 / \sigma} - z}{2(1-z)}
\]

with \(z\) solved so probabilities sum to 1.

## Selection policy

`select_devig_method_from_folds` chooses per `(book, market, horizon)` using prior-fold Brier / log-loss / ECE, shrinking to multiplicative when evidence is thin (`shrink_min_n ≈ 200`).

!!! quote "Design intent"
    No single de-vig is assumed universally correct — the code treats method choice as an empirical, time-safe decision.
