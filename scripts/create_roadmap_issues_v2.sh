#!/usr/bin/env bash
# Roadmap v3 delta: retitle renumbered issues and create new task issues.
# Idempotent: skips issues whose titles already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="ShaneHurley/BasketballElo"

M1="Epic 1 — Player & lineup rolling form features"
M2="Epic 2 — Player-impact-weighted team strength"
M3="Epic 3 — Rating engine upgrades"
M4="Epic 4 — Modeling, simulation & validation rigor"
M5="Epic 5 — Market & betting edge layer"
M6="Epic 6 — Engineering, CI/CD & platform"

issue_exists() {
  gh issue list --repo "$REPO" --search "in:title \"$1\"" --state all --json title --jq '.[].title' | grep -qxF "$1"
}

retitle() {
  local old="$1" new="$2"
  if issue_exists "$new"; then echo "Already retitled: $new"; return; fi
  local num
  num=$(gh issue list --repo "$REPO" --search "in:title \"$old\"" --state all --json number,title --jq ".[] | select(.title==\"$old\") | .number" | head -1)
  if [ -n "$num" ]; then
    gh issue edit "$num" --repo "$REPO" --title "$new" >/dev/null
    echo "Retitled #$num: $old -> $new"
  else
    echo "Not found (skip): $old"
  fi
}

make_issue() {
  local milestone="$1" title="$2" body="$3"
  if issue_exists "$title"; then echo "Issue exists, skipping: $title"; return; fi
  gh issue create --repo "$REPO" --title "$title" --milestone "$milestone" --body "$body" >/dev/null
  echo "Created issue: $title"
}

# ---- Retitles for renumbered tasks ----
retitle "1.4 Team-level rolling additions" "1.5 Team-level rolling additions"
retitle "1.5 Validation & leak hygiene for Epic 1" "1.9 Validation & leak hygiene for Epic 1"
retitle "2.4 Bayesian now-casting for player ratings (DRIP-style)" "2.4 Kalman-filter now-casting (DARKO-style)"
retitle "2.5 Player-aware strength of schedule" "2.9 Player-aware strength of schedule"
retitle "4.3 Monte Carlo game simulation layer" "4.4 Monte Carlo game simulation layer"
retitle "4.4 Honest benchmarking & validation" "4.7 Honest benchmarking & validation"

# ---- Epic 1 new tasks ----
make_issue "$M1" "1.4 Rolling 3-man / duo lineup windows" "## Subtasks
- [ ] Extend \`lineup_form.py\` to 3-man units (larger samples than 5-man, stabilizes faster)
- [ ] Rolling duo net rating to complement \`chemistry.py\` residuals with raw form
- [ ] Same shrinkage + decision-gate protocol as 1.3 (OOF log-loss/MAE + SHAP)"

make_issue "$M1" "1.6 Tempo-dictation matchup features" "## Subtasks
- [ ] Lineup-vs-lineup pace clash features: who dictates tempo when high-pace unit meets slow half-court unit
- [ ] Extend \`pace_mean/var/q10/q90\` with lineup-weighted expected pace distribution
- [ ] Feed into totals head (5.6) as well as spread head"

make_issue "$M1" "1.7 Player PER / BPM / VORP from derived boxes" "## Subtasks
- [ ] Implement PER (league-avg 15), BPM (per-100 impact), VORP (BPM x minutes) on Task 1.1 boxes — VORP was top value signal in Li & Jia 2024
- [ ] Minutes-weighted roster VORP/BPM sums as team-strength features (bridge to Epic 2)
- [ ] Decision gate: ablate vs existing Glicko/EPM features; keep only incremental signal"

make_issue "$M1" "1.8 Spatial & shot-quality rolling integration" "## Subtasks
- [ ] Extend existing \`shot_quality.py\`/\`shot_zones.py\` into rolling 10-game shot-quality-differential features (quality of looks created vs allowed) — EXTEND, do not rebuild
- [ ] Expose rim/three rate + PPS by zone as form windows
- [ ] Decision gate per stat"

# ---- Epic 2 new tasks ----
make_issue "$M2" "2.5 Padding method for high-variance stats" "## Subtasks
- [ ] Implement Medvedovsky padding: add baseline makes/attempts (e.g. 85/242 for 3P%) so estimates regress to league mean until sample surpasses noise threshold
- [ ] Apply to 3P%, FT% and other high-variance rate stats in Epic 1 rolling features
- [ ] Unit tests: small-sample shrinkage correct; large-sample converges to raw"

make_issue "$M2" "2.6 Luck adjustment (LEBRON-style)" "## Subtasks
- [ ] Strip opponent uncontested-3P% variance from defensive ratings (expected vs actual shooting)
- [ ] Garbage-time / rubber-band score-state adjustment audit vs existing \`GARBAGE_TIME_WEIGHT\`
- [ ] Ablate luck-adjusted vs raw defensive form features"

make_issue "$M2" "2.7 Role-aware archetype priors" "## Subtasks
- [ ] Offensive archetype labels stabilize box-score priors to player role (LEBRON-style; feeds 4.2 clustering)
- [ ] Use archetype prior when a player's role changes mid-season (trade, coach change)"

make_issue "$M2" "2.8 Box Creation passing metric" "## Subtasks
- [ ] Implement Ben Taylor's Box Creation polynomial (usage x passing volume x 3P-proficiency sigmoid for spacing gravity)
- [ ] Replace/augment raw assist rates as the offensive-initiation feature
- [ ] Decision gate vs rolling AST% from 1.2"

# ---- Epic 3 new tasks ----
make_issue "$M3" "3.6 Markov state-transition game-flow model" "## Subtasks
- [ ] Model possessions as Markov states (e.g. defensive rebound -> transition score); estimate transition matrices per team/lineup from PBP
- [ ] Simulate play-by-play flows -> alternative game-flow prior for pace/totals
- [ ] Decision gate: compare sim-based totals calibration vs existing score-pair heads"

# ---- Epic 4 new tasks ----
make_issue "$M4" "4.3 Heteroscedastic Huber loss for spread MAE" "## Direct MAE lever.

## Subtasks
- [ ] Replace MSE with Huber loss on margin head: quadratic inside eps, linear beyond — blowouts stop skewing weights
- [ ] Tune eps via Optuna on OOF MAE (never ROI); consider per-context eps (heteroscedastic)
- [ ] Acceptance: OOF spread MAE improvement vs current head on identical folds"

make_issue "$M4" "4.5 Zero-lookahead CV hardening" "## Subtasks
- [ ] Extend \`PastOnlyGroupCV\` with hard chronological gap enforcement between folds
- [ ] SHA-256 dataset hash + timestamp assertion pre-fit: no training row timestamp >= inference timestamp
- [ ] CI test that tampered chronology fails the assertion"

make_issue "$M4" "4.6 SHAP interpretability program" "## Subtasks
- [ ] Global summary plots per model head (quantify each rolling metric's contribution)
- [ ] Dependence plots for key interactions (e.g. Box Creation x elite Defensive EPM)
- [ ] Scheduled SHAP-driven pruning via \`shap_prune.py\`; publish per-run top-features in run manifests"

# ---- Epic 5 new tasks ----
make_issue "$M5" "5.5 Venn-Abers bounded-probability execution" "## Extends existing \`venn_abers.py\` — do not rebuild.

## Subtasks
- [ ] Surface calibrated upper/lower probability bounds on the moneyline head
- [ ] Execution flag when bookmaker implied probability falls entirely outside the calibrated bound
- [ ] Backtest bound-based filtering vs flat edge threshold"

make_issue "$M5" "5.6 Totals decomposition & Skellam margin clusters" "## Extends existing \`skellam.py\` — do not rebuild.

## Subtasks
- [ ] Decompose totals into expected possessions (1.6 tempo) x expected efficiency, not raw point regression
- [ ] Skellam PMF over independent Poisson home/away expectations -> exact margin-cluster probabilities (P(diff=3,5,7...)) for alternate-spread pricing
- [ ] Totals calibration report: Brier/ECE per totals bucket in dashboard"

make_issue "$M5" "5.7 Temporal de-vigging" "## Extends existing \`devig.py\` — do not rebuild.

## Subtasks
- [ ] Time-to-tipoff de-vig matrix: Shin method (sharp-money flow) for lines <2h from tip; Power method for early low-liquidity lines
- [ ] Validate de-vig method choice against closing-line consensus accuracy"

make_issue "$M5" "5.8 CVaR portfolio staking" "## Subtasks
- [ ] Move beyond per-bet Kelly: nightly slate as a correlated portfolio
- [ ] Gaussian copulas to map covariance between same-night games; simulate slate-level outcomes
- [ ] CVaR constraint: size bets so expected loss in worst 5% tail <= configured drawdown limit
- [ ] Backtest vs current fractional Kelly (\`stake_profiles.py\`): max drawdown, ruin probability, ROI"

# ---- Epic 6 new tasks ----
make_issue "$M6" "6.5 Performance engineering" "## Subtasks
- [ ] Profile rating-update loops (\`ratings.py\`, \`hierarchical.py\`, \`game_updates.py\`); vectorize hot Pandas loops (NumPy/Polars/PyArrow)
- [ ] Numba JIT on stint-level aggregation inner loops if profiling justifies
- [ ] CI benchmark gate: full-suite runtime regression check"

make_issue "$M6" "6.6 Automation agents" "## Subtasks
- [ ] Scheduled GitHub Action / Cursor Automation: nightly EPM + injury + odds refresh with failure alerts
- [ ] Dependabot dependency/security updates with CI gate
- [ ] Literature-watch automation: monthly digest issue of new relevant papers"

echo "Done: https://github.com/$REPO/issues"
