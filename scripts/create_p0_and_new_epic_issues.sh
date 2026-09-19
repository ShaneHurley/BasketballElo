#!/usr/bin/env bash
# Adversarial-review follow-up: P0 verified-bug issues + Epic 7/8/9 task issues.
# Source: docs/adversarial_review_2026.md + docs/roadmap.md (P0 section).
# Idempotent: skips issues whose titles already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="ShaneHurley/BasketballElo"

P0_MS="P0 — Verified correctness bugs (adversarial review 2026-09)"
M7="Epic 7 — Wire up what already exists"
M8="Epic 8 — Unify n-man synergy estimators"
M9="Epic 9 — Rating update quality (Kalman + evidence weighting)"

ensure_milestone() {
  local title="$1"
  if ! gh api "repos/$REPO/milestones" --jq '.[].title' | grep -qxF "$title"; then
    gh api "repos/$REPO/milestones" -f title="$title" >/dev/null
    echo "Created milestone: $title"
  fi
}

issue_exists() {
  gh issue list --repo "$REPO" --search "in:title \"$1\"" --state all --limit 50 --json title --jq '.[].title' | grep -qxF "$1"
}

make_issue() {
  local milestone="$1" title="$2" body="$3"
  if issue_exists "$title"; then echo "Issue exists, skipping: $title"; return; fi
  gh issue create --repo "$REPO" --title "$title" --milestone "$milestone" --body "$body" >/dev/null
  echo "Created issue: $title"
}

ensure_milestone "$P0_MS"
ensure_milestone "$M7"
ensure_milestone "$M8"
ensure_milestone "$M9"

# ---- P0 verified bugs (evidence in docs/adversarial_review_2026.md) ----

make_issue "$P0_MS" "P0.1 Chemistry & lineup-Elo train on home lineups only" "**Severity: critical. Verified by direct code read.**

\`code/pipeline/game_updates.py:120-123\` is the only call site: \`chemistry_tracker.update_stint(hp, ap, xh, xa, ...)\` and \`lineup_elo_tracker.update_stint(hp, ap, xh, xa, ..., True)\` are each called **once per stint, home lineup as offense**.

- \`chemistry.py:36-58\` only writes \`duo_off\`/\`trio_off\`/\`on_off[\"on\"]\` for \`off_ids\`; \`on_off[\"off\"]\`/\`[\"off_p\"]\` (declared at \`chemistry.py:23\`) are **never written anywhere** — the on/off differential is half-implemented.
- \`lineup_elo.py:59-75\` updates only \`key_off\`; \`lineup_elo.py:72\` (\`self.dff[key_off] += k * (-err * 0.5)\`) feeds the *offense's own* error into the defensive field. The defending unit's \`dff\` is never updated by any code path.
- Contrast proof this is a bug not a design: \`hierarchical.py:107-122\` updates both \`cb_off\` and \`cb_def\` combos symmetrically.

**Impact:** ~half of all stint data silently dropped for two of six rating engines; road-game chemistry never recorded; lineup5 defensive ratings are noise.

## Fix
- [ ] Mirror the update call with \`(ap, hp, xa, xb)\` per stint (or make both trackers internally symmetric like \`hierarchical.py\`)
- [ ] Implement the \`off\` leg of \`on_off\`
- [ ] Regression test \`test_lineup_trackers_update_both_sides\`
- [ ] \`LEAK_REGISTRY.md\` entry: \`asymmetric_lineup_training_bias\` (broaden registry to confirmed non-temporal defects)"

make_issue "$P0_MS" "P0.2 spread_kelly_fraction payout ratio inverted for juice ≠ −110" "**Severity: high. Verified.**

\`market.py:574\`: \`b = 100.0/110.0 if juice == -110 else abs(juice)/100.0\`. For −120 the correct payout is \`100/120 = 0.833\`; the code returns \`1.2\` — a **44% stake overstatement**. Production-reachable: \`stake_profiles.py:82\` ← \`metrics.py:818-830\` passes real \`SPREAD_PRICE\`/\`JUICE\` whenever \`abs(juice) >= 100\`.

## Fix
- [ ] \`b = 100.0/abs(juice) if juice < 0 else juice/100.0\`
- [ ] Property test vs \`american_to_decimal\` (\`b == dec - 1\`) across a juice grid {-130, -120, -115, -110, -105, +100, +105, +120}"

make_issue "$P0_MS" "P0.3 Moneyline de-vig silently no-ops on single-sided feeds" "**Severity: high. Verified.**

\`market.py:280-296\` \`fair_probs_from_ml_pair\`: when \`market_ml_away\` is missing, \`p_away = implied_probability(-market_ml_home)\` — American-odds negation sums to exactly 1.0, so \`devig_two_way\` (\`market.py:270-277\`) normalizes by 1.0 and does nothing. Every downstream ML EV/edge/calibration target from single-sided historical feeds (\`load_modern_odds\`) uses the **raw vigged** probability labeled as 'fair'.

## Fix
- [ ] When only one side is known, apply an assumed-vig shrink toward 0.5 (league-average vig factor) OR flag \`ml_fair_probs_estimated=True\` and exclude from ML calibration targets
- [ ] Never present negation output as de-vigged; test that single-sided input is flagged"

make_issue "$P0_MS" "P0.4 Elo calibrator prediction interval frozen at ±12 pts" "**Severity: medium-high. Verified.**

\`elo_calibration.py:238-244\` \`predict_interval\` reads \`self._resid_q\` default 12.0; \`update_residuals\` (\`elo_calibration.py:243\`) is **never called anywhere** in \`code/pipeline/\` (grep-confirmed). Every interval-based gate downstream reads a hardcoded constant.

## Fix
- [ ] Call \`update_residuals\` after each graded game in the live loop, OR delete the dead API and route callers to \`SpreadCalibrator.predict_interval\` (\`market.py:1271-1278\`), which already implements a working rolling-quantile interval"

make_issue "$P0_MS" "P0.5 Player RD is a games-played counter, not Glicko-2 uncertainty" "**Severity: medium-high. Verified.**

\`ratings.py:485-500\`: \`rd_new = rd * (0.99 + 0.02 * min(abs_err, 0.15))\` — multiplier ∈ [0.99, 0.993], so RD **always shrinks** regardless of outcome surprise, hitting its floor in ~244 stints (< 1 season). Combined with \`k_mult = max(0.5, 2.0*exp(-games/24.4))\` (\`ratings.py:470\`), effective learning rate decays ~40× within a season — the exact slow-reaction problem Epic 2.4 (Kalman) is meant to solve. RD feeds \`h_rating_uncertainty\`, \`elo_consistency\`, \`uncertainty_diff\` in \`SAFE_FEATURE_COLS\`.

## Fix
- [ ] Replace with scalar Kalman update (process noise \`q\`, observation noise \`r(poss)\`, gain \`K = rd²/(rd²+r)\`) — Epic 9.1
- [ ] Until then: stop feeding RD-derived features to the model as if they were uncertainty (or relabel honestly)"

make_issue "$P0_MS" "P0.6 HAPM shrinkage is a constant, not empirical Bayes" "**Severity: medium. Verified.**

\`hapm.py:71,76\`: \`coef * (SHRINK / (SHRINK + 50))\` = 80/130 ≈ 0.615 applied to every dyad/trio coefficient regardless of possession count — no \`n\` in the formula. A 5,000-possession duo and a barely-seen duo get identical damping.

## Fix
- [ ] \`coef * n/(n + SHRINK)\` using actual per-key possession counts (mirror \`chemistry.py:33-34\`'s real EB shrink), or subsume into Epic 8's unified RAPM"

make_issue "$P0_MS" "P0.7 Stale HOME_PPP_BOOST=0.024 landmine defaults (12× tuned 0.002)" "**Severity: medium (latent). Verified.**

\`feature_utils.py:71\` \`elo_cfg.get(\"HOME_PPP_BOOST\", 0.024)\` and \`lineup_elo.py:15\` constructor default \`home_boost=0.024\` vs \`config.py:64\` \`HOME_PPP_BOOST = 0.002\`. Currently latent (live cfgs carry the key; \`lineup_elo.expected_margin\` is only called from dead locals at \`lineup_elo.py:66-67\`), but any stripped cfg / test double / refactor silently reactivates a 12× home boost.

## Fix
- [ ] Remove both defaults; import from \`config.py\` or require explicit kwargs
- [ ] Delete dead \`exp_margin\`/\`act_margin\` locals in \`lineup_elo.py:66-67\`"

make_issue "$P0_MS" "P0.8 _weight_stint garbage-time discount mis-fires and double-applies" "**Severity: medium. Verified.**

\`ratings.py:506-515\`: \`if abs(margin) >= 25: weight *= gt_w\` has **no period guard** (a 25-pt Q2 lead counts as 'garbage'), and for a true 4th-quarter blowout with \`stint_ctx[\"garbage\"]=True\` the weight is multiplied by \`gt_w\` **twice** (0.3² = 0.09 instead of the configured single 0.3).

## Fix
- [ ] \`elif\`-chain the branches with period guards
- [ ] Unit test the four quarter × margin combinations"

make_issue "$P0_MS" "P0.9 Venn-Abers filter fails open on invalid width" "**Severity: medium. Verified.**

\`venn_abers.py:34-37\`: \`passes_venn_abers_filter\` returns \`True\` when width is None/NaN — a risk-limiting filter that approves bets it couldn't evaluate. Used in \`predict.py:809-811\` and \`simulate.py:894-901\`.

## Fix
- [ ] Return \`False\` on non-finite width (fail-closed, matching the codebase's date/odds philosophy)
- [ ] Add regression test"

make_issue "$P0_MS" "P0.10 fair_spread_vigfree is the raw vigged spread, mislabeled" "**Severity: low-medium. Verified.**

\`market.py:562-565\`: \`fair_spread = market_spread  # placeholder when single-book\` — passed through unchanged, then fed to the model as a named 'vig-free' feature (\`model.py\` \`MARKET_MICRO_COLS\`). Misleads SHAP audits and feature-importance reads.

## Fix
- [ ] Rename to \`market_spread_raw\` until real multi-book de-vig exists (Epic 5.1), or drop the column"

make_issue "$P0_MS" "P0.11 CLV dead in latest full run — investigate before trusting ROI" "**Severity: high (diagnostic). Verified.**

\`output/20260914_224422_dash_standard/checkpoints/review.json\`: \`n_actionable: 0\`, \`n_finite_clv: 0\`, \`mean_clv: NaN\` across 5,247 backtest games (spread MAE 11.51 ± 0.64, ECE 0.065). Likely downstream of the still-\`confirmed\` \`quote_tip_proxy\` leak — decision≠close pairs never materialize.

## Fix
- [ ] Trace \`odds_provenance.json\` / \`quote_source\` distribution for the run
- [ ] If tip-proxy-only: prioritize roadmap 5.4 (provenance debt) above all other Epic 5 work"

# ---- Epic 7 — wire up what exists ----

make_issue "$M7" "7.1 Wire Shin/Power devig into live path, bucketed by time-to-tip" "\`devig.py\`'s Shin/power/odds-ratio methods + \`select_devig_method_from_folds\` are built and tested but never imported by \`market.py\`/\`predict.py\`/\`backtest.py\`. \`market_snapshots.py\` already computes \`minutes_before_tip\`. Implements roadmap 5.7.

- [ ] Bucket quotes by horizon (>240min / 60–240min / <60min) using existing open/decision quote selectors
- [ ] Per-bucket fold selection of devig method; wire chosen method into \`fair_probs_from_ml_pair\` path
- [ ] Validate method choice vs closing-line consensus accuracy"

make_issue "$M7" "7.2 Wire robust Kelly / slate optimization / bankroll sim into production staking" "\`stake_profiles.py\`'s \`kelly_fraction\`, \`robust_fractional_kelly\`, \`optimize_slate_stakes\`, \`simulate_bankroll\`, \`recommend_fraction_under_risk\` are built and unit-tested but never called from production. Live staking stacks ad hoc multiplicative penalties (\`interval_penalty\` × \`unc_penalty\` double-count correlated uncertainty).

- [ ] Replace penalty stack in \`compute_stake\` with one uncertainty-shrunk probability → Kelly
- [ ] Wire \`optimize_slate_stakes\` covariance handling into \`apply_daily_caps\`
- [ ] Add drawdown-based circuit breaker via \`simulate_bankroll\` abstain path
- [ ] Backtest vs current fractional Kelly: max drawdown, ruin probability, ROI"

make_issue "$M7" "7.3 Venn-Abers interval vs market implied prob on the ML head" "Roadmap 5.5's actual ask — today \`venn_abers.py\` output is a width-only ATS filter, never compared against the market's own implied probability.

- [ ] Surface calibrated \`(p0, p1)\` bounds on the moneyline head
- [ ] Execution flag when bookmaker implied prob falls entirely outside \`[p0, p1]\`
- [ ] Backtest bound-based filtering vs flat edge threshold"

make_issue "$M7" "7.4 Compose minutes_forecast into EPM team rating (closes 2.2)" "Both halves exist: \`minutes_forecast.py::forecast_team\` projects minutes; \`epm_priors.py::get_as_of\` has versioned EPM. Never composed — \`blend_lineup_off\` (\`epm_priors.py:114\`) blends offense-only at a static 0.35 weight (\`epm_priors.py:27\`) with no in-season decay.

- [ ] Team O/D rating = Σ(projected minutes × player EPM-O/D)
- [ ] Game-count-based decay of EPM blend weight toward internal rating (mirror \`TeamXpppTracker\` warm-start decay)
- [ ] Benchmark vs D&T minutes-weighted team EPM RMSE (~12.1)"

make_issue "$M7" "7.5 Surface monitoring + negative controls on the dashboard" "\`monitoring.py\` (drift/weekly report) and \`negative_controls.py\` (permutation-null leak checks) exist but are visible nowhere.

- [ ] Dashboard health section: weekly drift report, PSI/KS per feature vs training distribution
- [ ] Surface permutation-null results per run in Run Lab"

make_issue "$M7" "7.6 Minimum-sample floors for calibration slices" "- [ ] \`venn_abers.py\`: add \`min_samples\` gate on isotonic fits (currently none — contrast \`WalkForwardEloCalibrator min_samples=80\`)
- [ ] \`calibration_registry.py\`: assert each of the 5 disjoint calibrator sub-slices meets its own minimum N; fail loudly with a recommendation (fewer targets or larger tail) instead of silently fitting isotonic on whatever an equal 5-way split produces"

# ---- Epic 8 — unify synergy estimators ----

make_issue "$M8" "8.1 Prerequisite: P0.1 fixed" "The unified RAPM reads the same stint stream as \`chemistry.py\`/\`lineup_elo.py\`. Fix the home-only training bug first or the consolidated model inherits the asymmetry."

make_issue "$M8" "8.2 Single sparse regularized possession regression (RAPM/PIPM)" "Three independent duo/trio synergy estimates with three inconsistent shrinkage formulas (\`chemistry.py\` EB-shrink, \`hapm.py\` fixed 0.615 scalar, \`hierarchical.py\` tier-weighted additive) all feed the stacker simultaneously, plus \`lineup_composite.py\`'s blend emitted in addition to its own raw ingredients.

- [ ] One design matrix, one regularization path, consistent n-aware shrinkage for player/duo/trio coefficients
- [ ] Replace \`hapm.py\` + \`chemistry.py\` duo/trio pieces
- [ ] Keep \`lineup_elo.py\` 5-man James-Stein as top tier"

make_issue "$M8" "8.3 Ablation gate: unified vs triple-redundant feature set" "- [ ] OOF MAE/log-loss comparison on identical folds
- [ ] SHAP audit post-consolidation (cleaner signal for \`shap_prune.py\`)
- [ ] Document verdict in \`docs/pipeline.md\`"

# ---- Epic 9 — rating update quality ----

make_issue "$M9" "9.1 Scalar Kalman filter for player ratings (fixes P0.5, supersedes 2.4)" "Replace \`ratings.py::_update_ratings\` compounded \`k_mult × rd/350\` decay with a real Kalman update: process noise \`q\`, observation noise \`r\` as a function of stint possessions, gain \`K = rd²/(rd²+r)\`, \`μ ← μ + K·error\`, \`rd² ← (1-K)·rd²\`.

- [ ] Implement + unit tests (small-sample shrink, large-sample convergence)
- [ ] DELTA-style rating-drift feature from the innovation term
- [ ] Ablate vs current decay on OOF MAE"

make_issue "$M9" "9.2 Age-conditioned offseason reversion" "\`ratings.py::offseason_revert\` reverts every player by the same flat 0.15. Needs a new \`player_id → age\` lookup (does not exist in the codebase today).

- [ ] Source player age/birthdate data
- [ ] Young players revert less / expected to improve; 30+ revert toward decline-adjusted prior
- [ ] Wire into season-start prior (roadmap 3.4)"

make_issue "$M9" "9.3 Fatigue-weighted evidentiary discount in _weight_stint" "A tired team's poor stint is weaker evidence of true talent — same logic as the existing garbage-time discount. \`fatigue.py\`/\`travel.py\` state is already computed in \`game_features.py\` but never fed back into the rating update.

- [ ] Thread rest/travel/tz state into \`process_stint\` → \`_weight_stint\` multiplier
- [ ] Ablate on OOF MAE"

make_issue "$M9" "9.4 Referee-crew multiplier in _context_multiplier" "\`refs.py::RefTracker\` already collects crew pace/foul-rate data (\`game_updates.py:227-231\`); nothing consumes it as a rating-update context multiplier.

- [ ] Crew whistle-rate prior → per-stint multiplier in \`ratings.py::_context_multiplier\`
- [ ] Decision gate vs existing context multipliers"

make_issue "$M9" "9.5 Altitude × rest interaction in pace model" "\`ALTITUDE_TEAMS\` exists in \`config.py\`; \`is_altitude\` is a bare additive flag (\`game_features.py:368\`). Denver/Utah altitude interacts with back-to-back fatigue supra-additively.

- [ ] \`is_altitude × (rest<=1)\` interaction term
- [ ] Altitude-specific pace/turnover multiplier in \`PaceTracker.get_expected_pace\`"

echo "Done."
