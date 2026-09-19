# Adversarial Review — September 2026

**Method:** three independent deep-dive passes (calibration/market/staking, rating-engine
internals, ML architecture/CV/dashboard) read the actual code in `code/pipeline/` and
`code/dashboard/`, cross-checked against `docs/roadmap.md` (v3, 44 tasks), `LEAK_REGISTRY.md`,
and the math docs, then were synthesized here with epic/stat/calibration recommendations.
Grounded against a real run: `output/20260914_224422_dash_standard` (2021–2025 seasons, 5,247
backtest games, git rev `b49c5d5`) — **spread MAE 11.51 ± 0.64, ECE 0.065 ± 0.023,
CLV n_actionable = 0** (all NaN).

Posture: assume broken until proven otherwise. This is not a criticism of the project's
overall discipline — the leak-hunting culture here is genuinely rare — it's a search for the
next layer of bugs that discipline hasn't caught yet.

---

## 0. Executive summary

1. **One live correctness bug is severe enough to fix before anything else lands:**
   `ChemistryTracker`/`LineupEloTracker` are only ever updated with the home lineup as
   "offense" (`game_updates.py:120-123`). Away-side chemistry/lineup5 data is never trained,
   and the lineup "defensive" rating field is fed the *offense's own* error term, not a
   defensive signal. This roughly halves effective sample size for two of your six rating
   engines and introduces a home/away asymmetry into features presented as symmetric. Not in
   `LEAK_REGISTRY.md` yet — should be Task 0 of any new work.
2. **A second live bug inflates recommended stakes:** `spread_kelly_fraction` (`market.py:571-577`)
   computes the payout ratio `b` backwards for any American price other than exactly −110
   (e.g. −120 juice → `b=1.2` instead of the correct `0.833`, a 44% overstatement of Kelly
   stake). This is reachable from `stake_profiles.compute_stake` in production whenever
   `abs(juice) != 110`.
3. **A third live bug silently disables de-vig on your most common odds feed:** when the away
   moneyline is missing (`load_modern_odds`, the common historical single-sided case),
   `fair_probs_from_ml_pair` negates the home price, which by definition already sums to 1.0
   — so `devig_two_way`'s normalization is a no-op and the "fair" probability used everywhere
   downstream is just the raw vigged number.
4. **You've already built more than the roadmap gives itself credit for.** `refs.py`,
   `fatigue.py`, `travel.py`, `team_volatility.py`, `upset_classifier.py`,
   `market_disagreement.py`, `matchup_rating.py`, `monitoring.py`, `negative_controls.py`,
   `selection_bias.py`, `devig.py`'s Shin/power methods, and `stake_profiles.py`'s
   `optimize_slate_stakes`/`simulate_bankroll`/`robust_fractional_kelly` all exist, are often
   well-tested in isolation, and are **not wired into the live pipeline.** The single highest-
   leverage category of work here is *plumbing, not invention* — see Epic 7 below.
5. **The "soft leak" pattern:** dozens of `config.py` constants (`MIN_CONFIDENCE_SCORE=55`,
   `EDGE_AVOID_BAND`, `MAX_QUANTILE_WIDTH=28.0`, `CONFIDENCE_TIER_2_STAKE_MULT=0.0`, etc.)
   carry code comments explicitly saying they were chosen by inspecting full walk-forward
   results across all seasons, then hardcoded as if they were priors. This is the same class
   of leak the anti-roadmap forbids Optuna from doing — just executed by hand across roadmap
   iterations instead of by a tuner. This deserves the same rigor (hold out a config-tuning
   season, or explicitly document it as accepted risk) that leaked model behavior gets.
6. **CLV is completely dead in your most recent full run** (`n_actionable=0`). Given how much
   of the promotion-gate philosophy depends on CLV, this is worth investigating before trusting
   any ROI claim from that run — likely downstream of the still-`confirmed` (not `fixed`)
   `quote_tip_proxy` leak.
7. **Redundant, uncoordinated synergy estimators.** Three separate statistical estimates of
   duo/trio synergy (`chemistry.py`, `hapm.py`, `hierarchical.py`) with three different, mutually
   inconsistent shrinkage formulas, all feed the stacker simultaneously alongside a "composite"
   that's supposed to disentangle them but is emitted *in addition to*, not *instead of*, the
   raw ingredients. This is a multicollinearity/interpretability risk baked into the feature set.
8. **Documentation has drifted from code.** `docs/pipeline.md` says "98 modules"; actual count
   is 134. `docs/math/calibration.md` describes Shin/Power de-vig as production reality; it's
   test-only, never called from `market.py`/`predict.py`/`backtest.py`. `hapm.py`'s docstring
   calls its shrinkage "empirical Bayes"; it's a fixed scalar unrelated to sample size.

---

## 1. Correctness bugs (ranked by severity, all cite file:line)

| # | Severity | Bug | Location |
|---|---|---|---|
| 1 | **Critical** | Chemistry/LineupElo only ever trained with home lineup as "offense"; away lineup data never recorded; lineup "defense" field fed the offense's own error | `game_updates.py:120-123`, `chemistry.py:36-58`, `lineup_elo.py:59-75` |
| 2 | **High** | `spread_kelly_fraction` computes payout ratio `b` backwards for juice ≠ −110 (inflates stakes up to ~44%) | `market.py:571-577` |
| 3 | **High** | Moneyline devig is a no-op whenever away price is missing — "fair" probability = raw vigged probability | `market.py:280-296` |
| 4 | **High** | `WalkForwardEloCalibrator.predict_interval` returns a frozen ±12pt interval forever — `update_residuals` is never called anywhere in the pipeline | `elo_calibration.py:238-244` |
| 5 | **Medium** | `PlayerRatingTracker`'s RD is not outcome-responsive Glicko-2 RD — it's a disguised games-played counter that hits its floor in ~244 stints (well within one season), after which effective learning rate is pinned near minimum for a player's career unless benched 60+ days | `ratings.py:485-500, 469-472` |
| 6 | **Medium** | `HapmPriorTracker` "shrinkage" is a fixed scalar (0.615) applied to every dyad/trio coefficient regardless of actual possession count — no `n` in the formula at all | `hapm.py:64-77` |
| 7 | **Medium** | `feature_utils.py::engine_implied_margins` reads a stale `HOME_PPP_BOOST` default of `0.024` (12× the real current `0.002` in `config.py`) that activates on any missing-key fallback | `feature_utils.py:71` vs `config.py:64` |
| 8 | **Medium** | `_weight_stint`'s blowout discount has no period guard on one branch and double-applies the garbage-time weight multiplicatively on true 4th-quarter blowouts (0.3×0.3=0.09 instead of 0.3) | `ratings.py:506-515` |
| 9 | **Medium** | `passes_venn_abers_filter` defaults to **pass** (not reject) on NaN/invalid interval width — wrong default for a risk-limiting filter | `venn_abers.py:34-37` |
| 10 | **Medium** | No model/feature schema-version check on `.pkl` load — a model trained under `FEATURE_SCHEMA_VERSION=2` (absolute margin) can silently score under v3 (decision-residual) semantics with zero error raised | `model.py:1274-1279`, `config.py:187` |
| 11 | **Medium** | `_dynamic_weights` gates 5-man-tier trust on the max possession count of *any* lineup league-wide, not the specific lineup being rated — a brand-new never-seen five-man unit inherits inflated trust once any popular lineup crosses the threshold | `hierarchical.py:70-76` |
| 12 | **Low-Med** | `market_microstructure_features`'s `fair_spread_vigfree` is literally the raw vigged spread passed through unchanged, but is fed to the model and SHAP as a named "vig-free" feature | `market.py:550-567` |
| 13 | **Low-Med** | Pervasive bare `except Exception: pass`/silent fallback in live-serving code paths (pace model, structured score, hybrid blend, rotation scenarios, total head) — a systemic bug in any of these degrades every live prediction with zero visible signal | `game_features.py:198-213,437-460,461-479`; `model.py:1216-1223`; `predict.py:487-490,499-522`; `shap_prune.py:20-31` |
| 14 | Low | `O_sigma`/`D_sigma` "volatility" computed, persisted, never read by any consumer — fully inert state in every saved artifact | `ratings.py:488,501,191-192` |
| 15 | Low | `EpmPriorTracker.priors` property is a live back door around the as-of/versioning fix — unused today but not deprecated or guarded against future misuse | `epm_priors.py:80-85` |
| 16 | Low | `fetch_espn_injuries()` has no timestamp/as-of gating at all — safe only because it's currently called solely from the live path (`run_daily.py:242`); nothing stops a future backtest caller from reintroducing a leak identical to the fixed `t60_actual_lineup_leak` | `availability.py:72-90` |
| 17 | Low | `docs/pipeline.md` claims "98 modules"; actual is 134 — stale doc | `docs/pipeline.md:3` |

**Recommendation:** fix #1–#4 immediately (they change live behavior today, not just future
risk). #1 in particular should get its own `LEAK_REGISTRY.md`-style entry (call it
`asymmetric_lineup_training_bias`) even though it's not a temporal leak, because the registry's
whole purpose is "confirmed defect + regression test + fix status," and this qualifies.

---

## 2. Methodological / statistical weaknesses

**Calibration:**
- Isotonic regression is refit on small slices with no minimum-sample guard in `venn_abers.py`
  (contrast: `WalkForwardEloCalibrator` requires `min_samples=80`, `SpreadCalibrator` requires 30).
- `calibration_registry.py`'s disjoint-slice guarantee mechanically divides an already-small
  calibration tail 5 ways (for 5 targets) with no lower-bound check on the resulting per-
  calibrator sample size — leak-safety was bought by shrinking each calibrator's effective N,
  with no corresponding minimum-N gate.
- No proper scoring rule (CRPS, log-loss) is used as a *training* objective anywhere in the
  calibration stack — `crps_gaussian` exists but is diagnostic-only.
- `SpreadCalibrator`'s 50-game rectangular rolling window reacts to genuine regime shifts (new
  season, rule change) with a ~50-game lag in either direction, and its slope clip `(0.3, 1.7)`
  further dampens exactly the fast adaptation a regime shift needs.

**Staking:**
- Production staking (`stake_profiles.compute_stake`) stacks independently-tuned multiplicative
  penalties (`interval_penalty`, `unc_penalty`, `edge_stake_mult`, `volatility_mult`,
  `confidence_tier_stake_mult`) rather than deriving stake from one coherent uncertainty-aware
  Kelly formula — `conf_width` and `rating_uncertainty` are correlated proxies for the same
  thing and risk double-counting.
- The more rigorous machinery (`kelly_fraction`, `robust_fractional_kelly`,
  `optimize_slate_stakes`, `simulate_bankroll`, `recommend_fraction_under_risk`) is fully built,
  unit-tested in isolation, and **never called from production code.** There is no live
  drawdown-based circuit breaker — `simulate_bankroll` can compute "abstain if max drawdown
  exceeds tolerance" but nothing in the live path ever asks it to.
- P&L reporting uses flat unit-stake sums (`calculate_financials`) while the theoretical staking
  model assumes compounding bankroll growth (`simulate_bankroll`) — these two philosophies are
  never reconciled.

**Devig:**
- The rigorous Shin/power/odds-ratio methods in `devig.py` (with `_check_output` guards) are
  never imported by `market.py`, `predict.py`, `simulate.py`, or `backtest.py`. Production devig
  is the ad hoc `devig_two_way`, which has none of those guards and silently returns `(0.5,0.5)`
  on bad input instead of raising.
- No market-type differentiation (spread vs total vs ML carry different vig magnitudes and
  favorite-longshot bias in the literature) and no time-to-tip conditioning, despite
  `market_snapshots.py` already computing `minutes_before_tip` for exactly this purpose.

**Rating engine:**
- RD widening (Elo layer) is fed downstream as if it were a genuine confidence signal, but per
  bug #5 it's nearly deterministic in games-played — "uncertainty theater," not real uncertainty.
- EPM blend weight is a static 35% on opening night and Game 82 alike — no in-season decay
  toward the internal rating (contrast: `TeamXpppTracker.get_rolling_xppp` does this correctly
  with an explicit `warm_start_games` decay).
- Combined `k_mult × (rd/350)` decay in the Elo update compounds into a ~40× reduction in
  effective learning rate within a single season — this is the actual root cause the roadmap's
  "Kalman now-casting" ask is trying to fix, and it's visible today, not just a missing feature.
- No multicollinearity diagnosis (VIF, orthogonalization) between `elo_net`, `hier_net`,
  `matchup_margin`, `lineup5_net`, `chem_net`, `hapm_net_diff` before they all land in
  `SAFE_FEATURE_COLS` together — several are near-deterministic functions of the same four
  underlying numbers.

**ML architecture / CV:**
- Default ensemble is only 2 base learners (Ridge + CatBoost; LightGBM gated off by default),
  weaker diversity than Epic 4.1 assumes.
- `PastOnlyGroupCV` has no embargo/purge gap between folds — rolling-window features are
  autocorrelated across adjacent dates, so a "strictly past" fold boundary can still leak
  near-identical rolling-stat information across the boundary. (Roadmap Task 4.5.1 already
  flags this as open.)
- Roadmap Task 4.5.2 (SHA-256 dataset hash + pre-fit timestamp assertion) does not exist — don't
  conflate it with `calibration_registry.py`'s SHA-256, which fingerprints row-id sets for a
  completely different purpose (calibrator slice disjointness, not dataset-tamper detection).
- SHAP-based feature pruning (`shap_prune.py`) computes SHAP on training data, not OOF/held-out
  predictions — an overfitting base model can over-credit noise features it memorized, causing
  the pruning step to keep the wrong features. No test file exists for this module at all,
  despite it gating which features survive into every future model.
- No calibration curves / reliability diagrams / residual-vs-time diagnostics surfaced anywhere,
  despite Brier/ECE being computed as scalars (Task 4.7.3 still open).

---

## 3. Architecture & engineering gaps

- **Config-as-hidden-hyperparameter-search.** The single biggest gap between stated leak-rigor
  culture and actual practice: dozens of `config.py` constants were chosen by a human looking at
  full-history walk-forward results, then hardcoded as global priors for all future runs. This
  is architecturally the same risk as Optuna-on-the-scoreboard, just manual.
- **No auth/CORS/rate-limiting on the dashboard**, mitigated today only by a default
  `127.0.0.1` bind that a user can override with `--host`. `POST /api/jobs` can fork arbitrary
  training/backtest subprocesses. Fine for solo local use; needs an explicit "never expose this"
  guard rail (or bearer token) if that assumption ever changes.
- **SSE job-events endpoint has no timeout/watchdog** — a hung job holds an ASGI connection open
  forever with a 1-second poll loop.
- **No model cards / artifact self-description.** A `.pkl` in `state/` carries no manifest of
  what feature schema, SAFE_FEATURE_COLS list, or walk-forward window it was validated against.
- **No canary/shadow evaluation before promotion** — a new model is promoted straight from
  walk-forward backtest to the live `state/latest_*.pkl` slot with zero live shadow period to
  catch data-source regressions that only show up in production (odds feed schema changes,
  injury report format drift).
- **No data contracts** (Pandera/pydantic) at pipeline boundaries — `.fillna(0.0)` blanket
  imputation before scaling turns genuinely-missing features (flagged via a `*_degraded` column)
  into a numeric value the model reads as a real, often extreme, observation.
- **Redundant synergy estimators** (see §2) are an architecture problem as much as a stats one —
  three modules solving the same problem independently is genuine technical debt.

---

## 4. Roadmap critique — what's further along, and what's actively regressing, vs. v3's framing

The v3 roadmap (`docs/roadmap.md`) is already unusually well-grounded, but this review found it
undersells some existing capability and oversells the readiness of others:

**Further along than the roadmap admits:**
- Epic 1.8 (shot-quality rolling): `shot_quality.py` already computes rolling xeFG/rim-rate/
  three-rate with prior-season blending — the roadmap's gap is really just the "differential
  over time" framing, not the underlying machinery.
- Epic 2.1 (EPM versioning): fully done via the `epm_prior_unversioned_leak` fix.
- Epic 2.3 (injury/rotation v2): `availability.py::expected_availability_impact` already computes
  something close to "impact × expected-minutes deltas," just via a hand-tuned linear formula
  rather than a principled replacement-EPM calculation.
- Epic 2.6 (luck adjustment): `luck_pts`/`elo_luck_adj_net` already exist; the gap is upgrading
  from possession-share allocation to a true LEBRON-style opponent-shooting-variance strip.

**Actively working against the roadmap's own stated goal:**
- Epic 2.4 (Kalman now-casting) isn't just "absent" — the *current* rating-update mechanism
  (compounded `k_mult × rd/350` decay) is the literal opposite of a Kalman gain: it monotonically
  stiffens instead of adaptively weighting new evidence. This should be reframed from "add
  Kalman" to "replace the current anti-Kalman decay with a real Kalman gain" — a more precise
  and more urgent task than the current roadmap wording implies.
- Epic 2.2 (minutes-weighted impact team rating): both halves of the bridge already exist in
  isolation (`minutes_forecast.py::forecast_team` projects minutes; `epm_priors.py` has
  versioned EPM) but are **never composed** — `epm_priors.py::blend_lineup_off` doesn't consume
  `minutes_forecast.py` at all. This is a one-file integration task, not a research problem —
  worth pulling forward in priority.

**New gap not on the roadmap at all:** the three-way redundant synergy estimator problem (§2/§3)
should become its own cleanup task before Epic 1.3/1.4 (lineup rolling form) adds a *fourth*
independent lineup-signal source on top of three that already disagree with each other.

---

## 5. Recommended changes to current epics

- **Epic 1 (rolling form):** add a **Task 1.0 — fix the home-only lineup training bug** before
  any new rolling-window feature is added on top of `chemistry.py`/`lineup_elo.py` state that is
  currently only half-trained. Building Task 1.3/1.4 on top of the current bug means the new
  rolling features inherit the same home/away asymmetry.
- **Epic 2:** re-scope 2.2 as an integration task ("compose `minutes_forecast.py` output with
  `epm_priors.py`'s `as_of` lookups into a single `epm_team_net` feature") rather than new
  research — it's mostly done. Re-scope 2.4 as "replace `_update_ratings`'s decay with a real
  scalar Kalman filter" rather than "add now-casting" — be explicit that this is a *fix*, not an
  *addition*, since the current mechanism actively fights the goal.
- **Epic 3:** add a task to fix RD (bug #5) as a prerequisite for 3.1 (dual-window ratings) —
  dual windows are much less useful if the underlying single-window RD isn't a real uncertainty
  signal to begin with.
- **Epic 4:** add "de-duplicate the n-man synergy estimators" (a new Task 4.x, described as
  Epic 8 below) ahead of any further ensembling work — feeding a stacker three correlated
  versions of the same signal is a worse starting point than feeding it one well-shrunk version.
- **Epic 5:** re-scope 5.5/5.6/5.7/5.8 as "wire up what already exists" rather than "build" —
  `devig.py`, `skellam.py`, `stake_profiles.py`'s advanced functions are written and tested in
  isolation; the work is integration plus fixing bugs #2/#3 first.
- **Epic 6:** add "make `LEAK_REGISTRY.md` cover non-temporal defects too" — bug #1's asymmetric-
  training-bias class isn't a T-60 leak but is exactly the kind of confirmed defect the registry
  exists to track; broaden its scope note accordingly.

## New Epic 7 — Wire up what you already built

Every task below is integration, not invention. This is the highest ROI-per-hour epic on the
board because the hard statistical work is already done and tested in isolation.

1. Wire `devig.py`'s Shin/Power methods into `market.py`'s live path via
   `select_devig_method_from_folds` + `market_snapshots.py`'s `minutes_before_tip` buckets
   (directly implements roadmap 5.7, and the pieces already exist independently).
2. Wire `robust_fractional_kelly`/`optimize_slate_stakes`/`simulate_bankroll` into
   `stake_profiles.compute_stake` and `apply_daily_caps`, replacing the ad hoc multiplicative
   penalty stack — closes the CVaR-adjacent staking gap (roadmap 5.8) largely for free.
3. Wire `venn_abers.py`'s `(p0, p1)` interval against the market's own implied probability on
   the ML head specifically (roadmap 5.5's actual ask — today it's a width-only ATS filter).
4. Wire `minutes_forecast.py::forecast_team` output into `epm_priors.py::blend_lineup_off`
   (closes roadmap 2.2 — the two halves already exist).
5. Actually call `WalkForwardEloCalibrator.update_residuals` after every graded game in the
   live loop, or delete the dead API and route callers to `SpreadCalibrator`'s working
   conformal-style rolling-quantile interval instead (fixes bug #4).
6. Surface `monitoring.py`'s drift/weekly-report and `negative_controls.py`'s permutation-null
   checks on the dashboard's Documentation/health tab — both exist, neither is visible anywhere.
7. Investigate why the latest full run shows `n_actionable=0` for CLV before trusting any ROI
   number from it — likely tied to the still-open `quote_tip_proxy` leak.

## New Epic 8 — Unify the n-man synergy estimators (RAPM/PIPM consolidation)

Replace `hapm.py` + `chemistry.py`'s duo/trio pieces with a single sparse regularized
possession-level regression (RAPM/PIPM-style: one design matrix, one regularization path,
consistent n-aware shrinkage for player/duo/trio coefficients simultaneously). Keep
`lineup_elo.py`'s 5-man James-Stein layer as the top tier over that. This fixes the fake-shrink
bug (#6), removes three-way multicollinearity, and gives SHAP/feature-pruning a much cleaner
signal to reason about. Companion task: fix the home-only training bug (#1) first, since it
affects the input possession stream this consolidation would read from.

## New Epic 9 — Real-time rating update quality (Kalman, aging, fatigue-weighted evidence)

1. Replace `_update_ratings`'s compounded games-played decay with an actual scalar Kalman
   filter (process noise `q`, observation noise `r` as a function of stint possessions, real
   Kalman gain `K = rd²/(rd²+r)`) — fixes bugs #5 and the roadmap's Epic 2.4 in one change.
2. Age-conditioned `offseason_revert` multiplier (young players revert less, 30+ reverts toward
   a decline-adjusted prior) instead of one flat 0.15 for every player.
3. Fatigue-weighted evidentiary discount inside `_weight_stint` itself (a tired team's poor
   stint is weaker evidence of true talent, same logic as the existing garbage-time discount) —
   sourced from state already computed in `fatigue.py`/`travel.py`, just not fed back into the
   rating update.
4. Referee-crew-tendency multiplier inside `_context_multiplier` — `refs.py`'s `RefTracker` is
   already collecting the data; it's never consumed as a per-stint rating-update multiplier.
5. Altitude × rest interaction term (not just an additive `is_altitude` flag) inside
   `PaceTracker.get_expected_pace` — `ALTITUDE_TEAMS` already exists in `config.py`.

---

## 6. New advanced stats — building on what's already there, not duplicating it

Everything in this section explicitly avoids re-proposing referee tendency, fatigue, travel, or
drift-monitoring features, since those already exist in the codebase (`refs.py`, `fatigue.py`,
`travel.py`, `monitoring.py`) — the ask here is genuinely new signal.

1. **On/off net rating, both legs, both home and away** — cheap to add once bug #1 is fixed,
   since `ratings.py::process_stint` already has the possession-level attribution needed;
   `chemistry.py::on_off["off"]` is declared but never written. This is qualitatively different
   information from the Elo/hierarchical residuals (captures "does the team's *style*, not just
   its Elo, depend on this player being on the floor").
2. **RAPM/PIPM consolidated coefficients** (Epic 8) — a genuinely new, single, well-shrunk
   synergy signal replacing three redundant weaker ones.
3. **Kalman rating-drift feature** (rate of change of the Kalman-filtered talent estimate,
   Epic 9.1) — DARKO's own "DELTA" concept; slow ratings underrate risers, and once you have a
   real Kalman filter this is a nearly-free byproduct (the innovation term itself).
4. **Graph-embedding lineup synergy** — once bug #1 is fixed and 5-man sample sizes roughly
   double, a skip-gram-style player embedding trained on duo/trio co-occurrence residuals would
   generalize to never-seen five-man combinations far better than the current "shrink toward
   additive player-Elo sum" fallback (`lineup_elo.py:47`), which has zero information about *how*
   two specific players' skills interact (e.g. two ball-dominant guards).
5. **Player-aware strength of schedule** (roadmap 2.9, worth calling out again here since it's a
   genuinely new *stat*, not just a plumbing fix): weight past opponents by who actually played
   (minutes-weighted opponent EPM), not team net rating — `_compute_sos` currently ignores
   who was actually on the floor.
6. **CLV decomposition regression** — not a player stat, but a genuinely new *market* stat: use
   the already-separated `point_clv`/`price_clv` (never multiplied together per existing rule)
   as regression targets against model-edge, devig-method choice, and market-microstructure
   noise, to diagnose whether calibration improvements are translating into real line-beating
   skill or just riding market drift.
7. **Aging curve prior** as a first-class player attribute (needs a `player_id → age/experience`
   lookup that doesn't currently exist in any reviewed file) — feeds both Epic 9.2 and could
   become its own rolling feature (`years_experience`, `age_adjusted_prior_delta`).
8. **Box Creation (Ben Taylor) as a genuine passing-gravity metric** — already on the roadmap
   (2.8), worth flagging here as higher priority than it currently reads, since it's the one
   "new stat" idea in Epic 2 that doesn't depend on any other unfinished infrastructure (only
   needs the box-derivation layer from Epic 1.1).

---

## 7. Calibration improvements, ranked by effort-to-impact

1. **Fix bugs #2, #3, #4, #9 first** — no calibration technique is worth adding on top of a
   payout ratio that's wrong, a devig that's a no-op, a dead interval, and a filter that fails
   open.
2. **EWMA-weighted rolling recalibration** in `SpreadCalibrator`, replacing the rectangular
   50-game window — reacts to regime shifts without discarding history at a hard boundary.
3. **Temperature-scaled Platt scaling** as a companion to isotonic in `elo_calibration.py`'s
   group-2 refit, specifically for thin early-season slices where isotonic's unregularized
   step function overfits on <100 games.
4. **Mondrian Venn-Abers with a `min_samples` gate**, stratified by market/edge-bucket, mirroring
   `WalkForwardEloCalibrator`'s existing pattern.
5. **CRPS as an explicit calibration objective**, not just a diagnostic — `crps_gaussian`
   already exists and `promotion_gates.py` already has a `tol=0.5` CRPS gate key; nothing
   currently optimizes it directly, so the mean (Huber) and width (ad hoc quantile) are fit
   separately when they should be fit jointly.
6. **Regime-switching / partial-pooling shrinkage per (book, market, horizon) cell** for devig
   method choice in `store_devig_choice`, replacing the current hard `n_obs >= shrink_min_n`
   cutoff with continuous empirical-Bayes shrinkage toward the global win rate.
7. **Minimum-N floor in `calibration_registry.py`'s disjoint-slice partitioning** — assert each
   of the 5 calibrator sub-slices meets its own `min_samples` requirement, and fail loudly
   (recommend fewer disjoint targets or a larger tail) rather than silently fitting isotonic on
   whatever fraction falls out of an equal 5-way split.

---

## 8. Other improvements (engineering / process)

1. Add a model-card manifest (feature schema version, `SAFE_FEATURE_COLS` hash, validated
   walk-forward window, known caveats) written alongside every `state/latest_*.pkl`, checked at
   `MetaScoreModel.load()` — closes the version-skew hazard (bug #10) cheaply.
2. Pandera/pydantic data contracts at the `generate_features → model.fit` and
   `build_game_features → predict_game` boundaries — turns the "silent 0.0 imputation of a
   degraded feature" class of bug into an enforced, loud failure.
3. Mutation testing (`mutmut`/`cosmic-ray`) targeted at `cv.py` and `oof.py` specifically —
   given how much project credibility rests on "the leak tests actually catch leaks," this is a
   stronger validation than more example-based tests alone.
4. Property-based tests (Hypothesis) for CV boundary conditions — single-fold degeneracy,
   embargo behavior, tiny `n_groups` — the exact edges the fixed-example test suite doesn't hit.
5. Make `ruff check` fail-hard (already on the roadmap as 6.1.1, worth re-flagging as low-effort/
   high-signal) and add `pytest --cov=pipeline --cov-report=term-missing` to CI so "which modules
   have zero coverage" becomes an objective, continuously-checked number instead of a manual
   audit like the one in this review.
6. A lightweight shadow/canary period before promoting a new model artifact to `state/latest_*.pkl`
   — logs predictions from the new and old model side by side on live games before the switch,
   catching production-only regressions (odds feed schema drift, injury report format changes)
   that a retrospective walk-forward backtest structurally cannot see.
7. Standardize on structured logging instead of mixed `print()`/`logging` calls, specifically so
   the many silent `except Exception: pass` blocks (bug #13) can become cheap, visible warnings
   surfaced on the dashboard's Jobs tab instead of disappearing entirely.
8. Fix the `docs/pipeline.md` module-count drift (98 → 134) and audit other docs (`docs/math/
   calibration.md`, `docs/math/devig.md`) for similar "describes code that isn't wired up yet"
   drift — both currently describe the Shin/power devig machinery as if it's in the live path.

---

## Appendix — background review agents

This report synthesizes three parallel adversarial passes:
- Calibration, staking, devig, and market/odds handling
- Rating engine internals (Glicko-2, hierarchical n-man, HAPM, lineup Elo, chemistry, EPM,
  minutes/injury projections)
- ML architecture, cross-validation/leak hardening, SHAP, and the dashboard

Each pass read the actual source files line-by-line and cross-referenced claims against
`docs/roadmap.md`, `LEAK_REGISTRY.md`, and the math docs; every finding above is traceable to a
specific file and function.
