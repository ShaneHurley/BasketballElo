# NBA Elo → Betting Model: Complete Project History & Formula Evolution

*A detailed report on the evolution of your NBA prediction model from 2021 to 2026, reconstructed from 15+ notebooks, a real git history (16 commits), and the current modular codebase.*

---

## Executive Summary

Over roughly five years, you built what started as a simple possession-level player Elo notebook into a production-grade, modular Python pipeline with rigorous anti-leakage testing, multi-layer calibration, and a full betting/staking engine. The project went through four distinct eras:

1. **2021–2024: Notebook experimentation era** — pure Elo math, constant tuning, one abandoned detour
2. **May 2026: Rapid restart** — full PPP-based rewrite, first real ATS results (60%+ claimed, later self-flagged as overfit)
3. **June 2026: Leak-hunting era** — "may have leaks" → walk-forward everything → "consistent 2-4% edge"
4. **July 2026: Production modularization** — single notebook → 98-module tested package with a formal `LEAK_REGISTRY.md`, FastAPI T-60 dashboard, and staged full-suite runner

The most important arc: **exciting overfit backtest → sober leak-hunting → production-grade modular pipeline with tests and a leak registry → bug fixes exposed by the refactor → empirically re-tuned thresholds.**

---

## Part 1: Chronological Evolution (2021–2026)

### Era 1 — Origin: Possession-Level Player Elo (Oct 2021)

**File:** `NBA ELO marc 2(3).ipynb`

The very first working version. Tracks separate **Offensive Elo (OELO)** and **Defensive Elo (DELO)** per *player*, keyed off which 5-man lineup is on court, processing raw play-by-play possession-by-possession.

**Core formulas:**
- Expected score: `expectedScored = 1.0 / (1 + 10^(eloDiff / 400))`, then rescaled by `0.9475 * (2/3)` (empirically fit, because scoring is normalized to a 0–3 scale)
- Update: `add = 64 * ((scored/3) - expectedScored)` — flat **K=64**
- Starting rating: **1000** (not 1500)
- **No home court advantage, no margin-of-victory multiplier, no season regression**

This is the seed of the entire project — pure, textbook Elo applied at the player-possession level.

### Era 2 — Constant Tuning Experiments (May 2022)

Four near-identical notebooks (`NBA ELO(1).ipynb`, `marc 2.ipynb`, `marc 2(2).ipynb`, `marc 2(4).ipynb`) differing only in K-factor and calibration constant. K oscillates between 16, 32, 36, and 64 depending on how "scored" is normalized.

**Key innovation:** first appearance of a FIDE-style **provisional-rating boost** — players with <40 possessions get a 33% K boost (`add = 36 * (4/3) * (scored - expectedScored)`). This is the proto-version of what later becomes the `k_mult` veteran-dampening curve.

### Era 3 — Stint/Lineup-Change Elo (2023)

**File:** `Player NBA ELO.ipynb`

Moves from single-possession updates to **lineup-stint** updates (accumulate score over stretches where the same 5 players are on court). Introduces a league-wide `pointsPerPos` normalizer (~1.247).

**Core formulas:**
- `addD = pos * 32 * (expectedScored - (score/(pointsPerPos*2)))` — K=32, now **scaled by stint length**
- `addO = -addD` (zero-sum between offense and defense)

### Era 4 — Two Parallel Branches (Feb 2024)

**4a. `NBA ELO.ipynb`** — same architecture, adds `*pos` scaling to the final assignment (doubles down on possession-weighting).

**4b. `Copy of NBA ELO.ipynb`** — a **divergent experimental branch that abandons logistic-Elo entirely** in favor of an exponentially-weighted moving average of points-per-possession efficiency:

```python
expectedScored = (lineUpData[O][6] + lineUpData[D][7]) / 2  # avg of O/D ratings directly (no logistic!)
lineUpData[O][6] = pointsPerPos * (score/expectedScored) * (pos/totalPosO) + lineUpData[O][6] * ((totalPosO - pos) / totalPosO)
```

This was **not carried forward** — all subsequent versions return to classic Elo-logistic updates. A notable "what if" moment in the project's history.

### Era 5 — Git Commit `b875a72` (Oct 27, 2023): `NBA_ELO.ipynb`

*Commit message: "I've done lots of changes over time but I do it on google colab"*

First commit to the git repo. Byte-for-byte identical to Era 4a minus the `*pos` scaling. Same K=32 possession-scaled formula, starting Elo 1000, no home court, no MOV, no regression.

### Era 6 — Git Commit `5bb2b29` (May 19, 2026): `BasketballEloSim.ipynb`

*Commit message: "This has been a consistent side project for me over the last 5 years on a and off and made a lot of improvements"*

**The single biggest architectural leap in the entire project.**

**Base constants:**
- `BASE_ELO = 1500` (first time starting rating moves from 1000)
- `ELO_SCALING_FACTOR = 1000`
- `K_OFF = 0.3`, `K_DEF = 0.7`
- `HOME_PPP_BOOST = 0.024`

**Unit change:** ratings now feed directly into a **points-per-possession (PPP) model** instead of a 400-divisor logistic:
```python
xPPP_A = league_avg_ppp + HOME_PPP_BOOST + ((rating_A['O_Elo'] - rating_B['D_Elo']) / ELO_SCALING_FACTOR)
```

**Game-level win probability** (separate divisor from player-Elo scaling):
```python
margin_per_100 = net_ppp * 100
win_prob = 1.0 / (1.0 + 10.0 ** (-margin_per_100 / 15.0))  # divisor 15, not 400
```

**Dynamic K-factor** replaces the classic `ln(margin)*(2.2/...)` MOV multiplier with a **residual-based** update scaled by 5 multiplicative modifiers:
```python
res_A = eff_pts_A - (xPPP_A * poss)  # actual - expected points
k_mult = np.clip(avg_exp/500.0, 0.2, 1.0)  # experience dampener
leverage = 2.0 if period>4 else clip(0.5+1.5*exp(-0.05*lead)*(period/4), 0.3, 2.0)  # clutch leverage
chaos_damp = 0.7 if poss>25 else (0.9 if poss>18 else 1.0)  # garbage-stint dampener
clutch = 1.4 if (period>=4 and lead<=3 and res>0) else (1.2 if lead<=7 else 1.0)
dynamic_k_off = K_OFF * k_mult * leverage * chaos_damp * clutch * b2b_penalty
```

**Age-dependent season regression** (replaces flat regression):
```python
base = 0.85 if age<=23 else (0.60 if age>=33 else 0.75)  # carryover
carryover = clip(base + arch_boost, 0.55, 0.90)
O_Elo = O_Elo*carryover + BASE_ELO*(1-carryover)
```
Regression-to-1500 ranges from ~10% (young players) to ~45% (old players).

**New inputs:** back-to-back penalty, usage-share credit assignment, HAPM-style dyad/triad/quintet lineup-synergy terms, Ridge-regression RAPM residual bonus for clutch performance.

### Era 7 — Git Commit `f05c34c` (May 19, 2026): `basketballevaluatorPASTYEARS.ipynb`

Same core engine as Era 6, plus:
- **Recency amplifier:** `recency_amp = 1.0 + (0.20 * season_progress)` — ratings move up to 20% faster late in the season
- **Meta-model layer:** Elo output becomes an input feature to an **XGBoost regressor + Platt-scaling logistic-regression calibrator** — the first time Elo is no longer the final prediction

### Era 8 — Git Commit `ec8f006` (May 20, 2026): `2026ELOcode.ipynb`

*Commit message: "code that gets above 57% accuracy verse the spread of this (may be over fit for this year)"*

- **Inactivity-based regression** replaces age/season-based: players inactive 60+ days regress 25% toward 1500
- **Shot-zone xPoints** replaces raw box-score points (proto-xPoints model)
- **Full ATS backtest with confidence tiers:**
```python
confidence_score = min(100, round((abs_edge / 7.5) * 100, 1))
```

**Recorded backtest results by edge tier (1,012 bets):**

| Min Edge | Bets | Win % | ROI |
|---|---|---|---|
| 0.0+ | 1012 | 60.8% | 16.0% |
| 3.0+ | 969 | 61.0% | 16.4% |
| 4.5+ | 866 | 61.7% | 17.7% |
| 6.0+ | 753 | 62.3% | 18.9% |
| 8.0+ | 588 | 63.9% | 22.1% |

The commit message's "above 57%" is a conservative undersell — but the parenthetical "(may be over fit for this year)" is the author's own caveat that this was calibrated and evaluated on the same 2025-26 season slice, i.e. **no true holdout**.

### Era 9 — Git Commit `a8e2cb9` (June 18, 2026): `nba_unified_prediction_pipeline.ipynb`

*Commit message: "may have leaks in pre 2026 data"*

**Full rewrite into a Glicko-2-inspired rating system** — the biggest formula-family change since Era 6.

Each player now has `O_mu`/`D_mu` (rating), `O_rd`/`D_rd` (rating deviation — uncertainty), and `O_sigma`/`D_sigma` (volatility).

**Key changes:**
- Inactivity decay now **inflates uncertainty (RD)** instead of only regressing the mean: `p["O_rd"] = min(350.0, p["O_rd"] * (1 + 0.1 * (days/7)))`
- **Roster-continuity-aware offseason reversion:** returning players regress 25%, departing/traded players regress 50%
- **Games-played exponential-decay learning rate** replaces the multi-factor multiplier stack: `k_mult = max(0.5, 2.0 * exp(-games/15))`
- **Garbage-time down-weighting** replaces flat chaos-damper
- **Hierarchical n-man lineup possession engine** (1-man/2-man/3-man/5-man weights: 0.50/0.25/0.15/0.10)
- **CatBoost + HuberRegressor stacking ensemble** meta-model
- **Injury-report data** as a live feature
- Optuna-tuned hyperparameters

The commit message flags a data-leakage concern rather than an accuracy claim — this version was later specifically fixed for leakage.

### Era 10 — Git Commit `ade9434` (June 23, 2026): `nba_unified_pipeline_colab.ipynb`

*Commit message: "new unifided pipeline no leaks consitent 2-4% edge"*

Explicitly self-described as **"leak-free"** in its own markdown header. Key formula deltas:
- `OFFSEASON_REVERSION` reduced from 0.25 → **0.15**
- `HOME_PPP_BOOST` reduced from 0.024 → **0.002** (a large recalibration, likely correcting an inflated home-edge that had been contaminated by the leak)
- **Split defensive ratings by shot location:** `D_rim_mu` / `D_peri_mu` (rim vs. perimeter defense tracked separately)
- Walk-forward, chronological cross-validation everywhere

The "consistent 2-4% edge" is a much more modest but credible number than the possibly-overfit 60%+ win rates claimed in Era 8 — reflecting a maturing, more skeptical evaluation methodology.

### Era 11 — Git Commit `7c23af2` (July 23, 2026): Modularization

*Commit message: "Add modular code/ package and rebuild import-based Colab notebook."*

**The pivotal architectural commit.** Exactly one month after Era 10, the single giant Colab notebook is decomposed into a proper importable Python package: `BasketballElo/code/pipeline/` (today **98 modules**) with a large regression suite under `code/tests/`.

Most importantly, this commit introduces **`LEAK_REGISTRY.md`** — a formal document that retroactively documents **every leak found and fixed** during modularization, confirming that the "may have leaks" suspicion from Era 9 was well-founded. Subsequent local work also added a **FastAPI T-60 analysis dashboard** (`code/dashboard/`) and **`run_full_suite.py`** (ordered stages 0–8 with checkpoints and anti-overfit review gates). Leaks catalogued as fixed:

1. `score_corruption` — final scores computed from garbage-time-zeroed stints instead of raw scores
2. `date_corruption` — malformed dates silently fabricated as `prev_date + 1 day`
3. `close_line_conflation` — bet line and closing line loaded from the same snapshot, making CLV identically zero
4. `stack_cv_future_leakage` — stacking base learners trained on future rows
5. `elo_stack_in_sample_leak` — Elo-margin ridge fit on entire training set
6. `hapm_before_split_leak` — HAPM fit once on entire training slice before chronological split
7. `static_to_rolling_calibration_mismatch` — calibrator fit once on entire slice, applied uniformly
8. `calibration_slice_reuse` — multiple calibrators fit on identical tail slice
9. `roi_tuning_objective_and_overfit_sign_bug` — Optuna objective had inverted overfit-penalty sign
10. `t60_actual_lineup_leak` — T-60 features accidentally used the actual game's own lineup
11. `epm_prior_unversioned_leak` — unversioned EPM CSV let future-dated ratings join to past games

### Era 12 — Final Commits (July 23, 2026): Bug Fixes + Threshold Tuning

- `e017c9b`: Fix missing `ATSClassifier` import (classic "worked in monolithic notebook, broke once split into modules" bug)
- `8287ca2`: Fix diagnostics `KeyError` on `ats_push` when a season has zero actionable bets
- `b49c5d5`: **Relax over-strict ATS gates** — the final empirical tuning pass:

| Gate | Before | After | Reason |
|---|---|---|---|
| `MAX_QUANTILE_WIDTH` | 22.0 | 28.0 | Median model uncertainty width ≈26pts, so 22 was rejecting ~all bets (~3% bet rate) |
| `MIN_DISAGREEMENT_TRUST` | 0.85 | 0.70 | 0.85 silenced season-1 walks whenever Elo and meta-model disagreed slightly |
| `SKIP_TIGHT_SPREAD` | True | False | Tight markets are common; hard-rejecting them killed most available edge |
| `EDGE_AVOID_BAND` | (7.0, 9.5) | None | Diagnostics showed the 7–9.5pt edge band had the *strongest* lean-ATS performance |

---

## Part 2: Current Production Pipeline (98 modules)

**Canonical source of truth:** [`BasketballElo/code/`](BasketballElo/code/) inside the git repo at `github.com/ShaneHurley/BasketballElo` (`code/pipeline/` = **98 modules**). An older, untracked copy may also exist at `/Users/shurley/Documents/basketball/pipeline` (fewer modules) — do not treat that path as production.

Here's the complete execution flow:

### Pipeline Execution Flow

**Training / backtest path** (`run_backtest.py` / `run_training_experiment.py`):

1. **Ingest raw play-by-play** → `ingest.py` normalizes raw PBP schemas
2. **Canonical game table** → `game_results.py` independently derives final scores two ways and raises on disagreement
3. **Preprocess PBP → possessions** → `preprocess.py` computes expected points (xPoints) per event using `shot_zones.py` walk-forward zone calibration
4. **Build stints** → `stints.py` collapses PBP into lineup-vs-lineup possession "stints"
5. **Date/schedule resolution** → `dates.py` guarantees monotonic, leak-free chronological ordering
6. **Load market odds** → `market.py` builds date/team-keyed odds dict
7. **Hyperparameter tuning** → `tuning.py` (Optuna, walk-forward, per simulated season)
8. **Feature generation with online rating updates** → `features.py` loops game-by-game: `build_game_features` (pre-game, T-60-safe) then `update_trackers_after_game` (post-game)
9. **ELO calibration** → `elo_calibration.py` (3-group Huber regression: engine → uncertainty → market)
10. **Meta score model fit** → `model.py` (Ridge+CatBoost stacked via `ManualOOFStacker`, walk-forward OOF, Huber meta-learner)
11. **Win-probability model** → `model.py` (logistic regression blended with Elo-only logistic head, isotonic/Platt calibration)
12. **Auxiliary calibrators** → fit on disjoint calibration slices (`calibration_registry.py` enforces no slice reuse)
13. **Walk-forward backtest loop** → `backtest.py` orchestrates season-by-season
14. **Bet selection & staking** → `bet_selection.py` → `market.py` → `bet_confidence.py` → `stake_profiles.py` → `bet_grading.py`
15. **Metrics / diagnostics / artifacts** → `metrics.py`, `diagnostics.py`, `artifacts.py`

**Live/daily path** (`run_daily.py`): Load persisted engine state → fetch odds/injuries → `predict_game` → append to prediction log.

**Full suite path** (`run_full_suite.py`): Ordered stages 0–8 (toggles → stints → integrity → engines/calib/walk-forward → **review stop** → policy gates → persist) with checkpoints and promotion bars that require finite CLV — not ATS/ROI alone.

**T-60 dashboard** (`code/dashboard/`): Local FastAPI UI for browsing run outputs, launching suite/backtest jobs with SSE progress, and ATS/ML/totals calibration charts (Plotly, night-mode).

### Core Formulas (Current Production)

**Team Elo** (`team_elo.py`):
- Predicted margin: `margin = (spread_mu[home] − spread_mu[away]) / 25 + HOME_ADV`, `HOME_ADV = 2.5`
- Update with decaying K: `k = K_SPREAD / (1 + games[home]·0.02)`, `K_SPREAD=20.0`

**Player Rating Tracker** (`ratings.py`) — Glicko-2-inspired:
- Expected PPP: `exp_ppp_A = league_xppp + HOME_PPP_BOOST + (off_A − def_B)/ELO_SCALING_FACTOR` (scaling=1000, `HOME_PPP_BOOST=0.002`)
- Update: `delta = error · k_effective · (usage_share) · k_mult`, where `k_effective = k_base·(RD/350)` (uncertainty-weighted) and `k_mult = max(0.5, 2·exp(−games/half_life))` (`K_MULT_HALF_LIFE=24.4`)
- Context multipliers: clutch (`CLUTCH_BOOST=1.30`), garbage time (`GARBAGE_TIME_WEIGHT=0.30`), turnover-heavy (`TOV_PENALTY=0.85`), foul-drawing (`FOUL_DRAW_BOOST=1.10`), high-3PA-rate (`VARIANCE_DAMPEN=0.90`)

**Elo Calibration** (`elo_calibration.py`) — 3-stage walk-forward:
- Sequential residual regression: `pred1 = Huber(engine_features)`; `resid1 = y − pred1`; `pred2 = Huber(uncertainty_features)`; `resid2 = resid1 − pred2`; `pred3 = Huber(market_features)`; final `elo_margin_calibrated = pred1+pred2+pred3`
- Huber epsilon 1.35, `hier_blend_elo=0.6 / hier_blend_hier=0.4`

**Lineup Elo** (`lineup_elo.py`) — James-Stein-style shrinkage:
- `w = n/(n+SHRINK_K)`, `SHRINK_K=100`, `MIN_POSSESSIONS=50`

**EPM Priors** (`epm_priors.py`) — external metric blend, point-in-time versioned:
- `scaled_impact = epm · 50`, `blend_lineup_off = (1−blend)·internal + blend·(1500+mean(impacts))`, `BLEND_WEIGHT=0.35`

**HAPM** (`hapm.py`) — Hierarchical/Sparse-Ridge Adjusted Plus-Minus:
- `xPPP_home − league_xppp ≈ Σ β_{dyad} · 1[dyad in lineup] + Σ β_{trio} · 1[trio in lineup]`, fit via Ridge(alpha=100), possession-weighted
- Applied prior shrinkage: `coef * SHRINK/(SHRINK+50)`, `SHRINK=80`

**Hierarchical Possession Engine** (`hierarchical.py`):
- Weighted rating: `off = Σ_l W[l]·mean(off_store[combo_l])` for `l ∈ {1,2,3,5}`
- Update K decays by combo size: `K_off = {1: k_off, 2: k_off·0.5, 3: k_off·0.25, 5: k_off·0.1}`

**Chemistry** (`chemistry.py`) — pairwise/triple on-court chemistry residuals:
- Shrinkage: `shrink(raw, n) = raw · n/(n+SHRINK)`, `SHRINK=80`

**Meta Score Model** (`model.py`) — the central meta-learner:
- Manual OOF stacking target (optionally on **closing-line residual**, not raw margin): `target = actual_margin − closing_spread`
- Elo blend at inference: `raw_margin = (1−α)·stack_pred + α·elo_margin_calibrated`, dynamic `α∈[0, ELO_BLEND_ALPHA_MAX]` (`ELO_BLEND_ALPHA_MAX=0.35` in current `config.py`)
- Win probability: isotonic/Platt calibrator on `raw_margin`, fallback `sigmoid(margin/12)`
- Quantile heads (CatBoost `Quantile:alpha=0.1/0.9`) give `spread_quantile_width` used as uncertainty gate

**Devig** (`devig.py`) — four candidate de-vig transforms:
- Multiplicative: `p_i = q_i/Σq`
- Power (Shin/Jullien): solve `Σ q_i^k = 1` for `k` via bisection
- Odds-ratio: solve `Σ (c·q_i)/(1−q_i+c·q_i) = 1` for `c`
- Shin (1992/1993 insider-trading model): `p_i = (√(z² + 4(1−z)q_i²/σ) − z) / (2(1−z))`
- Method chosen per `(book, market, horizon)` using only strictly prior folds' OOS loss

**Spread Cover Probability** (`market.py`):
- Gaussian: `z = (model_spread + market_spread) / use_sigma`, `p_home = norm.cdf(z)`
- σ floors from `conf_width/2.5`, matchup volatility, or scenario-mixture σ

**Kelly Staking** (`stake_profiles.py`):
- Full Kelly: `f* = (bp − q)/b`, `b=100/110` at −110 juice
- Robust fractional Kelly shrinks `p` toward market-fair probability by `uncertainty` before applying a capped fraction (`≤0.50`)

**Venn-Abers Calibration** (`venn_abers.py`):
- Fits isotonic regression twice (once assuming label 0, once assuming label 1) to get `(p0, p1)` bounds
- Optimal point: `p1/(1−p0+p1)`, interval width `p1−p0` as native uncertainty measure

**ATS Confidence Score** (`bet_confidence.py`):
- Hand-engineered, weighted linear composite of 15 terms: |edge| slope, historical bucket ATS lift, quantile-interval-width bonus, rating-uncertainty penalty, Elo/meta agreement bonus, cover-probability term, matchup-volatility penalty, market-disagreement trust bonus, phantom-injury penalty, ATS-classifier probability term, win-probability term, raw Elo-margin/alignment bonus
- Each weight bounded/clamped and tuned per-season via walk-forward grid search

---

## Part 3: Comparison to Textbook Elo

| Dimension | Textbook Elo | Your Current System |
|---|---|---|
| **Unit of rating** | Team | Player (off/def/rim/peri splits) **+** lineup5 **+** duo/trio chemistry **+** team-Elo **+** hierarchical combo (1/2/3/5-man) **+** HAPM ridge — five distinct, cross-validated rating systems blended together |
| **Margin of victory** | Ignored (win/loss only) | Every engine trains directly on point-differential / xPPP, not win/loss |
| **Home court** | Fixed constant, if any | `HOME_PPP_BOOST`, `HOME_ADV`, `home_boost_rtg`, plus learned `hca_net` from rolling home/road split per team |
| **K-factor** | Fixed | Decays with games played, scales with rating uncertainty (Glicko RD), decays by combo granularity, separately tuned via Optuna every walk-forward season |
| **Uncertainty** | None | Full Glicko-2-style RD per player/side, inflated on inactivity, drives stake sizing and confidence scoring |
| **Rest / fatigue / travel** | None | `fatigue.py`, `travel.py` (haversine distance + timezone shift), schedule-density buckets |
| **Injuries / availability** | None | `availability.py`, `injury_reports.py` (point-in-time versioned), rotation scenario mixture |
| **Referees** | None | `refs.py` rolling crew pace/foul bias |
| **Shot quality / zones** | None | `shot_zones.py` walk-forward zone-level xPPS calibration, `shot_quality.py` rolling xEFG/rim-rate/three-rate |
| **External priors** | None | EPM/RAPTOR blend with observation-date versioning |
| **Market information** | None | Market spread/ML/total are themselves *model inputs*, plus explicit devig, disagreement, and CLV machinery |
| **Probability model** | Logistic (Elo formula) | Gaussian cover-prob with heteroscedastic σ, Skellam PMF option, isotonic/Platt/Venn-Abers calibration layers, direct ATS classifier |
| **Ensembling** | None | Ridge+CatBoost OOF stack with Elo-blend anchor, quantile heads, blowout heads |
| **Regularization / shrinkage** | None | James-Stein-style shrinkage everywhere, Huber robust regression, Ridge, empirical-Bayes bucket lift shrinkage |
| **Anti-leakage rigor** | N/A | Purpose-built `PastOnlyGroupCV`, calibration-slice registry, versioned EPM/injury snapshots, canonical dual-verified score labels |
| **Betting layer** | N/A | Full Kelly-derived stake sizing, portfolio/slate covariance shrinkage, bankroll Monte Carlo, CLV tracking, edge-vs-ROI curve fitting |

---

## Part 4: Most Sophisticated / State-of-the-Art Components

1. **`ManualOOFStacker` + `PastOnlyGroupCV`** — hand-built, leak-proof stacking/OOF framework that raises hard `LeakageError`s if any fold violates strict chronology; goes well beyond what most hobby models attempt
2. **`devig.py`** — four-method de-vig toolkit with per-`(book, market, horizon)` method selection based purely on strictly-prior-fold OOS loss — genuinely research-grade market microstructure handling
3. **`calibration_registry.py`** — fingerprint-based registry that *provably* prevents multiple calibrators from being fit on the same row slice
4. **`WalkForwardEloCalibrator`** — sequential 3-stage Huber-regression residual calibration (engine → uncertainty → market), cleanly separating "what the model knows" from "what the market knows"
5. **`epm_priors.py` / `injury_reports.py`** — point-in-time versioned external data with hard schema validation, preventing any retroactive-data leak
6. **`scenario_mixer.py` + `rotation_scenarios.py`** — exact (non-Monte-Carlo) finite-mixture posterior-predictive combination of correlated multi-player rotation scenarios via the law of total expectation/variance
7. **`hapm.py`** — sparse Ridge dyad/trio Adjusted Plus-Minus with dedicated past-only cross-fitting
8. **`stake_profiles.py::simulate_bankroll` / `optimize_slate_stakes`** — Gaussian-copula correlated-Bernoulli bankroll Monte Carlo plus PSD-projected shrunk-covariance joint slate optimization

---

## Part 5: Key Lessons & "Best Of" Synthesis

### What worked best at each era:

| Era | Best Idea | Why It Mattered |
|---|---|---|
| 2021 | Possession-level player O/D Elo | Foundation — granular ratings beat team-level |
| 2022 | Provisional K-boost for new players | First recognition that sample size should affect learning rate |
| 2023 | Stint-based updates + `pointsPerPos` normalizer | Moved from event-level to lineup-level thinking |
| 2024 | (Abandoned moving-average branch) | Valuable negative result — confirmed Elo-logistic was the right family |
| May 2026 | PPP-based unit + explicit home-court constant | First real predictive signal; unit change unlocked everything downstream |
| May 2026 | Dynamic K-factor (leverage/clutch/chaos/b2b) | First context-aware rating updates |
| May 2026 | Elo → XGBoost → Platt stack | First recognition that Elo alone isn't enough — needs ML meta-layer |
| Jun 2026 | Glicko-2 (μ/RD/σ) | Uncertainty quantification enabled confidence-based betting |
| Jun 2026 | Walk-forward everything + leak hunting | The single most important methodological improvement — turned overfit 60% into credible 2-4% |
| Jul 2026 | Modular package + LEAK_REGISTRY | Made the system maintainable, testable, and auditable |
| Jul 2026 | Empirical gate relaxation | Data-driven threshold tuning beat intuition |

### The "best of what it did" — a recommended architecture:

1. **Player-level Glicko-2 ratings** (off/def/rim/peri splits) with uncertainty-aware K-decay
2. **Hierarchical n-man lineup engine** (1/2/3/5-man) with sample-size-adaptive weights
3. **EPM/RAPTOR external priors** with strict point-in-time versioning
4. **Walk-forward 3-stage calibration** (engine → uncertainty → market) with Huber robust regression
5. **Ridge+CatBoost OOF stacking** with past-only CV and hard leakage assertions
6. **Multi-method devig** with per-market method selection
7. **Venn-Abers + isotonic calibration** for probability estimates with native uncertainty intervals
8. **Fractional Kelly staking** with slate-level covariance shrinkage and bankroll Monte Carlo
9. **Formal leak registry** with regression tests for every confirmed leak
10. **Empirical gate tuning** via diagnostics rather than intuition

---

*Report generated September 18, 2026; corrected September 19, 2026 for canonical path/module count. Sources: 15+ Jupyter notebooks (2021-2026), git history in `BasketballElo/` (github.com/ShaneHurley/BasketballElo), current production codebase at `BasketballElo/code/pipeline/` (98 modules), `code/dashboard/`, `run_full_suite.py`, and `code/LEAK_REGISTRY.md`.*
