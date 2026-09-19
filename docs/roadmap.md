# Roadmap

Ranked by expected model lift and engineering value. Each epic has 3–10 tasks; each task has
2–5+ subtasks with file touch-points and acceptance criteria. Research sources cited inline.

**Primary optimization targets:** lower spread **MAE**, higher **winner** accuracy, better
**totals** calibration, and positive **CLV**. §"Prediction-target map" below shows which tasks
move which metric.

**Where we are:** the stack is already deep on possession-level ratings (player Glicko-2 O/D,
hierarchical n-man, lineup Elo, chemistry, HAPM), team multi-window four-factors form
(3/5/10/20), minutes/injury scaffolding, and anti-leakage machinery (past-only OOF stacker,
leak registry, promotion gates). Confirmed white space: **player counting-stat features**,
**lineup-level rolling box/pace windows**, **player-impact-weighted team strength**,
**Kalman-style now-casting**, and **portfolio-level staking**.

**Already exists — extend, don't rebuild** (audit-confirmed):
`code/pipeline/venn_abers.py` (Venn-Abers calibration), `skellam.py` (Skellam PMF),
`devig.py` (de-vigging), `shot_quality.py` + `shot_zones.py` (spatial shot quality),
`cv.py` `PastOnlyGroupCV` (zero-lookahead CV), `epm_priors.py` (versioned EPM),
`shap_prune.py` (SHAP auditing), `minutes_forecast.py` / `rotation_scenarios.py`
(minutes projections), `chemistry.py` (duo/trio residuals), `trackers.py` (rolling pace/xPPP).

**Research base:** Osken & Sahin 2022 (archetype clustering + GA-NN, ~70–76%); Houde 2021
(Elo strongest single predictor, z-score diffs, 65–72% ceiling); Lu/Chen/Zhu 2019
(recency-weighted regression, per-team HCA, RMSE ~12 vs market ~11.85); Li & Jia 2024
(CatBoost + Optuna + SHAP, VORP top value signal); Dunks & Threes EPM (SPM-prior RAPM,
nightly since 2001–02, no lookahead, minutes-weighted team RMSE 12.1); CraftedNBA
(depth-chart minutes × impact); Opta DRIP (Bayesian now-casting, opponent-neutral talent);
DARKO (Kalman filters + padding: add 85 makes / 242 attempts baseline to stabilize 3P%);
LEBRON (luck adjustment, role-aware priors); Ben Taylor Box Creation; Neil Paine
(dual-window ratings, 15-game regression, playoff 3×, market composite); Stats Insider
(10k Monte Carlo, fair-odds edge flags, line shopping); SportsVisio (four-factors / TS% /
AST% / pace formulas).

---

## Epic 1 — Player & lineup rolling form features

*Gap confirmed: no player counting stats in features; lineup layer is rating residuals only.
`nba_player_stats_2026.csv` sits on disk unused. Every new stat ships behind an ablation
toggle and must earn its place via OOF metrics + SHAP — investigate, don't assume.*

### Task 1.1 — Player box-score derivation layer
- [ ] 1.1.1 Derive per-player per-game boxes (PTS, AST, REB, TOV, FGA, FTA, 3PM, MIN, USG) from V3 PBP stints (`stints.py`, `stint_context.py`) — no external box API needed
- [ ] 1.1.2 Reconcile derived boxes against `nba_player_stats_2026.csv` as validation oracle; parity test with tolerances
- [ ] 1.1.3 Add `player_game_box` artifact to ingest (`ingest.py`, `preprocess.py`); schema-version in `config.py`

### Task 1.2 — Rolling player efficiency stats (windows 5/10/20)
- [ ] 1.2.1 Rolling **AST%** per player (share of teammate FGM assisted while on floor)
- [ ] 1.2.2 Rolling **TS%** `PTS / (2 × (FGA + 0.44 × FTA))` and **eFG%** `(FGM + 0.5 × 3PM) / FGA`
- [ ] 1.2.3 Rolling **PPG**, **USG%**, **TOV%**, **OREB%/DREB%**
- [ ] 1.2.4 Minutes-weighted team aggregation via `minutes_forecast.py` projections (NOT actuals — T-60 leak rule)
- [ ] 1.2.5 Emit `h_/a_/_diff` columns in `game_features.build_game_features`; register in `model.py` `SAFE_FEATURE_COLS`; bump `FEATURE_SCHEMA_VERSION`

### Task 1.3 — Rolling 5-man lineup form, last 10 games *(investigate as winners)*
- [ ] 1.3.1 New `code/pipeline/lineup_form.py`: rolling lineup **pace** (poss/48 from stints) over last 10 appearances
- [ ] 1.3.2 Rolling lineup **PPG** + xPPP comparison, last 10
- [ ] 1.3.3 Rolling lineup **AST%** and **TOV%** last 10 (cohesion proxy; strongest early-season when RD is high)
- [ ] 1.3.4 Shrinkage to player-mean for low-possession lineups (reuse `MIN_POSSESSIONS`/`SHRINK_K` from `lineup_elo.py`)
- [ ] 1.3.5 **Decision gate:** keep each stat only if walk-forward OOF log-loss/MAE improves AND SHAP importance non-trivial; document verdicts in `docs/pipeline.md`

### Task 1.4 — Rolling 3-man / duo lineup windows
- [ ] 1.4.1 Extend `lineup_form.py` to 3-man units (larger samples than 5-man, stabilizes faster)
- [ ] 1.4.2 Rolling duo net rating to complement `chemistry.py` residuals with raw form
- [ ] 1.4.3 Same shrinkage + decision-gate protocol as 1.3

### Task 1.5 — Team-level rolling additions
- [ ] 1.5.1 Rolling team **TS% differential** (team TS% − opp TS% allowed), windows 5/10/20
- [ ] 1.5.2 Rolling team **AST% + TOV%** chemistry feature
- [ ] 1.5.3 Z-score differential transforms vs league mean/std (Houde's top recommendation)

### Task 1.6 — Tempo-dictation matchup features
- [ ] 1.6.1 Lineup-vs-lineup pace clash features: expected tempo when high-pace unit meets slow half-court unit (who dictates tempo historically)
- [ ] 1.6.2 Extend `pace_mean/var/q10/q90` with lineup-weighted expected pace distribution
- [ ] 1.6.3 Feed into totals head (Epic 5.6) as well as spread head

### Task 1.7 — Player PER / BPM / VORP from derived boxes
- [ ] 1.7.1 Implement PER (league-avg 15), BPM (per-100 impact), VORP (BPM × minutes) on top of Task 1.1 boxes — VORP was the top value signal in Li & Jia
- [ ] 1.7.2 Minutes-weighted roster VORP/BPM sums as team-strength features (bridge to Epic 2)
- [ ] 1.7.3 Decision gate: ablate vs existing Glicko/EPM features; keep only incremental signal

### Task 1.8 — Spatial & shot-quality rolling integration
- [ ] 1.8.1 Extend existing `shot_quality.py`/`shot_zones.py` into rolling 10-game shot-quality-differential features (quality of looks created vs allowed)
- [ ] 1.8.2 Rolling rim/three rate + PPS by zone already partly present — expose as form windows
- [ ] 1.8.3 Decision gate per stat

### Task 1.9 — Validation & leak hygiene for Epic 1
- [ ] 1.9.1 Past-only tests: no rolling window includes same-game/future data (`test_feature_parity.py`, new `test_lineup_form_t60.py`)
- [ ] 1.9.2 Live/serve parity in `predict.py`/`simulate.py`
- [ ] 1.9.3 SHAP audit (`shap_prune.py`); prune noise features
- [ ] 1.9.4 Update `LEAK_REGISTRY.md` for any new leak class

---

## Epic 2 — Player-impact-weighted team strength (EPM / DRIP / DARKO / LEBRON)

*The biggest documented weakness of result-based ratings: slow reaction to trades, injuries,
early-season churn. Every pro model (EPM, DRIP, DARKO, CARM-Elo, CraftedNBA) solves it the
same way: minutes-weighted player impact → team strength, with Bayesian stabilization.*

### Task 2.1 — EPM data pipeline hardening
- [ ] 2.1.1 Automate nightly EPM prior refresh with versioned `observation_date` (extends `epm_priors.py`); fail loudly on stale data
- [ ] 2.1.2 Backfill historical daily EPM (nightly values since 2001–02, no lookahead — backtest-safe)

### Task 2.2 — Minutes-weighted impact team rating
- [ ] 2.2.1 Team O/D rating = Σ(projected minutes × player EPM-O/D) via `minutes_forecast.py` + `rotation_scenarios.py`
- [ ] 2.2.2 Features (`epm_team_net`, `epm_vs_elo_gap`) + season-start prior (CARM-Elo trick: player prior first, game results dominate as sample grows)
- [ ] 2.2.3 Benchmark vs D&T minutes-weighted team EPM RMSE (~12.1)

### Task 2.3 — Injury & rotation adjustment v2
- [ ] 2.3.1 Replace/augment star-out heuristics with impact × expected-minutes deltas (missing EPM − replacement EPM)
- [ ] 2.3.2 Depth-chart diff signal: rotation minute shifts as leading indicator (CraftedNBA)
- [ ] 2.3.3 Historical injury archive so backtests use the same path as live ESPN fetch (`availability.py`, `injury_reports.py`)

### Task 2.4 — Kalman-filter now-casting (DARKO-style)
- [ ] 2.4.1 Kalman filter per player-stat: recursive true-talent estimate updated per game, replacing raw season averages (DRIP/DARKO approach)
- [ ] 2.4.2 Rookie priors from age/draft/measurables; regression-to-mean falls out of the filter
- [ ] 2.4.3 DELTA-style rating-drift feature (rate of change — slow ratings underrate risers)
- [ ] 2.4.4 Keep talent estimates opponent-neutral; apply opponent/pace/rest/HCA as separate context layer

### Task 2.5 — Padding method for high-variance stats
- [ ] 2.5.1 Implement Medvedovsky padding: add baseline makes/attempts (e.g. 85/242 for 3P%) so estimates regress to league mean until sample surpasses the noise threshold
- [ ] 2.5.2 Apply to 3P%, FT%, and other high-variance rate stats in the rolling features of Epic 1
- [ ] 2.5.3 Unit tests: small-sample estimates shrink correctly; large-sample estimates converge to raw

### Task 2.6 — Luck adjustment (LEBRON-style)
- [ ] 2.6.1 Strip opponent uncontested-3P% variance from defensive ratings (expected vs actual teammate/opponent shooting)
- [ ] 2.6.2 Garbage-time / rubber-band score-state adjustment audit vs existing `GARBAGE_TIME_WEIGHT`
- [ ] 2.6.3 Ablate luck-adjusted vs raw defensive form features

### Task 2.7 — Role-aware archetype priors (LEBRON / Osken & Sahin bridge)
- [ ] 2.7.1 Offensive archetype labels stabilize box-score priors to the player's role (feeds Epic 4.2 clustering)
- [ ] 2.7.2 Use archetype prior when a player's role changes mid-season (trade, coach change)

### Task 2.8 — Box Creation passing metric
- [ ] 2.8.1 Implement Ben Taylor's Box Creation polynomial (usage × passing volume × 3P-proficiency sigmoid for spacing gravity)
- [ ] 2.8.2 Replace/augment raw assist rates as the offensive-initiation feature
- [ ] 2.8.3 Decision gate vs rolling AST% from 1.2

### Task 2.9 — Player-aware strength of schedule
- [ ] 2.9.1 Weight past opponents by who actually played (minutes-weighted opponent impact), not just team rating

---

## Epic 3 — Rating engine upgrades

### Task 3.1 — Dual-window team ratings (Neil Paine)
- [ ] 3.1.1 Short window (~last 50 games, 15-game mean regression) for regular season
- [ ] 3.1.2 Long window (~last 110 games, no regression), playoff games triple-weighted, for postseason
- [ ] 3.1.3 Time-horizon uncertainty widening for far-future predictions

### Task 3.2 — Recency-weighted closed-form baseline
- [ ] 3.2.1 Weighted least-squares / Massey-style ratings on margin of victory as benchmark head (best variant SD 11.98)
- [ ] 3.2.2 Publish RMSE vs ~12.0 model / ~11.85 closing-spread benchmarks in `docs/history.md`

### Task 3.3 — Per-team home-court advantage
- [ ] 3.3.1 Team-specific HCA instead of global `HOME_ADV=2.5`; shrink low-sample teams to league mean
- [ ] 3.3.2 Re-estimate per season; track HCA decay trend over eras

### Task 3.4 — Season-start priors
- [ ] 3.4.1 Prior = prior-season tail + explicit ~15-game mean regression; combine with Epic 2.2 impact prior
- [ ] 3.4.2 Revisit `OFFSEASON_REVERSION` knob against new prior; tune within leak rules

### Task 3.5 — Playoff weighting
- [ ] 3.5.1 Upweight playoff games (~3×) in rating updates

### Task 3.6 — Markov state-transition game-flow model
- [ ] 3.6.1 Model possessions as Markov states (e.g. defensive rebound → transition score); estimate transition probability matrices per team/lineup from PBP
- [ ] 3.6.2 Use transition matrices to simulate play-by-play flows → alternative game-flow prior for pace/totals
- [ ] 3.6.3 Decision gate: compare sim-based totals calibration vs existing score-pair heads

---

## Epic 4 — Machine learning architecture & interpretability

### Task 4.1 — Gradient-boosted stacking ensemble
- [ ] 4.1.1 CatBoost head (native high-cardinality categoricals: coach IDs, venue, archetype clusters) + XGBoost + LightGBM on identical OOF splits
- [ ] 4.1.2 Keep Elo/hier as features inside the ensemble — ensemble, don't replace (Elo strongest single predictor in Houde)
- [ ] 4.1.3 Beat the 65.1% GNB baseline (Houde) and report vs 65–72% realistic ceiling

### Task 4.2 — Player-archetype clustering features (Osken & Sahin)
- [ ] 4.2.1 Seasonal k-means + fuzzy c-means on box + efficiency + shot zones (incl. Above-Break-3 and Restricted-Area usage)
- [ ] 4.2.2 Minutes-weighted cluster-share game representation as high-weight categorical features for CatBoost
- [ ] 4.2.3 Staggered in-season refits (every ~10–30 games), past-only

### Task 4.3 — Heteroscedastic Huber loss for spread MAE *(direct MAE lever)*
- [ ] 4.3.1 Replace MSE with Huber loss on the margin head: quadratic inside ε, linear beyond — blowouts stop skewing regression weights
- [ ] 4.3.2 Tune ε via Optuna on OOF MAE (never ROI); consider per-context ε (heteroscedastic: rivalry/blowout-risk games get wider ε)
- [ ] 4.3.3 Acceptance: OOF spread MAE improvement vs current head on identical folds

### Task 4.4 — Monte Carlo game simulation layer
- [ ] 4.4.1 Simulate margins/totals 10k× from predicted mean + variance (wire onto existing Skellam/score-pair heads)
- [ ] 4.4.2 Derive spread/total/ML fair prices from sim distribution; cross-check vs analytic probabilities

### Task 4.5 — Zero-lookahead CV hardening
- [ ] 4.5.1 Extend `PastOnlyGroupCV` with hard chronological gap enforcement between folds
- [ ] 4.5.2 SHA-256 dataset hash + timestamp assertion pre-fit: no training row timestamp ≥ inference timestamp
- [ ] 4.5.3 CI test that tampered chronology fails the assertion

### Task 4.6 — SHAP interpretability program
- [ ] 4.6.1 Global summary plots per model head (quantify each rolling metric's contribution)
- [ ] 4.6.2 Dependence plots for key interactions (e.g. Box Creation × elite Defensive EPM)
- [ ] 4.6.3 Scheduled SHAP-driven pruning via `shap_prune.py`; publish per-run top-features in run manifests

### Task 4.7 — Honest benchmarking & validation
- [ ] 4.7.1 Always report vs naive home-win (~58%), win%-only baseline, closing-spread RMSE
- [ ] 4.7.2 In-season forward validation harness: train prior seasons, score current season weekly
- [ ] 4.7.3 Calibration slope check (research models run ~0.91 — shrink overconfidence); surface Brier/ECE/log-loss in dashboard
- [ ] 4.7.4 Optuna expansion within leak-registry rules (never ROI objective; never rolling-window tuning on scoreboard seasons)

---

## Epic 5 — Market calibration, totals & risk execution

### Task 5.1 — Multi-book line shopping
- [ ] 5.1.1 Ingest multiple books; compute edge vs **best available price**, not a single reference book
- [ ] 5.1.2 Track CLV against close per book; keep quote-level provenance (promotion gate requirement)

### Task 5.2 — Edge presentation
- [ ] 5.2.1 Fair-odds conversion + edge % column on dashboard ATS/ML/Totals tabs
- [ ] 5.2.2 One-glance positive-EV flag with min-edge threshold respecting the anti-roadmap

### Task 5.3 — Market signal features & composite mode
- [ ] 5.3.1 Extend `spread_move`/RLM/steam with timestamped move velocity from Pinnacle snapshots
- [ ] 5.3.2 Composite mode (Paine): optional market-odds blend to absorb news — stats-only mode preserved for backtests

### Task 5.4 — Provenance debt
- [ ] 5.4.1 Resolve `quote_tip_proxy` → restore `promotion_eligible` or backfill real tip times

### Task 5.5 — Venn-Abers bounded-probability execution *(extends existing `venn_abers.py`)*
- [ ] 5.5.1 Surface calibrated upper/lower probability bounds on the moneyline head
- [ ] 5.5.2 Execution flag when bookmaker implied probability falls entirely outside the calibrated bound (high-leverage games)
- [ ] 5.5.3 Backtest bound-based filtering vs flat edge threshold

### Task 5.6 — Totals decomposition & Skellam margin clusters *(extends existing `skellam.py`)*
- [ ] 5.6.1 Decompose totals into expected possessions (Epic 1.6 tempo) × expected efficiency, instead of raw point regression
- [ ] 5.6.2 Skellam PMF over independent Poisson home/away expectations → exact margin-cluster probabilities (P(diff = 3, 5, 7…)) for alternate-spread pricing
- [ ] 5.6.3 Totals calibration report: Brier/ECE per totals bucket in dashboard

### Task 5.7 — Temporal de-vigging *(extends existing `devig.py`)*
- [ ] 5.7.1 Time-to-tipoff de-vig matrix: **Shin method** (accounts for sharp-money/insider flow) for lines <2h from tip; **Power method** for early low-liquidity lines
- [ ] 5.7.2 Validate de-vig method choice against closing-line consensus accuracy

### Task 5.8 — CVaR portfolio staking
- [ ] 5.8.1 Move beyond per-bet Kelly: nightly slate as a correlated portfolio
- [ ] 5.8.2 Gaussian copulas to map covariance between same-night games; simulate slate-level outcomes
- [ ] 5.8.3 CVaR constraint: size bets so expected loss in worst 5% of tail scenarios ≤ configured drawdown limit
- [ ] 5.8.4 Backtest vs current fractional Kelly (`stake_profiles.py`): compare max drawdown, ruin probability, ROI

---

## Epic 6 — Engineering, automation & platform

### Task 6.1 — CI hardening
- [ ] 6.1.1 Make `ruff check` fail-hard once backlog is clean (currently `|| true`)
- [ ] 6.1.2 Expand GHA pytest matrix beyond core leak/smoke set toward the full ~100-module suite
- [ ] 6.1.3 `pre-commit install` documented + enforced

### Task 6.2 — Experiment tracking
- [ ] 6.2.1 Index `output/<run>/README.md` manifests in a lightweight JSON store (or MLflow)
- [ ] 6.2.2 Dashboard Run Lab reads the index for cross-run comparison

### Task 6.3 — Serving & reproducibility
- [ ] 6.3.1 `predict_game` FastAPI route alongside the dashboard
- [ ] 6.3.2 Docker image for the full suite; document raw PBP/odds data not in git
- [ ] 6.3.3 Dashboard tabs: player rolling stats, lineup rolling form, edge/fair-odds view

### Task 6.4 — Knowledge sharing
- [ ] 6.4.1 Public write-up: *"Finding chronological leaks in my own NBA model"*
- [ ] 6.4.2 Docs page per epic as features land

### Task 6.5 — Performance engineering
- [ ] 6.5.1 Profile rating-update loops (`ratings.py`, `hierarchical.py`, `game_updates.py`); vectorize hot Pandas loops (NumPy/Polars/PyArrow)
- [ ] 6.5.2 Numba JIT on stint-level aggregation inner loops if profiling justifies
- [ ] 6.5.3 CI benchmark gate: full-suite runtime regression check

### Task 6.6 — Automation agents
- [ ] 6.6.1 Scheduled GitHub Action / Cursor Automation: nightly EPM + injury + odds refresh with failure alerts
- [ ] 6.6.2 Automated dependency/security updates (Dependabot) with CI gate
- [ ] 6.6.3 Literature-watch automation: monthly digest issue of new relevant papers (automates what this roadmap did manually)

---

## Prediction-target map (what moves which metric)

| Target | Highest-leverage tasks |
|--------|------------------------|
| **Spread MAE ↓** | 4.3 Huber loss, 3.2 recency-weighted margin baseline, 2.2 impact-weighted ratings, 3.3 per-team HCA, 1.3/1.4 lineup form |
| **Winner accuracy ↑** | 4.1 GBM ensemble, 4.2 archetype clusters, 2.4 Kalman now-casting, 3.1 dual-window ratings, 3.4 season priors |
| **Totals calibration ↑** | 5.6 totals decomposition + Skellam clusters, 1.6 tempo-dictation, 3.6 Markov game-flow, 4.4 Monte Carlo sims |
| **CLV / ROI ↑** | 5.1 line shopping, 5.5 Venn-Abers bounds, 5.7 temporal de-vig, 5.8 CVaR staking, 5.3 composite mode |

## Explicit non-goals (anti-roadmap)

From `code/README.md` / leak registry:

- Do not promote on ATS/ROI when CLV is NaN
- Do not Optuna-tune the rolling window for scoreboard seasons
- Do not loosen min-edge only to restore bet volume after edge compression
- Do not let `elo_blend_alpha` hug the market past `ELO_BLEND_ALPHA_MAX`
- Do not ship rolling features without past-only T-60 isolation tests
- Do not assume a new stat is a winner — ablate, SHAP-audit, then keep or drop
- Do not rebuild what exists (Venn-Abers, Skellam, de-vig, shot quality, past-only CV) — extend it
- Respect the ~65–72% winner-accuracy ceiling literature; chase calibration and CLV, not raw accuracy records

## Recently landed (kept for context)

- [x] Modular `code/pipeline/` package + leak registry
- [x] FastAPI T-60 dashboard
- [x] Staged `run_full_suite.py` with review gates
- [x] Portfolio README, MIT license, `pyproject.toml`
- [x] GitHub Actions CI (pytest + ruff) + MkDocs site
- [x] Roadmap v2 research synthesis (12-agent literature + codebase audit)
