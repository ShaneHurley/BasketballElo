# Roadmap

Ranked by expected model lift and engineering value. Each epic lists tasks → subtasks with
file touch-points and acceptance criteria. Research sources are cited inline.

**Where we are:** the stack is already deep on possession-level ratings (player Glicko-2 O/D,
hierarchical n-man, lineup Elo, chemistry, HAPM), team multi-window four-factors form
(3/5/10/20), minutes/injury scaffolding, and anti-leakage machinery (OOF stacker, leak
registry, promotion gates). The biggest white space, confirmed by codebase audit and the
research below: **player counting-stat features**, **lineup-level rolling box/pace windows**,
**player-impact-weighted team strength**, and **market/simulation layers**.

**Research base:** Osken & Sahin 2022 (player-archetype clustering + GA-NN, ~70–76% acc);
Houde 2021 (Elo strongest single predictor, z-score differentials, 65–72% realistic ceiling);
Lu/Chen/Zhu 2019 (recency-weighted regression, per-team HCA, RMSE ~12 vs market ~11.85);
Li & Jia 2024 (CatBoost + Optuna + SHAP, VORP top value signal); Dunks & Threes EPM
(SPM-prior RAPM, nightly since 2001–02, no lookahead, minutes-weighted team EPM RMSE 12.1);
CraftedNBA (depth-chart minutes × impact → game-day strength); Opta DRIP (Bayesian
now-casting, opponent-neutral talent + separate context layer); Neil Paine (dual-window
ratings, 15-game mean regression, playoff 3× weight, market composite mode); Stats Insider
(10k Monte Carlo sims, fair-odds edge flags, line shopping); SportsVisio (four-factors /
TS% / AST% / pace formulas).

---

## Epic 1 — Player & lineup rolling form features

*User-priority epic. Gap confirmed: no player counting stats in features; lineup layer is
rating residuals only, no rolling box windows. `nba_player_stats_2026.csv` sits on disk
unused. Each new stat ships behind an ablation toggle and must earn its place via OOF
metrics + SHAP — investigate, don't assume.*

### Task 1.1 — Player box-score derivation layer
- [ ] 1.1.1 Derive per-player per-game box scores (PTS, AST, REB, TOV, FGA, FTA, 3PM, MIN, USG) from existing V3 PBP stints (`code/pipeline/stints.py`, `stint_context.py`) so no external box-score API is needed
- [ ] 1.1.2 Reconcile derived boxes against `nba_player_stats_2026.csv` as a validation oracle; add a parity test with tolerance thresholds
- [ ] 1.1.3 Add a `player_game_box` artifact to the ingest stage (`ingest.py`, `preprocess.py`) with schema versioning in `config.py`

### Task 1.2 — Rolling player efficiency stats (windows 5/10/20, mirroring `FORM_ROLL_WINDOWS`)
- [ ] 1.2.1 Rolling **AST%** (share of teammate FGM assisted while on floor — SportsVisio formula) per player
- [ ] 1.2.2 Rolling **TS%** and **eFG%** per player (`PTS / (2 × (FGA + 0.44 × FTA))`, `(FGM + 0.5 × 3PM) / FGA`)
- [ ] 1.2.3 Rolling **PPG**, **USG%**, **TOV%**, **OREB%/DREB%** per player
- [ ] 1.2.4 Aggregate to team level minutes-weighted by `minutes_forecast.py` projected minutes (not actuals — T-60 leak rule)
- [ ] 1.2.5 Emit `h_/a_` + `_diff` columns in `game_features.build_game_features`; register in `model.py` `SAFE_FEATURE_COLS`; bump `FEATURE_SCHEMA_VERSION`

### Task 1.3 — Rolling lineup form (last 10 games) — *investigate as potential winners*
- [ ] 1.3.1 New `code/pipeline/lineup_form.py`: per-5-man-lineup rolling **pace** (possessions per 48 from stint data) over last 10 lineup appearances
- [ ] 1.3.2 Rolling lineup **PPG** (and xPPP alongside for comparison) last 10
- [ ] 1.3.3 Rolling lineup **AST%** and **TOV%** last 10 (cohesion proxy — strongest early-season when Elo RD is high)
- [ ] 1.3.4 Shrinkage toward player-mean for low-possession lineups (reuse `MIN_POSSESSIONS`/`SHRINK_K` pattern from `lineup_elo.py`)
- [ ] 1.3.5 Emit expected-lineup-weighted team aggregates on the T-60 row; ablation toggles in `ablation.py`
- [ ] 1.3.6 **Decision gate:** keep each stat only if walk-forward OOF log-loss/MAE improves AND SHAP importance is non-trivial; document keep/drop verdict per stat in `docs/pipeline.md`

### Task 1.4 — Team-level additions
- [ ] 1.4.1 Rolling team **TS% differential** (team TS% − opponent TS% allowed) windows 5/10/20
- [ ] 1.4.2 Rolling team **AST% + TOV%** chemistry feature (SportsVisio cohesion proxy)
- [ ] 1.4.3 Z-score differential transforms of top features vs league mean/std (Houde's top recommendation)

### Task 1.5 — Validation & leak hygiene for Epic 1
- [ ] 1.5.1 Past-only tests: no rolling window may include same-game or future data (`test_feature_parity.py`, new `test_lineup_form_t60.py`)
- [ ] 1.5.2 Live/serve parity check in `predict.py` / `simulate.py`
- [ ] 1.5.3 SHAP audit of new features (`shap_prune.py`); prune noise
- [ ] 1.5.4 Update `LEAK_REGISTRY.md` if any new leak class is found

---

## Epic 2 — Player-impact-weighted team strength (EPM / DRIP / CraftedNBA style)

*The single biggest documented weakness of result-based ratings: slow reaction to trades,
injuries, and early-season roster churn. Every researched pro model (EPM, DRIP, DARKO,
CARM-Elo, CraftedNBA) solves it the same way: minutes-weighted player impact → team strength.*

### Task 2.1 — EPM data pipeline hardening
- [ ] 2.1.1 Automate nightly EPM prior refresh with versioned `observation_date` (extends `epm_priors.py`); fail loudly on stale data
- [ ] 2.1.2 Backfill historical daily EPM where available (nightly values exist since 2001–02, no lookahead — safe for backtests)

### Task 2.2 — Minutes-weighted impact team rating
- [ ] 2.2.1 Compute team O/D rating = Σ(projected minutes × player EPM-O/EPM-D) using `minutes_forecast.py` + `rotation_scenarios.py`
- [ ] 2.2.2 Expose as features (`epm_team_net`, `epm_vs_elo_gap`) and as a **season-start prior** replacing flat mean regression (CARM-Elo trick: player prior first, game-based ratings dominate as sample grows)
- [ ] 2.2.3 Benchmark vs Dunks & Threes' published minutes-weighted team EPM RMSE (~12.1)

### Task 2.3 — Injury & rotation adjustment v2
- [ ] 2.3.1 Replace/augment star-out heuristics with impact × expected-minutes deltas (missing player EPM − replacement EPM)
- [ ] 2.3.2 Depth-chart diff signal: monitor rotation minute shifts as a leading indicator before results show it (CraftedNBA approach)
- [ ] 2.3.3 Build a historical injury archive so backtests use the same adjustment path as live ESPN fetch (`availability.py`, `injury_reports.py`)

### Task 2.4 — Bayesian now-casting for player ratings (DRIP-style)
- [ ] 2.4.1 Exponentially-decayed per-stat "current true talent" estimates with priors (rookie prior from age/draft/measurables) instead of raw season averages
- [ ] 2.4.2 DELTA-style rating-drift feature: rate of player/team rating change as an input — slow-moving ratings systematically underrate risers
- [ ] 2.4.3 Keep talent estimates opponent-neutral; apply opponent/pace/rest/HCA as a separate context layer (cleaner to validate)

### Task 2.5 — Player-aware strength of schedule
- [ ] 2.5.1 Weight past opponents by who actually played (minutes-weighted opponent EPM), not just opponent team rating

---

## Epic 3 — Rating engine upgrades

### Task 3.1 — Dual-window team ratings (Neil Paine)
- [ ] 3.1.1 Short window (~last 50 games, 15-game mean regression) for regular season
- [ ] 3.1.2 Long window (~last 110 games, no regression) with playoff games triple-weighted for postseason predictions
- [ ] 3.1.3 Time-horizon uncertainty widening for far-future predictions

### Task 3.2 — Recency-weighted closed-form baseline (Lu/Chen/Zhu)
- [ ] 3.2.1 Implement weighted least-squares / Massey-style ratings on margin of victory as a benchmark head
- [ ] 3.2.2 Publish RMSE vs the ~12.0 (model) / ~11.85 (closing spread) benchmarks in `docs/history.md`

### Task 3.3 — Per-team home-court advantage
- [ ] 3.3.1 Estimate team-specific HCA instead of global `HOME_ADV=2.5`; shrink low-sample teams toward league mean

### Task 3.4 — Season-start priors
- [ ] 3.4.1 Prior = prior-season tail + explicit ~15-game mean regression (not flat revert); combine with Epic 2.2.2 impact prior
- [ ] 3.4.2 Revisit `OFFSEASON_REVERSION` knob against the new prior; tune within leak rules

### Task 3.5 — Playoff weighting
- [ ] 3.5.1 Upweight playoff games (~3×) in rating updates; postseason results are disproportionately informative

---

## Epic 4 — Modeling, simulation & validation rigor

### Task 4.1 — ML benchmark bake-off
- [ ] 4.1.1 CatBoost and LightGBM heads vs current stack on identical OOF splits (Li & Jia: CatBoost + Optuna won; Houde: GNB/logreg strong baselines)
- [ ] 4.1.2 Keep Elo/hier as features inside the ML model — ensemble, don't replace (Elo was the single strongest predictor in Houde)

### Task 4.2 — Player-archetype roster features (Osken & Sahin)
- [ ] 4.2.1 Seasonal k-means + fuzzy c-means player clustering on box + efficiency + shot zones
- [ ] 4.2.2 Minutes-weighted cluster-share game representation as features
- [ ] 4.2.3 Staggered in-season cluster refits (every ~10–30 games), past-only

### Task 4.3 — Monte Carlo game simulation layer (Stats Insider)
- [ ] 4.3.1 Simulate margins/totals 10k× from predicted mean + variance (Skellam/score-pair heads already exist — wire sims on top)
- [ ] 4.3.2 Derive spread/total/ML fair prices from sim distribution; cross-check against analytic probabilities

### Task 4.4 — Honest benchmarking & validation
- [ ] 4.4.1 Always report vs naive home-win (~58%), win%-only baseline, and closing-spread RMSE
- [ ] 4.4.2 In-season forward validation harness: train on prior seasons, score current season weekly
- [ ] 4.4.3 Calibration slope check (research models run ~0.91 — shrink overconfident predictions); Brier/ECE/log-loss already in `calibration_metrics.py` — surface in dashboard
- [ ] 4.4.4 Optuna expansion for new knobs within leak-registry rules (never ROI objective, never rolling-window tuning on scoreboard seasons)

---

## Epic 5 — Market & betting edge layer

### Task 5.1 — Multi-book line shopping
- [ ] 5.1.1 Ingest multiple books; compute edge vs **best available price**, not a single reference book
- [ ] 5.1.2 Track CLV against close per book; keep quote-level provenance (promotion gate requirement)

### Task 5.2 — Edge presentation
- [ ] 5.2.1 Fair-odds conversion + edge % column on dashboard ATS/ML/Totals tabs
- [ ] 5.2.2 One-glance positive-EV flag (Stats Insider "green smiley" analog) with min-edge threshold that respects the anti-roadmap (no loosening to restore volume)

### Task 5.3 — Market signal features
- [ ] 5.3.1 Extend `spread_move` / RLM / steam with timestamped move velocity from Pinnacle snapshots
- [ ] 5.3.2 Composite mode (Paine): optional blend of model probabilities with market odds to absorb news the model can't see — **stats-only mode preserved for honest backtesting**

### Task 5.4 — Provenance debt
- [ ] 5.4.1 Resolve `quote_tip_proxy` (tip inferred from last quote) → restore `promotion_eligible` for affected games or backfill real tip times

---

## Epic 6 — Engineering, CI/CD & platform

### Task 6.1 — CI hardening
- [ ] 6.1.1 Make `ruff check` fail-hard once backlog is clean (currently `|| true`)
- [ ] 6.1.2 Expand GHA pytest matrix beyond core leak/smoke set toward the full ~100-module suite
- [ ] 6.1.3 `pre-commit install` documented + enforced in contributor docs

### Task 6.2 — Experiment tracking
- [ ] 6.2.1 Index `output/<run>/README.md` manifests in a lightweight JSON store (or MLflow)
- [ ] 6.2.2 Dashboard Run Lab reads the index for cross-run comparison

### Task 6.3 — Serving & reproducibility
- [ ] 6.3.1 `predict_game` FastAPI route alongside the dashboard
- [ ] 6.3.2 Docker image for the full suite; document that raw PBP/odds data is not in git
- [ ] 6.3.3 Dashboard tabs for new features: player rolling stats, lineup rolling form, edge/fair-odds view

### Task 6.4 — Knowledge sharing
- [ ] 6.4.1 Public write-up: *"Finding chronological leaks in my own NBA model"*
- [ ] 6.4.2 Docs page per epic as features land (`docs/pipeline.md` + math pages)

---

## Explicit non-goals (anti-roadmap) — unchanged

From `code/README.md` / leak registry:

- Do not promote on ATS/ROI when CLV is NaN
- Do not Optuna-tune the rolling window for scoreboard seasons
- Do not loosen min-edge only to restore bet volume after edge compression
- Do not let `elo_blend_alpha` hug the market past `ELO_BLEND_ALPHA_MAX`
- Do not ship rolling features without past-only T-60 isolation tests
- Do not assume a new stat is a winner — ablate, SHAP-audit, then keep or drop (applies to rolling lineup PPG/AST%/pace explicitly)

## Recently landed (kept for context)

- [x] Modular `code/pipeline/` package + leak registry
- [x] FastAPI T-60 dashboard
- [x] Staged `run_full_suite.py` with review gates
- [x] Portfolio README, MIT license, `pyproject.toml`
- [x] GitHub Actions CI (pytest + ruff) + MkDocs site
