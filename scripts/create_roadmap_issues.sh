#!/usr/bin/env bash
# Creates GitHub milestones (epics) and issues (tasks) for docs/roadmap.md v2.
# Idempotent-ish: skips milestones/issues whose titles already exist.
# Usage: ./scripts/create_roadmap_issues.sh
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="ShaneHurley/BasketballElo"

milestone_number() {
  gh api "repos/$REPO/milestones?state=all&per_page=100" --jq ".[] | select(.title==\"$1\") | .number"
}

ensure_milestone() {
  local title="$1" desc="$2"
  local num
  num=$(milestone_number "$title" || true)
  if [ -z "$num" ]; then
    num=$(gh api "repos/$REPO/milestones" -f title="$title" -f description="$desc" --jq '.number')
    echo "Created milestone: $title (#$num)" >&2
  else
    echo "Milestone exists: $title (#$num)" >&2
  fi
  echo "$title"
}

issue_exists() {
  gh issue list --repo "$REPO" --search "in:title \"$1\"" --json title --jq '.[].title' | grep -qxF "$1"
}

make_issue() {
  local milestone="$1" title="$2" body="$3"
  if issue_exists "$title"; then
    echo "Issue exists, skipping: $title"
    return
  fi
  gh issue create --repo "$REPO" --title "$title" --milestone "$milestone" --body "$body" >/dev/null
  echo "Created issue: $title"
}

# ---------------- Milestones (Epics) ----------------
# Milestones already created; reference by title.

M1="Epic 1 — Player & lineup rolling form features"
M2="Epic 2 — Player-impact-weighted team strength"
M3="Epic 3 — Rating engine upgrades"
M4="Epic 4 — Modeling, simulation & validation rigor"
M5="Epic 5 — Market & betting edge layer"
M6="Epic 6 — Engineering, CI/CD & platform"
# ---------------- Epic 1 issues ----------------

make_issue "$M1" "1.1 Player box-score derivation layer" "## Goal
Derive per-player per-game box scores from existing PBP so no external box API is needed.

## Subtasks
- [ ] Derive per-player per-game box (PTS, AST, REB, TOV, FGA, FTA, 3PM, MIN, USG) from V3 PBP stints (\`code/pipeline/stints.py\`, \`stint_context.py\`)
- [ ] Reconcile derived boxes against \`nba_player_stats_2026.csv\` (currently on disk, unused) as validation oracle; add parity test with tolerances
- [ ] Add \`player_game_box\` artifact to ingest stage (\`ingest.py\`, \`preprocess.py\`) with schema versioning in \`config.py\`

## Acceptance
Derived season totals match the CSV oracle within tolerance; artifact is schema-versioned."

make_issue "$M1" "1.2 Rolling player efficiency stats (5/10/20)" "## Goal
Rolling player stats mirroring \`FORM_ROLL_WINDOWS\`, aggregated to team level by projected minutes.

## Subtasks
- [ ] Rolling AST% (share of teammate FGM assisted while on floor)
- [ ] Rolling TS% \`PTS / (2 × (FGA + 0.44 × FTA))\` and eFG% \`(FGM + 0.5 × 3PM) / FGA\`
- [ ] Rolling PPG, USG%, TOV%, OREB%/DREB%
- [ ] Minutes-weighted team aggregation using \`minutes_forecast.py\` projections (NOT actuals — T-60 leak rule)
- [ ] Emit \`h_/\`a_\` + \`_diff\` columns in \`game_features.build_game_features\`; register in \`model.py\` \`SAFE_FEATURE_COLS\`; bump \`FEATURE_SCHEMA_VERSION\`"

make_issue "$M1" "1.3 Rolling lineup form, last 10 games (investigate as winners)" "## Goal
New \`code/pipeline/lineup_form.py\` — rolling box stats per 5-man lineup. User-requested: investigate whether these are winners, don't assume.

## Subtasks
- [ ] Rolling lineup pace (possessions per 48 from stints) over last 10 lineup appearances
- [ ] Rolling lineup PPG (and xPPP for comparison) last 10
- [ ] Rolling lineup AST% and TOV% last 10 (cohesion proxy; strongest early-season when RD is high)
- [ ] Shrinkage to player-mean for low-possession lineups (reuse \`MIN_POSSESSIONS\`/\`SHRINK_K\` from \`lineup_elo.py\`)
- [ ] Expected-lineup-weighted team aggregates on T-60 row; ablation toggles in \`ablation.py\`
- [ ] **Decision gate:** keep each stat only if walk-forward OOF log-loss/MAE improves AND SHAP importance is non-trivial; document verdicts in \`docs/pipeline.md\`"

make_issue "$M1" "1.4 Team-level rolling additions" "## Subtasks
- [ ] Rolling team TS% differential (team TS% − opponent TS% allowed), windows 5/10/20
- [ ] Rolling team AST% + TOV% chemistry feature
- [ ] Z-score differential transforms of top features vs league mean/std (Houde 2021's top recommendation)"

make_issue "$M1" "1.5 Validation & leak hygiene for Epic 1" "## Subtasks
- [ ] Past-only tests: no rolling window includes same-game/future data (\`test_feature_parity.py\`, new \`test_lineup_form_t60.py\`)
- [ ] Live/serve parity check in \`predict.py\`/\`simulate.py\`
- [ ] SHAP audit of new features (\`shap_prune.py\`); prune noise
- [ ] Update \`LEAK_REGISTRY.md\` if any new leak class is found"

# ---------------- Epic 2 issues ----------------

make_issue "$M2" "2.1 EPM data pipeline hardening" "## Subtasks
- [ ] Automate nightly EPM prior refresh with versioned \`observation_date\` (extends \`epm_priors.py\`); fail loudly on stale data
- [ ] Backfill historical daily EPM where available (nightly values since 2001–02, no lookahead — safe for backtests)"

make_issue "$M2" "2.2 Minutes-weighted impact team rating" "## Goal
Team O/D rating = Σ(projected minutes × player EPM-O/EPM-D). The standard fix (EPM/DRIP/DARKO/CARM-Elo/CraftedNBA) for slow reaction to trades/injuries.

## Subtasks
- [ ] Compute via \`minutes_forecast.py\` + \`rotation_scenarios.py\`
- [ ] Expose features (\`epm_team_net\`, \`epm_vs_elo_gap\`) and use as season-start prior replacing flat mean regression
- [ ] Benchmark vs Dunks & Threes minutes-weighted team EPM RMSE (~12.1)"

make_issue "$M2" "2.3 Injury & rotation adjustment v2" "## Subtasks
- [ ] Replace/augment star-out heuristics with impact × expected-minutes deltas (missing EPM − replacement EPM)
- [ ] Depth-chart diff signal: rotation minute shifts as leading indicator (CraftedNBA approach)
- [ ] Historical injury archive so backtests use same adjustment path as live ESPN fetch (\`availability.py\`, \`injury_reports.py\`)"

make_issue "$M2" "2.4 Bayesian now-casting for player ratings (DRIP-style)" "## Subtasks
- [ ] Exponentially-decayed per-stat current-true-talent estimates with priors (rookie prior from age/draft/measurables)
- [ ] DELTA-style rating-drift feature (rate of change as input — slow ratings underrate risers)
- [ ] Keep talent estimates opponent-neutral; apply opponent/pace/rest/HCA as separate context layer"

make_issue "$M2" "2.5 Player-aware strength of schedule" "## Subtasks
- [ ] Weight past opponents by who actually played (minutes-weighted opponent EPM), not just opponent team rating"

# ---------------- Epic 3 issues ----------------

make_issue "$M3" "3.1 Dual-window team ratings (Neil Paine)" "## Subtasks
- [ ] Short window (~last 50 games, 15-game mean regression) for regular season
- [ ] Long window (~last 110 games, no regression), playoff games triple-weighted, for postseason
- [ ] Time-horizon uncertainty widening for far-future predictions"

make_issue "$M3" "3.2 Recency-weighted closed-form baseline" "## Subtasks
- [ ] Weighted least-squares / Massey-style ratings on margin of victory as benchmark head (Lu/Chen/Zhu 2019: best variant SD 11.98)
- [ ] Publish RMSE vs ~12.0 model / ~11.85 closing-spread benchmarks in \`docs/history.md\`"

make_issue "$M3" "3.3 Per-team home-court advantage" "## Subtasks
- [ ] Team-specific HCA instead of global \`HOME_ADV=2.5\`; shrink low-sample teams toward league mean"

make_issue "$M3" "3.4 Season-start priors" "## Subtasks
- [ ] Prior = prior-season tail + explicit ~15-game mean regression (not flat revert); combine with impact prior from 2.2
- [ ] Revisit \`OFFSEASON_REVERSION\` knob against new prior; tune within leak rules"

make_issue "$M3" "3.5 Playoff weighting" "## Subtasks
- [ ] Upweight playoff games (~3×) in rating updates — postseason results are disproportionately informative"

# ---------------- Epic 4 issues ----------------

make_issue "$M4" "4.1 ML benchmark bake-off" "## Subtasks
- [ ] CatBoost and LightGBM heads vs current stack on identical OOF splits (Li & Jia 2024: CatBoost+Optuna won; Houde: GNB/logreg strong baselines)
- [ ] Keep Elo/hier as features inside the ML model — ensemble, don't replace (Elo was strongest single predictor in Houde)"

make_issue "$M4" "4.2 Player-archetype roster features (Osken & Sahin)" "## Subtasks
- [ ] Seasonal k-means + fuzzy c-means player clustering on box + efficiency + shot zones
- [ ] Minutes-weighted cluster-share game representation as features
- [ ] Staggered in-season cluster refits (every ~10–30 games), past-only"

make_issue "$M4" "4.3 Monte Carlo game simulation layer" "## Subtasks
- [ ] Simulate margins/totals 10k× from predicted mean + variance (wire onto existing Skellam/score-pair heads)
- [ ] Derive spread/total/ML fair prices from sim distribution; cross-check vs analytic probabilities"

make_issue "$M4" "4.4 Honest benchmarking & validation" "## Subtasks
- [ ] Always report vs naive home-win (~58%), win%-only baseline, closing-spread RMSE
- [ ] In-season forward validation harness: train prior seasons, score current season weekly
- [ ] Calibration slope check (research models run ~0.91 — shrink overconfidence); surface Brier/ECE/log-loss in dashboard
- [ ] Optuna expansion for new knobs within leak-registry rules (never ROI objective; never rolling-window tuning on scoreboard seasons)"

# ---------------- Epic 5 issues ----------------

make_issue "$M5" "5.1 Multi-book line shopping" "## Subtasks
- [ ] Ingest multiple books; compute edge vs best available price, not a single reference book
- [ ] Track CLV against close per book; keep quote-level provenance (promotion gate requirement)"

make_issue "$M5" "5.2 Edge presentation" "## Subtasks
- [ ] Fair-odds conversion + edge % column on dashboard ATS/ML/Totals tabs
- [ ] One-glance positive-EV flag (Stats Insider-style) with min-edge threshold respecting anti-roadmap (no loosening to restore volume)"

make_issue "$M5" "5.3 Market signal features" "## Subtasks
- [ ] Extend \`spread_move\`/RLM/steam with timestamped move velocity from Pinnacle snapshots
- [ ] Composite mode (Paine): optional blend with market odds to absorb news — stats-only mode preserved for honest backtesting"

make_issue "$M5" "5.4 Provenance debt" "## Subtasks
- [ ] Resolve \`quote_tip_proxy\` (tip inferred from last quote) → restore \`promotion_eligible\` for affected games or backfill real tip times"

# ---------------- Epic 6 issues ----------------

make_issue "$M6" "6.1 CI hardening" "## Subtasks
- [ ] Make \`ruff check\` fail-hard once backlog is clean (currently \`|| true\`)
- [ ] Expand GHA pytest matrix beyond core leak/smoke set toward full ~100-module suite
- [ ] \`pre-commit install\` documented + enforced in contributor docs"

make_issue "$M6" "6.2 Experiment tracking" "## Subtasks
- [ ] Index \`output/<run>/README.md\` manifests in lightweight JSON store (or MLflow)
- [ ] Dashboard Run Lab reads the index for cross-run comparison"

make_issue "$M6" "6.3 Serving & reproducibility" "## Subtasks
- [ ] \`predict_game\` FastAPI route alongside the dashboard
- [ ] Docker image for full suite; document raw PBP/odds data not in git
- [ ] Dashboard tabs for new features: player rolling stats, lineup rolling form, edge/fair-odds view"

make_issue "$M6" "6.4 Knowledge sharing" "## Subtasks
- [ ] Public write-up: 'Finding chronological leaks in my own NBA model'
- [ ] Docs page per epic as features land (\`docs/pipeline.md\` + math pages)"

echo "Done. View milestones: https://github.com/$REPO/milestones"
