#!/usr/bin/env bash
# Epic 10 — Synthetic formula test bench & adversarial stress harness.
# Source: docs/roadmap.md Epic 10. Idempotent: skips existing issue titles.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="ShaneHurley/BasketballElo"
M10="Epic 10 — Synthetic formula test bench"

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

ensure_milestone "$M10"

make_issue "$M10" "10.1 Synthetic data factory layer" "## Subtasks
- [ ] 10.1.1 \`code/tests/synth/factories.py\`: \`make_stint\`, \`make_game\`, \`make_odds\`, \`make_player_season\` — schema-identical to \`stints.py\` / odds loaders; explicit \`seed=\` everywhere
- [ ] 10.1.2 Extreme presets in \`code/tests/synth/presets.py\`: \`BLOWOUT_60\`, \`ZERO_POSSESSION_FT_STINT\`, \`ONE_SIDED_ODDS_FEED\`, \`JUICE_EXTREMES\`, \`ROOKIE_VS_5000_GAME_VET\`, \`NAN_STORM\`, \`SINGLE_LINEUP_ALL_SEASON\`, \`DISJOINT_LINEUPS_EVERY_STINT\`
- [ ] 10.1.3 Schema-parity test against real ingest validators
- [ ] 10.1.4 \`code/tests/synth/README.md\`: bench is invariant/adversarial only — never for tuning constants

**Do not rebuild** \`pipeline/synthetic_research.py\` (TVAE/SDV augmentation gates — different purpose)."

make_issue "$M10" "10.2 Stage-by-stage formula invariant battery" "## Subtasks
- [ ] 10.2.1 \`test_bench_ratings.py\` — home/away symmetry, RD bounds, K-decay monotonicity, garbage-weight idempotence (P0.8)
- [ ] 10.2.2 \`test_bench_lineup_chemistry.py\` — mirrored-update invariant (P0.1), on/off \`off\` leg, shrinkage limits
- [ ] 10.2.3 \`test_bench_market.py\` — devig sums to 1 incl. single-sided (P0.3), Kelly \`b == american_to_decimal − 1\` (P0.2), vigfree honesty (P0.10)
- [ ] 10.2.4 \`test_bench_calibration.py\` — interval responds to \`update_residuals\` (P0.4), VA fail-closed (P0.9), min-sample floors
- [ ] 10.2.5 \`test_bench_staking_grading.py\` — stake ≥ 0, push ⇒ 0 profit, caps before profit, bankroll under 100-loss streak"

make_issue "$M10" "10.3 Property-based testing (Hypothesis)" "## Subtasks
- [ ] 10.3.1 Add \`hypothesis>=6\` to \`requirements-dev.txt\` + \`pyproject.toml\` \`[dev]\`
- [ ] 10.3.2 \`test_bench_properties.py\`: randomized \`PastOnlyGroupCV\`/\`ManualOOFStacker\` groups; residual↔margin round-trip
- [ ] 10.3.3 Freeze each new Hypothesis failure into a permanent example test
- [ ] 10.3.4 CI \`@settings(max_examples=200, deadline=None, derandomize=True)\`; separate local profile"

make_issue "$M10" "10.4 Leak canary harness (planted-signal tests)" "## Subtasks
- [ ] 10.4.1 \`code/tests/synth/canaries.py\`: \`PsychicFeature\`, \`TimeTravelerTracker\`, \`FutureOddsQuote\`
- [ ] 10.4.2 Assert every gate catches its canary (a gate that passes a canary = failing test)
- [ ] 10.4.3 Chronology-tamper harness for roadmap 4.5.3 (SHA-256 hash fails on row-order permutation)
- [ ] 10.4.4 Extend \`negative_controls.py\` permutation nulls into the bench runner per stage"

make_issue "$M10" "10.5 Golden-master differential oracles" "## Subtasks
- [ ] 10.5.1 \`code/tests/synth/golden.py\`: fixed synthetic 40-game season (seed-locked); snapshot JSON
- [ ] 10.5.2 Differential checks: four \`devig.py\` methods; Skellam vs Gaussian; \`fit_static\` vs \`replay_rolling\`
- [ ] 10.5.3 Golden diffs fail CI on numeric drift
- [ ] 10.5.4 Version golden files with \`FEATURE_SCHEMA_VERSION\`"

make_issue "$M10" "10.6 P0 regression battery (acceptance for current bug fixes)" "## Subtasks
- [ ] 10.6.1 One bench-style test per P0.1–P0.10, each citing GitHub issue + planned \`LEAK_REGISTRY.md\` ID. Red until fixed, green after
- [ ] 10.6.2 P0.11: synthetic two-snapshot odds fixture (decision ≠ close) proving CLV path can produce finite values

Scaffold already includes P0.2 (Kelly payout identity) and P0.9 (VA fail-closed) as the working template."

make_issue "$M10" "10.7 CI & reporting integration" "## Subtasks
- [ ] 10.7.1 \`@pytest.mark.bench\`; separate CI job with <5 min budget (\`.github/workflows/ci.yml\`)
- [ ] 10.7.2 Bench summary → \`output/bench/latest.json\` + dashboard Documentation tab (extends 7.5)
- [ ] 10.7.3 Coverage gate on the bench run (\`pytest --cov=pipeline\`)"

make_issue "$M10" "10.8 Registry & docs integration" "## Subtasks
- [ ] 10.8.1 \`LEAK_REGISTRY.md\` scope note: \`bench_<name>\` IDs; broaden to non-temporal defects
- [ ] 10.8.2 \`docs/pipeline.md\`: how to add a stage (\`test_bench_<module>.py\`)
- [ ] 10.8.3 Anti-roadmap already updated: do not tune constants against the bench; do not treat bench-green as leak-proof"

echo "Done."
