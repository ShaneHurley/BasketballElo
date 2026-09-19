#!/usr/bin/env bash
# Epic 11 — Quick Wins (P0 fixes).
# Source: docs/roadmap.md Epic 11. Idempotent: skips existing issue titles.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="ShaneHurley/BasketballElo"
M11="Epic 11 — Quick wins (P0 fixes)"

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

ensure_milestone "$M11"

make_issue "$M11" "11.1 P0.7: Remove stale HOME_PPP_BOOST landmine defaults" "## Subtasks
- [ ] Files: \`code/pipeline/feature_utils.py:71\`, \`code/pipeline/lineup_elo.py:15\`
- [ ] Replace \`elo_cfg.get(\"HOME_PPP_BOOST\", 0.024)\` with \`elo_cfg[\"HOME_PPP_BOOST\"]\` (KeyError on missing = fail loud)
- [ ] In \`lineup_elo.py\`, change constructor default \`home_boost: float = 0.024\` to \`home_boost: float = 0.002\`; add a comment citing \`config.py:64\`
- [ ] Bench test \`test_bench_p0_home_boost.py\`: \`LineupEloTracker()\` with no args has \`home_boost == 0.002\`; tracker with a \`cfg\` lacking the key raises \`KeyError\`

**Acceptance:** bench green; \`grep -rn \"0.024\" code/pipeline/\` returns zero hits.

**Agent prompt size:** ~200 tokens (two file paths, two line numbers, the fix)."

make_issue "$M11" "11.2 P0.8: Fix _weight_stint garbage-time double-apply and missing period guard" "## Subtasks
- [ ] File: \`code/pipeline/ratings.py:506-515\`
- [ ] Restructure to a single \`elif\` chain: \`if stint_ctx and stint_ctx.get(\"garbage\"): weight *= gt_w\` / \`elif period >= 4 and abs(margin) >= 15: weight *= gt_w\`
- [ ] Remove the unguarded \`if abs(margin) >= 25: weight *= gt_w\` entirely, or gate it with \`elif period >= 4 and abs(margin) >= 25: weight *= gt_w\`
- [ ] Bench test \`test_bench_p0_garbage_weight.py\`: four cases — (Q1, margin=30) no discount; (Q4, margin=20, garbage=True) exactly one \`gt_w\` application; (Q4, margin=30, garbage=True) exactly one application (not two); (Q4, margin=10) no discount

**Acceptance:** bench green; the four quarter×margin combinations each produce the expected weight multiplier.

**Agent prompt size:** ~150 tokens."

make_issue "$M11" "11.3 P0.10: Rename fair_spread_vigfree to honest label" "## Subtasks
- [ ] File: \`code/pipeline/market.py:562-565\`
- [ ] Rename the dict key to \`market_spread_raw\`; update the comment
- [ ] Check \`model.py\` \`MARKET_MICRO_COLS\` for the old name and update it there too
- [ ] Bench test \`test_bench_p0_vigfree_honesty.py\`: call \`market_microstructure_features(spread_move=0, public_home_pct=0.5, market_spread=-3.5)\`, assert \`\"market_spread_raw\" in result\` and \`\"fair_spread_vigfree\" not in result\`

**Acceptance:** bench green; \`grep -rn \"fair_spread_vigfree\" code/\` returns zero hits.

**Agent prompt size:** ~150 tokens."

make_issue "$M11" "11.4 P0.4: Wire update_residuals into live loop or delete dead API" "## Subtasks
- [ ] Files: \`code/pipeline/elo_calibration.py:238-244\`, \`code/pipeline/backtest.py\` (find where graded games are processed)
- [ ] In \`backtest.py\`'s walk-forward loop, after each graded game, call \`elo_calibrator.update_residuals(residuals_list)\`
- [ ] If no natural call site exists, delete \`predict_interval\`/\`update_residuals\` from \`WalkForwardEloCalibrator\` and document that \`SpreadCalibrator.predict_interval\` (\`market.py:1271-1278\`) is the working interval source
- [ ] Bench test \`test_bench_p0_interval_updates.py\`: create calibrator, call \`predict_interval\` (expect ±12), call \`update_residuals([5]*30)\`, call \`predict_interval\` again (expect width ≠ 24)

**Acceptance:** bench green; \`grep -rn \"update_residuals\" code/pipeline/\` returns at least one call site outside the definition.

**Agent prompt size:** ~250 tokens (needs to find the backtest loop)."

make_issue "$M11" "11.5 P0.6: HAPM shrinkage uses real possession counts" "## Subtasks
- [ ] File: \`code/pipeline/hapm.py:71,76\`
- [ ] Replace \`coef * (SHRINK / (SHRINK + 50))\` with \`coef * (n / (n + SHRINK))\` where \`n\` is the actual possession count for that dyad/trio key
- [ ] Track per-key possession counts in the tracker (add \`self._dyad_poss\` / \`self._trio_poss\` dicts, incremented during fit)
- [ ] Bench test \`test_bench_p0_hapm_shrink.py\`: fit HAPM on synthetic stints where duo A has 5000 possessions and duo B has 10; assert duo A's shrunk coefficient is closer to its raw value than duo B's

**Acceptance:** bench green; shrinkage factor differs by possession count.

**Agent prompt size:** ~300 tokens."

echo "Done."
