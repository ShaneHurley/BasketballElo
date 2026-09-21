# Phase 2 Execution Plan — Synthetic Formula Test Bench (Epic 10)

*Status: in execution. Phase 0/1 complete (9/11 P0s fixed, P0.5 deferred to Epic 9.1, P0.11
open diagnostic). Phase 2 "done when": full bench suite green, CI bench job passes,
`pytest -m bench` exits 0.*

## Gap audit (what remains of Epic 10)

Verified by direct repo read on 2026-09-19:

| Task | State |
| ---- | ----- |
| 10.1.1 `make_game_sequence` / `make_season` factories | **missing** |
| 10.1.2 presets | done (8 presets) |
| 10.1.3 schema-parity test | **missing** |
| 10.1.4 README | done |
| 10.2.1 `test_bench_ratings.py` | **missing** |
| 10.2.2 `test_bench_lineup_chemistry.py` | file exists — verify it covers all 3 invariants (mirrored update, on/off off-leg, shrinkage limits) |
| 10.2.3 `test_bench_market.py` | **missing** |
| 10.2.4 `test_bench_calibration.py` | **missing** |
| 10.2.5 `test_bench_staking_grading.py` | **missing** |
| 10.3.2 `test_bench_properties.py` (Hypothesis) | **missing** |
| 10.4.1 `SliceReuseAttack` canary | **missing** |
| 10.4.2 `test_bench_canaries.py` | **missing** |
| 10.4.3 `test_bench_chronology.py` | **missing** |
| 10.4.4 `test_bench_permutation.py` | **missing** |
| 10.5.1 golden season generator | **missing** (`golden.py` has write/load only) |
| 10.5.2 `test_bench_differential.py` | **missing** |
| 10.5.3 golden-diff CI gate | **missing** |
| 10.6.1 P0 battery | done except P0.11 (→ 10.6.2) |
| 10.6.2 `test_bench_p0_clv.py` | **missing** |
| 10.7.2 bench summary report + dashboard service | **missing** |
| 10.7.3 coverage gate in CI | **missing** |
| CI bench job | exists but only runs 3 bench files — must be expanded to all `test_bench_*` |

## Improvements over the roadmap text

1. **Merge 10.1.1 + 10.1.3 into one agent** — both touch `synth/factories.py` consumers;
   one agent avoids schema drift between the builder and its parity test.
2. **Expand the CI bench job to a glob** (`tests/test_bench_*.py`) instead of enumerating
   files — the enumerated list is exactly the kind of thing that silently goes stale.
3. **10.2.2 gets a verify-and-complete pass**, not a blind rewrite — the file exists post-P0.1
   fix, so the `xfail(strict=True)` markers must have been resolved; an agent confirms.
4. **Golden master (10.5.x) runs after all stage-invariant files land** so the snapshot
   captures the final formula state; regenerating goldens mid-flight would bake in churn.
5. **10.7.2 bench report writes to `output/bench/latest.json` via a conftest hook** scoped to
   bench-marked runs only, so ordinary test runs don't clobber the report.
6. **Every agent prompt carries the roadmap's verbatim Context text plus the adversarial
   protocol** (attempt a counterexample that passes the test but violates the invariant;
   tighten if found).

## Wave plan (dependency-driven, per roadmap graph)

- **Wave 1 (parallel, 6 agents):** A=10.1.1+10.1.3 factories/parity; B=10.2.1 ratings;
  C=10.2.3 market; D=10.2.4 calibration; E=10.2.5 staking/grading; F=10.6.2 CLV fixture.
- **Wave 2 (parallel, 4 agents):** G=10.2.2 verify/complete; H=10.3.2 Hypothesis properties;
  I=10.4.1+10.4.2 canaries + gate tests; J=10.4.3 chronology + 10.4.4 permutation nulls.
- **Wave 3 (parallel, 2 agents):** K=10.5.1 golden generator; L=10.5.2 differential checks.
- **Wave 4 (sequential, 1 agent):** M=10.5.3 CI golden gate + expand CI bench glob + 10.7.3
  coverage gate + 10.7.2 bench report/dashboard service + roadmap checkbox updates.
  Also investigate: `tests/test_promotion_gates.py` has a pre-existing import error
  (`ats_promote`) at HEAD, unrelated to bench work — triage and either fix or exclude
  with a documented reason.

## Agent contract (every worker)

1. Work in `/Users/shurley/Documents/basketball/BasketballElo`.
2. Create only the files named in the task; read at most the 3 source modules cited.
3. All tests marked `@pytest.mark.bench`; deterministic seeds via `np.random.default_rng`.
4. Run `cd code && python3 -m pytest tests/<new_file> -q --tb=short` until green.
5. Adversarial self-review: construct one counterexample input that passes the test but
   violates the invariant; if found, tighten the test. Report the attempt.
6. Do NOT tune constants against bench output (synthetic overfitting — anti-roadmap).
7. Return: files created/edited, test count, pytest result, adversarial attempt outcome.
