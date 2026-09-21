# Synthetic formula test bench (`code/tests/synth/`)

Roadmap: **Epic 10** — Synthetic formula test bench & adversarial stress harness.

## Purpose

Feed deliberately **unrealistic**, extreme, and adversarial fake data through every
pipeline stage so mathematical invariants (symmetry, conservation, monotonicity,
boundedness) become unmissable. Real NBA data clusters in a narrow band where
formula bugs hide (home-only training, inverted Kelly payout at juice ≠ −110,
frozen ±12 intervals, etc.).

## What this is not

- **Not** `pipeline/synthetic_research.py` — that module gates TVAE/SDV *augmentation*
  fidelity for optional research training. Different purpose; do not import its
  generators into production features.
- **Not** a place to tune `config.py` constants. Bench-derived constants are a new
  soft-leak class (anti-roadmap). The bench asserts *invariants*, never accuracy
  targets.

## Layout

| Path | Role |
|------|------|
| `factories.py` | `make_stint`, `make_game`, `make_odds`, `make_player_season` |
| `presets.py` | Extreme scenarios (`BLOWOUT_60`, `JUICE_EXTREMES`, …) |
| `canaries.py` | Planted leak signals (Task 10.4 — stub until implemented) |
| `golden.py` | Seed-locked golden season + stage-output snapshot (Task 10.5.1) |
| `golden_snapshots/golden_season.json` | Committed golden file (drift gate: `test_bench_golden.py`; regenerate with `pytest tests/test_bench_golden.py --regenerate-golden` + a `FEATURE_SCHEMA_VERSION` bump) |
| `../test_bench_p0_*.py` | P0 acceptance battery (Task 10.6) |

## Markers

All bench tests use `@pytest.mark.bench`. Run:

```bash
cd code
pytest -m bench -q
```

CI runs the bench suite in a separate job with a runtime budget (Task 10.7).

## Adding a stage

1. Add `tests/test_bench_<module>.py` with invariant assertions.
2. Prefer factory/preset inputs over hand-rolled dicts.
3. Cite the P0 / `LEAK_REGISTRY.md` / GitHub issue ID in the docstring.
4. Never assert accuracy targets against synthetic outcomes.
