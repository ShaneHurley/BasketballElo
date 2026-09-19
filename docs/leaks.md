# Leak registry

The formal catalog lives at [`code/LEAK_REGISTRY.md`](https://github.com/ShaneHurley/BasketballElo/blob/main/code/LEAK_REGISTRY.md). This page is the portfolio-facing summary.

!!! success "Why this document matters"
    Most personal sports models hide leakage. This project **names** leaks, attaches regression tests, and refuses to promote results that violate provenance rules.

## How to read a row

| Field | Meaning |
|-------|---------|
| **ID** | Stable slug used in tests and commits |
| **Description** | Root cause in plain language |
| **Regression test** | Pytest that fails if the leak returns |
| **Status** | `fixed` / `confirmed` / guarded / out of scope |

## Headline leaks (fixed)

| ID | One-line |
|----|----------|
| `score_corruption` | Finals built from garbage-filtered stints instead of raw scores |
| `date_corruption` | Bad dates silently fabricated as `prev_date + 1` |
| `close_line_conflation` | Decision line == close → CLV identically zero |
| `stack_cv_future_leakage` | Stacking base learners saw future rows |
| `elo_stack_in_sample_leak` | Elo stack feature fit in-sample |
| `hapm_before_split_leak` | HAPM fit on full span before chronological split |
| `static_to_rolling_calibration_mismatch` | Static calibrator applied to its own fit slice |
| `calibration_slice_reuse` | Multiple calibrators shared one tail slice |
| `roi_tuning_objective_and_overfit_sign_bug` | Optuna rewarded the wrong overfit sign + ROI terms |
| `t60_actual_lineup_leak` | T-60 features used the game's own actual lineup |
| `epm_prior_unversioned_leak` | Unversioned EPM joined future snapshots to past games |

## Later guards (T-60 / promotion)

Additional rows in the registry cover tip-proxy odds provenance, decision-residual training targets, pace scale mismatches, score-pair market feature bans, and promotion gates that **block ROI claims** without `promotion_eligible` odds.

## Engineering pattern

1. Suspect a result  
2. Write a failing test that demonstrates lookahead / train-serve mismatch  
3. Fix the root cause  
4. Record the leak in the registry  
5. Invalidate caches / artifacts  

That loop is the project's real product.
