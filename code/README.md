# Production code package

Canonical modular NBA spread pipeline for this repository (`BasketballElo/code`).

| Path | Role |
|------|------|
| `pipeline/` | Library modules (ingest → features → backtest → predict) |
| `dashboard/` | Local T-60 analysis UI (FastAPI + Jinja + jobs/SSE) |
| `tests/` | Regression suite (integrity, leaks, evaluation) |
| `scripts/` | Utilities including Colab notebook builders + `run_dashboard.py` |
| `run_full_suite.py` | **Ordered full-run suite** (stages 0–8, checkpoints, anti-overfit pause) |
| `run_backtest.py` / `run_daily.py` / `run_training_experiment.py` | Entrypoints |
| `LEAK_REGISTRY.md` | Confirmed leaks and fix status |

## Local analysis dashboard

Browse runs, player ratings, and ATS/ML/totals calibration reports with background jobs (progress + ETA). See [`dashboard/README.md`](dashboard/README.md).

```bash
pip install -r dashboard/requirements.txt
python scripts/run_dashboard.py   # http://127.0.0.1:8765
```

## Full suite (recommended local walkthrough)

```bash
cd BasketballElo/code
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1

# Clean calib run (years with real schedule/odds coverage). Do not trust calib3h ROI.
python -u run_full_suite.py \
  --years 2021,2022,2023,2024 \
  --window 4 --elo-trials 12 --hier-trials 12 --meta-trials 18 \
  --stints rebuild --yes --label calib_clean

# Mini run (newest 2 seasons), reuse stints if data unchanged, pause at review
python run_full_suite.py --pause --stints auto

# After editing files under basketballData/data — force rebuild stints
python run_full_suite.py --stints rebuild --yes

# Resume from review stage
python run_full_suite.py --run-dir ../output/<run_id>_suite --from-stage 6 --pause
```

Stage order: toggles → stints → integrity → engines/calib/walk-forward → **review stop** → policy gates → persist.

- Seasons with >5% interpolated PBP↔schedule dates are **dropped** unless `--allow-interpolated-dates`.
- Odds join fail-closed if match rate < 50% (`--skip-odds-gate` only for debug).
- Scoreboard priority: **spread MAE → Brier/ECE/Murphy → CLV → ATS (secondary)**.
- Do **not** train from quarantined `newest data/` baselines.
- Promotion bar: finite CLV when decision≠close exists; do not promote on ATS/ROI alone.

## Anti-roadmap (calib3h-informed)

| Temptation | Reject — do instead |
|---|---|
| Trust +ROI / ATS with NaN CLV | Require finite CLV + multi-season bet counts |
| Keep 2019–21 with fake dates | Drop until schedule source covers them |
| Loosen edge to 3.5 for volume | Diagnose \|edge\| / CLV first; `policy_tuning` only (`scripts/calibrate_edge_policy.py`) |
| High Optuna `elo_blend_alpha` (market hug) | Cap via `ELO_BLEND_ALPHA_MAX` (diagnose: `scripts/diagnose_edge_compression.py`) |
| Delete `TOTAL_HEAD_CV_SANITY_CAP` | Fix residual/data; raise only with logged evidence |
| CPCV / DSR as promote gates | Season path variance after CLV works |
| Ignore 0% odds match | Fail closed |
| Elo Optuna ATS-miss while widening bounds | MAE-first objective |
| Window Optuna / CPCV-as-train | Keep chronological walk-forward as live path |

See also `LEAK_REGISTRY.md` (`close_line_conflation`) and suite stage-6 CLV gate.

## Edge policy calibration (more bets?)

Post-hoc only — grids min `|edge|` on an existing `backtest_results.csv` (volume / ATS / CLV). Does **not** write `MIN_EDGE_BUCKET`.

```bash
python scripts/calibrate_edge_policy.py \
  ../output/<run_id>/betting/backtest_results.csv \
  --compare-avoid
# → writes betting/edge_policy_calib/{edge_policy_grid.csv,edge_policy_recommendation.json}
```

If lean `|edge|` is compressed (median ~2–3) vs a prior smoke (~6), compare runs before changing gates:

```bash
python scripts/diagnose_edge_compression.py \
  --smoke-dir ../output/<smoke_run> \
  --full-dir  ../output/<full_run>
```

Live Floors stay at **5.5**. Elo market-hug is capped by `ELO_BLEND_ALPHA_MAX` (default 0.20).

## Colab notebook

```bash
python scripts/build_colab_notebook_full.py
python scripts/build_colab_notebook.py
```

Historical notebooks live under `../old code/`.
