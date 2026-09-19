# T-60 Analysis Dashboard

Local FastAPI site for browsing run outputs, launching full suite / backtest jobs (notebook-equivalent), and re-running ATS / ML / Totals calibration reports without freezing the UI.

Night-mode UI (black / grey). Jobs stream progress + logs via SSE.

## Install

From `BasketballElo/code`:

```bash
python3 -m venv .venv_dashboard
source .venv_dashboard/bin/activate   # Windows: .venv_dashboard\Scripts\activate
pip install -r dashboard/requirements.txt
# Also need the project’s usual scientific stack (pandas/numpy/sklearn) for heavy jobs.
```

## Run

**Double-click (app-style):** from the `BasketballElo/` folder:

| OS | Launcher |
|----|----------|
| macOS | `Open T-60 Dashboard.command` |
| Windows | `Open T-60 Dashboard.bat` |
| Linux | `./open_dashboard.sh` |

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Default bind is **localhost only**.

```bash
python scripts/run_dashboard.py --host 127.0.0.1 --port 8765
```

## Tabs

| Tab | Purpose |
|-----|---------|
| All Data | Run picker, summary, results table, artifacts, charts, `ats_status` JSON viewer |
| **Run Lab** | Full suite / backtest forms (years, trials, stints, stages, flags) + presets |
| Player | Rating snapshot table, CSV export, smoke rebuild |
| ATS | Calibration forms, reliability / edge-policy jobs, cover + ROI + CLV charts |
| ML | Win-prob / calib form + reliability chart |
| Totals | O/U form + residual charts |
| Jobs | History, dismiss, **View log** |

## Run Lab (notebook-equivalent)

Presets (dropdown):

| Preset | Years | Elo / Hier / Meta | Notes |
|--------|-------|-------------------|-------|
| Smoke | (mini / empty) | 6 / 6 / 8 | `--fast-tuning`, skip integrity |
| Standard | 2021–2024 | 12 / 12 / 18 | fast-tuning |
| Full | 2020–2025 | 24 / 24 / 36 | stints=rebuild, no fast-tuning |

Flags passed through to `run_full_suite.py` (see `dashboard/jobs/schemas.py`):

`--years`, `--start-season`, `--end-season`, `--elo-trials`, `--hier-trials`, `--meta-trials`,
`--window`, `--stints`, `--fast-tuning`, `--label`, `--from-stage`, `--to-stage`, `--run-dir` (resume),
`--skip-integrity-tests`, `--allow-interpolated-dates`, `--venn-abers`, `--rolling-z`,
`--skip-odds-gate`, `--no-cache`, `--force-stints-cache`.

Backtest form maps to `run_backtest.py`: `--quick`, `--elo-trials`, `--hier-trials`, `--meta-trials`, `--window`, `--fast-tuning`.

Heavy jobs use `pipeline_python()` (interpreter that can `import sklearn`) so the thin dashboard venv does not break suite imports.

## Job modules

| id | heaviness | Notes |
|----|-----------|--------|
| `suite_smoke` | heavy | Mini suite |
| `suite_custom` | heavy | Full flag pass-through (Run Lab) |
| `backtest_quick` | heavy | Walk-forward backtest |
| `player_rating_smoke` | heavy | Quick backtest for ratings |
| `edge_policy_calib` | light | Edge floor grid (+ real calibrator when possible) |
| `ats_reliability` | light | Cover reliability / season table |
| `ml_calibration_report` | light | Brier / logloss / ECE |
| `totals_calibration_report` | light | MAE / RMSE / bias |
| `promotion_gates` | light | ats / ml / total promote |
| `export_slice` | light | Filtered CSV export |

Progress: `dashboard_data/jobs/{id}/progress.json` + SSE `/api/jobs/{id}/events`.  
Logs: every progress line appends to `job.log`; UI **View log** → `GET /api/jobs/{id}/log?tail=400`.

## Chart IDs

- All: `season_ats`, `rolling_mae`, `edge_hist`, `clv_scatter`, `bankroll`
- ATS: `cover_calibration`, `roi_by_conf` (ATS_WIN −110 fallback), `clv_by_season`
- ML: `ml_reliability`
- Totals: `pred_vs_actual_total`, `residual_vs_market`

Plotly is vendored at `dashboard/static/js/vendor/plotly.min.js` (no CDN required).

## Data locations

- Reads: `BasketballElo/output/<run>/`, `BasketballElo/code/state/`
- Writes: `BasketballElo/dashboard_data/{jobs,exports,ui_state.json,job_timings.json}`

## Safety

- Path sandbox: `output/`, `state/`, `dashboard_data/`
- tip-proxy / blocked `ats_status` → research-only banner
- Never auto-promotes frozen baseline

## Smoke checklist

1. Open all tabs including **Run Lab**
2. ATS charts render (≥3) with dark theme
3. Launch light job; open **View log**
4. Launch Smoke preset; confirm progress + log lines
5. Export CSV; dismiss finished jobs
