# T-60 analysis dashboard

Local FastAPI app under `code/dashboard/` for browsing walk-forward runs, launching suite/backtest jobs, and inspecting ATS / ML / totals calibration — without freezing the UI.

## Run

From `BasketballElo/code`:

```bash
pip install -r dashboard/requirements.txt
# Also need the scientific stack (pandas/numpy/sklearn/catboost)
python scripts/run_dashboard.py   # http://127.0.0.1:8765
```

Or double-click from the repo root:

| OS | Launcher |
|----|----------|
| macOS | `Open T-60 Dashboard.command` |
| Windows | `Open T-60 Dashboard.bat` |
| Linux | `./open_dashboard.sh` |

Default bind is **localhost only**.

## Tabs

| Tab | Purpose |
|-----|---------|
| All Data | Run picker, summary, results, artifacts, charts |
| Run Lab | Full suite / backtest forms + presets (Smoke / Standard / Full) |
| Player | Rating snapshot table + CSV export |
| ATS | Cover reliability, edge policy, CLV charts |
| ML | Win-prob / calibration report |
| Totals | O/U residual charts |
| Jobs | History, dismiss, live log via SSE |

## Why it's in the portfolio

- Real async job runner with progress + ETA (`dashboard/jobs/`)  
- Sandboxed paths under `output/`, `state/`, `dashboard_data/`  
- Vendored Plotly (no CDN required)  
- Research-only banners when tip-proxy / blocked `ats_status` would overclaim  

!!! note "Screenshots"
    Capture a few dark-theme screenshots locally after `mkdocs serve` if you want them embedded here under `docs/assets/`. The dashboard README in `code/dashboard/README.md` has a smoke checklist.
