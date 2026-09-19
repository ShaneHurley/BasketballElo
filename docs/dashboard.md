# T-60 analysis dashboard

Local FastAPI app under `code/dashboard/` for browsing walk-forward runs, launching suite/backtest jobs, and inspecting ATS / ML / totals calibration — without freezing the UI.

The UI follows the [Carbon Design System](https://carbondesignsystem.com/): IBM Plex Sans, a gray-100 light theme (`#f4f4f4` background, white cards, `#161616` text), blue-60 (`#0f62fe`) primary actions, an 8px spacing grid, and sharp 0-radius rectangles.

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

### Browser permission prompt

The launcher scripts **ask before opening a browser window**:

```text
Open the T-60 Dashboard in your browser? [Y/n]
```

Press Enter (or `y`) to open the dashboard in your default browser once the server is up; answer `n` to skip. The server keeps running either way — you can always open <http://127.0.0.1:8765> manually.

Escape hatches for headless / scripted use:

```bash
./open_dashboard.sh --no-browser      # flag
T60_NO_BROWSER=1 ./open_dashboard.sh  # env var
```

The same flag and env var work for the `.command` and `.bat` launchers. When stdin is not an interactive terminal, the launchers skip the browser automatically instead of popping windows.

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
| Documentation | In-app overview: pipeline stages, metric glossary, links to this docs site, `LEAK_REGISTRY.md`, and the roadmap |

## First-run experience

When no runs exist under `output/` yet, the All Data tab shows a welcome empty state that introduces the project, offers a one-click **suite smoke** to generate a first run, and points to the Documentation tab.

## Why it's in the portfolio

- Real async job runner with progress + ETA (`dashboard/jobs/`)
- Sandboxed paths under `output/`, `state/`, `dashboard_data/`
- Vendored Plotly (no CDN required); IBM Plex fonts load from CDN with system-font fallback
- Research-only banners when tip-proxy / blocked `ats_status` would overclaim

!!! note "Screenshots"
    Capture a few light-theme screenshots locally after `mkdocs serve` if you want them embedded here under `docs/assets/`. The dashboard README in `code/dashboard/README.md` has a smoke checklist.
