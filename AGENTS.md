# AGENTS.md

## Project overview

This repository is a collection of Jupyter notebooks for **NBA / basketball game prediction**. The
core idea is a lineup- and team-level **ELO rating system** computed from play-by-play data, layered
with ML models (XGBoost, CatBoost, Ridge/Logistic regression, Optuna tuning, isotonic calibration)
to predict game outcomes and point spreads.

Notebooks:

- `NBA_ELO.ipynb` — the foundational lineup-ELO model (the core `calc` ELO update lives here).
- `BasketballEloSim.ipynb`, `2026ELOcode.ipynb`, `basketballevaluatorPASTYEARS.ipynb` — ELO simulation/evaluation variants across seasons.
- `nba_unified_prediction_pipeline.ipynb` — end-to-end pipeline (data ingest via `nba_api`, feature build, model training/tuning, calibration).

There is no application server, package, or test suite — the "app" is the set of notebooks run in Jupyter.

## Cursor Cloud specific instructions

- **Runtime**: System Python 3.12 with packages installed via `pip install --break-system-packages`
  (no virtualenv). The full scientific + NBA stack is installed by the startup update script
  (pandas==2.2.3, numpy, scipy, scikit-learn, xgboost, catboost, optuna, tqdm, nba_api, nbainjuries,
  matplotlib, requests) plus Jupyter (`jupyterlab`, `notebook`, `ipykernel`, `nbconvert`).
- **Jupyter scripts live in `~/.local/bin`**, which is NOT on `PATH` by default. Either run via
  `python3 -m jupyter ...` / `python3 -m jupyterlab` (recommended, always works) or prepend
  `export PATH="$HOME/.local/bin:$PATH"`.
- **Run the dev app (JupyterLab)**:
  `python3 -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token="" --ServerApp.password=""`
  then open `http://127.0.0.1:8888/lab`.
- **Headless execution / "build" check**: execute a notebook with
  `python3 -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 <nb>.ipynb`.
  Validate all notebooks parse with `nbconvert --to script --stdout`.
- **No automated tests or linter config** exist in this repo; "verification" means a notebook executes
  cleanly end-to-end.
- **Data is NOT in the repo.** The notebooks were authored on Google Colab and read CSVs from a mounted
  Google Drive, e.g. cells that do `from google.colab import drive` / `drive.mount(...)` and paths like
  `/drive/MyDrive/basketballData/...`. These cells will fail in this environment — skip the Drive-mount
  cells and either point the file paths at locally-available data or use `nba_api` to fetch data.
- **`nba_api` and `nbainjuries` hit external network** (stats.nba.com etc.). Those endpoints may be slow,
  rate-limited, or blocked in the sandbox; cells depending on live fetches can hang or error for reasons
  unrelated to code changes.
- **numpy is 2.x.** If a notebook relies on removed numpy 1.x APIs you may need to adjust calls (the
  current notebooks import cleanly).
- The core ELO update (`calc` in `NBA_ELO.ipynb`) uses a large effective K-factor (`32 * possessions`),
  so an instantaneous ELO is a fast-tracking random walk around true strength; average over many updates
  to recover a stable latent rating.
