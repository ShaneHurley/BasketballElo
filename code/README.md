# Production code package

Canonical modular NBA spread pipeline for this repository.

| Path | Role |
|------|------|
| `pipeline/` | Library modules (ingest → features → backtest → predict) |
| `tests/` | Regression suite (integrity, leaks, evaluation) |
| `scripts/` | Utilities including `build_colab_notebook.py` |
| `run_backtest.py` / `run_daily.py` | Entrypoints |
| `LEAK_REGISTRY.md` | Confirmed leaks and fix status |
| `requirements.txt` | Python deps |

The Colab/local notebook at the repo root (`nba_unified_pipeline_colab.ipynb`) adds this directory to `sys.path` and imports `pipeline` from here.

Rebuild the notebook after code changes:

```bash
python code/scripts/build_colab_notebook.py
```

Historical notebooks live under `../old code/`. Example odds CSVs live under `../example data/`.
