# BasketballElo

NBA spread prediction pipeline.

- **`code/`** — current production package (`pipeline`, `tests`, scripts, entrypoints)
- **`nba_unified_pipeline_colab.ipynb`** — import-based Colab/local notebook (uses `code/`)
- **`old code/`** — historical notebooks
- **`example data/`** — sample odds / schedule CSVs
- **`analysis_plots/`** — diagnostic plots from prior runs

```bash
# Local quick test
cd code && python -m pytest tests/test_evaluation_stage4.py -q

# Rebuild notebook
python code/scripts/build_colab_notebook.py
```
