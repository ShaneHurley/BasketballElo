# Roadmap

Ranked by recruiter signal and genuine engineering value.

## Near term (in progress / just landed)

- [x] Modular `code/pipeline/` package + leak registry  
- [x] FastAPI T-60 dashboard  
- [x] Staged `run_full_suite.py` with review gates  
- [x] Portfolio README, MIT license, `pyproject.toml`  
- [x] GitHub Actions CI (pytest + ruff) + MkDocs site  

## Next

1. **Tighten CI** — make `ruff check` fail-hard once the backlog is clean; expand the pytest matrix beyond the core leak/smoke set.  
2. **Pre-commit locally** — `pre-commit install` using `.pre-commit-config.yaml`.  
3. **Experiment tracking** — index `output/<run>/README.md` manifests (MLflow or lightweight JSON store).  
4. **Predict API** — expose `predict_game` as a FastAPI route alongside the dashboard.  
5. **Docker** — reproducible suite image; document that raw PBP/odds data is not in git.  
6. **Write-up** — short public post: *“Finding chronological leaks in my own NBA model.”*

## Explicit non-goals (anti-roadmap)

From `code/README.md` / leak registry:

- Do not promote on ATS/ROI when CLV is NaN  
- Do not Optuna-tune the rolling window for scoreboard seasons  
- Do not loosen min-edge only to restore bet volume after edge compression  
- Do not let `elo_blend_alpha` hug the market past `ELO_BLEND_ALPHA_MAX`
