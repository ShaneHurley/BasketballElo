# Five-year evolution

A condensed history of the rating formulas and engineering practice. Full narrative: see the corrected `PROJECT_HISTORY_REPORT.md` in the parent workspace; this page is the portfolio-facing version grounded in the git repo at `BasketballElo/code/`.

## Eras

### 2021 — Possession-level player Elo

Standalone notebooks (`NBA ELO marc 2(3).ipynb` and kin). Separate offensive/defensive Elo per player from PBP, logistic divisor 400, flat **K=64**, start rating **1000**. No home-court term, no season regression.

### 2022 — K-tuning + provisional boost

Near-duplicate notebooks varying K ∈ {16, 32, 36, 64}. First FIDE-style provisional boost for players with &lt;40 possessions.

### 2023–2024 — Stint updates

Move from event-level to lineup-stint updates; league `pointsPerPos` normalizer; K scaled by possessions. One experimental branch abandons logistic Elo for a moving-average PPP update — later abandoned (valuable negative result).

### May 2026 — PPP rewrite

Git commits `5bb2b29` → `ec8f006`. Start rating **1500**, unit switch to points-per-possession, explicit `HOME_PPP_BOOST`, multi-factor dynamic K, then shot-zone xPoints and ATS confidence tiers. A same-season backtest printed ~60% ATS — **author caveat: may be overfit**.

### June 2026 — Leak hunting

Commit `a8e2cb9` ("may have leaks") introduces Glicko-2-style μ/RD/σ, hierarchical n-man engine, CatBoost stacking. Commit `ade9434` claims a leak-free unified pipeline and a more modest **2–4% edge**; `HOME_PPP_BOOST` drops **0.024 → 0.002**, `OFFSEASON_REVERSION` **0.25 → 0.15**.

### July 2026 — Modular package

Commit `7c23af2` decomposes the notebook into `code/pipeline/` with tests and `LEAK_REGISTRY.md`. Follow-ups fix import/empty-frame bugs and relax over-strict ATS gates (`MAX_QUANTILE_WIDTH` 22→28, etc.). Local work since then adds the **T-60 dashboard**, **`run_full_suite.py`**, and expands the package to **98 modules**.

## What to take from the arc

```mermaid
flowchart LR
    notebook[Notebook Elo] --> overfit[Exciting overfit ATS]
    overfit --> distrust[Author distrust]
    distrust --> leaks[Leak hunt + walk-forward]
    leaks --> modular[Modular package + tests]
    modular --> gates[Empirical gate tuning]
```

The hireable story is the middle of that chain: **distrust → catalog → fix → test**.
