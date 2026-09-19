<p class="carbon-eyebrow">NBA analytics &nbsp;/&nbsp; walk-forward ML</p>

# BasketballElo

Leak-aware NBA spread prediction — from a five-year Elo side project to a walk-forward, modular Python pipeline with a formal leak registry.

!!! tip "What hiring managers should notice"
    This project is not about claiming a magic ATS percentage. It is about **distrusting a good-looking backtest**, documenting every chronological leak found, and rebuilding until the evaluation would survive a skeptical review.

<div class="carbon-grid" markdown>
<div class="carbon-tile" markdown>

### 98

Modules in `code/pipeline/`

</div>
<div class="carbon-tile" markdown>

### 100+

Pytest regression modules

</div>
<div class="carbon-tile" markdown>

### 11

Confirmed leaks, documented & tested

</div>
<div class="carbon-tile" markdown>

### 0–8

Staged suite phases with an anti-overfit review stop

</div>
</div>

[:octicons-mark-github-16: View source](https://github.com/ShaneHurley/BasketballElo){ .md-button }
[Read the leak registry](leaks.md){ .md-button .md-button--secondary }

## Architecture

```mermaid
flowchart TD
    pbp[Raw play-by-play] --> stints[Build stints]
    odds[Market odds] --> calib
    injuries[Injury reports] --> meta
    epm[EPM priors] --> meta
    stints --> features[Game features T-60]
    features --> elo[Player Glicko-2]
    features --> hier[Hierarchical n-man]
    features --> hapm[HAPM ridge]
    features --> chem[Chemistry]
    features --> lineup[Lineup Elo]
    features --> team[Team Elo]
    elo --> meta[MetaScore stack]
    hier --> meta
    hapm --> meta
    chem --> meta
    lineup --> meta
    team --> meta
    meta --> calib[3-stage calibration]
    calib --> prob[Cover probability]
    prob --> gates[Bet-selection gates]
    odds --> gates
    gates --> stake[Fractional Kelly]
    stake --> output[Bet recommendation]
```

## Start reading

1. [History](history.md) — how the formulas evolved (2021 → 2026)
2. [Leak registry](leaks.md) — every confirmed leak and its regression test
3. [Pipeline](pipeline.md) — stage-by-stage execution
4. [Math](math/elo.md) — Elo → PPP → Glicko-2 → calibration → Kelly
5. [Roadmap](roadmap.md) — what to build next

## Credible claims vs. overfit claims

| Claim | Status |
|-------|--------|
| Same-season ~60% ATS (May 2026 notebook) | Self-flagged as possibly overfit in the commit message |
| "Consistent 2–4% edge" after leak-free rewrite | Author's walk-forward claim — still requires CLV/provenance gates for promotion |
| Formal leak catalog with tests | Verifiable in-repo today |

## Quick install

```bash
pip install -e ".[dev]"
cd code && python -m pytest tests/test_leak_registry.py -q
```
