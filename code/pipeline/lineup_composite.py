"""Composite lineup rating from player Elo, chemistry, and lineup5 Elo."""
from __future__ import annotations

# lineup5 weight is fixed at 0.50 minimum per project spec (legacy).
# When evidence pooling is enabled, reliability replaces the hard floor.
DEFAULT_WEIGHTS = {
    "lineup5": 0.50,
    "player": 0.25,
    "duo": 0.15,
    "trio": 0.10,
}

LINEUP5_SHRINK_K = 100.0


def lineup5_reliability(sample_poss: float, shrink_k: float = LINEUP5_SHRINK_K) -> float:
    """Evidence-based weight in [0, 1] for five-man ratings."""
    n = max(float(sample_poss or 0.0), 0.0)
    return float(n / (n + float(shrink_k)))


def composite_lineup_rating(
    *,
    player_off_delta: float,
    player_def_delta: float,
    lineup5_net: float,
    chem_duo_net: float,
    chem_trio_net: float = 0.0,
    weights: dict | None = None,
    lineup5_sample_poss: float | None = None,
    evidence_pooling: bool | None = None,
) -> float:
    """Single lineup-strength scalar in ~PPP-margin units."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    if evidence_pooling is None:
        try:
            from pipeline.config import LINEUP_COMPOSITE_EVIDENCE_POOLING
            evidence_pooling = bool(LINEUP_COMPOSITE_EVIDENCE_POOLING)
        except Exception:
            evidence_pooling = False

    if evidence_pooling and lineup5_sample_poss is not None:
        # Reliability pools toward player/chemistry parents when sample is thin.
        rel = lineup5_reliability(lineup5_sample_poss)
        w["lineup5"] = float(rel * 0.65)  # cap lineup5 share even with huge samples
        rem = 1.0 - w["lineup5"]
        other = w.get("player", 0.25) + w.get("duo", 0.15) + w.get("trio", 0.10)
        if other > 0:
            scale = rem / other
            w["player"] = w.get("player", 0.25) * scale
            w["duo"] = w.get("duo", 0.15) * scale
            w["trio"] = w.get("trio", 0.10) * scale
    else:
        w["lineup5"] = max(0.50, float(w.get("lineup5", 0.50)))
        rem = 1.0 - w["lineup5"]
        other = w.get("player", 0.25) + w.get("duo", 0.15) + w.get("trio", 0.10)
        if other > 0:
            scale = rem / other
            w["player"] = w.get("player", 0.25) * scale
            w["duo"] = w.get("duo", 0.15) * scale
            w["trio"] = w.get("trio", 0.10) * scale

    player_component = (float(player_off_delta) - float(player_def_delta)) / 1000.0
    lineup5_component = float(lineup5_net) / 100.0
    duo_component = float(chem_duo_net) / 100.0
    trio_component = float(chem_trio_net) / 100.0

    return (
        w["player"] * player_component
        + w["duo"] * duo_component
        + w["trio"] * trio_component
        + w["lineup5"] * lineup5_component
    )
