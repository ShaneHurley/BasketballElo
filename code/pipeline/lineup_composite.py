"""Composite lineup rating from player Elo, chemistry, and lineup5 Elo."""
from __future__ import annotations

# lineup5 weight is fixed at 0.50 minimum per project spec
DEFAULT_WEIGHTS = {
    "lineup5": 0.50,
    "player": 0.25,
    "duo": 0.15,
    "trio": 0.10,
}


def composite_lineup_rating(
    *,
    player_off_delta: float,
    player_def_delta: float,
    lineup5_net: float,
    chem_duo_net: float,
    chem_trio_net: float = 0.0,
    weights: dict | None = None,
) -> float:
    """Single lineup-strength scalar in ~PPP-margin units."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
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
