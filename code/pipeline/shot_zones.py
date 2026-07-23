"""Shot zone taxonomy, coordinate normalization, and walk-forward calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
CORNER3_Y_MAX_FT = 14.0
CORNER3_DIST_MIN_FT = 22.0
SHRINK_K = 100.0

CANONICAL_ZONES = (
    "restricted", "paint", "short_mid", "midrange", "corner3", "abovebreak3", "other",
)

LEGACY_ZONE_ALIAS = {
    "ab3": "abovebreak3",
    "rim": "restricted",
    "long_mid": "midrange",
    "three": "abovebreak3",
    "mid2": "midrange",
}

DEFAULT_FG_PCT = {
    "restricted": 0.64,
    "paint": 0.50,
    "short_mid": 0.42,
    "midrange": 0.40,
    "corner3": 0.38,
    "abovebreak3": 0.35,
    "other": 0.40,
}

DEFAULT_POINT_VALUE = {
    "restricted": 2.0,
    "paint": 2.0,
    "short_mid": 2.0,
    "midrange": 2.0,
    "corner3": 3.0,
    "abovebreak3": 3.0,
    "other": 2.0,
}


def legacy_zone_alias(zone: str) -> str:
    if zone is None or (isinstance(zone, float) and pd.isna(zone)):
        return "other"
    z = str(zone).strip().lower()
    if z in CANONICAL_ZONES:
        return z
    if z in LEGACY_ZONE_ALIAS:
        return LEGACY_ZONE_ALIAS[z]
    if "corner" in z and "3" in z:
        return "corner3"
    if "restricted" in z or z == "rim":
        return "restricted"
    if "paint" in z:
        return "paint"
    if "mid" in z and "3" not in z:
        return "midrange" if "long" in z or "mid2" in z else "short_mid"
    if "3" in z or "three" in z:
        return "abovebreak3"
    return "other"


@dataclass
class ZoneCalibration:
    """Walk-forward zone make rates × shot value."""

    fg_pct: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FG_PCT))
    point_value: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_POINT_VALUE))
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def pps(self) -> dict[str, float]:
        out = {}
        for z in set(list(self.fg_pct.keys()) + list(DEFAULT_FG_PCT.keys())):
            zc = legacy_zone_alias(z)
            fg = self.fg_pct.get(zc, DEFAULT_FG_PCT.get(zc, 0.40))
            pv = self.point_value.get(zc, DEFAULT_POINT_VALUE.get(zc, 2.0))
            out[zc] = fg * pv
        return out

    def xpoints(self, zone: str) -> float:
        z = legacy_zone_alias(zone)
        n = self.counts.get(z, 0)
        raw_fg = self.fg_pct.get(z, DEFAULT_FG_PCT.get(z, 0.40))
        default_fg = DEFAULT_FG_PCT.get(z, 0.40)
        fg = (n * raw_fg + SHRINK_K * default_fg) / (n + SHRINK_K) if n > 0 else default_fg
        pv = self.point_value.get(z, DEFAULT_POINT_VALUE.get(z, 2.0))
        return float(fg * pv)

    def fg_pct_for_zone(self, zone: str) -> float:
        z = legacy_zone_alias(zone)
        n = self.counts.get(z, 0)
        raw_fg = self.fg_pct.get(z, DEFAULT_FG_PCT.get(z, 0.40))
        default_fg = DEFAULT_FG_PCT.get(z, 0.40)
        if n <= 0:
            return default_fg
        return float((n * raw_fg + SHRINK_K * default_fg) / (n + SHRINK_K))

    def to_flat_pps(self) -> dict[str, float]:
        """Backward-compatible flat zone_pps dict."""
        return self.pps


def normalize_court_coords(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return x_ft, y_ft from tracking columns when present."""
    x_ft = pd.Series(np.nan, index=df.index, dtype=float)
    y_ft = pd.Series(np.nan, index=df.index, dtype=float)
    cols = {c.lower(): c for c in df.columns}
    ox = cols.get("original_x") or cols.get("xlegacy") or cols.get("x_legacy")
    oy = cols.get("original_y") or cols.get("ylegacy") or cols.get("y_legacy")
    if ox and oy:
        x = pd.to_numeric(df[ox], errors="coerce")
        y = pd.to_numeric(df[oy], errors="coerce")
        # 2026 tracking: 0-100 half-court scale → feet
        if (x.abs().max(skipna=True) or 0) <= 100 and (y.abs().max(skipna=True) or 0) <= 100:
            x_ft = (x / 100.0) * COURT_LENGTH_FT
            y_ft = (y / 100.0) * COURT_WIDTH_FT
        else:
            # V3 xLegacy/yLegacy: tenths-of-foot from basket (do not use for distance)
            x_ft = x / 10.0
            y_ft = y / 10.0
    return x_ft, y_ft


def assign_shot_zone_scalar(
    dist_ft: float,
    x_ft: float = np.nan,
    y_ft: float = np.nan,
    *,
    is_3pt_text: bool = False,
    area: str | None = None,
    area_detail: str | None = None,
) -> str:
    joined = f"{area or ''} {area_detail or ''}".lower()
    if "corner" in joined and ("3" in joined or "three" in joined):
        return "corner3"
    if "restricted" in joined or joined.strip() == "rim":
        return "restricted"
    if "paint" in joined:
        return "paint"
    if "above" in joined and "break" in joined:
        return "abovebreak3"
    if "3" in joined or "three" in joined:
        return "abovebreak3"

    d = float(dist_ft) if pd.notna(dist_ft) and dist_ft >= 0 else np.nan
    if pd.notna(x_ft) and pd.notna(y_ft):
        d_coord = float(np.sqrt(x_ft ** 2 + y_ft ** 2))
        if pd.isna(d) or d < 0:
            d = d_coord
        else:
            d = min(d, d_coord) if d_coord > 0 else d

    is_3 = bool(is_3pt_text)
    if pd.notna(d):
        if d >= CORNER3_DIST_MIN_FT and (is_3 or is_3pt_text):
            if pd.notna(y_ft) and abs(y_ft) <= CORNER3_Y_MAX_FT:
                return "corner3"
            return "abovebreak3"
        if is_3:
            return "abovebreak3"
        if d <= 4:
            return "restricted"
        if d <= 10:
            return "paint"
        if d <= 16:
            return "short_mid"
        if d < CORNER3_DIST_MIN_FT:
            return "midrange"
        return "abovebreak3" if is_3 else "midrange"

    return "abovebreak3" if is_3pt_text else "other"


def assign_shot_zones_df(df: pd.DataFrame, dist: pd.Series, combo: pd.Series) -> pd.Series:
    """Vectorized zone assignment with coordinate corner-3 override."""
    x_ft, y_ft = normalize_court_coords(df)
    is_3pt = combo.str.contains(r"3pt|three|3-pt| corner ", case=False, na=False)
    if "shot_value" in df.columns:
        sv = pd.to_numeric(df["shot_value"], errors="coerce")
        is_3pt = is_3pt | (sv >= 3)

    d = pd.to_numeric(dist, errors="coerce").fillna(-1)
    has_coord = x_ft.notna() & y_ft.notna() & (x_ft.abs() + y_ft.abs() > 0)
    # Prefer official shotDistance when present (especially V3)
    d_official = d.copy()
    d_coord = np.sqrt(x_ft.fillna(0) ** 2 + y_ft.fillna(0) ** 2)
    use_coord = has_coord & ((d_official < 0) | ((d_official > 0) & (d_coord > 0) & (d_coord < d_official * 0.5)))
    d = np.where(d_official >= 0, d_official, np.where(use_coord, d_coord, d_official))
    d = np.where((d <= 0) & is_3pt, 23.0, d)
    d = np.where((d <= 0) & ~is_3pt & (d_official < 0), -1, d)

    is_3pt = is_3pt | (d >= 23.75)

    area = df["area"].fillna("").astype(str).str.lower() if "area" in df.columns else pd.Series("", index=df.index)
    area_detail = (
        df["area_detail"].fillna("").astype(str).str.lower()
        if "area_detail" in df.columns else pd.Series("", index=df.index)
    )

    corner3 = (d >= CORNER3_DIST_MIN_FT) & is_3pt & (y_ft.abs() <= CORNER3_Y_MAX_FT) & has_coord
    abovebreak3 = (d >= CORNER3_DIST_MIN_FT) & is_3pt & ~corner3
    restricted = (d >= 0) & (d <= 4) & ~is_3pt
    paint = (d > 4) & (d <= 10) & ~is_3pt
    short_mid = (d > 10) & (d <= 16) & ~is_3pt
    midrange = (d > 16) & (d < CORNER3_DIST_MIN_FT) & ~is_3pt

    zone = np.select(
        [corner3, abovebreak3, restricted, paint, short_mid, midrange, is_3pt],
        ["corner3", "abovebreak3", "restricted", "paint", "short_mid", "midrange", "abovebreak3"],
        default="other",
    )
    zones = pd.Series(zone, index=df.index)

    joined = (area + " " + area_detail + " " + combo).str.strip()
    corner_text = joined.str.contains("corner", na=False) & joined.str.contains(r"3|three|3pt", case=False, na=False)
    zones = np.where(corner_text, "corner3", zones)
    restricted_text = joined.str.contains("restricted|at rim", na=False)
    zones = np.where(restricted_text, "restricted", zones)
    paint_text = joined.str.contains("paint", na=False)
    zones = np.where(paint_text & ~corner_text, "paint", zones)

    return pd.Series(zones, index=df.index).map(legacy_zone_alias)


def compute_zone_calibration(df: pd.DataFrame) -> ZoneCalibration:
    """Leak-free zone calibration from a season of PBP (for the *next* season)."""
    if df is None or df.empty or "EVENTMSGTYPE" not in df.columns:
        return ZoneCalibration()
    is_shot = df["EVENTMSGTYPE"].isin([1, 2])
    if not is_shot.any():
        return ZoneCalibration()
    if "shot_zone" not in df.columns:
        return ZoneCalibration()

    sub = df.loc[is_shot, ["shot_zone"]].copy()
    sub["zone"] = sub["shot_zone"].map(legacy_zone_alias)
    sub["made"] = df.loc[is_shot, "is_fg_make"].astype(float)
    sub["point_value"] = np.where(sub["zone"].isin(["corner3", "abovebreak3"]), 3.0, 2.0)

    fg = sub.groupby("zone")["made"].mean()
    counts = sub.groupby("zone")["made"].count()
    pv = sub.groupby("zone")["point_value"].first()

    calib = ZoneCalibration(
        fg_pct={str(z): float(v) for z, v in fg.items()},
        point_value={str(z): float(pv.get(z, DEFAULT_POINT_VALUE.get(str(z), 2.0))) for z in fg.index},
        counts={str(z): int(counts.get(z, 0)) for z in fg.index},
    )
    return calib


def compute_zone_pps(df: pd.DataFrame) -> dict[str, float]:
    """Backward-compatible wrapper returning flat PPS by zone."""
    return compute_zone_calibration(df).to_flat_pps()
