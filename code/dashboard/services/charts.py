"""Plotly chart builders. Add a chart = one function + one CHARTS entry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from dashboard.jobs.util import ensure_ats_win
from dashboard.paths import results_csv_path


@dataclass
class ChartSpec:
    id: str
    tab: str
    title: str
    required_columns: list[str]
    builder: Callable[[pd.DataFrame], dict]


def _empty(title: str, reason: str) -> dict:
    return {
        "data": [],
        "layout": {
            "title": title,
            "annotations": [{"text": reason, "xref": "paper", "yref": "paper",
                             "x": 0.5, "y": 0.5, "showarrow": False}],
        },
    }


def _json_safe(obj: Any) -> Any:
    """Convert NaN/inf and numpy scalars so FastAPI JSONResponse can serialize."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    return obj


def _has(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def chart_season_ats(df: pd.DataFrame) -> dict:
    title = "Season ATS hit rate"
    season = "simulated_season_window" if "simulated_season_window" in df.columns else None
    if not season or "ATS_WIN" not in df.columns:
        return _empty(title, "Need simulated_season_window + ATS_WIN")
    g = df.groupby(season)["ATS_WIN"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    return {
        "data": [{"type": "bar", "x": [str(i) for i in g.index], "y": g.values.tolist(), "name": "ATS"}],
        "layout": {"title": title, "yaxis": {"title": "Hit rate", "range": [0, 1]}},
    }


def chart_rolling_mae(df: pd.DataFrame) -> dict:
    title = "Rolling spread MAE"
    err_col = "SPREAD_ERR" if "SPREAD_ERR" in df.columns else None
    if err_col is None:
        return _empty(title, "Need SPREAD_ERR")
    err = pd.to_numeric(df[err_col], errors="coerce").abs().dropna()
    if len(err) < 20:
        return _empty(title, "Not enough rows")
    roll = err.rolling(50, min_periods=10).mean()
    return {
        "data": [{"type": "scatter", "mode": "lines", "y": roll.values.tolist(), "name": "MAE50"}],
        "layout": {"title": title, "yaxis": {"title": "MAE"}},
    }


def chart_edge_hist(df: pd.DataFrame) -> dict:
    title = "Edge histogram"
    if "EDGE" not in df.columns:
        return _empty(title, "Need EDGE")
    e = pd.to_numeric(df["EDGE"], errors="coerce").dropna()
    return {
        "data": [{"type": "histogram", "x": e.values.tolist(), "nbinsx": 40, "name": "EDGE"}],
        "layout": {"title": title, "xaxis": {"title": "EDGE"}},
    }


def chart_clv_scatter(df: pd.DataFrame) -> dict:
    title = "CLV scatter"
    clv = None
    for c in ("POINT_CLV", "CLV", "CLV_POINTS"):
        if c in df.columns:
            clv = c
            break
    if clv is None or "EDGE" not in df.columns:
        return _empty(title, "Need EDGE + CLV column")
    x = pd.to_numeric(df["EDGE"], errors="coerce")
    y = pd.to_numeric(df[clv], errors="coerce")
    m = x.notna() & y.notna()
    return {
        "data": [{"type": "scatter", "mode": "markers", "x": x[m].tolist(), "y": y[m].tolist(),
                  "marker": {"size": 4, "opacity": 0.5}}],
        "layout": {"title": title, "xaxis": {"title": "EDGE"}, "yaxis": {"title": clv}},
    }


def chart_cover_calibration(df: pd.DataFrame) -> dict:
    """Cover calibration — defaults to actionable bets when ACTIONABLE is present."""
    title = "Cover calibration"
    work = df.copy()
    population = "all_graded"
    if "ACTIONABLE" in work.columns:
        act = pd.to_numeric(work["ACTIONABLE"], errors="coerce").fillna(0) > 0.5
        if int(act.sum()) >= 20:
            work = work.loc[act].copy()
            population = "actionable"
            title = "Cover calibration (actionable)"
        else:
            title = "Cover calibration (lean-graded; few actionable)"
            population = "lean_graded"
    pcol = "CALIBRATED_COVER_PROB" if "CALIBRATED_COVER_PROB" in work.columns else (
        "COVER_PROB_CALIBRATED" if "COVER_PROB_CALIBRATED" in work.columns else None)
    if pcol is None or "ATS_WIN" not in work.columns:
        return _empty(title, "Need calibrated cover prob + ATS_WIN")
    p = pd.to_numeric(work[pcol], errors="coerce")
    y = pd.to_numeric(work["ATS_WIN"], errors="coerce")
    m = p.notna() & y.notna()
    tmp = pd.DataFrame({"p": p[m], "y": y[m]})
    if tmp.empty:
        return _empty(title, "No overlapping cover prob / ATS_WIN rows")
    tmp["bucket"] = pd.cut(tmp["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
    g = tmp.groupby("bucket", observed=False).agg(mean_p=("p", "mean"), hit=("y", "mean"), n=("y", "size"))
    g = g.dropna(subset=["mean_p", "hit"])
    g = g[g["n"] > 0]
    if g.empty:
        return _empty(title, "All calibration buckets empty")
    return {
        "data": [
            {"type": "scatter", "mode": "lines+markers", "x": g["mean_p"].tolist(), "y": g["hit"].tolist(),
             "name": "Empirical", "text": [f"n={int(n)}" for n in g["n"]]},
            {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "name": "Ideal",
             "line": {"dash": "dash"}},
        ],
        "layout": {
            "title": title,
            "xaxis": {"title": f"Predicted ({population})"},
            "yaxis": {"title": "Observed", "range": [0, 1]},
        },
        "meta": {"population": population, "n": int(len(tmp))},
    }


def chart_roi_by_conf(df: pd.DataFrame) -> dict:
    title = "ROI by confidence tier"
    work = df.copy()
    conf = None
    for c in ("CONFIDENCE_TIER", "CONFIDENCE", "ATS_CONF", "conf_tier"):
        if c in work.columns:
            conf = c
            break
    if conf is None:
        return _empty(title, "Need CONFIDENCE_TIER / CONFIDENCE")

    if "PNL" in work.columns:
        pnl = pd.to_numeric(work["PNL"], errors="coerce")
    elif "ROI" in work.columns:
        pnl = pd.to_numeric(work["ROI"], errors="coerce")
    elif "ATS_WIN" in work.columns:
        # -110 American: win +100/110, loss −1
        y = pd.to_numeric(work["ATS_WIN"], errors="coerce")
        pnl = y.map(lambda v: (100.0 / 110.0) if v == 1 else (-1.0 if v == 0 else np.nan))
    else:
        return _empty(title, "Need PNL/ROI or ATS_WIN")

    work = work.assign(_pnl=pnl).dropna(subset=["_pnl"])
    if work.empty:
        return _empty(title, "No graded bets for ROI")
    g = work.groupby(conf)["_pnl"].mean()
    return {
        "data": [{"type": "bar", "x": [str(i) for i in g.index], "y": g.values.tolist(), "name": "mean_pnl"}],
        "layout": {"title": title, "yaxis": {"title": "Mean PNL (−110)"}},
    }


def chart_clv_by_season(df: pd.DataFrame) -> dict:
    title = "CLV by season"
    clv = None
    for c in ("POINT_CLV", "CLV", "CLV_POINTS"):
        if c in df.columns:
            clv = c
            break
    season = "simulated_season_window" if "simulated_season_window" in df.columns else None
    if clv is None or season is None:
        return _empty(title, "Need POINT_CLV + simulated_season_window")
    tmp = df[[season, clv]].copy()
    tmp[clv] = pd.to_numeric(tmp[clv], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return _empty(title, "No finite CLV rows")
    g = tmp.groupby(season)[clv].mean()
    return {
        "data": [{"type": "bar", "x": [str(i) for i in g.index], "y": g.values.tolist(), "name": "mean_clv"}],
        "layout": {"title": title, "yaxis": {"title": "Mean point CLV"}},
    }


def chart_ml_reliability(df: pd.DataFrame) -> dict:
    title = "ML reliability"
    wp = "WIN_PROB" if "WIN_PROB" in df.columns else ("P_HOME" if "P_HOME" in df.columns else None)
    if wp is None or "ACTUAL_HOME" not in df.columns:
        return _empty(title, "Need WIN_PROB + scores")
    p = pd.to_numeric(df[wp], errors="coerce")
    y = (pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
         > pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")).astype(float)
    m = p.notna() & y.notna()
    tmp = pd.DataFrame({"p": p[m], "y": y[m]})
    if tmp.empty:
        return _empty(title, "No overlapping win-prob rows")
    tmp["bucket"] = pd.cut(tmp["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
    g = tmp.groupby("bucket", observed=False).agg(mean_p=("p", "mean"), hit=("y", "mean"), n=("y", "size"))
    g = g.dropna(subset=["mean_p", "hit"])
    g = g[g["n"] > 0]
    if g.empty:
        return _empty(title, "All reliability buckets empty")
    return {
        "data": [
            {"type": "scatter", "mode": "lines+markers", "x": g["mean_p"].tolist(), "y": g["hit"].tolist()},
            {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "line": {"dash": "dash"}},
        ],
        "layout": {"title": title},
    }


def chart_pred_vs_actual_home(df: pd.DataFrame) -> dict:
    title = "Predicted vs actual home score"
    if "PRED_HOME" not in df.columns or "ACTUAL_HOME" not in df.columns:
        return _empty(title, "Need PRED_HOME + ACTUAL_HOME")
    pred = pd.to_numeric(df["PRED_HOME"], errors="coerce")
    act = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
    m = pred.notna() & act.notna()
    return {
        "data": [{"type": "scatter", "mode": "markers", "x": act[m].tolist(), "y": pred[m].tolist(),
                  "marker": {"size": 4, "opacity": 0.4}, "name": "home"}],
        "layout": {"title": title, "xaxis": {"title": "Actual home"}, "yaxis": {"title": "Predicted home"}},
        "meta": {"population": "all_games", "forecast_source": (
            str(df["FORECAST_SOURCE"].mode().iloc[0]) if "FORECAST_SOURCE" in df.columns and len(df) else None
        )},
    }


def chart_pred_vs_actual_away(df: pd.DataFrame) -> dict:
    title = "Predicted vs actual away score"
    if "PRED_AWAY" not in df.columns or "ACTUAL_AWAY" not in df.columns:
        return _empty(title, "Need PRED_AWAY + ACTUAL_AWAY")
    pred = pd.to_numeric(df["PRED_AWAY"], errors="coerce")
    act = pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
    m = pred.notna() & act.notna()
    return {
        "data": [{"type": "scatter", "mode": "markers", "x": act[m].tolist(), "y": pred[m].tolist(),
                  "marker": {"size": 4, "opacity": 0.4}, "name": "away"}],
        "layout": {"title": title, "xaxis": {"title": "Actual away"}, "yaxis": {"title": "Predicted away"}},
        "meta": {"population": "all_games"},
    }


def chart_score_residual_corr(df: pd.DataFrame) -> dict:
    title = "Home vs away score residuals"
    if not {"PRED_HOME", "PRED_AWAY", "ACTUAL_HOME", "ACTUAL_AWAY"}.issubset(df.columns):
        return _empty(title, "Need predicted and actual scores")
    rh = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce") - pd.to_numeric(df["PRED_HOME"], errors="coerce")
    ra = pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce") - pd.to_numeric(df["PRED_AWAY"], errors="coerce")
    m = rh.notna() & ra.notna()
    return {
        "data": [{"type": "scatter", "mode": "markers", "x": rh[m].tolist(), "y": ra[m].tolist(),
                  "marker": {"size": 4, "opacity": 0.35}}],
        "layout": {"title": title, "xaxis": {"title": "Home residual"}, "yaxis": {"title": "Away residual"}},
        "meta": {"population": "all_games", "n": int(m.sum())},
    }


def chart_pred_vs_actual_total(df: pd.DataFrame) -> dict:
    title = "Predicted vs actual total"
    if "PRED_TOTAL" not in df.columns or "ACTUAL_HOME" not in df.columns:
        return _empty(title, "Need PRED_TOTAL + scores")
    pred = pd.to_numeric(df["PRED_TOTAL"], errors="coerce")
    act = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce") + pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
    m = pred.notna() & act.notna()
    return {
        "data": [{"type": "scatter", "mode": "markers", "x": act[m].tolist(), "y": pred[m].tolist(),
                  "marker": {"size": 4, "opacity": 0.4}}],
        "layout": {"title": title, "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"}},
    }


def chart_residual_vs_market(df: pd.DataFrame) -> dict:
    title = "Residual vs market total"
    if "PRED_TOTAL" not in df.columns or "MARKET_TOTAL" not in df.columns or "ACTUAL_HOME" not in df.columns:
        return _empty(title, "Need PRED_TOTAL, MARKET_TOTAL, scores")
    pred = pd.to_numeric(df["PRED_TOTAL"], errors="coerce")
    mkt = pd.to_numeric(df["MARKET_TOTAL"], errors="coerce")
    act = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce") + pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
    resid = pred - act
    m = resid.notna() & mkt.notna()
    return {
        "data": [{"type": "scatter", "mode": "markers", "x": mkt[m].tolist(), "y": resid[m].tolist(),
                  "marker": {"size": 4, "opacity": 0.4}}],
        "layout": {"title": title, "xaxis": {"title": "Market total"}, "yaxis": {"title": "Pred − actual"}},
    }


def chart_bankroll(df: pd.DataFrame) -> dict:
    title = "Cumulative PNL"
    if "PNL" not in df.columns:
        return _empty(title, "Need PNL")
    pnl = pd.to_numeric(df["PNL"], errors="coerce").fillna(0)
    return {
        "data": [{"type": "scatter", "mode": "lines", "y": pnl.cumsum().tolist(), "name": "bankroll"}],
        "layout": {"title": title},
    }


CHARTS: dict[str, ChartSpec] = {}


def _reg(spec: ChartSpec) -> None:
    CHARTS[spec.id] = spec


_reg(ChartSpec("season_ats", "all", "Season ATS hit rate", ["ATS_WIN"], chart_season_ats))
_reg(ChartSpec("rolling_mae", "all", "Rolling spread MAE", ["SPREAD_ERR"], chart_rolling_mae))
_reg(ChartSpec("edge_hist", "all", "Edge histogram", ["EDGE"], chart_edge_hist))
_reg(ChartSpec("clv_scatter", "all", "CLV scatter", ["EDGE"], chart_clv_scatter))
_reg(ChartSpec("bankroll", "all", "Cumulative PNL", ["PNL"], chart_bankroll))
_reg(ChartSpec("cover_calibration", "ats", "Cover calibration", ["ATS_WIN"], chart_cover_calibration))
_reg(ChartSpec("roi_by_conf", "ats", "ROI by confidence", [], chart_roi_by_conf))
_reg(ChartSpec("clv_by_season", "ats", "CLV by season", ["POINT_CLV"], chart_clv_by_season))
_reg(ChartSpec("ml_reliability", "ml", "ML reliability", ["ACTUAL_HOME"], chart_ml_reliability))
_reg(ChartSpec("pred_vs_actual_home", "all", "Pred vs actual home", ["PRED_HOME", "ACTUAL_HOME"], chart_pred_vs_actual_home))
_reg(ChartSpec("pred_vs_actual_away", "all", "Pred vs actual away", ["PRED_AWAY", "ACTUAL_AWAY"], chart_pred_vs_actual_away))
_reg(ChartSpec("score_residual_corr", "all", "Score residual correlation", ["PRED_HOME", "PRED_AWAY"], chart_score_residual_corr))
_reg(ChartSpec("pred_vs_actual_total", "totals", "Pred vs actual total", ["PRED_TOTAL"], chart_pred_vs_actual_total))
_reg(ChartSpec("residual_vs_market", "totals", "Residual vs market", ["PRED_TOTAL", "MARKET_TOTAL"], chart_residual_vs_market))


def list_charts(tab: str | None = None) -> list[dict]:
    specs = CHARTS.values()
    if tab:
        specs = [s for s in specs if s.tab == tab or tab == "all"]
    return [{"id": s.id, "tab": s.tab, "title": s.title, "required_columns": s.required_columns} for s in specs]


def build_chart(chart_id: str, run_id: str | None) -> dict[str, Any]:
    spec = CHARTS.get(chart_id)
    if spec is None:
        return _empty(chart_id, f"Unknown chart: {chart_id}")
    path = results_csv_path(run_id)
    if path is None or not path.exists():
        return _empty(spec.title, "No results CSV")
    df = pd.read_csv(path)
    df = ensure_ats_win(df)
    missing = [c for c in spec.required_columns if c not in df.columns]
    # Still try builder; it handles soft deps
    try:
        fig = spec.builder(df)
    except Exception as exc:
        fig = _empty(spec.title, str(exc))
    fig["_meta"] = {"chart_id": chart_id, "missing_required": missing, "n_rows": len(df)}
    return _json_safe(fig)
