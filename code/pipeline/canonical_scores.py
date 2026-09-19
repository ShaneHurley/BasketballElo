"""Apply canonical paired-score forecasts into predict/simulate outputs."""
from __future__ import annotations

import numpy as np


def _elo_pair_prior(feat: dict) -> tuple[float | None, float | None]:
    """Elo/matchup-implied home/away points when available."""
    mh = feat.get("matchup_home_pp100")
    ma = feat.get("matchup_away_pp100")
    if mh is not None and ma is not None and np.isfinite(float(mh)) and np.isfinite(float(ma)):
        return float(mh), float(ma)
    total = feat.get("matchup_total")
    margin = feat.get("matchup_margin", feat.get("elo_margin_calibrated", feat.get("elo_margin")))
    if total is None or margin is None:
        return None, None
    if not (np.isfinite(float(total)) and np.isfinite(float(margin))):
        return None, None
    t, m = float(total), float(margin)
    return 0.5 * (t + m), 0.5 * (t - m)


def apply_canonical_score_pair(
    preds: dict,
    feat: dict,
    score_pair_model,
    *,
    legacy_margin: float | None = None,
    cover_calibrator=None,
) -> dict:
    """Overwrite preds with pair-derived home/away as sole score source of truth.

    Structured/hybrid and optional Elo prior blends are applied to home/away
    *before* deriving margin/total. Market lines only price probabilities.
    """
    from pipeline.config import (
        SCORE_PAIR_ELO_BETA,
        SCORE_PAIR_STRUCT_BETA,
        USE_CANONICAL_SCORE_PAIR,
        USE_HYBRID_STRUCTURED_BLEND,
        HYBRID_TOTAL_BETA,
        HYBRID_MARGIN_BETA,
    )
    from pipeline.score_targets import (
        apply_structured_to_pair,
        assert_pair_algebra,
        margin_home_cover_prob,
        margin_win_prob,
        score_predictive_intervals,
        total_over_prob,
    )

    if score_pair_model is None or not getattr(score_pair_model, "fitted", False):
        return preds

    pair_out = score_pair_model.predict_scores(feat)
    raw_h = float(pair_out["pred_home"])
    raw_a = float(pair_out["pred_away"])
    preds["raw_pred_home"] = raw_h
    preds["raw_pred_away"] = raw_a
    preds["forecast_source"] = "score_pair"

    ph, pa = raw_h, raw_a
    if USE_HYBRID_STRUCTURED_BLEND and (
        feat.get("struct_home_pts") is not None or feat.get("struct_away_pts") is not None
    ):
        beta = float(SCORE_PAIR_STRUCT_BETA)
        if beta <= 0:
            beta = 0.5 * (float(HYBRID_TOTAL_BETA) + float(HYBRID_MARGIN_BETA))
        ph, pa = apply_structured_to_pair(
            ph,
            pa,
            struct_home=feat.get("struct_home_pts"),
            struct_away=feat.get("struct_away_pts"),
            home_beta=beta,
        )
        preds["struct_blend_beta"] = beta

    elo_beta = float(SCORE_PAIR_ELO_BETA)
    if elo_beta > 0:
        eh, ea = _elo_pair_prior(feat)
        if eh is not None and ea is not None:
            ph, pa = apply_structured_to_pair(
                ph, pa, struct_home=eh, struct_away=ea, home_beta=elo_beta,
            )
            preds["elo_blend_beta"] = elo_beta

    margin = ph - pa
    total = ph + pa
    assert_pair_algebra(ph, pa, margin, total)

    sigma_h = float(pair_out.get("sigma_home", 10.0))
    sigma_a = float(pair_out.get("sigma_away", 10.0))
    corr = float(pair_out.get("score_residual_corr", 0.35))
    intervals = score_predictive_intervals(
        ph, pa, sigma_home=sigma_h, sigma_away=sigma_a, corr=corr,
    )

    preds["pred_home"] = ph
    preds["pred_away"] = pa
    preds["pred_home_pts"] = ph
    preds["pred_away_pts"] = pa
    preds["pred_margin"] = margin
    preds["pred_total"] = total
    preds["sigma_home"] = intervals["sigma_home"]
    preds["sigma_away"] = intervals["sigma_away"]
    preds["sigma_margin"] = intervals["sigma_margin"]
    preds["sigma_total"] = intervals["sigma_total"]
    preds["score_residual_corr"] = intervals["corr"]
    preds["score_residual_cov"] = intervals["cov"]
    preds["CONF_LOWER"] = intervals["CONF_LOWER"]
    preds["CONF_UPPER"] = intervals["CONF_UPPER"]
    preds["CONF_WIDTH"] = intervals["CONF_WIDTH"]
    for k, v in intervals.items():
        if k.startswith(("home_q", "away_q", "margin_q", "total_q", "spread_q")):
            preds[k] = v
        if k == "spread_quantile_width":
            preds[k] = v

    preds["win_prob_margin"] = margin_win_prob(margin, intervals["sigma_margin"])
    decision = feat.get("decision_spread", feat.get("market_spread"))
    if decision is not None and np.isfinite(float(decision)):
        raw_cover = margin_home_cover_prob(
            margin, float(decision), intervals["sigma_margin"],
        )
        preds["home_cover_prob_raw"] = float(raw_cover)
        if cover_calibrator is not None and getattr(cover_calibrator, "fitted", False):
            preds["home_cover_prob"] = float(cover_calibrator.transform(raw_cover))
            preds["cover_prob_source"] = "pair_margin_oof_calibrated"
        else:
            preds["home_cover_prob"] = float(raw_cover)
            preds["cover_prob_source"] = "pair_margin_gaussian"
    mkt_tot = feat.get("market_total")
    if mkt_tot is not None and np.isfinite(float(mkt_tot)):
        preds["p_over"] = total_over_prob(total, float(mkt_tot), intervals["sigma_total"])
        preds["p_under"] = 1.0 - float(preds["p_over"])

    if USE_CANONICAL_SCORE_PAIR and legacy_margin is not None:
        preds["legacy_margin"] = float(legacy_margin)
        preds["margin_disagreement"] = abs(float(margin) - float(legacy_margin))
        preds["score_pair_margin"] = margin

    return preds


def prefer_canonical_margin(preds: dict, fallback_margin: float) -> float:
    """Return pair-derived margin when canonical mode is on and available."""
    from pipeline.config import USE_CANONICAL_SCORE_PAIR

    if USE_CANONICAL_SCORE_PAIR and preds.get("forecast_source") == "score_pair":
        m = preds.get("pred_margin")
        if m is not None and np.isfinite(float(m)):
            return float(m)
    return float(fallback_margin)
