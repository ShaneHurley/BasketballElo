#!/usr/bin/env python3
"""Permutation importance on calib slice features (suggest drops via ablation only)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from pipeline.model import MetaScoreModel, SAFE_FEATURE_COLS, build_feature_row


def main():
    parser = argparse.ArgumentParser(description="Feature permutation importance on a feature CSV")
    parser.add_argument("features_csv", type=Path, help="CSV with SAFE_FEATURE_COLS + actual_margin")
    parser.add_argument("--out", type=Path, default=Path("feature_importance.csv"))
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    if "actual_margin" not in df.columns:
        raise SystemExit("features_csv must include actual_margin column")

    cols = [c for c in SAFE_FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0)
    y = df["actual_margin"].values

    model = MetaScoreModel(feature_cols=cols)
    if "actual_home" in df.columns and "actual_away" in df.columns:
        model.fit(df, df["actual_home"], df["actual_away"])
    else:
        raise SystemExit("Need actual_home and actual_away to fit MetaScoreModel")

    raw = model._predict_raw(df)
    base_mae = mean_absolute_error(y, raw["pred_margin"])

    class _Wrapper:
        fitted = True
        features = cols

        def predict(self, X_in):
            if isinstance(X_in, pd.DataFrame):
                sub = X_in
            else:
                sub = pd.DataFrame(X_in, columns=cols)
            return model._predict_raw(sub)["pred_margin"]

    wrapper = _Wrapper()
    result = permutation_importance(
        wrapper, X, y, scoring="neg_mean_absolute_error", n_repeats=args.n_repeats, random_state=42,
    )
    out = pd.DataFrame({
        "feature": cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    out["baseline_mae"] = base_mae
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Baseline MAE={base_mae:.3f}  wrote {len(out)} rows → {args.out}")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
