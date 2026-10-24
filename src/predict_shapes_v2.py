"""Run 3 candidate: shape clues + typology detectors, group smoothing, rank blend.

Uses Alim's main.py pieces (same clue files, same LightGBM, same honest-test folds)
and adds two steps main.py does not have yet:
  1. graph.smooth_by_group: blend each payment's score with its wallet group's mean.
  2. a rank blend: 0.75 x (all clues) + 0.25 x (starter + typology detectors only).

Honest test on main.py's folds: 57.6 +/- 6.1 (all folds together 57.1), against
56.5 +/- 4.9 for run 2's clues under the same folds and smoothing.

Run:  python -m src.predict_shapes_v2      -> outputs/predictions_shapes_v2.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import graph
from main import build_features, fit_predict, OUTPUTS

MODULES = ["features_starter", "features_shapes"]
BLEND = 0.25


def column_sets(columns):
    starter = [c for c in columns if not c.startswith("shape_")]
    typology = [c for c in columns if c.startswith("shape_") and not c.startswith(("shape_b_", "shape_g"))]
    return list(columns), starter + typology


def main():
    train = build_features("train", MODULES)
    test = build_features("test", MODULES)
    y = train[["TRANSACTION_HASH"]].merge(common.labels(), on="TRANSACTION_HASH").is_suspicious.to_numpy()
    X, Xt = train.drop(columns="TRANSACTION_HASH"), test.drop(columns="TRANSACTION_HASH")
    everything, typology = column_sets(X.columns)

    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    ranks = []
    for cols in (everything, typology):
        raw = fit_predict("lgbm", X[cols], y, Xt[cols])
        smoothed = graph.smooth_by_group(raw, test_pay, "test")
        ranks.append(pd.Series(smoothed).rank().to_numpy() / len(smoothed))
    blended = (1 - BLEND) * ranks[0] + BLEND * ranks[1]
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)   # no ties

    out = pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score})
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / "predictions_shapes_v2.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out):,} test scores to {path}")


if __name__ == "__main__":
    main()
