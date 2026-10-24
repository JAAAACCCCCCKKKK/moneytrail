"""Shape clues + typology detectors, group smoothing, rank blend.

Uses Alim's main.py pieces (same clue files, same LightGBM, same honest-test folds)
and adds two steps main.py does not have:
  1. graph.smooth_by_group: blend each payment's score with its wallet group's mean.
  2. a rank blend: 0.75 x (all clues) + 0.25 x (starter + typology detectors only).

Run 3 (defaults):     python -m src.predict_shapes_v2
    honest test 57.6 +/- 6.1 on main.py's folds; leaderboard 62.0.
Run 4 candidate:      python -m src.predict_shapes_v2 --windows 120 300 600 1200 --medium-weight 6 \
                          --out predictions_shapes_v3.csv
    Medium schemes are the same shapes as easy ones but slow (median gap about 18
    minutes against 1 minute), so they only link into a wallet group at 20 minutes,
    and the model sees only 225 of them; the 20-minute window and the extra weight
    on medium training rows target both. Gain over run 3 on three fold seeds:
    +1.2, +0.2 and +0.3 (fold means), small but in the same direction every time.
    The weight uses the training label's `difficulty`, only when fitting; it is
    never a clue.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common
import graph
from main import build_features, make_model, OUTPUTS

MODULES = ["features_starter", "features_shapes"]
BLEND = 0.25


def column_sets(columns):
    starter = [c for c in columns if not c.startswith("shape_")]
    typology = [c for c in columns if c.startswith("shape_") and not c.startswith(("shape_b_", "shape_g"))]
    return list(columns), starter + typology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=list(graph.GROUP_WINDOWS),
                        help="wallet-group windows in seconds for smoothing (run 3: 120 300 600)")
    parser.add_argument("--medium-weight", type=float, default=1.0,
                        help="training weight for medium suspicious payments (run 3: 1)")
    parser.add_argument("--out", default="predictions_shapes_v2.csv", help="file name inside outputs/")
    args = parser.parse_args()

    train = build_features("train", MODULES)
    test = build_features("test", MODULES)
    lab = train[["TRANSACTION_HASH"]].merge(common.labels(), on="TRANSACTION_HASH")
    y = lab.is_suspicious.to_numpy()
    weight = np.where((lab.difficulty == "medium").to_numpy(), args.medium_weight, 1.0)
    X = train.drop(columns="TRANSACTION_HASH").astype(float)
    Xt = test.drop(columns="TRANSACTION_HASH").astype(float)
    everything, typology = column_sets(X.columns)

    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    ranks = []
    for cols in (everything, typology):
        model = make_model("lgbm")
        if args.medium_weight == 1.0:
            model.fit(X[cols], y)            # exactly run 3
        else:
            model.fit(X[cols], y, sample_weight=weight)
        raw = model.predict_proba(Xt[cols])[:, 1]
        smoothed = graph.smooth_by_group(raw, test_pay, "test", windows=tuple(args.windows))
        ranks.append(pd.Series(smoothed).rank().to_numpy() / len(smoothed))
    blended = (1 - BLEND) * ranks[0] + BLEND * ranks[1]
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)   # no ties

    out = pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score})
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / args.out
    out.to_csv(path, index=False)
    print(f"wrote {len(out):,} test scores to {path} "
          f"(windows {args.windows}, medium weight {args.medium_weight})")


if __name__ == "__main__":
    main()
