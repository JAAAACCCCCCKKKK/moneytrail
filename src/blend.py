"""Run 5 candidate: rank blend of run 3 (Jack's shape model) and Violet's detector v2.

The two models disagree a lot (rank correlation about 0.5), so each catches cases the other
misses. Make both prediction files first, then blend:

  python3 -m src.predict_shapes_v2
  python3 detector/predict.py --train data/train --test data/test-data --out outputs/predictions_violet_v2.csv
  python3 -m src.blend

Honest test (each model's out-of-fold scores, blended with the same weight): run 3 56.8
(folds 57.1 +/- 5.9), Violet 57.4 (57.1 +/- 7.2), blend 50 / 50 58.6 (58.9 +/- 6.3), better
than Violet alone in 4 of 5 folds and than run 3 in 4 of 5. Leaderboard: run 3 62.0, Violet 65.1.
"""
from __future__ import annotations

import argparse

import pandas as pd

from main import OUTPUTS

WEIGHT = 0.5    # share of Violet's model in the blend


def rank01(s: pd.Series) -> pd.Series:
    return s.rank() / len(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run3", default="predictions_shapes_v2.csv", help="file inside outputs/")
    parser.add_argument("--violet", default="predictions_violet_v2.csv", help="file inside outputs/")
    parser.add_argument("--weight", type=float, default=WEIGHT, help="share of Violet's model")
    parser.add_argument("--out", default="predictions_blend_run3_violet.csv", help="file name inside outputs/")
    args = parser.parse_args()

    a = pd.read_csv(OUTPUTS / args.run3)
    b = pd.read_csv(OUTPUTS / args.violet)
    d = a.merge(b, on="TRANSACTION_HASH", suffixes=("_run3", "_violet"), validate="one_to_one")
    if len(d) != len(a) or len(d) != len(b):
        raise ValueError(f"files cover different transactions ({len(a)}, {len(b)}, {len(d)} shared)")
    blended = (1 - args.weight) * rank01(d.score_run3) + args.weight * rank01(d.score_violet)
    score = blended.rank(method="first") / len(blended)   # keep the order, break ties
    out = pd.DataFrame({"TRANSACTION_HASH": d.TRANSACTION_HASH, "score": score})
    out.to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(out):,} blended scores to {OUTPUTS / args.out} (Violet weight {args.weight})")


if __name__ == "__main__":
    main()
