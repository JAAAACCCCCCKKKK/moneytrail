"""Keep one ranking's top k, order everything below by another ranking (Jack).

  python -m src.protect_top        # outputs/predictions_protect_top.csv

Average precision punishes a false alarm near the top far more than it rewards a gain lower
down: on our honest-test ranking one extra false alarm at the very top costs about 0.4 points,
while moving one medium case up 1,300 places adds about 0.01. On the leaderboard the 66.7 recipe's
top 1% was 95% precise and the 67.1 recipe's 93%, while the 67.1 recipe ranks medium cases higher
further down. So: the top k of --top in its own order, then every other payment in the order of
--rest. Both default inputs are the reproducible rebuilds of those two recipes.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from main import OUTPUTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", default="predictions_blend_union_signals_violet.csv", help="file inside outputs/")
    parser.add_argument("--rest", default="predictions_blend_two_q_violet.csv", help="file inside outputs/")
    parser.add_argument("--k", type=int, default=402, help="how many of --top's highest to keep first")
    parser.add_argument("--out", default="predictions_protect_top.csv", help="file name inside outputs/")
    args = parser.parse_args()

    top = pd.read_csv(OUTPUTS / args.top)
    rest = pd.read_csv(OUTPUTS / args.rest)
    if set(top.TRANSACTION_HASH) != set(rest.TRANSACTION_HASH):
        raise ValueError("the two files cover different transactions")
    first = top.sort_values(["score", "TRANSACTION_HASH"], ascending=[False, True]).TRANSACTION_HASH.head(args.k)
    later = rest[~rest.TRANSACTION_HASH.isin(first)].sort_values(["score", "TRANSACTION_HASH"], ascending=[False, True]).TRANSACTION_HASH
    order = pd.concat([first, later], ignore_index=True)
    score = np.arange(len(order), 0, -1) / len(order)          # distinct, highest first
    out = pd.DataFrame({"TRANSACTION_HASH": order, "score": score})
    out = rest[["TRANSACTION_HASH"]].merge(out, on="TRANSACTION_HASH")   # keep the usual row order
    out.to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(out):,} scores to {OUTPUTS / args.out} (top {args.k} from {args.top}, rest from {args.rest})")


if __name__ == "__main__":
    main()
