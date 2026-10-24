"""Run 5 candidate: rank blend of run 3 (Jack's shape model) and Violet's detector v2.

The two models disagree a lot (rank correlation about 0.5), so each catches cases the other
misses. Make both prediction files first, then blend:

  python3 -m src.predict_shapes_v2
  python3 detector/predict.py --train data/train --test data/test-data --out outputs/predictions_violet_v2.csv
  python3 -m src.blend

Honest test (each model's out-of-fold scores, blended with the same weight): run 3 56.8
(folds 57.1 +/- 5.9), Violet 57.4 (57.1 +/- 7.2), blend 50 / 50 58.6 (58.9 +/- 6.3), better
than Violet alone in 4 of 5 folds and than run 3 in 4 of 5. Leaderboard: run 3 62.0, Violet 65.1.

Gate (--gate): multiply the combined model's probability by the structure model's probability
instead of mixing their ranks, then optionally rank-mix in Violet's model. Both input files must
come from `src.predict_union --probabilities` (ranks lose what the product needs):

  python3 -m src.predict_union --probabilities --out predictions_union_prob.csv
  python3 -m src.predict_union --target structure --probabilities --out predictions_structure_prob.csv
  python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_prob.csv \
      --violet predictions_violet_v2_fixed.csv --structure predictions_structure_prob.csv --out predictions_gate_v10.csv

Honest test (main.py folds, out-of-fold for every model), against the 66.7 upload's 62.5 (folds
62.7 +/- 5.6): combined x structure 63.8 (64.4 +/- 5.3, better in 4 of 5 folds); with Violet's
share 0.1, 63.7 (64.3 +/- 5.2, 5 of 5). Multiplying ranks instead gives 62.7.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from main import OUTPUTS

WEIGHT = 0.5    # share of Violet's model in the blend


def rank01(s: pd.Series) -> pd.Series:
    return s.rank() / len(s)


SMOOTH_SCALES = (120, 300, 600, 900)


def smooth_gate(gated: pd.Series, hashes: pd.Series, weight: float, week: str) -> pd.Series:
    """Blend each payment's gate score (log scale) with the mean gate score of its gap clusters.

    A gap cluster joins payments that share a wallet within G seconds (detector/clusters.py), at
    2, 5, 10 and 15 minutes. Labels are pure per scheme, so a payment whose cluster-mates look
    guilty is lifted, and an isolated burst inside an innocent cluster is pulled down. Done on
    the log probabilities, before ranking: on ranks it hurts (Violet, 14:35).
    """
    import sys
    import common
    sys.path.insert(0, str(common.ROOT / "detector"))
    from base import load                      # Violet's loader (same table as detector/)
    from clusters import gap_clusters
    base, _ = load(common.ROOT / "data" / week)
    base = base.sort_values(["TIMESTAMP", "TRANSACTION_INDEX"], kind="mergesort").reset_index(drop=True)
    T = base.TIMESTAMP.to_numpy(np.int64); S = base.ACCOUNT.to_numpy(); R = base.DEST.to_numpy()
    g = gated.set_axis(hashes.to_numpy()).loc[base.TRANSACTION_HASH].to_numpy()
    cluster_mean = np.zeros(len(g))
    for G in SMOOTH_SCALES:
        ids = gap_clusters(T, S, R, G)
        cluster_mean += pd.Series(g).groupby(ids).transform("mean").to_numpy() / len(SMOOTH_SCALES)
    smoothed = pd.Series((1 - weight) * g + weight * cluster_mean, index=base.TRANSACTION_HASH.to_numpy())
    return smoothed.loc[hashes.to_numpy()].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run3", default="predictions_shapes_v2.csv", help="file inside outputs/")
    parser.add_argument("--violet", default="predictions_violet_v2.csv", help="file inside outputs/")
    parser.add_argument("--weight", type=float, default=WEIGHT, help="share of Violet's model")
    parser.add_argument("--structure", default=None,
                        help="optional third file inside outputs/: the structure model (src.predict_union --target structure)")
    parser.add_argument("--structure-weight", type=float, default=0.4, help="share of the structure model")
    parser.add_argument("--gate", action="store_true",
                        help="multiply the --run3 file's probability by the --structure file's probability "
                             "(both from src.predict_union --probabilities) instead of mixing ranks")
    parser.add_argument("--gate-power", type=float, default=1.0, help="power on the structure probability")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="with --gate: share of the multi-scale gap-cluster mean in the gate score (e.g. 0.2)")
    parser.add_argument("--week", default="test-data", help="data/<week> the files score (for --smooth)")
    parser.add_argument("--relay", default=None,
                        help="with --gate: relay multiplier file inside outputs/ (src.relay_groups), applied to the gate")
    parser.add_argument("--out", default="predictions_blend_run3_violet.csv", help="file name inside outputs/")
    args = parser.parse_args()
    if args.gate and not args.structure:
        parser.error("--gate needs --structure")
    if args.relay and not args.gate:
        parser.error("--relay needs --gate")

    a = pd.read_csv(OUTPUTS / args.run3)
    b = pd.read_csv(OUTPUTS / args.violet)
    d = a.merge(b, on="TRANSACTION_HASH", suffixes=("_run3", "_violet"), validate="one_to_one")
    if len(d) != len(a) or len(d) != len(b):
        raise ValueError(f"files cover different transactions ({len(a)}, {len(b)}, {len(d)} shared)")
    if args.structure:
        st = pd.read_csv(OUTPUTS / args.structure).set_index("TRANSACTION_HASH").score.loc[d.TRANSACTION_HASH]
        st = st.reset_index(drop=True)
    if args.gate:
        gated = np.log(d.score_run3.clip(1e-9, 1)) + args.gate_power * np.log(st.clip(1e-9, 1))
        if args.smooth > 0:
            gated = smooth_gate(gated, d.TRANSACTION_HASH, args.smooth, args.week)
        if args.relay:
            mult = pd.read_csv(OUTPUTS / args.relay).set_index("TRANSACTION_HASH").score.loc[d.TRANSACTION_HASH]
            gated = gated + np.log(mult.reset_index(drop=True))
        blended = (1 - args.weight) * rank01(gated) + args.weight * rank01(d.score_violet)
    else:
        blended = (1 - args.weight) * rank01(d.score_run3) + args.weight * rank01(d.score_violet)
        if args.structure:
            blended = (1 - args.structure_weight) * blended + args.structure_weight * rank01(st)
    score = blended.rank(method="first") / len(blended)   # keep the order, break ties
    out = pd.DataFrame({"TRANSACTION_HASH": d.TRANSACTION_HASH, "score": score})
    out.to_csv(OUTPUTS / args.out, index=False)
    if args.gate:
        extra = f", gated by structure^{args.gate_power}" + (f", smoothed {args.smooth}" if args.smooth else "")
    else:
        extra = f", structure weight {args.structure_weight}" if args.structure else ""
    print(f"wrote {len(out):,} blended scores to {OUTPUTS / args.out} (Violet weight {args.weight}{extra})")


if __name__ == "__main__":
    main()
