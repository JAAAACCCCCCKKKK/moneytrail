"""Shape clues + typology detectors, group smoothing, rank blend.

Uses Alim's main.py pieces (same clue files, same LightGBM, same honest-test folds)
and adds two steps main.py does not have:
  1. graph.smooth_by_group: blend each payment's score with its wallet group's mean.
  2. a rank blend: 0.75 x (all clues) + 0.25 x (starter + typology detectors only).

Run 3 (defaults, leaderboard 62.0):   python -m src.predict_shapes_v2
Run 4 (leaderboard 60.8, dropped):     --windows 120 300 600 1200 --medium-weight 6

Options aimed at the train-to-test shift (the README warns the test week has
variations absent from training, and run 4 gained in the honest test but lost on
the leaderboard):
  --drop-conversion  remove the conversion and asset clues. The README says asset and
                     conversion choices ignore the labels, and conversion share moves
                     from 41% (train) to 51% (test).
  --week-rank        replace every amount-based clue with its percentile within its own
                     week. Test amounts run about 12% lower and amount spreads higher
                     (0.58 -> 0.69), the strongest per-clue drift we measured.
  --long-groups      add 30-minute, 1-hour and 2-hour motif clues (features_motifs_long.py).
  --once-groups      add slow-scheme clues: wallet groups at 30 and 60 minutes linked only
                     through once-a-week payment pairs (features_motifs_once.py).
  --self-train       one round of self-training on the test week: take our own top 2% of
                     test scores as suspicious and the bottom 50% as innocent, at half
                     weight, retrain, and score again. Uses only our predictions, never test
                     answers. Simulated on held-out training folds (each fold playing an
                     unseen week): +0.7 and +1.3 on two fold seeds, 7 of 10 folds better.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common
import graph
from main import build_features, make_model, OUTPUTS

MODULES = ["features_starter", "features_shapes"]
LONG_MODULE = "features_motifs_long"
ONCE_MODULE = "features_motifs_once"
BLEND = 0.25
PSEUDO_TOP, PSEUDO_BOTTOM, PSEUDO_WEIGHT = 0.02, 0.5, 0.5
CONVERSION_KEYS = ("conversion", "bridged", "n_legs", "n_assets")
AMOUNT_KEYS = ("usd", "value", "spread")


def column_sets(columns):
    starter = [c for c in columns if not c.startswith(("shape_", "long_", "once_"))]
    typology = [c for c in columns if c.startswith("shape_") and not c.startswith(("shape_b_", "shape_g"))]
    return list(columns), starter + typology


def week_rank(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Percentile of each value within its own week (blanks stay blank)."""
    out = frame.copy()
    for c in columns:
        out[c] = frame[c].rank(pct=True)
    return out


def prepare(drop_conversion=False, rank_amounts=False, long_groups=False, once_groups=False):
    """Train and test clue tables for the chosen options, plus labels."""
    modules = MODULES + ([LONG_MODULE] if long_groups else []) + ([ONCE_MODULE] if once_groups else [])
    train = build_features("train", modules)
    test = build_features("test", modules)
    lab = train[["TRANSACTION_HASH"]].merge(common.labels(), on="TRANSACTION_HASH")
    X = train.drop(columns="TRANSACTION_HASH").astype(float)
    Xt = test.drop(columns="TRANSACTION_HASH").astype(float)
    if drop_conversion:
        keep = [c for c in X.columns if not any(k in c for k in CONVERSION_KEYS)]
        X, Xt = X[keep], Xt[keep]
    if rank_amounts:
        amounts = [c for c in X.columns if any(k in c.lower() for k in AMOUNT_KEYS)]
        X, Xt = week_rank(X, amounts), week_rank(Xt, amounts)
    return train, test, lab, X, Xt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=list(graph.GROUP_WINDOWS),
                        help="wallet-group windows in seconds for smoothing (run 3: 120 300 600)")
    parser.add_argument("--medium-weight", type=float, default=1.0,
                        help="training weight for medium suspicious payments (run 3: 1)")
    parser.add_argument("--drop-conversion", action="store_true")
    parser.add_argument("--week-rank", action="store_true")
    parser.add_argument("--long-groups", action="store_true")
    parser.add_argument("--self-train", action="store_true")
    parser.add_argument("--once-groups", action="store_true")
    parser.add_argument("--out", default="predictions_shapes_v2.csv", help="file name inside outputs/")
    args = parser.parse_args()

    train, test, lab, X, Xt = prepare(args.drop_conversion, args.week_rank, args.long_groups, args.once_groups)
    y = lab.is_suspicious.to_numpy()
    weight = np.where((lab.difficulty == "medium").to_numpy(), args.medium_weight, 1.0)
    everything, typology = column_sets(X.columns)

    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")

    def score_test(fit_X, fit_y, fit_w):
        ranks = []
        for cols in (everything, typology):
            model = make_model("lgbm")
            if fit_w is None:
                model.fit(fit_X[cols], fit_y)            # exactly run 3
            else:
                model.fit(fit_X[cols], fit_y, sample_weight=fit_w)
            raw = model.predict_proba(Xt[cols])[:, 1]
            smoothed = graph.smooth_by_group(raw, test_pay, "test", windows=tuple(args.windows))
            ranks.append(pd.Series(smoothed).rank().to_numpy() / len(smoothed))
        return (1 - BLEND) * ranks[0] + BLEND * ranks[1]

    blended = score_test(X, y, None if args.medium_weight == 1.0 else weight)
    if args.self_train:
        order = pd.Series(blended).rank(pct=True, ascending=False).to_numpy()
        pos, neg = np.flatnonzero(order <= PSEUDO_TOP), np.flatnonzero(order >= 1 - PSEUDO_BOTTOM)
        fit_X = pd.concat([X, Xt.iloc[pos], Xt.iloc[neg]], ignore_index=True)
        fit_y = np.r_[y, np.ones(len(pos)), np.zeros(len(neg))]
        fit_w = np.r_[weight, np.full(len(pos) + len(neg), PSEUDO_WEIGHT)]
        print(f"self-training: {len(pos):,} test payments as suspicious, {len(neg):,} as innocent")
        blended = score_test(fit_X, fit_y, fit_w)
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)   # no ties

    out = pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score})
    OUTPUTS.mkdir(exist_ok=True)
    path = OUTPUTS / args.out
    out.to_csv(path, index=False)
    flags = [f for f in ("drop_conversion", "week_rank", "long_groups", "self_train", "once_groups") if getattr(args, f)]
    print(f"wrote {len(out):,} test scores to {path} (windows {args.windows}, "
          f"medium weight {args.medium_weight}, options {flags or 'none'}; {X.shape[1]} clues)")


if __name__ == "__main__":
    main()
