"""Two questions instead of one, with slow-scheme groups (Jack).

  python -m src.predict_two_q            # outputs/predictions_two_q.csv

Why: medium schemes are the same shapes as easy ones run slowly (a median of 18 minutes
between payments). Our clues look 2-10 minutes wide, so a medium payment looks like an
isolated background payment and sits at a median rank of about 3,900 in the honest test,
under hundreds of relay and background payments. Two facts make it recoverable:
  - 96% of planted shape-scheme payments (easy, medium and their honest look-alikes) are
    once-a-week pairs, against 28% of background and 36% of relay payments, so "is this
    part of a planted scheme?" is a much easier question than "is this suspicious?";
  - among planted schemes, slow ones are still about 1 in 4 suspicious (median gap
    960-1,920s), far above any background or relay payment.
So two LightGBM models on the combined clues plus the once-a-week-pair group clues
(features_motifs_once.py): (1) trained on every row to say whether a payment belongs to a
planted shape scheme, suspicious or not (`family` is a training target only, never a
clue); (2) trained only on scheme rows to say `is_suspicious`. Score = (1) x (2), no group
smoothing (it mixes in short-window groups and hurt in the honest test). The final score
is a 50/50 rank blend with the combined model (src.predict_union_signals), which keeps a
safety net if the test week has a kind of scheme model (1) does not recognise.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common
import graph
from main import make_model, OUTPUTS
from src.predict_shapes_v2 import prepare
from src.predict_union_signals import LEFT_OUT, violet_clues

NOT_SCHEME = ("background", "aggregate_relay", "distributed_relay", "ambiguous")
BLEND = 0.5    # share of the two-question score


def rank01(s) -> np.ndarray:
    return pd.Series(s).rank(pct=True).to_numpy()


def tables():
    """Combined clues with and without the once-a-week-pair group clues, both weeks."""
    train, test, lab, X, Xt = prepare(once_groups=True, signals=True)
    X, Xt = X.drop(columns=list(LEFT_OUT)), Xt.drop(columns=list(LEFT_OUT))
    U2 = pd.concat([X.reset_index(drop=True), violet_clues("train", train.TRANSACTION_HASH)], axis=1)
    Ut2 = pd.concat([Xt.reset_index(drop=True), violet_clues("test", test.TRANSACTION_HASH)], axis=1)[U2.columns]
    once = [c for c in U2.columns if c.startswith("once_")]
    return train, test, lab, U2.drop(columns=once), Ut2.drop(columns=once), U2, Ut2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="predictions_two_q.csv", help="file name inside outputs/")
    args = parser.parse_args()

    train, test, lab, U, Ut, U2, Ut2 = tables()
    y = lab.is_suspicious.to_numpy()
    scheme = (~lab.family.isin(NOT_SCHEME)).to_numpy().astype(int)
    in_scheme = scheme == 1

    single = make_model("lgbm").fit(U, y).predict_proba(Ut)[:, 1]          # = src.predict_union_signals
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    single = graph.smooth_by_group(single, test_pay, "test")
    p_scheme = make_model("lgbm").fit(U2, scheme).predict_proba(Ut2)[:, 1]
    p_within = make_model("lgbm").fit(U2[in_scheme], y[in_scheme]).predict_proba(Ut2)[:, 1]

    blended = (1 - BLEND) * rank01(single) + BLEND * rank01(p_scheme * p_within)
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)   # no ties
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(score):,} test scores to {OUTPUTS / args.out} ({U2.shape[1]} clues)")


if __name__ == "__main__":
    main()
