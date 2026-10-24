"""Guilt by association: a second round that learns from how suspicious the money around
each payment looks.

Round 1 is Jack's model (src/train.py). Its scores are turned into neighbour clues: for each
payment from A to B, the mean, max and count of round-1 scores for money into A, A's other
outgoing payments, B's other incoming payments, and money out of B, always leaving the
payment itself out. Round 2 retrains with those clues added.

The honest test is nested so nothing leaks: inside each of Jack's 5 folds, round 1 is
retrained on the other four folds only (with an inner out-of-fold split for those rows), so
no score used to build a held-out payment's clues ever saw a held-out answer.

Run:
  python3 -m src.stage2 --cv-only                     # nested honest test only
  python3 -m src.stage2 --out outputs/predictions.csv # honest test, then test predictions
  python3 -m src.stage2 --cv-only --shuffle-labels    # leak check: should score about 3
  add --fast for quicker, rougher experiments
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from .data import load_labels
from .features import build_features
from .train import CACHE, DATA, PARAMS, SMOOTHERS, groups_for, report, smooth

DIRECTIONS = {
    # name: (column the group is keyed on, column used to look the group up)
    "into_sender": ("dst", "src"),
    "sender_out": ("src", "src"),
    "receiver_in": ("dst", "dst"),
    "out_of_receiver": ("src", "dst"),
}


def _group_stats(keys: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Per key: count, sum, highest and second-highest score."""
    d = pd.DataFrame({"k": keys, "s": s}).sort_values(["k", "s"], ascending=[True, False])
    d["r"] = d.groupby("k").cumcount()
    g = d.groupby("k").s
    stats = pd.DataFrame({"n": g.size(), "sum": g.sum()})
    stats["max1"] = d[d.r == 0].set_index("k").s
    stats["max2"] = d[d.r == 1].set_index("k").s
    return stats


def neighbour_features(p: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """Mean, max and count of the scores around each payment, leaving the payment out."""
    out = pd.DataFrame(index=p.index)
    for name, (group_col, lookup_col) in DIRECTIONS.items():
        stats = _group_stats(p[group_col].to_numpy(), s)
        lookup = p[lookup_col]
        # the payment sits inside its own group when the group and lookup keys match it
        inside = (p[group_col].to_numpy() == lookup.to_numpy()).astype(int)
        n = lookup.map(stats.n).fillna(0).to_numpy() - inside
        total = lookup.map(stats["sum"]).fillna(0).to_numpy() - inside * s
        top1 = lookup.map(stats.max1).to_numpy()
        top2 = lookup.map(stats.max2).to_numpy()
        highest = np.where((inside == 1) & (s >= top1), top2, top1)
        out[f"gba_{name}_mean"] = np.where(n > 0, total / np.maximum(n, 1), np.nan)
        out[f"gba_{name}_max"] = np.where(n > 0, highest, np.nan)
        out[f"gba_{name}_n"] = n
    return out


def fit_score(params, X_fit, y_fit, X_score):
    return lgb.LGBMClassifier(**params).fit(X_fit, y_fit).predict_proba(X_score)[:, 1]


def out_of_fold(params, X, y, groups, rows, n_splits):
    """Round-1 scores for `rows`, each from a model that never saw it."""
    scores = np.zeros(len(rows))
    for a, b in GroupKFold(n_splits).split(X.iloc[rows], y[rows], groups[rows]):
        scores[b] = fit_score(params, X.iloc[rows[a]], y[rows[a]], X.iloc[rows[b]])
    return scores


def nested_cv(params, p, X, y, groups, folds):
    n = len(X)
    round1, round2 = np.zeros(n), np.zeros(n)
    for k, (fit, check) in enumerate(folds, 1):
        s = np.zeros(n)
        s[fit] = out_of_fold(params, X, y, groups, fit, 4)
        s[check] = fit_score(params, X.iloc[fit], y[fit], X.iloc[check])
        round1[check] = s[check]
        X2 = pd.concat([X, neighbour_features(p, s)], axis=1)
        round2[check] = fit_score(params, X2.iloc[fit], y[fit], X2.iloc[check])
        print(f"  fold {k}: round 1 {100 * average_precision_score(y[check], round1[check]):5.1f}"
              f"  round 2 {100 * average_precision_score(y[check], round2[check]):5.1f}", flush=True)
    return round1, round2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-only", action="store_true")
    ap.add_argument("--skip-cv", action="store_true", help="go straight to predictions")
    ap.add_argument("--smoothing", default=None, help="name from SMOOTHERS; default: best round 2 in CV")
    ap.add_argument("--fast", action="store_true", help="fewer trees, for quick experiments")
    ap.add_argument("--shuffle-labels", action="store_true", help="leak check: AP should fall to about 3")
    ap.add_argument("--out", default="outputs/predictions.csv")
    args = ap.parse_args()

    params = dict(PARAMS, n_estimators=250, learning_rate=0.06) if args.fast else dict(PARAMS)
    p, X, ids = build_features(DATA / "train", cache=CACHE)
    lab = load_labels(DATA / "train").set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].reset_index()
    y = lab.is_suspicious.to_numpy()
    if args.shuffle_labels:
        y = np.random.default_rng(0).permutation(y)
    groups = groups_for(lab)
    difficulty = lab.difficulty.to_numpy()
    folds = list(GroupKFold(5).split(X, y, groups=groups))

    choice = args.smoothing
    if not args.skip_cv:
        print(f"{len(X):,} payments, {y.sum():,} suspicious, {X.shape[1]} clues + 12 neighbour clues")
        round1, round2 = nested_cv(params, p, X, y, groups, folds)
        print()
        results = {}
        for label, scores in (("round 1", round1), ("round 2", round2)):
            for name, weights in SMOOTHERS.items():
                s = smooth(scores, ids, weights)
                report(f"{label} | {name}", y, s, folds, difficulty)
                results[f"{label} | {name}"] = 100 * average_precision_score(y, s)
        best2 = max((k for k in results if k.startswith("round 2")), key=results.get)
        best1 = max((k for k in results if k.startswith("round 1")), key=results.get)
        w1, w2 = SMOOTHERS[best1.split(" | ")[1]], SMOOTHERS[best2.split(" | ")[1]]
        wins = sum(average_precision_score(y[c], smooth(round2, ids, w2)[c])
                   > average_precision_score(y[c], smooth(round1, ids, w1)[c]) for _, c in folds)
        print(f"\nbest round 1: {best1} ({results[best1]:.1f})")
        print(f"best round 2: {best2} ({results[best2]:.1f}), better in {wins} of 5 folds")
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/stage2_cv.json").write_text(json.dumps(
            dict(results=results, best_round1=best1, best_round2=best2, folds_won=int(wins),
                 fast=args.fast, shuffled=args.shuffle_labels), indent=2))
        choice = choice or best2.split(" | ")[1]
        if args.cv_only:
            return

    choice = choice or "raw + all three"
    # Round 1 for the training week (out of fold) and the test week (all training data).
    s_train = np.zeros(len(X))
    for fit, check in folds:
        s_train[check] = fit_score(params, X.iloc[fit], y[fit], X.iloc[check])
    pt, Xt, idt = build_features(DATA / "test-data", cache=CACHE)
    Xt = Xt[X.columns]
    s_test = fit_score(params, X, y, Xt)
    # Round 2: each week's neighbour clues come from its own payments only.
    X2 = pd.concat([X, neighbour_features(p, s_train)], axis=1)
    Xt2 = pd.concat([Xt, neighbour_features(pt, s_test)], axis=1)[X2.columns]
    model = lgb.LGBMClassifier(**params).fit(X2, y)
    raw = model.predict_proba(Xt2)[:, 1]
    score = smooth(raw, idt, SMOOTHERS[choice]) + 1e-9 * raw
    score = (score - score.min()) / (score.max() - score.min())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": pt.TRANSACTION_HASH, "score": score}).to_csv(args.out, index=False)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=X2.columns)
    print("neighbour clues by gain rank:",
          ", ".join(f"{c} #{int(gain.rank(ascending=False)[c])}" for c in gain.filter(like="gba_").sort_values(ascending=False).index[:6]))
    print(f"wrote {len(pt):,} test scores to {args.out} using '{choice}'")


if __name__ == "__main__":
    main()
