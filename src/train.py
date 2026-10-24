"""Train the shape detector, measure it honestly, and score the test split.

Honest test: 5-fold GroupKFold grouped by the training scenario_id, with every
background payment as its own group. Labels are pure per scenario, so a split
that puts one scenario on both sides would flatter the score.

Smoothing: because labels are pure per scenario, a payment's final score is
blended with the mean score of the motif it belongs to. Catching one member of
a planted shape then lifts the whole shape.

Run:
  python -m src.train                 # cross-validate, then write predictions.csv
  python -m src.train --cv-only       # cross-validate only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from .data import load_labels
from .features import build_features

SEED = 42
DATA = Path("data")
CACHE = DATA / "cache"
PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=SEED, verbose=-1,
)


def groups_for(labels: pd.DataFrame) -> np.ndarray:
    sid = labels.scenario_id.astype("string")
    filler = pd.Series("bg_" + np.arange(len(labels)).astype(str), index=labels.index, dtype="string")
    return sid.fillna(filler).to_numpy()


def smooth(raw: np.ndarray, ids: dict[int, np.ndarray], weights: dict) -> np.ndarray:
    """Blend each payment's score with its motif's mean score at chosen scales."""
    out = weights.get("raw", 1.0) * raw
    for scale, w in weights.items():
        if scale == "raw" or w == 0:
            continue
        motif_mean = pd.Series(raw).groupby(ids[scale]).transform("mean").to_numpy()
        out = out + w * motif_mean
    return out


SMOOTHERS = {
    "raw only": {"raw": 1.0},
    "raw + m120 mean": {"raw": 0.5, 120: 0.5},
    "raw + m300 mean": {"raw": 0.5, 300: 0.5},
    "raw + m600 mean": {"raw": 0.5, 600: 0.5},
    "raw + m120 + m300": {"raw": 0.4, 120: 0.3, 300: 0.3},
    "raw + all three": {"raw": 0.25, 120: 0.25, 300: 0.25, 600: 0.25},
}


def report(name: str, y: np.ndarray, s: np.ndarray, folds, difficulty) -> None:
    per_fold = [average_precision_score(y[va], s[va]) for _, va in folds]
    thr = np.sort(s)[::-1][max(int(len(s) * 0.01) - 1, 0)]
    rec = {d: float((s[difficulty == d] >= thr).mean()) for d in ("easy", "medium", "hard")}
    print(f"  {name:22s} AP {100*average_precision_score(y, s):5.1f}  "
          f"folds {100*np.mean(per_fold):5.1f} +/- {100*np.std(per_fold):4.1f}   "
          f"top1% recall easy {rec['easy']:.2f} medium {rec['medium']:.2f} hard {rec['hard']:.2f}")


def cross_validate(p, X, ids, labels):
    lab = labels.set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].reset_index()
    y = lab.is_suspicious.to_numpy()
    groups = groups_for(lab)
    difficulty = lab.difficulty.to_numpy()
    folds = list(GroupKFold(5).split(X, y, groups=groups))

    oof = np.zeros(len(X))
    importance = np.zeros(X.shape[1])
    for tr, va in folds:
        model = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        importance += model.booster_.feature_importance("gain")

    print(f"{len(X):,} payments, {y.sum():,} suspicious, {X.shape[1]} features")
    best_name, best_ap = None, -1.0
    for name, w in SMOOTHERS.items():
        s = smooth(oof, ids, w)
        report(name, y, s, folds, difficulty)
        ap = average_precision_score(y, s)
        if ap > best_ap:
            best_name, best_ap = name, ap

    top = pd.Series(importance, index=X.columns).sort_values(ascending=False)
    print("\ntop 20 features by gain:")
    print("  " + ", ".join(top.head(20).index))
    print(f"\nbest smoothing: {best_name} ({100*best_ap:.1f})")
    return best_name, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-only", action="store_true")
    ap.add_argument("--smoothing", default=None, help="name from SMOOTHERS; default: best in CV")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    p, X, ids = build_features(DATA / "train", cache=CACHE)
    labels = load_labels(DATA / "train")
    best, _ = cross_validate(p, X, ids, labels)
    if args.cv_only:
        return

    choice = args.smoothing or best
    y = labels.set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].is_suspicious.to_numpy()
    model = lgb.LGBMClassifier(**PARAMS).fit(X, y)

    pt, Xt, idt = build_features(DATA / "test-data", cache=CACHE)
    raw = model.predict_proba(Xt[X.columns])[:, 1]
    score = smooth(raw, idt, SMOOTHERS[choice])
    # rescale into [0, 1] keeping the order; break exact ties with the raw score
    score = score + 1e-9 * raw
    score = (score - score.min()) / (score.max() - score.min())
    pd.DataFrame({"TRANSACTION_HASH": pt.TRANSACTION_HASH, "score": score}).to_csv(args.out, index=False)
    print(f"wrote {len(pt):,} test scores to {args.out} using '{choice}'")


if __name__ == "__main__":
    main()
