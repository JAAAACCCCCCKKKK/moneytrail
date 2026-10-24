"""Compare the starter notebook's own held-back score with an honest, grouped score.

The notebook splits rows at random. Labels are pure per scenario_id, so members of
one planted motif land on both sides of that split and the score is inflated.
This script scores the same feature set three ways on train/ only.

Run:  python src/honest_eval.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notebook_features import make_features  # noqa: E402
from payments import load_payments  # noqa: E402

TRAIN = Path("data/train")


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def main():
    df = pd.DataFrame(load_payments(TRAIN))
    lab = pd.read_csv(TRAIN / "labels.csv")
    df = df.merge(lab, on="TRANSACTION_HASH")
    feats = make_features(df)
    y = df.set_index("TRANSACTION_HASH").is_suspicious.loc[feats.TRANSACTION_HASH].to_numpy()
    meta = df.set_index("TRANSACTION_HASH").loc[feats.TRANSACTION_HASH]
    X = feats.drop(columns="TRANSACTION_HASH")
    groups = np.where(meta.scenario_id.notna(), meta.scenario_id.astype(str),
                      "bg_" + np.arange(len(meta)).astype(str))

    print(f"{len(X):,} rows, {y.sum():,} suspicious ({y.mean():.3%}); random guessing scores {100*y.mean():.1f}")

    xf, xc, yf, yc = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    m = model().fit(xf, yf)
    print(f"notebook's own split (leaky)      {100*average_precision_score(yc, m.predict_proba(xc)[:,1]):5.1f}")

    for name, cv, g in [("stratified 5-fold (still leaky)", StratifiedKFold(5, shuffle=True, random_state=0), None),
                        ("grouped 5-fold by scenario", GroupKFold(5), groups)]:
        oof = np.zeros(len(X)); fold = []
        for tr, va in cv.split(X, y, groups=g):
            m = model().fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            fold.append(average_precision_score(y[va], oof[va]))
        print(f"{name:34s}{100*average_precision_score(y, oof):5.1f}   "
              f"per fold {100*np.mean(fold):.1f} +/- {100*np.std(fold):.1f}")

    oof = np.zeros(len(X))
    for tr, va in GroupKFold(5).split(X, y, groups=groups):
        oof[va] = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    thr = np.quantile(oof, 0.99)
    print("\nrecall inside the top 1% of scores, by difficulty (grouped scores):")
    for d, grp in pd.Series(oof).groupby(meta.difficulty.to_numpy()):
        if d != "normal":
            print(f"  {d:7s} {(grp >= thr).mean():.3f}  ({len(grp)} transactions)")
    print("\nAP if a detector ranked every easy and medium case perfectly and missed every hard one:")
    ideal = np.zeros(len(y)); ideal[(y == 1) & (meta.difficulty != "hard").to_numpy()] = 1.0
    ideal += np.random.default_rng(0).uniform(0, 1e-6, len(y))
    print(f"  {100*average_precision_score(y, ideal):.1f}")


if __name__ == "__main__":
    main()
