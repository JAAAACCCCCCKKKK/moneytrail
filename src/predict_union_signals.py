"""Alim's combined model (src/predict_union.py) with Jack's deck signals added.

Identical to `python3 -m src.predict_union` (one LightGBM on Jack's run 3 clues plus
Violet's detector clues, then run 3's group smoothing), except that Jack's clues are
built with `prepare(signals=True)`, which adds features_shape_signals.py (the loop-asset
clue is left out: the README says asset choices ignore the labels). Neither teammate's
file changes.

Violet's clue tables take several minutes per week to build, so they are cached in
data/cache under a hash of detector/*.py: any change to her code rebuilds them.

  python -m src.predict_union_signals --build-week train    cache Violet's clues for one week
  python -m src.predict_union_signals --cv                  honest test (main.py folds), with and without
  python -m src.predict_union_signals                       outputs/predictions_union_signals.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from main import SEED, make_model, OUTPUTS
from src.predict_shapes_v2 import prepare

DETECTOR = common.ROOT / "detector"
sys.path.insert(0, str(DETECTOR))
from base import load                                  # noqa: E402  (Violet's loader)
from predict import features as violet_features        # noqa: E402  (Violet's clues)

warnings.filterwarnings("ignore")
LEFT_OUT = ("sig_loop_n_assets",)


def detector_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(DETECTOR.glob("*.py")):
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


def violet_table(week: str) -> pd.DataFrame:
    cache = common.DATA / "cache" / f"violet_clues_{detector_hash()}_{week}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t0 = time.time()
    F = violet_features(load(common.WEEKS[week])[0])
    cache.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(cache)
    print(f"  Violet's clues for {week}: {F.shape[1] - 1} ({time.time() - t0:.0f}s), cached", flush=True)
    return F


def violet_clues(week: str, hashes) -> pd.DataFrame:
    """As src.predict_union.violet_clues, from the cache."""
    F = violet_table(week).set_index("TRANSACTION_HASH").loc[hashes]
    F.columns = ["v_" + c for c in F.columns]
    return F.reset_index(drop=True).astype(float)


def union_tables(signals: bool):
    train, test, lab, X, Xt = prepare(signals=signals)
    if signals:
        X, Xt = X.drop(columns=list(LEFT_OUT)), Xt.drop(columns=list(LEFT_OUT))
    U = pd.concat([X.reset_index(drop=True), violet_clues("train", train.TRANSACTION_HASH)], axis=1)
    Ut = pd.concat([Xt.reset_index(drop=True), violet_clues("test", test.TRANSACTION_HASH)], axis=1)[U.columns]
    return train, test, lab, U, Ut


def cross_validate():
    results = {}
    for name, signals in (("combined model", False), ("combined model + signals", True)):
        t0 = time.time()
        train, _, lab, U, _ = union_tables(signals)
        y = lab.is_suspicious.to_numpy()
        pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
        groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + pay.FROM_ADDRESS)
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(U, y, groups))
        oof = np.zeros(len(y))
        for tr, va in folds:
            oof[va] = make_model("lgbm").fit(U.iloc[tr], y[tr]).predict_proba(U.iloc[va])[:, 1]
        s = graph.smooth_by_group(oof, pay, "train")
        per = [100 * aps(y[va], s[va]) for _, va in folds]
        top5 = s >= np.sort(s)[::-1][int(len(s) * 0.05)]
        recall = {d: float(top5[(lab.difficulty == d).to_numpy()].mean()) for d in ("easy", "medium")}
        results[name] = per
        print(f"  {name:26s} {100 * aps(y, s):5.1f} (folds {np.mean(per):.1f} +/- {np.std(per):.1f} "
              f"[{', '.join(f'{x:.1f}' for x in per)}])  top-5% recall easy {recall['easy']:.2f} "
              f"medium {recall['medium']:.2f}  ({U.shape[1]} clues, {time.time() - t0:.0f}s)", flush=True)
    a, b = results["combined model"], results["combined model + signals"]
    print(f"  gain {np.mean(b) - np.mean(a):+.1f}; folds better {sum(x > y_ for x, y_ in zip(b, a))}/5")


def predict(out: str):
    train, test, lab, U, Ut = union_tables(signals=True)
    raw = make_model("lgbm").fit(U, lab.is_suspicious.to_numpy()).predict_proba(Ut)[:, 1]
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    smoothed = graph.smooth_by_group(raw, test_pay, "test")
    score = pd.Series(smoothed).rank(method="first").to_numpy() / len(smoothed)   # no ties
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / out, index=False)
    print(f"wrote {len(score):,} test scores to {OUTPUTS / out} ({U.shape[1]} clues)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--build-week", choices=["train", "test"])
    parser.add_argument("--out", default="predictions_union_signals.csv")
    args = parser.parse_args()
    if args.build_week:
        violet_table(args.build_week)
    elif args.cv:
        cross_validate()
    else:
        predict(args.out)
