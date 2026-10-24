"""MoneyTrail: load the payments, add everyone's clues, test honestly, train, predict.

Usage:
  python3 main.py                  # honest test with LightGBM, then outputs/predictions.csv
  python3 main.py --model logreg   # the starter notebook's model, for comparison
  python3 main.py --no-predict     # honest test only

Clue files are picked up automatically if they exist: features_starter.py,
features_wallet.py (Natasha), features_shapes.py (Jack), features_xrpl.py (Violet).
Each one provides add_features(payments, week) returning TRANSACTION_HASH plus new
columns, one row per transaction.
"""
import argparse
import importlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common

SEED = 42
FEATURE_MODULES = ["features_starter", "features_wallet", "features_shapes", "features_xrpl"]
OUTPUTS = common.ROOT / "outputs"


def build_features(week, modules):
    """Every available clue file's columns for one week, joined on TRANSACTION_HASH."""
    payments = common.payments(week)
    table = payments[["TRANSACTION_HASH"]].copy()
    for name in modules:
        started = time.time()
        part = importlib.import_module(name).add_features(payments, week)
        if part.TRANSACTION_HASH.duplicated().any() or len(part) != len(payments):
            raise ValueError(f"{name} must return exactly one row per transaction ({len(part)} vs {len(payments)})")
        table = table.merge(part, on="TRANSACTION_HASH", how="left")
        print(f"  {name:<18} {part.shape[1] - 1:>3} clues  ({time.time() - started:.0f}s, {week})")
    return table


def make_model(kind, seed=SEED):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, n_jobs=-1, verbose=-1)


def fit_predict(kind, X_fit, y_fit, X_score):
    X_fit, X_score = X_fit.astype(float), X_score.astype(float)
    if kind == "logreg":  # logistic regression can't handle blanks; LightGBM can
        X_fit, X_score = X_fit.fillna(0), X_score.fillna(0)
    return make_model(kind).fit(X_fit, y_fit).predict_proba(X_score)[:, 1]


def honest_test(kind, X, y, groups, info):
    """5 parts, each planted scheme (scenario_id) kept on one side. Normal payments
    without a scenario are kept together by sender."""
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof, scores = np.zeros(len(y)), []
    for fit_idx, check_idx in folds.split(X, y, groups):
        oof[check_idx] = fit_predict(kind, X.iloc[fit_idx], y[fit_idx], X.iloc[check_idx])
        scores.append(100 * average_precision_score(y[check_idx], oof[check_idx]))
    overall = 100 * average_precision_score(y, oof)
    print(f"\nHONEST TEST ({kind}): {np.mean(scores):.1f} ± {np.std(scores):.1f} across folds "
          f"[{', '.join(f'{s:.1f}' for s in scores)}], all folds together {overall:.1f}")

    top = pd.Series(oof).rank(ascending=False) <= max(1, int(len(oof) * 0.01))
    caught = {}
    for difficulty in ["easy", "medium", "hard"]:
        mask = (info.difficulty == difficulty).to_numpy() & (y == 1)
        caught[difficulty] = float(top[mask].mean())
    ambiguous = (info.family == "ambiguous").to_numpy()
    caught["observable hard"] = float(top[(info.difficulty == "hard").to_numpy() & (y == 1) & ~ambiguous].mean())
    print("Caught in the top 1%: " + ", ".join(f"{k} {v:.0%}" for k, v in caught.items()))
    return dict(fold_scores=scores, mean=float(np.mean(scores)), spread=float(np.std(scores)),
                overall=overall, caught_top1=caught)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "logreg"], default="lgbm")
    parser.add_argument("--no-predict", action="store_true")
    args = parser.parse_args()

    modules = [m for m in FEATURE_MODULES if (common.ROOT / f"{m}.py").exists()]
    print("Clue files found:", ", ".join(modules))
    train = build_features("train", modules)
    info = train[["TRANSACTION_HASH"]].merge(common.labels(), on="TRANSACTION_HASH", how="left")
    y = info.is_suspicious.to_numpy()
    sender = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH").FROM_ADDRESS
    groups = np.where(info.scenario_id.notna(), "s:" + info.scenario_id.astype(str), "w:" + sender)
    X = train.drop(columns="TRANSACTION_HASH")
    print(f"{len(X):,} training transactions, {int(y.sum()):,} suspicious, {X.shape[1]} clues")

    result = honest_test(args.model, X, y, groups, info)
    result.update(model=args.model, clue_files=modules, clues=list(X.columns))

    if not args.no_predict:
        test = build_features("test", modules)
        raw = fit_predict(args.model, X, y, test.drop(columns="TRANSACTION_HASH")[X.columns])
        # Ranks keep the order but break any ties, which average precision penalises.
        score = pd.Series(raw).rank(method="first").to_numpy() / len(raw)
        everyone = common.payments("test")[["TRANSACTION_HASH"]]
        predictions = everyone.merge(pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}),
                                     on="TRANSACTION_HASH", how="left")
        if predictions.score.isna().any():
            raise ValueError(f"{predictions.score.isna().sum()} test transactions have no score")
        OUTPUTS.mkdir(exist_ok=True)
        predictions.to_csv(OUTPUTS / "predictions.csv", index=False)
        print(f"\nSaved {len(predictions):,} predictions to outputs/predictions.csv")

    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "last_run.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
