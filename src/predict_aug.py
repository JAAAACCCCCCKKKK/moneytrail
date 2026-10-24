"""The 67.1 recipe trained on the augmented week (training week + scheme copies, tools/augment.py).

  python tools/augment.py --copies 2
  python -m src.predict_aug --cv       # honest test on the ORIGINAL rows, copies kept with their original
  python -m src.predict_aug            # outputs/predictions_aug.csv

Same three models and blend as src.predict_two_q; only the training rows change: every planted
scheme appears 1 + copies times, each copy with other wallets and at another time of the week, so
the models see each shape in several contexts. The honest test groups each copy with its original
(fold key = original scenario_id), fits on the other folds' originals and copies, and scores only
original rows, so the numbers compare directly with the 67.1 recipe's.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from main import SEED, build_features, make_model, OUTPUTS
from src.predict_shapes_v2 import MODULES, ONCE_MODULE, SIGNALS_MODULE
from src.predict_two_q import BLEND, NOT_SCHEME, rank01
from src.predict_union_signals import LEFT_OUT, violet_clues


def week_tables(week: str):
    """Jack's clues (with once-groups and signals) plus Violet's, for one week."""
    F = build_features(week, MODULES + [ONCE_MODULE, SIGNALS_MODULE])
    X = F.drop(columns=["TRANSACTION_HASH"] + list(LEFT_OUT)).astype(float).reset_index(drop=True)
    U2 = pd.concat([X, violet_clues(week, F.TRANSACTION_HASH)], axis=1)
    return F[["TRANSACTION_HASH"]].reset_index(drop=True), U2


def labels_for(week: str, hashes):
    lab = pd.read_csv(common.WEEKS[week] / "labels.csv")
    return pd.DataFrame({"TRANSACTION_HASH": hashes}).merge(lab, on="TRANSACTION_HASH", how="left")


def fit_three(U, U2, y, scheme):
    once = [c for c in U2.columns if c.startswith("once_")]
    in_scheme = scheme == 1
    return (make_model("lgbm").fit(U, y),
            make_model("lgbm").fit(U2, scheme),
            make_model("lgbm").fit(U2[in_scheme], y[in_scheme]))


def score_with(models, U, U2, pay, week):
    m1, m2, m3 = models
    single = graph.smooth_by_group(m1.predict_proba(U)[:, 1], pay, week)
    return (1 - BLEND) * rank01(single) + BLEND * rank01(m2.predict_proba(U2)[:, 1] * m3.predict_proba(U2)[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--out", default="predictions_aug.csv")
    parser.add_argument("--weeks", nargs="+", default=["aug"],
                        help="training weeks to stack, e.g. train aug2 (aug2 = copies in a separate week, originals untouched)")
    args = parser.parse_args()
    t0 = time.time()
    parts = [week_tables(w) for w in args.weeks]
    ids = pd.concat([i.assign(week=w) for (i, _), w in zip(parts, args.weeks)], ignore_index=True)
    U2 = pd.concat([u for _, u in parts], ignore_index=True)[parts[0][1].columns]
    U = U2.drop(columns=[c for c in U2.columns if c.startswith("once_")])
    lab = pd.concat([labels_for(w, i.TRANSACTION_HASH) for (i, _), w in zip(parts, args.weeks)], ignore_index=True)
    y = lab.is_suspicious.to_numpy().astype(int)
    scheme = (~lab.family.isin(NOT_SCHEME)).to_numpy().astype(int)
    pay = pd.concat([i[["TRANSACTION_HASH"]].merge(common.payments(w), on="TRANSACTION_HASH") for (i, _), w in zip(parts, args.weeks)], ignore_index=True)
    original = (ids.week == args.weeks[0]).to_numpy() & ~lab.scenario_id.astype(str).str.startswith("aug:").to_numpy()
    print(f"augmented week: {len(y):,} rows ({original.sum():,} original), {U2.shape[1]} clues ({time.time() - t0:.0f}s)", flush=True)

    if args.cv:
        root = lab.scenario_id.astype(str).str.replace(r"^aug:\d+:", "", regex=True)
        groups = np.where(lab.scenario_id.notna(), "s:" + root, "w:" + pay.FROM_ADDRESS)
        med = (lab.difficulty == "medium").to_numpy() & original
        easy = (lab.difficulty == "easy").to_numpy() & original
        summary = {}
        for fs in (SEED, 7, 2026):
            # folds defined on the original rows only (same keys as main.py), copies follow their original
            orig_idx = np.flatnonzero(original)
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(U.iloc[orig_idx], y[orig_idx], groups[orig_idx]))
            res = {}
            for name, use_copies in (("recipe (originals only)", False), ("recipe + copies", True)):
                s = np.full(len(y), np.nan)
                for tr_o, va_o in folds:
                    va = orig_idx[va_o]
                    held = set(groups[va])
                    tr = np.flatnonzero((original if not use_copies else np.ones(len(y), bool)) & ~pd.Series(groups).isin(held).to_numpy())
                    models = fit_three(U.iloc[tr], U2.iloc[tr], y[tr], scheme[tr])
                    s[va] = score_with(models, U.iloc[orig_idx], U2.iloc[orig_idx], pay.iloc[orig_idx].reset_index(drop=True), "train")[va_o]
                res[name] = s
                so = s[orig_idx]; yo = y[orig_idx]
                per = [100 * aps(yo[va_o], so[va_o]) for _, va_o in folds]
                r = pd.Series(so).rank(ascending=False).to_numpy()
                summary.setdefault(name, []).append((100 * aps(yo, so), np.mean(per), per))
                print(f"  seed {fs} {name:26s} pooled {100 * aps(yo, so):5.1f} folds {np.mean(per):.1f} "
                      f"| med med-rank {np.median(r[med[orig_idx]]):.0f} | top400 prec {np.mean(yo[np.argsort(-so)[:400]]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        print("\nSUMMARY (3 seeds: pooled, fold mean, folds better of 15)")
        base = summary["recipe (originals only)"]
        for k, v in summary.items():
            better = sum(sum(a > b for a, b in zip(x[2], y_[2])) for x, y_ in zip(v, base))
            print(f"  {k:26s} {np.mean([x[0] for x in v]):5.1f} {np.mean([x[1] for x in v]):5.1f} {better:2d}/15")
        return

    ids_t, Ut2 = week_tables("test")
    Ut = Ut2.drop(columns=[c for c in Ut2.columns if c.startswith("once_")])[U.columns]
    Ut2 = Ut2[U2.columns]
    test_pay = ids_t.merge(common.payments("test"), on="TRANSACTION_HASH")
    blended = score_with(fit_three(U, U2, y, scheme), Ut, Ut2, test_pay, "test")
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": ids_t.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(score):,} test scores to {OUTPUTS / args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
