"""One LightGBM on Jack's run 3 clues and Violet's detector clues together, then group smoothing.

  python3 -m src.predict_union        # writes outputs/predictions_union.csv

Jack's clues come from src.predict_shapes_v2.prepare() (run 3 defaults, 187 clues); Violet's
from detector/ (225 clues, prefixed v_). Neither file is changed. Same LightGBM settings and
group smoothing as run 3.

Honest test (main.py folds): 60.0 (folds 60.4 +/- 5.3), against run 3 56.8 and Violet v2 57.4.
Blended 50/50 with Violet v2 (src.blend): 60.6 (folds 60.7 +/- 5.9), better than our best
upload so far (30% run 3 + 70% Violet, 58.4) in 5 of 5 folds.

  python3 -m src.predict_union --target structure    # writes outputs/predictions_structure.csv

The structure model answers a different question on the same clues: is this payment part of a
planted shape scheme, laundering OR innocent look-alike? (`family` from the training labels is
used only as the training target, never as a clue.) Medium schemes can't be told from their
look-alikes, but both can be told from relays and background, which outrank them. Mixed into
the blend at 0.4 (src.blend --structure): honest test 60.6 -> 62.5, better in 5 of 5 folds;
medium in the top 1,250 16% -> 23%.

  python3 -m src.predict_union --signals --probabilities --out predictions_union_signals_prob.csv
  python3 -m src.predict_union --target structure --signals --once-groups --probabilities --out predictions_scheme_prob.csv

Jack's clues added: his deck signals in both models, and his once-a-week-pair groups in the
structure model only (his scheme model in src.predict_two_q). Multiplied with src.blend --gate.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

import numpy as np
import pandas as pd

import common
import graph
from main import make_model, OUTPUTS
from src.predict_shapes_v2 import prepare, CONVERSION_KEYS
from src.predict_union_signals import LEFT_OUT

SHAPES = ["bipartite", "chain", "cycle", "fan_in", "fan_out", "fan_out_in", "multi_stage", "random", "stack"]

sys.path.insert(0, str(common.ROOT / "detector"))
from base import load                                  # noqa: E402  (Violet's loader)
from predict import features as violet_features        # noqa: E402


def violet_clues(week, hashes):
    """Violet's clues for one week, in the given transaction order. Cached in data/cache under a
    hash of detector/*.py, so any change to her code rebuilds them."""
    code = b"".join(f.read_bytes() for f in sorted((common.ROOT / "detector").glob("*.py")))
    path = common.DATA / "cache" / f"violet_{week}_{hashlib.sha256(code).hexdigest()[:12]}.parquet"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        violet_features(load(common.WEEKS[week])[0]).to_parquet(path)
    F = pd.read_parquet(path).set_index("TRANSACTION_HASH").loc[hashes]
    F.columns = ["v_" + c for c in F.columns]
    return F.reset_index(drop=True).astype(float)


def make_xgb(seed):
    """XGBoost with settings close to the LightGBM ones in main.make_model."""
    import xgboost as xgb
    return xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=5,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             random_state=seed, n_jobs=-1, verbosity=0)


def fit_seeds(X, y, X_score, n_seeds, use_xgb=False):
    """Average LightGBM probabilities over seeds 42, 43, ... (n_seeds=1 is the original single model).

    With use_xgb, also average XGBoost over the same seeds and give the two model kinds equal weight.
    """
    p = np.zeros(len(X_score))
    for k in range(n_seeds):
        p += make_model("lgbm", seed=42 + k).fit(X, y).predict_proba(X_score)[:, 1]
    p /= n_seeds
    if not use_xgb:
        return p
    q = np.zeros(len(X_score))
    for k in range(n_seeds):
        q += make_xgb(42 + k).fit(X, y).predict_proba(X_score)[:, 1]
    return (p + q / n_seeds) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["suspicious", "structure"], default="suspicious",
                        help="structure: learn 'part of a planted shape scheme, guilty or not'")
    parser.add_argument("--self-train-from", default=None,
                        help="file inside outputs/ whose test ranking gives pseudo-labels for one round of "
                             "self-training: its top --pseudo-top share as suspicious, bottom half as innocent")
    parser.add_argument("--pseudo-top", type=float, default=0.02)
    parser.add_argument("--pseudo-weight", type=float, default=0.5)
    parser.add_argument("--no-conversion", action="store_true",
                        help="drop the 25 conversion clues (13 Jack's, 12 Violet's): the one clear train-to-test "
                             "shift (41%% to 51%% conversions), and the README says asset choices ignore the labels")
    parser.add_argument("--probabilities", action="store_true",
                        help="write the model's probabilities instead of ranks, for src.blend --gate "
                             "(an intermediate file, never uploaded: probabilities can tie)")
    parser.add_argument("--signals", action="store_true",
                        help="add Jack's deck signals (features_shape_signals.py) without the loop-asset clue, "
                             "as src.predict_union_signals")
    parser.add_argument("--once-groups", action="store_true",
                        help="add Jack's once-a-week-pair group clues (features_motifs_once.py)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="average this many LightGBM seeds (42, 43, ...); 1 = the original single model")
    parser.add_argument("--xgb", action="store_true",
                        help="also train XGBoost (same seeds) and average it 50/50 with LightGBM")
    parser.add_argument("--out", default=None, help="file name inside outputs/")
    args = parser.parse_args()
    out_name = args.out or ("predictions_union.csv" if args.target == "suspicious" else "predictions_structure.csv")

    train, test, lab, X, Xt = prepare(once_groups=args.once_groups, signals=args.signals)
    if args.signals:
        X, Xt = X.drop(columns=list(LEFT_OUT)), Xt.drop(columns=list(LEFT_OUT))
    U = pd.concat([X.reset_index(drop=True), violet_clues("train", train.TRANSACTION_HASH)], axis=1)
    Ut = pd.concat([Xt.reset_index(drop=True), violet_clues("test", test.TRANSACTION_HASH)], axis=1)[U.columns]
    if args.no_conversion:
        drop = [c for c in U.columns if (not c.startswith("v_") and any(k in c for k in CONVERSION_KEYS))
                or (c.startswith("v_") and any(k in c.lower() for k in ("conv", "legs", "asset", "sm_slack")))]
        U, Ut = U.drop(columns=drop), Ut.drop(columns=drop)
    n_violet = sum(c.startswith("v_") for c in U.columns)
    print(f"clues: Jack {U.shape[1] - n_violet}, Violet {n_violet}, together {U.shape[1]}")

    if args.target == "structure":
        y = lab.family.isin(SHAPES).to_numpy().astype(int)
        final = fit_seeds(U, y, Ut, args.seeds, args.xgb)                               # no smoothing, as tested
    elif args.self_train_from:
        # one round of self-training on the test week: uses only our own earlier test scores, never test answers
        prior = test[["TRANSACTION_HASH"]].merge(pd.read_csv(OUTPUTS / args.self_train_from), on="TRANSACTION_HASH").score
        pct = prior.rank(ascending=False, pct=True).to_numpy()
        hi, lo = np.flatnonzero(pct <= args.pseudo_top), np.flatnonzero(pct > 0.5)
        Xs = pd.concat([U, Ut.iloc[hi], Ut.iloc[lo]], ignore_index=True)
        ys = np.r_[lab.is_suspicious.to_numpy(), np.ones(len(hi)), np.zeros(len(lo))]
        ws = np.r_[np.ones(len(U)), np.full(len(hi) + len(lo), args.pseudo_weight)]
        raw = make_model("lgbm").fit(Xs, ys, sample_weight=ws).predict_proba(Ut)[:, 1]
        test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
        final = graph.smooth_by_group(raw, test_pay, "test")
    else:
        raw = fit_seeds(U, lab.is_suspicious.to_numpy(), Ut, args.seeds, args.xgb)
        test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
        final = graph.smooth_by_group(raw, test_pay, "test")
    if args.probabilities:
        score = final
    else:
        score = pd.Series(final).rank(method="first").to_numpy() / len(final)   # no ties

    out = pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score})
    OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(OUTPUTS / out_name, index=False)
    print(f"wrote {len(out):,} {args.target} scores to {OUTPUTS / out_name}")


if __name__ == "__main__":
    main()
