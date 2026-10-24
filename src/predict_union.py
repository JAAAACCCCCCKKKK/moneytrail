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
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import common
import graph
from main import make_model, OUTPUTS
from src.predict_shapes_v2 import prepare

SHAPES = ["bipartite", "chain", "cycle", "fan_in", "fan_out", "fan_out_in", "multi_stage", "random", "stack"]

sys.path.insert(0, str(common.ROOT / "detector"))
from base import load                                  # noqa: E402  (Violet's loader)
from predict import features as violet_features        # noqa: E402


def violet_clues(week, hashes):
    """Violet's clues for one week, in the given transaction order."""
    F = violet_features(load(common.WEEKS[week])[0]).set_index("TRANSACTION_HASH").loc[hashes]
    F.columns = ["v_" + c for c in F.columns]
    return F.reset_index(drop=True).astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["suspicious", "structure"], default="suspicious",
                        help="structure: learn 'part of a planted shape scheme, guilty or not'")
    parser.add_argument("--out", default=None, help="file name inside outputs/")
    args = parser.parse_args()
    out_name = args.out or ("predictions_union.csv" if args.target == "suspicious" else "predictions_structure.csv")

    train, test, lab, X, Xt = prepare()
    U = pd.concat([X.reset_index(drop=True), violet_clues("train", train.TRANSACTION_HASH)], axis=1)
    Ut = pd.concat([Xt.reset_index(drop=True), violet_clues("test", test.TRANSACTION_HASH)], axis=1)[U.columns]
    print(f"clues: Jack {X.shape[1]}, Violet {U.shape[1] - X.shape[1]}, together {U.shape[1]}")

    if args.target == "structure":
        y = lab.family.isin(SHAPES).to_numpy().astype(int)
        final = make_model("lgbm").fit(U, y).predict_proba(Ut)[:, 1]          # no smoothing, as tested
    else:
        raw = make_model("lgbm").fit(U, lab.is_suspicious.to_numpy()).predict_proba(Ut)[:, 1]
        test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
        final = graph.smooth_by_group(raw, test_pay, "test")
    score = pd.Series(final).rank(method="first").to_numpy() / len(final)   # no ties

    out = pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score})
    OUTPUTS.mkdir(exist_ok=True)
    out.to_csv(OUTPUTS / out_name, index=False)
    print(f"wrote {len(out):,} {args.target} scores to {OUTPUTS / out_name}")


if __name__ == "__main__":
    main()
