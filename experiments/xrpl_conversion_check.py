"""Violet's XRPL check: does Jack's model need the conversion clues?

Conversions are 41% of training payments but 51% of test payments, so clues built
on them could behave differently on the test week. This runs the same honest test
as `python3 -m src.train --cv-only`, once with every clue and once without the
conversion clues, so we can see what they're worth.

  python3 -m experiments.xrpl_conversion_check
"""
from src.data import load_labels
from src.features import build_features
from src.train import CACHE, DATA, cross_validate

CONVERSION_CLUES = ["is_conversion", "is_bridged", "conversion_spread", "n_legs"]

p, X, ids = build_features(DATA / "train", cache=CACHE)
labels = load_labels(DATA / "train")
dropped = CONVERSION_CLUES + [c for c in X.columns
                              if c.endswith(("conversion_share", "bridged_share", "n_assets"))]
dropped = [c for c in dropped if c in X.columns]

print("=== WITH every clue ===")
cross_validate(p, X, ids, labels)
print(f"\n=== WITHOUT the {len(dropped)} conversion clues: {', '.join(dropped)} ===")
cross_validate(p, X.drop(columns=dropped), ids, labels)
