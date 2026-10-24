"""MoneyTrail detector: train on the labelled week, score the test week.

  python3 predict.py --train path/to/train --test path/to/test-data --out predictions.csv

Deterministic (fixed seeds). About 10 minutes on a laptop, under 4 GB RAM.
"""
import argparse, time, warnings
import pandas as pd
from base import load
from model import build, train_and_score
from clusters import cluster_features
warnings.filterwarnings("ignore")

def features(base):
    return build(base).merge(cluster_features(base), on="TRANSACTION_HASH")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train"); ap.add_argument("--test", default="test-data")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args(); t = time.time()
    base_tr, _ = load(a.train); base_te, _ = load(a.test)
    lab = pd.read_csv(f"{a.train}/labels.csv")
    F_tr = features(base_tr); F_te = features(base_te)
    print(f"features: {F_tr.shape[1]-1} clues ({time.time()-t:.0f}s)")
    pred = train_and_score(base_tr, F_tr, lab, base_te, F_te, seeds=(0, 1, 2))
    pred.to_csv(a.out, index=False)
    print(f"wrote {len(pred):,} scores to {a.out} ({time.time()-t:.0f}s)")
