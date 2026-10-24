"""Honest test: 5-fold StratifiedGroupKFold grouped by scenario_id, both stages on the same folds.
  python3 run_cv.py --train path/to/train
"""
import argparse, warnings, pandas as pd
from base import load
from predict import features
from model import cross_validate
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--train", default="train"); a = ap.parse_args()
base, _ = load(a.train); lab = pd.read_csv(f"{a.train}/labels.csv")
cross_validate(base, features(base), lab)
