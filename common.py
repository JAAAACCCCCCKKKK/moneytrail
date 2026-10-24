"""Shared helpers: where each week's data lives, and loading it the same way for everyone."""
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
WEEKS = {"train": DATA / "train", "validation": DATA / "validation", "test": DATA / "test-data",
         "aug": DATA / "aug", "aug2": DATA / "aug2"}   # the training week plus scheme copies, from tools/augment.py

sys.path.insert(0, str(DATA))  # the organisers' payments.py lives in data/
from payments import load_payments  # noqa: E402


@lru_cache(maxsize=None)
def payments(week):
    """One row per transaction: TRANSACTION_HASH, TIMESTAMP (seconds), FROM_ADDRESS,
    TO_ADDRESS and VALUE (delivered value in US dollars), from the organisers' payments.py."""
    df = pd.DataFrame(load_payments(WEEKS[week]))
    df["TIMESTAMP"] = df.TIMESTAMP.astype("int64")
    return df.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def raw(week, table):
    """A raw organiser file for one week: "transactions", "transfers" or "blocks"."""
    return pd.read_parquet(WEEKS[week] / f"{table}.parquet")


@lru_cache(maxsize=None)
def labels():
    """Training labels. is_suspicious is the answer; the other columns are only for
    splitting and reporting, never for clues."""
    return pd.read_csv(WEEKS["train"] / "labels.csv")
