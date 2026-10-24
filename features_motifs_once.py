"""Slow-scheme clues: wallet groups linked only through once-a-week payment pairs.

Owner: Jack. Same clue-file format as the others.

Why: medium schemes are the same shapes as easy ones run slowly (a median of 18
minutes between payments over about 2 hours). Plain wallet groups need a 30-minute
window to hold them, and at that width busy wallets chain 57% of all payments into
groups of 100 or more. But 95% of medium payments (97% of easy) belong to a sender-
receiver pair that occurs only once in the week, against 34% of innocent payments,
whose relationships recur. Linking only through those once-a-week payments, at one
hour, keeps every medium scheme together (median group of 17) while only 12% of
payments fall into groups of 100 or more. Payments whose pair recurs get blanks.
Hubs are set aside. Used when `src.predict_shapes_v2` is run with --once-groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common
import graph
from src import motifs as motif_lib
from src.data import load_payments as load_rich

WINDOWS = (1800, 3600)
CACHE = common.DATA / "cache"
VERSION = "v1"


def add_features(payments: pd.DataFrame, week: str) -> pd.DataFrame:
    cache = CACHE / f"motifs_once_{VERSION}_{week}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    p = payments.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    hub_set = graph.hubs(p, week)
    pairs = pd.Series(list(zip(p.FROM_ADDRESS, p.TO_ADDRESS)))
    once = (pairs.map(pairs.value_counts()) == 1).to_numpy()

    rich = load_rich(common.WEEKS[week]).set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].reset_index()
    sub = rich[once].reset_index(drop=True)
    as_team = sub.rename(columns={"src": "FROM_ADDRESS", "dst": "TO_ADDRESS", "ts": "TIMESTAMP"})
    frames = [p[["TRANSACTION_HASH"]], pd.DataFrame({"once_pair": once.astype(float)})]
    rows = np.flatnonzero(once)
    for w in WINDOWS:
        g = graph.wallet_groups(as_team, hub_set, w)
        described = motif_lib.describe(sub, g, prefix=f"once_g{w}_")
        block = pd.DataFrame(np.nan, index=range(len(p)), columns=described.columns)
        block.iloc[rows] = described.loc[g].to_numpy()
        frames.append(block)
    out = pd.concat(frames, axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


if __name__ == "__main__":
    import time
    for wk in ("train", "test"):
        t0 = time.time()
        f = add_features(common.payments(wk), wk)
        print(f"{wk}: {f.shape[1] - 1} once-pair clues, {time.time() - t0:.0f}s, "
              f"once-a-week pairs {f.once_pair.mean():.2f}")
