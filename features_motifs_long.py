"""Long-window motif clues: wallet groups at 30 minutes, 1 hour and 2 hours.

Owner: Jack. Same clue-file format as the others. Alim's test at 12:25 (motif
features at 1,800s, 3,600s and 7,200s on top of run 2's clues) gained 1.7 on the
`src.train` honest test, better in 4 of 5 folds. Hubs are set aside when linking.
Only used when `src.predict_shapes_v2` is run with --long-groups.
"""
from __future__ import annotations

import pandas as pd

import common
import graph
from src import motifs as motif_lib
from src.data import load_payments as load_rich

WINDOWS = (1800, 3600, 7200)
CACHE = common.DATA / "cache"
VERSION = "v1"


def add_features(payments: pd.DataFrame, week: str) -> pd.DataFrame:
    cache = CACHE / f"motifs_long_{VERSION}_{week}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    p = payments.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    hub_set = graph.hubs(p, week)
    rich = load_rich(common.WEEKS[week]).set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].reset_index()
    as_team = rich.rename(columns={"src": "FROM_ADDRESS", "dst": "TO_ADDRESS", "ts": "TIMESTAMP"})
    frames = [p[["TRANSACTION_HASH"]]]
    for w in WINDOWS:
        g = graph.wallet_groups(as_team, hub_set, w)
        described = motif_lib.describe(rich, g, prefix=f"long_g{w}_")
        frames.append(described.loc[g].reset_index(drop=True))
    out = pd.concat(frames, axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


if __name__ == "__main__":
    import time
    for wk in ("train", "test"):
        t0 = time.time()
        f = add_features(common.payments(wk), wk)
        print(f"{wk}: {f.shape[1] - 1} long-window clues, {time.time() - t0:.0f}s")
