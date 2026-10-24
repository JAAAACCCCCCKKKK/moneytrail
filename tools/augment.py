"""Data augmentation: copies of the planted training schemes with new wallets and new times (Jack).

  python tools/augment.py --copies 2      # writes data/aug/ (the training week plus the copies)

Why this is label-preserving here: wallet identity carries no signal (wallets are reused at random
and differ between weeks), and a scheme's label depends on its internal timing and shape, which a
whole-scheme time shift leaves unchanged. So every planted shape scheme (suspicious or honest) is
copied --copies times: its wallets are replaced by other non-hub wallets of the week (one consistent
map per copy), the whole scheme is shifted to another moment of the same week, hashes are renewed,
and SEQUENCE is recomputed for every account as the generator does (1, 2, 3... in time order). The
copies are added to the full training week, so their context (busy wallets, hubs, background) is
real. labels.csv gets the copies with scenario_id "aug:<k>:<original>", so an honest test can keep
every copy on the same side as its original.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN, AUG = ROOT / "data" / "train", ROOT / "data" / "aug"
NOT_SCHEME = ("background", "aggregate_relay", "distributed_relay", "ambiguous")


def new_hash(h: str, k: int) -> str:
    return hashlib.sha256(f"{h}:aug:{k}".encode()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true",
                        help="write a second week: background and relays plus the copies, WITHOUT the original schemes, "
                             "so the original week stays clean (data/aug2/)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / "data" / ("aug2" if args.replace else "aug")

    tx = pd.read_parquet(TRAIN / "transactions.parquet")
    tr = pd.read_parquet(TRAIN / "transfers.parquet")
    lab = pd.read_csv(TRAIN / "labels.csv")
    ts = tx.TIMESTAMP.astype("int64")
    week_lo, week_hi = int(ts.min()), int(ts.max())
    ledger_per_sec = (tx.LEDGER_INDEX.astype("int64").max() - tx.LEDGER_INDEX.astype("int64").min()) / (week_hi - week_lo)

    dest = tx.EXTRA_TRANSACTION_FIELDS.map(lambda s: json.loads(s)["Destination"])
    ends = pd.concat([tx.ACCOUNT, dest])
    inner = set(tr.FROM_ADDRESS) | set(tr.TO_ADDRESS)
    hubs = inner - set(ends)                                   # liquidity providers never send or receive a payment
    pool = np.array(sorted(set(ends) - hubs))

    scheme = lab[~lab.family.isin(NOT_SCHEME)]
    tx_by_hash = tx.set_index("TRANSACTION_HASH")
    tr_by_hash = tr.groupby("TRANSACTION_HASH")
    new_tx, new_tr, new_lab = [], [], []
    for k in range(args.copies):
        for sid, rows in scheme.groupby("scenario_id"):
            t = tx_by_hash.loc[rows.TRANSACTION_HASH].reset_index()
            t_ts = t.TIMESTAMP.astype("int64")
            wallets = sorted(set(t.ACCOUNT) | set(dest.loc[tx_by_hash.index.get_indexer(rows.TRANSACTION_HASH)]))
            repl = rng.choice(pool[~np.isin(pool, wallets)], len(wallets), replace=False)
            wmap = dict(zip(wallets, repl))
            offset = int(rng.integers(week_lo - int(t_ts.min()), week_hi - int(t_ts.max()) + 1))
            legs = pd.concat([tr_by_hash.get_group(h) for h in t.TRANSACTION_HASH])
            t = t.copy()
            fields = []
            for s in t.EXTRA_TRANSACTION_FIELDS:
                f = json.loads(s)
                f["Destination"] = wmap.get(f["Destination"], f["Destination"])
                fields.append(json.dumps(f, separators=(",", ":")))
            t["EXTRA_TRANSACTION_FIELDS"] = fields
            t["ACCOUNT"] = t.ACCOUNT.map(lambda w: wmap.get(w, w))
            t["TIMESTAMP"] = (t_ts + offset).map(Decimal)
            t["DATETIME"] = pd.to_datetime((t_ts + offset) * 1000, unit="ms", utc=True)
            t["LEDGER_INDEX"] = (t.LEDGER_INDEX.astype("int64") + int(round(offset * ledger_per_sec))).map(Decimal)
            t["TRANSACTION_HASH"] = [new_hash(h, k) for h in t.TRANSACTION_HASH]
            legs = legs.copy()
            legs["FROM_ADDRESS"] = legs.FROM_ADDRESS.map(lambda w: wmap.get(w, w))
            legs["TO_ADDRESS"] = legs.TO_ADDRESS.map(lambda w: wmap.get(w, w))
            legs["TIMESTAMP"] = (legs.TIMESTAMP.astype("int64") + offset).map(Decimal)
            legs["DATETIME"] = pd.to_datetime((legs.TIMESTAMP.astype("int64")) * 1000, unit="ms", utc=True)
            legs["LEDGER_INDEX"] = (legs.LEDGER_INDEX.astype("int64") + int(round(offset * ledger_per_sec))).map(Decimal)
            legs["TRANSACTION_HASH"] = [new_hash(h, k) for h in legs.TRANSACTION_HASH]
            l = rows.copy()
            l["TRANSACTION_HASH"] = [new_hash(h, k) for h in l.TRANSACTION_HASH]
            l["scenario_id"] = f"aug:{k}:{sid}"
            new_tx.append(t); new_tr.append(legs); new_lab.append(l)

    if args.replace:
        keep = ~tx.TRANSACTION_HASH.isin(scheme.TRANSACTION_HASH)
        tx, tr, lab = tx[keep], tr[~tr.TRANSACTION_HASH.isin(scheme.TRANSACTION_HASH)], lab[~lab.TRANSACTION_HASH.isin(scheme.TRANSACTION_HASH)]
    tx_all = pd.concat([tx] + new_tx, ignore_index=True)
    tr_all = pd.concat([tr] + new_tr, ignore_index=True)
    lab_all = pd.concat([lab] + new_lab, ignore_index=True)
    # SEQUENCE as the generator makes it: per account, 1, 2, 3... in time order
    order = tx_all.TIMESTAMP.astype("int64")
    tx_all["SEQUENCE"] = tx_all.assign(_t=order).groupby("ACCOUNT")._t.rank(method="first").astype(int).map(Decimal)
    tx_all = tx_all.sort_values(["TIMESTAMP", "TRANSACTION_HASH"], key=lambda s: s.astype(str) if s.name == "TRANSACTION_HASH" else s.astype("int64")).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    tx_all.to_parquet(out_dir / "transactions.parquet", index=False)
    tr_all.to_parquet(out_dir / "transfers.parquet", index=False)
    lab_all.to_csv(out_dir / "labels.csv", index=False)
    (out_dir / "market_rates.csv").write_bytes((TRAIN / "market_rates.csv").read_bytes())
    print(f"augmented week: {len(tx):,} + {len(tx_all) - len(tx):,} copied payments from {scheme.scenario_id.nunique()} schemes x {args.copies}; "
          f"{len(lab_all[lab_all.is_suspicious == 1]):,} suspicious rows; wrote {out_dir}")


if __name__ == "__main__":
    main()
