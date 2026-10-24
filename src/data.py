"""Load a split into one USD-valued payment edge per transaction.

Traps handled here, per the organisers' README:
  * one row per TRANSACTION_HASH, never one per transfer leg;
  * XRP amounts are drops, 1 XRP = 1,000,000 drops;
  * an asset is TOKEN_ADDRESS plus SYMBOL, so USD from two issuers is two assets;
  * DELIVERED_AMOUNT is what arrived; SendMax is only a ceiling;
  * EXTRA_TRANSACTION_FIELDS and DELIVERED_AMOUNT are JSON text;
  * values are compared in USD using the hour that contains the payment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

XRP_SENTINEL = "0x1111222211112222111122221111222211112222"


def _decode(wire) -> tuple[str, float]:
    """Return (asset key, amount in whole units) for an XRPL amount field."""
    if isinstance(wire, str):              # XRP, a string of integer drops
        return "XRP", int(wire) / 1_000_000
    return f"{wire['issuer']}:{wire['currency']}", float(wire["value"])


def load_rates(directory: Path) -> dict[tuple[int, str], int]:
    rates = pd.read_csv(directory / "market_rates.csv")
    hours = rates.TIMESTAMP.astype("int64").to_numpy()
    return dict(zip(zip(hours, rates.ASSET), rates.PRICE_MICRO_USD.astype("int64")))


def load_payments(directory: str | Path) -> pd.DataFrame:
    """One row per transaction: who paid whom, what arrived, when, valued in USD."""
    directory = Path(directory)
    tx = pd.read_parquet(directory / "transactions.parquet")
    rates = load_rates(directory)

    extra = [json.loads(s) for s in tx.EXTRA_TRANSACTION_FIELDS]
    delivered = [_decode(json.loads(s)) for s in tx.DELIVERED_AMOUNT]
    ts = tx.TIMESTAMP.astype("int64").to_numpy()
    asset = np.array([d[0] for d in delivered])
    units = np.array([d[1] for d in delivered])

    price = np.array([rates[(t // 3600 * 3600, a)] for t, a in zip(ts, asset)], dtype="float64")
    out = pd.DataFrame({
        "TRANSACTION_HASH": tx.TRANSACTION_HASH.to_numpy(),
        "ts": ts,
        "ledger": tx.LEDGER_INDEX.astype("int64").to_numpy(),
        "src": tx.ACCOUNT.to_numpy(),
        "dst": np.array([e.get("Destination") for e in extra]),
        "asset": asset,
        "units": units,
        "usd": units * price / 1_000_000,
        "fee_drops": tx.SOURCE_FEE.astype("int64").to_numpy(),
        # a conversion carries SendMax; an XRP-bridged conversion also carries Paths
        "is_conversion": np.array([("SendMax" in e) for e in extra]),
        "is_bridged": np.array([("Paths" in e) for e in extra]),
    })
    return out.sort_values("ts").reset_index(drop=True)


def load_legs(directory: str | Path) -> pd.DataFrame:
    """The executed transfer legs. Leg 0 is what the sender actually paid."""
    legs = pd.read_parquet(Path(directory) / "transfers.parquet")
    legs["TRANSFER_INDEX"] = legs.TRANSFER_INDEX.astype("int64")
    return legs


def leg_features(directory: str | Path) -> pd.DataFrame:
    """Per-transaction facts that only the legs can tell us, keyed by hash.

    The first leg is the real source debit, so comparing it with the delivered
    amount gives the spread the sender paid to convert.
    """
    legs = load_legs(directory)
    first = legs.sort_values("TRANSFER_INDEX").groupby("TRANSACTION_HASH").first()
    counts = legs.groupby("TRANSACTION_HASH").size().rename("n_legs")
    out = pd.DataFrame({
        "n_legs": counts,
        "first_leg_value": first.VALUE,
        "first_leg_asset": first.TOKEN_ADDRESS + "|" + first.SYMBOL,
    })
    return out.reset_index()


def load_labels(directory: str | Path) -> pd.DataFrame | None:
    path = Path(directory) / "labels.csv"
    return pd.read_csv(path) if path.exists() else None
