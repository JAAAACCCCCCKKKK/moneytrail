"""Public, label-free payment projection for the synthetic FX benchmark.

Reference prices are fictional contemporaneous valuations, not executable quotes.
Executed transfers retain the actual assets, amounts and market-maker inventory legs.
"""
import csv
import json
from decimal import Decimal
from pathlib import Path
import pyarrow.parquet as pq

XRP = "0x1111222211112222111122221111222211112222"
SCALE = 1_000_000


def load_rates(directory):
    with (Path(directory) / "market_rates.csv").open() as stream:
        return {(int(r["TIMESTAMP"]), r["ASSET"]): int(r["PRICE_MICRO_USD"])
                for r in csv.DictReader(stream)}


def price(rates, timestamp, asset):
    return rates[int(timestamp) // 3600 * 3600, asset]


def decode_amount(wire):
    if isinstance(wire, str):
        return "XRP", int(wire)
    return f"{wire['issuer']}:{wire['currency']}", int(Decimal(wire["value"]) * SCALE)


def project(transactions, rates):
    """One effective customer edge per payment, valued in synthetic reference USD."""
    result = []
    for tx in transactions:
        fields = json.loads(tx["EXTRA_TRANSACTION_FIELDS"])
        asset, units = decode_amount(json.loads(tx["DELIVERED_AMOUNT"]))
        reference = units * price(rates, tx["TIMESTAMP"], asset) / SCALE
        result.append(dict(TRANSACTION_HASH=tx["TRANSACTION_HASH"],
                           TIMESTAMP=tx["TIMESTAMP"], TRANSACTION_INDEX=tx.get("TRANSACTION_INDEX", 0),
                           FROM_ADDRESS=tx["ACCOUNT"],
                           TO_ADDRESS=fields["Destination"], SOURCE_VALUE=reference,
                           VALUE=reference / SCALE, TOKEN_ADDRESS="reference-valuation", SYMBOL="USD"))
    return result


def load_payments(directory):
    directory = Path(directory)
    if (directory / "market_rates.csv").exists():
        return project(pq.read_table(directory / "transactions.parquet").to_pylist(), load_rates(directory))
    return pq.read_table(directory / "transfers.parquet").to_pylist()
