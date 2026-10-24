"""Wallet-history clues (batch 1): age and pass-through speed.

Owner: wallet-history lead. Called by main.py as add_features(payments, week).

Rules this file follows (from the team plan):
- One row out per transaction, TRANSACTION_HASH plus new wallet_* columns.
- Built the same way for both weeks, only from that week's own payments.
- Backward-looking only: "last received" uses strictly earlier payments.
- Never uses labels. Divide-by-zero becomes NaN, never infinity.
"""
import numpy as np
import pandas as pd

# Column names as given by the team lead: FROM_ADDRESS, TO_ADDRESS,
# TIMESTAMP (whole seconds) and VALUE. Not yet checked against the real clean
# table: confirm that VALUE is already in US dollars, not the asset's own units.
COLS = {
    "hash": "TRANSACTION_HASH",
    "sender": "FROM_ADDRESS",
    "receiver": "TO_ADDRESS",
    "time": "TIMESTAMP",
    "usd": "VALUE",
}


def parse_time(series):
    """Numeric timestamps are seconds since 1970; anything else is parsed as a date."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s")
    return pd.to_datetime(series)

NEW_COLUMNS = [
    "wallet_sender_age_hours",
    "wallet_receiver_age_hours",
    "wallet_mins_since_received",
    "wallet_out_vs_last_in",
]


def add_features(payments, week):
    """payments: clean table for one week (one row per transaction).
    week: "train" or "test" (unused: both weeks are built identically).
    Returns TRANSACTION_HASH plus wallet_* columns, one row per transaction,
    in the same order as `payments`."""
    c = COLS
    df = payments[[c["hash"], c["sender"], c["receiver"], c["time"], c["usd"]]].copy()
    df.columns = ["TRANSACTION_HASH", "sender", "receiver", "time", "usd"]
    df["time"] = parse_time(df["time"])
    df["usd"] = pd.to_numeric(df["usd"], errors="coerce")
    df["rid"] = np.arange(len(df))  # original position, so the output keeps input order

    # Stable time order; rid breaks ties so identical timestamps are deterministic.
    df = df.sort_values(["time", "rid"], kind="mergesort").reset_index(drop=True)

    # --- Age: hours since the wallet first appears this week (as sender or receiver).
    # Note: a wallet that existed before the week looks "new" at its first appearance.
    appearances = pd.concat(
        [
            df[["sender", "time"]].rename(columns={"sender": "wallet"}),
            df[["receiver", "time"]].rename(columns={"receiver": "wallet"}),
        ]
    )
    first_seen = appearances.groupby("wallet")["time"].min()
    df["wallet_sender_age_hours"] = (
        df["time"] - df["sender"].map(first_seen)
    ).dt.total_seconds() / 3600
    df["wallet_receiver_age_hours"] = (
        df["time"] - df["receiver"].map(first_seen)
    ).dt.total_seconds() / 3600

    # --- Pass-through: the sender's most recent EARLIER receipt of money.
    # merge_asof with allow_exact_matches=False means strictly earlier, so a
    # payment can never see itself or anything at the same instant.
    receipts = (
        df[["receiver", "time", "usd"]]
        .rename(columns={"receiver": "sender", "time": "last_in_time", "usd": "last_in_usd"})
        .sort_values("last_in_time", kind="mergesort")
    )
    left = df[["rid", "sender", "time"]].sort_values("time", kind="mergesort")
    merged = pd.merge_asof(
        left,
        receipts,
        left_on="time",
        right_on="last_in_time",
        by="sender",
        allow_exact_matches=False,
    ).set_index("rid")

    df = df.set_index("rid").sort_index()
    df["wallet_mins_since_received"] = (
        df["time"] - merged["last_in_time"].reindex(df.index)
    ).dt.total_seconds() / 60

    last_in_usd = merged["last_in_usd"].reindex(df.index)
    df["wallet_out_vs_last_in"] = np.where(last_in_usd > 0, df["usd"] / last_in_usd, np.nan)

    out = df[["TRANSACTION_HASH"] + NEW_COLUMNS].reset_index(drop=True)
    assert len(out) == len(payments), "output must have one row per input transaction"
    return out
