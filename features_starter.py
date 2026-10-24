"""The organisers' starter clues, unchanged in substance.

Adapted from make_features in the organisers' starter_notebook.ipynb (supplied with
the training data). We did not write these rules; they are kept as our baseline set.
"""
import numpy as np
import pandas as pd

HOUR, TEN_MINUTES = 3600, 600


def count_in_window(times_by_account, account, t, before, after):
    """How many of an account's payments fall between t - before and t + after."""
    times = times_by_account.get(account)
    if times is None:
        return 0
    return int(np.searchsorted(times, t + after, "right") - np.searchsorted(times, t - before, "left"))


def add_features(payments, week):
    df = payments.sort_values("TIMESTAMP").reset_index(drop=True)
    t = df.TIMESTAMP.astype(int).to_numpy()
    value = df.VALUE.to_numpy()
    sends = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("FROM_ADDRESS")}
    gets = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    f = pd.DataFrame({"TRANSACTION_HASH": df.TRANSACTION_HASH})
    f["log_value"] = np.log1p(value)
    f["sender_payments"] = df.FROM_ADDRESS.map(df.FROM_ADDRESS.value_counts())
    f["receiver_payments"] = df.TO_ADDRESS.map(df.TO_ADDRESS.value_counts())
    f["receiver_forwards_1h"] = [count_in_window(sends, a, ti, 0, HOUR) > 0 for a, ti in zip(df.TO_ADDRESS, t)]
    f["sender_received_1h_before"] = [count_in_window(gets, a, ti, HOUR, 0) > 0 for a, ti in zip(df.FROM_ADDRESS, t)]
    f["sender_burst_10m"] = [count_in_window(sends, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.FROM_ADDRESS, t)]
    f["receiver_burst_10m"] = [count_in_window(gets, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.TO_ADDRESS, t)]

    sent_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("FROM_ADDRESS")}
    got_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    def forwards_similar(account, ti, amount):
        times = sends.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti, "right"), np.searchsorted(times, ti + HOUR, "right")
        return bool(np.any(np.abs(sent_values[account][lo:hi] - amount) <= 0.1 * amount))

    def received_similar(account, ti, amount):
        times = gets.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti - HOUR, "left"), np.searchsorted(times, ti, "left")
        return bool(np.any(np.abs(got_values[account][lo:hi] - amount) <= 0.1 * amount))

    def minutes_to_next_send(account, ti):
        times = sends.get(account)
        if times is None:
            return 60.0
        i = np.searchsorted(times, ti, "right")
        return min(60.0, (times[i] - ti) / 60) if i < len(times) else 60.0

    f["receiver_forwards_similar_1h"] = [forwards_similar(a, ti, v) for a, ti, v in zip(df.TO_ADDRESS, t, value)]
    f["sender_received_similar_1h"] = [received_similar(a, ti, v) for a, ti, v in zip(df.FROM_ADDRESS, t, value)]
    f["receiver_minutes_to_forward"] = [minutes_to_next_send(a, ti) for a, ti in zip(df.TO_ADDRESS, t)]

    sent_total = df.groupby("FROM_ADDRESS").VALUE.sum()
    got_total = df.groupby("TO_ADDRESS").VALUE.sum()
    pass_through = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["sender_pass_through"] = df.FROM_ADDRESS.map(pass_through).fillna(0)
    f["receiver_pass_through"] = df.TO_ADDRESS.map(pass_through).fillna(0)
    return f.astype({c: float for c in f.columns if c != "TRANSACTION_HASH"})
