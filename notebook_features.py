"""make_features copied verbatim from the organisers starter_notebook.ipynb.

Kept as a module so src/honest_eval.py can score the same features.
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


def make_features(df):
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    t = df.TIMESTAMP.astype(int).to_numpy()
    value = df.VALUE.to_numpy()
    sends = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("FROM_ADDRESS")}
    gets = {a: g.TIMESTAMP.astype(int).to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    f = pd.DataFrame({"TRANSACTION_HASH": df.TRANSACTION_HASH})

    # Rule: bigger payments are more interesting (log keeps huge ones in range).
    f["log_value"] = np.log1p(value)

    # Rule: how busy are the sender and the receiver over the whole week?
    f["sender_payments"] = df.FROM_ADDRESS.map(df.FROM_ADDRESS.value_counts())
    f["receiver_payments"] = df.TO_ADDRESS.map(df.TO_ADDRESS.value_counts())

    # Rule: the receiver sends money on within an hour (a "relay" or "mule").
    f["receiver_forwards_1h"] = [
        count_in_window(sends, a, ti, 0, HOUR) > 0 for a, ti in zip(df.TO_ADDRESS, t)
    ]

    # Rule: the sender only just received money (it is passing it along).
    f["sender_received_1h_before"] = [
        count_in_window(gets, a, ti, HOUR, 0) > 0 for a, ti in zip(df.FROM_ADDRESS, t)
    ]

    # Rule: fan-out / fan-in, many payments from one sender or into one
    # receiver within ten minutes.
    f["sender_burst_10m"] = [
        count_in_window(sends, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.FROM_ADDRESS, t)
    ]
    f["receiver_burst_10m"] = [
        count_in_window(gets, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(df.TO_ADDRESS, t)
    ]

    # Rule: the receiver passes on a similar amount (within 10%) within an hour,
    # a classic sign of money being layered through an account.
    sent_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("FROM_ADDRESS")}

    def forwards_similar(account, ti, amount):
        times = sends.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti, "right"), np.searchsorted(times, ti + HOUR, "right")
        later = sent_values[account][lo:hi]
        return bool(np.any(np.abs(later - amount) <= 0.1 * amount))

    f["receiver_forwards_similar_1h"] = [
        forwards_similar(a, ti, v) for a, ti, v in zip(df.TO_ADDRESS, t, value)
    ]

    # Rule: the sender received a similar amount within the hour before.
    got_values = {a: g.VALUE.to_numpy() for a, g in df.groupby("TO_ADDRESS")}

    def received_similar(account, ti, amount):
        times = gets.get(account)
        if times is None:
            return False
        lo, hi = np.searchsorted(times, ti - HOUR, "left"), np.searchsorted(times, ti, "left")
        earlier = got_values[account][lo:hi]
        return bool(np.any(np.abs(earlier - amount) <= 0.1 * amount))

    f["sender_received_similar_1h"] = [
        received_similar(a, ti, v) for a, ti, v in zip(df.FROM_ADDRESS, t, value)
    ]

    # Rule: how quickly does the receiver send money on? (minutes, capped at 60)
    def minutes_to_next_send(account, ti):
        times = sends.get(account)
        if times is None:
            return 60.0
        i = np.searchsorted(times, ti, "right")
        return min(60.0, (times[i] - ti) / 60) if i < len(times) else 60.0

    f["receiver_minutes_to_forward"] = [
        minutes_to_next_send(a, ti) for a, ti in zip(df.TO_ADDRESS, t)
    ]

    # Rule: pass-through accounts send on most of what they receive over the week.
    sent_total = df.groupby("FROM_ADDRESS").VALUE.sum()
    got_total = df.groupby("TO_ADDRESS").VALUE.sum()
    pass_through = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["sender_pass_through"] = df.FROM_ADDRESS.map(pass_through).fillna(0)
    f["receiver_pass_through"] = df.TO_ADDRESS.map(pass_through).fillna(0)

    # ---- Add your own rules here, for example: ----
    # f["round_amount"] = (value % 100 < 1)
    # f["night_time"] = (t % 86400) < 6 * 3600

    return f.astype({c: float for c in f.columns if c != "TRANSACTION_HASH"})


