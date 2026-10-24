"""The payment network for one week: hubs, wallet groups and wallet histories.

Owner: Jack. Built separately for each week, only from that week's own payments.

What we found when building it (train and test look the same):
  * Each week has 700 wallets. Exactly 8 of them are hubs: they act as liquidity
    providers inside conversion legs and have 235-280 distinct counterparties,
    against a median of 61 and a next-busiest wallet at about 126. In training,
    every one of their payments is background activity. Loops and paths that run
    through a hub mean nothing, so hubs are set aside before any shape search.
  * Over a whole week every ordinary wallet connects to every other, so "wallet
    groups" only make sense inside a time window. A group is a set of payments
    linked by a shared non-hub wallet within `window` seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

import common

GROUP_WINDOWS = (120, 300, 600)


# ---------------------------------------------------------------- hubs

def liquidity_providers(week: str) -> set[str]:
    """Wallets that appear inside a conversion's legs without being its sender or receiver."""
    legs = common.raw(week, "transfers")[["TRANSACTION_HASH", "FROM_ADDRESS", "TO_ADDRESS"]]
    ends = common.payments(week)[["TRANSACTION_HASH", "FROM_ADDRESS", "TO_ADDRESS"]]
    joined = legs.merge(ends, on="TRANSACTION_HASH", suffixes=("", "_end"))
    inner_from = joined.FROM_ADDRESS[(joined.FROM_ADDRESS != joined.FROM_ADDRESS_end)]
    inner_to = joined.TO_ADDRESS[(joined.TO_ADDRESS != joined.TO_ADDRESS_end)]
    return set(inner_from) | set(inner_to)


def distinct_peers(payments: pd.DataFrame) -> pd.Series:
    out = payments.groupby("FROM_ADDRESS").TO_ADDRESS.nunique()
    inc = payments.groupby("TO_ADDRESS").FROM_ADDRESS.nunique()
    return out.add(inc, fill_value=0).sort_values(ascending=False)


def hubs(payments: pd.DataFrame, week: str) -> set[str]:
    """The liquidity providers: 8 per week, and the 8 busiest wallets by counterparties.

    A peer-count threshold was tried and dropped: "more than twice the median" flagged
    two extra wallets in train (126 and 123 peers) but none in test, so it would build
    the clue differently per week.
    """
    return liquidity_providers(week)


# ---------------------------------------------------------------- the network

def network(payments: pd.DataFrame) -> nx.DiGraph:
    """Wallets as dots, one arrow per sender-receiver pair with count, USD and time range."""
    pairs = payments.groupby(["FROM_ADDRESS", "TO_ADDRESS"]).agg(
        n=("VALUE", "size"), usd=("VALUE", "sum"),
        first=("TIMESTAMP", "min"), last=("TIMESTAMP", "max")).reset_index()
    g = nx.DiGraph()
    for row in pairs.itertuples(index=False):
        g.add_edge(row.FROM_ADDRESS, row.TO_ADDRESS, n=row.n, usd=row.usd, first=row.first, last=row.last)
    return g


# ---------------------------------------------------------------- wallet histories

@dataclass
class History:
    """One wallet's money in and money out, each in time order."""
    out_t: np.ndarray = field(default_factory=lambda: np.empty(0, "int64"))
    out_v: np.ndarray = field(default_factory=lambda: np.empty(0))
    out_to: np.ndarray = field(default_factory=lambda: np.empty(0, object))
    out_i: np.ndarray = field(default_factory=lambda: np.empty(0, "int64"))
    in_t: np.ndarray = field(default_factory=lambda: np.empty(0, "int64"))
    in_v: np.ndarray = field(default_factory=lambda: np.empty(0))
    in_from: np.ndarray = field(default_factory=lambda: np.empty(0, object))
    in_i: np.ndarray = field(default_factory=lambda: np.empty(0, "int64"))


def histories(payments: pd.DataFrame) -> dict[str, History]:
    """`payments` must be sorted by TIMESTAMP; *_i holds the row position of each payment."""
    h: dict[str, History] = {}
    rows = np.arange(len(payments))
    for wallet, g in payments.assign(_i=rows).groupby("FROM_ADDRESS", sort=False):
        e = h.setdefault(wallet, History())
        e.out_t, e.out_v = g.TIMESTAMP.to_numpy(), g.VALUE.to_numpy()
        e.out_to, e.out_i = g.TO_ADDRESS.to_numpy(), g._i.to_numpy()
    for wallet, g in payments.assign(_i=rows).groupby("TO_ADDRESS", sort=False):
        e = h.setdefault(wallet, History())
        e.in_t, e.in_v = g.TIMESTAMP.to_numpy(), g.VALUE.to_numpy()
        e.in_from, e.in_i = g.FROM_ADDRESS.to_numpy(), g._i.to_numpy()
    return h


# ---------------------------------------------------------------- wallet groups

def wallet_groups(payments: pd.DataFrame, hub_set: set[str], window: int) -> np.ndarray:
    """Group number per payment: payments linked by a shared non-hub wallet within `window` s.

    A payment between two hubs, or one whose only links are through hubs, is its own group.
    """
    parent = np.arange(len(payments))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    last: dict[str, tuple[int, int]] = {}
    for i, (a, b, t) in enumerate(zip(payments.FROM_ADDRESS.to_numpy(),
                                      payments.TO_ADDRESS.to_numpy(),
                                      payments.TIMESTAMP.to_numpy())):
        for w in (a, b):
            if w in hub_set:
                continue
            prev = last.get(w)
            if prev is not None and t - prev[1] <= window:
                ra, rb = find(i), find(prev[0])
                if ra != rb:
                    parent[ra] = rb
            last[w] = (i, t)
    return np.array([find(i) for i in range(len(payments))])


# ---------------------------------------------------------------- for main.py

def smooth_by_group(scores: np.ndarray, payments: pd.DataFrame, week: str,
                    windows=GROUP_WINDOWS, own_weight: float = 0.25) -> np.ndarray:
    """Blend each payment's score with its wallet group's mean score at each window.

    Training labels are all-or-nothing per planted scheme, so catching one payment
    should lift the rest of its group. `payments` must be in the same order as `scores`.
    On run 2 this blend added about 2.6 points to the honest test.
    """
    order = np.argsort(payments.TIMESTAMP.to_numpy(), kind="stable")
    sorted_payments = payments.iloc[order].reset_index(drop=True)
    sorted_scores = np.asarray(scores, dtype=float)[order]
    hub_set = hubs(sorted_payments, week)
    blended = own_weight * sorted_scores
    share = (1 - own_weight) / len(windows)
    for w in windows:
        g = wallet_groups(sorted_payments, hub_set, w)
        blended += share * pd.Series(sorted_scores).groupby(g).transform("mean").to_numpy()
    out = np.empty_like(blended)
    out[order] = blended
    return out


if __name__ == "__main__":
    for week in ("train", "test"):
        p = common.payments(week)
        hs = hubs(p, week)
        peers = distinct_peers(p)
        print(f"{week}: {len(peers)} wallets, {len(hs)} hubs "
              f"(liquidity providers {len(liquidity_providers(week))}); "
              f"hub peer counts {sorted(peers[list(hs)].astype(int).tolist(), reverse=True)}; "
              f"next wallet {int(peers.drop(list(hs)).iloc[0])}")
        for w in GROUP_WINDOWS:
            g = wallet_groups(p, hs, w)
            sizes = pd.Series(g).value_counts()
            print(f"  groups at {w:>3}s: {len(sizes):,}, largest {sizes.iloc[0]:,}, "
                  f"payments in groups of 3+: {sizes[sizes >= 3].sum():,}")
