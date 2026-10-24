"""Shape clues: one detector per laundering typology from the training deck.

Owner: Jack. Format: add_features(payments, week) -> TRANSACTION_HASH plus shape_*
columns, one row per transaction. Built the same way for every week, only from
that week's own payments, never from labels.

Typology            Clues
fan-in              shape_fan_in_*, shape_intake_spread_*
fan-out             shape_fan_out_*, shape_payout_spread_*
gather-scatter      shape_gather_scatter_*, shape_gs_overlap_*, shape_hub_pass_seconds
scatter-gather      shape_two_hop_paths
cycle               shape_loop_hops, shape_loop_closes, shape_loop_value_back
                    (also short loops: shape_returns_2hop, shape_returns_3hop)
layered stack       shape_reach_depth, shape_reach_wallets
chain               shape_chain_forward, shape_chain_back, shape_chain_depth
bipartite           shape_bipartite_density, shape_bipartite_size
irregular network   shape_g{120,300,600}_* group features (hubs set aside)
context             shape_sender_is_hub, shape_receiver_is_hub, shape_first_pair, shape_pair_week_n
behaviour           shape_b_* per-payment tempo and amount clues from run 2 (src/features.py)

Hubs (the 8 liquidity providers per week) are set aside in every path, loop and
group search, because routes through an exchange-like wallet mean nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import common
import graph
from src import motifs as motif_lib
from src.data import load_payments as load_rich
from src.features import payment_features

WINDOWS = (600, 3600, 21600)       # 10 minutes, 1 hour, 6 hours
LOOP_WINDOW = 21600                # loops must close within 6 hours
CHAIN_WINDOW = 3600                # each chain hop within an hour
CHAIN_KEEP = (0.7, 1.02)           # "almost whole": 70% to 102% of the value carried on
HOP_WINDOW = 900                   # long-loop search: next hop within 15 minutes
MAX_HOPS = 8                       # planted cycles run 6 to 8 hops
MAX_REACH = 200                    # stop a search that has spread into the background
CACHE = common.DATA / "cache"
VERSION = "v4"


def _in_window(times, t, before, after):
    return np.searchsorted(times, t - before, "left"), np.searchsorted(times, t + after, "right")


def _spread(values):
    return float(values.std() / values.mean()) if len(values) > 1 and values.mean() > 0 else np.nan


def _reach(hist, hub_set, start, t0, direction, target=None):
    """Time-respecting search from `start` at time t0, forward (+1) or backward (-1).

    Each hop must follow (or, backward, precede) the previous one within HOP_WINDOW,
    stay within LOOP_WINDOW of t0, skip hubs and take any amount. Returns the reached
    wallets, the deepest hop count, and (forward only) the hop count and value of the
    first payment that lands back on `target`.
    """
    arrival = {start: (t0, 1)}
    frontier = [(t0 * direction, start, 1)]
    depth, closes, value_back = 1, 0, 0.0
    while frontier:
        frontier.sort()
        key, wallet, hops = frontier.pop(0)
        t = key * direction
        if hops >= MAX_HOPS:
            continue
        h = hist[wallet]
        if direction > 0:
            a, b = _in_window(h.out_t, t, 0, HOP_WINDOW)
            times, peers, values = h.out_t[a:b], h.out_to[a:b], h.out_v[a:b]
        else:
            a, b = _in_window(h.in_t, t, HOP_WINDOW, 0)
            times, peers, values = h.in_t[a:b], h.in_from[a:b], h.in_v[a:b]
        for tx, x, vx in zip(times, peers, values):
            if (tx - t) * direction <= 0 or x in hub_set or abs(tx - t0) > LOOP_WINDOW:
                continue
            if target is not None and x == target:
                if not closes:
                    closes, value_back = hops + 1, float(vx)
                continue
            prev = arrival.get(x)
            if prev is None or (tx - prev[0]) * direction < 0:
                arrival[x] = (tx, hops + 1)
                depth = max(depth, hops + 1)
                frontier.append((tx * direction, x, hops + 1))
        if len(arrival) > MAX_REACH:
            break
    return arrival, depth, closes, value_back


def _typology_clues(p: pd.DataFrame, hist, hub_set) -> pd.DataFrame:
    src, dst = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy()
    ts, val = p.TIMESTAMP.to_numpy(), p.VALUE.to_numpy()
    n = len(p)
    cols: dict[str, np.ndarray] = {}

    # ---- fan-in, fan-out, gather-scatter, at several windows
    for w in WINDOWS:
        fan_in, fan_out = np.zeros(n), np.zeros(n)
        in_spread, out_spread = np.full(n, np.nan), np.full(n, np.nan)
        gs, gs_overlap = np.zeros(n), np.zeros(n)
        for i in range(n):
            hd, hs = hist[dst[i]], hist[src[i]]
            a, b = _in_window(hd.in_t, ts[i], w, w)
            senders = set(hd.in_from[a:b])
            fan_in[i] = len(senders)
            in_spread[i] = _spread(hd.in_v[a:b])
            a2, b2 = _in_window(hs.out_t, ts[i], w, w)
            fan_out[i] = len(set(hs.out_to[a2:b2]))
            out_spread[i] = _spread(hs.out_v[a2:b2])
            # gather-scatter: the receiver collects from many and pays many
            a3, b3 = _in_window(hd.out_t, ts[i], w, w)
            receivers = set(hd.out_to[a3:b3])
            gs[i] = len(senders) * len(receivers)
            union = senders | receivers
            gs_overlap[i] = len(senders & receivers) / len(union) if union else 0.0
        cols[f"shape_fan_in_{w}"] = fan_in
        cols[f"shape_fan_out_{w}"] = fan_out
        cols[f"shape_intake_spread_{w}"] = in_spread
        cols[f"shape_payout_spread_{w}"] = out_spread
        cols[f"shape_gather_scatter_{w}"] = gs
        cols[f"shape_gs_overlap_{w}"] = gs_overlap

    # ---- the receiver's typical pass-through time over the week
    pass_seconds = {}
    for wallet, h in hist.items():
        if len(h.in_t) and len(h.out_t):
            j = np.searchsorted(h.out_t, h.in_t, "right")
            ok = j < len(h.out_t)
            pass_seconds[wallet] = float(np.median(h.out_t[j[ok]] - h.in_t[ok])) if ok.any() else np.nan
    cols["shape_hub_pass_seconds"] = np.array([pass_seconds.get(d, np.nan) for d in dst])

    # ---- scatter-gather: several middle wallets linking this sender to one destination
    two_hop = np.zeros(n)
    w = CHAIN_WINDOW
    for i in range(n):
        s, d, t = src[i], dst[i], ts[i]
        if d in hub_set:
            continue
        hd = hist[d]
        a, b = _in_window(hd.out_t, t, 0, w)
        finals = set(hd.out_to[a:b]) - hub_set - {s}
        if not finals:
            continue
        hs = hist[s]
        a2, b2 = _in_window(hs.out_t, t, w, w)
        middles = set(hs.out_to[a2:b2]) - hub_set - {d}
        best = 1
        for x in finals:
            count = 1
            for m in middles:
                hm = hist[m]
                a3, b3 = _in_window(hm.out_t, t, w, 2 * w)
                if x in set(hm.out_to[a3:b3]):
                    count += 1
            best = max(best, count)
        two_hop[i] = best
    cols["shape_two_hop_paths"] = two_hop

    # ---- cycle: does the money come back to the sender in 2 or 3 hops?
    back2, back3, value_back = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        s, d, t, v = src[i], dst[i], ts[i], val[i]
        if s in hub_set or d in hub_set:
            continue
        hd = hist[d]
        a, b = _in_window(hd.out_t, t, 0, LOOP_WINDOW)
        returned = 0.0
        for k in range(a, b):
            x, tx, vx = hd.out_to[k], hd.out_t[k], hd.out_v[k]
            if x == s:
                back2[i] = 1
                returned += vx
            elif x not in hub_set:
                hx = hist[x]
                a2, b2 = _in_window(hx.out_t, tx, 0, LOOP_WINDOW - (tx - t))
                hits = hx.out_to[a2:b2] == s
                if hits.any():
                    back3[i] = 1
                    returned += hx.out_v[a2:b2][hits].sum()
        value_back[i] = min(returned / v, 5.0) if v > 0 else 0.0
    cols["shape_returns_2hop"] = back2
    cols["shape_returns_3hop"] = back3
    cols["shape_cycle_value_back"] = value_back

    # ---- long cycles and layered depth: a time-respecting forward search.
    # Planted cycles run 6 to 8 hops and the amount drifts on the way (77 USD out,
    # 33 back), and layered stacks split and merge, so neither the 2-3 hop loop check
    # nor the "almost whole" chain finds them. Here each hop must follow the previous
    # one within HOP_WINDOW, at any amount, hubs set aside.
    loop_len, loop_value = np.zeros(n), np.zeros(n)
    on_loop = np.zeros(n)
    fwd_depth, bwd_depth = np.zeros(n), np.zeros(n)
    fwd_wallets, bwd_wallets = np.zeros(n), np.zeros(n)
    for i in range(n):
        s, d, t0 = src[i], dst[i], ts[i]
        if s in hub_set or d in hub_set:
            continue
        ahead, depth_a, closes, back_value = _reach(hist, hub_set, d, t0, +1, target=s)
        behind, depth_b, _, _ = _reach(hist, hub_set, s, t0, -1)
        fwd_depth[i], bwd_depth[i] = depth_a, depth_b
        fwd_wallets[i], bwd_wallets[i] = len(ahead), len(behind)
        loop_len[i] = closes
        loop_value[i] = min(back_value / max(val[i], 1e-9), 5.0) if closes else 0.0
        # on a loop: money leaving this receiver reaches a wallet that, earlier,
        # fed this sender (covers payments in the middle of a cycle)
        shared = (set(ahead) & set(behind)) - {s, d}
        on_loop[i] = float(bool(closes) or bool(shared))
    cols["shape_loop_hops"] = loop_len
    cols["shape_loop_closes"] = (loop_len > 0).astype(float)
    cols["shape_on_loop"] = on_loop
    cols["shape_loop_value_back"] = loop_value
    cols["shape_reach_forward"] = fwd_depth
    cols["shape_reach_back"] = bwd_depth
    cols["shape_reach_depth"] = fwd_depth + bwd_depth
    cols["shape_reach_wallets"] = fwd_wallets + bwd_wallets

    # ---- layered stack / chain: how far does the money travel on almost whole?
    lo, hi = CHAIN_KEEP
    fwd, bwd = np.zeros(n), np.zeros(n)
    for i in range(n):
        wallet, t, v, hops = dst[i], ts[i], val[i], 0
        while hops < 10 and wallet not in hub_set:
            h = hist[wallet]
            a, b = _in_window(h.out_t, t, 0, CHAIN_WINDOW)
            ok = np.flatnonzero((h.out_t[a:b] > t) & (h.out_v[a:b] >= lo * v) & (h.out_v[a:b] <= hi * v))
            if not ok.size:
                break
            k = a + ok[0]
            wallet, t, v, hops = h.out_to[k], h.out_t[k], h.out_v[k], hops + 1
        fwd[i] = hops
        wallet, t, v, hops = src[i], ts[i], val[i], 0
        while hops < 10 and wallet not in hub_set:
            h = hist[wallet]
            a, b = _in_window(h.in_t, t, CHAIN_WINDOW, 0)
            ok = np.flatnonzero((h.in_t[a:b] < t) & (v >= lo * h.in_v[a:b]) & (v <= hi * h.in_v[a:b]))
            if not ok.size:
                break
            k = a + ok[-1]
            wallet, t, v, hops = h.in_from[k], h.in_t[k], h.in_v[k], hops + 1
        bwd[i] = hops
    cols["shape_chain_forward"] = fwd
    cols["shape_chain_back"] = bwd
    cols["shape_chain_depth"] = fwd + bwd

    # ---- bipartite: the receiver's senders pay a shared set of receivers, densely
    density, size = np.zeros(n), np.zeros(n)
    w = CHAIN_WINDOW
    for i in range(n):
        d, t = dst[i], ts[i]
        if d in hub_set:
            continue
        hd = hist[d]
        a, b = _in_window(hd.in_t, t, w, w)
        payers = set(hd.in_from[a:b]) - hub_set
        if len(payers) < 2:
            continue
        links, receivers = 0, set()
        for m in payers:
            hm = hist[m]
            a2, b2 = _in_window(hm.out_t, t, w, w)
            outs = [x for x in hm.out_to[a2:b2] if x not in hub_set]
            receivers.update(outs)
            links += len(set(outs))
        if len(receivers) >= 2:
            density[i] = links / (len(payers) * len(receivers))
            size[i] = len(payers) + len(receivers)
    cols["shape_bipartite_density"] = density
    cols["shape_bipartite_size"] = size

    # ---- context
    cols["shape_sender_is_hub"] = np.isin(src, list(hub_set)).astype(float)
    cols["shape_receiver_is_hub"] = np.isin(dst, list(hub_set)).astype(float)
    pair = pd.Series(list(zip(src, dst)))
    cols["shape_pair_week_n"] = pair.map(pair.value_counts()).to_numpy().astype(float)
    cols["shape_first_pair"] = (~pair.duplicated()).to_numpy().astype(float)
    return pd.DataFrame(cols)


def _group_clues(rich: pd.DataFrame, hub_set) -> pd.DataFrame:
    """Irregular-network clues: the shape of each payment's wallet group, hubs set aside."""
    frames = []
    as_team = rich.rename(columns={"src": "FROM_ADDRESS", "dst": "TO_ADDRESS", "ts": "TIMESTAMP"})
    for w in graph.GROUP_WINDOWS:
        g = graph.wallet_groups(as_team, hub_set, w)
        described = motif_lib.describe(rich, g, prefix=f"shape_g{w}_")
        frames.append(described.loc[g].reset_index(drop=True))
    return pd.concat(frames, axis=1)


def add_features(payments: pd.DataFrame, week: str) -> pd.DataFrame:
    cache = CACHE / f"shapes_{VERSION}_{week}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    p = payments.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    hub_set = graph.hubs(p, week)
    hist = graph.histories(p)
    typology = _typology_clues(p, hist, hub_set)

    # asset-aware view of the same payments, aligned to p's order
    rich = load_rich(common.WEEKS[week]).set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].reset_index()
    groups = _group_clues(rich, hub_set)
    behaviour = payment_features(rich, common.WEEKS[week]).reset_index(drop=True).add_prefix("shape_b_")

    out = pd.concat([p[["TRANSACTION_HASH"]], typology, groups, behaviour], axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


if __name__ == "__main__":
    import time
    for wk in ("train", "test"):
        t0 = time.time()
        f = add_features(common.payments(wk), wk)
        print(f"{wk}: {f.shape[0]:,} rows, {f.shape[1] - 1} shape clues, {time.time() - t0:.0f}s")
