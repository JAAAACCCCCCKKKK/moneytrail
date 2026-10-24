"""Shape signals from the training deck's table that the other clue files missed.

Owner: Jack. Clue-file format: add_features(payments, week) -> TRANSACTION_HASH plus
sig_* columns, one row per transaction, built only from that week's payments, never
from labels. Hubs (the 8 liquidity providers) are set aside in every search.

Deck signal (shape)                                Columns
"senders that are new or do nothing else" (fan-in) sig_fanin_quiet_share_*, sig_fanin_new_share_*,
                                                   sig_fanin_sender_week_log_*
"exit amount close to the sum in" (fan-in)         sig_fanin_exit_closeness_*
"time from last deposit to exit" (fan-in)          sig_fanin_last_deposit_to_exit
"recipients that pass it on fast" (fan-out)        sig_fanout_fast_share_*, sig_fanout_fwd_median_log_*
"value out ~ value in" (scatter-gather)            sig_sg_value_balance
"middle wallets with no other activity" (s-g)      sig_sg_middle_quiet
"similar short hold times" (scatter-gather)        sig_sg_hold_median, sig_sg_hold_spread
"middle wallets used once" (layered stack)         sig_g*_middle_once_share, sig_g*_middle_week_log
"similar delays" (layered stack)                   sig_g*_hold_median, sig_g*_hold_spread
"few contacts outside the block" (bipartite)       sig_bip_inside_share
"many pairs active at once" (bipartite)            sig_bip_pairs_active
"similar amounts" (bipartite)                      sig_bip_amount_spread
"number of hops and time taken" (cycle)            sig_loop_seconds, sig_loop_total_hops
"assets changing along the way" (cycle)            sig_loop_n_assets (dropped by --drop-conversion:
                                                   the README says asset choices ignore the labels)
"distinct routes between two wallets" (irregular)  sig_g*_routes
"value into vs out of the group" (irregular)       sig_g*_outside_in_log, sig_g*_outside_out_log,
                                                   sig_g*_outside_balance, sig_g*_closure

Group clues (sig_g300_*, sig_g600_*) use the hub-aware wallet groups from graph.py;
sig_gonce_* use groups linked only through once-a-week pairs at one hour, which hold
slow (medium) schemes together. Note on wallet age: every account's XRPL Sequence starts
at 1 in each week and rises by one per payment, so it is a payment counter, not an age;
"new" here means first seen in this week within the last 6 hours.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import common
import graph
from src.data import load_payments as load_rich

WINDOWS = (600, 3600)          # fan-in / fan-out windows: 10 minutes and 1 hour
QUIET = 1800                   # "does nothing else": no other payment within +-30 minutes
NEW = 6 * 3600                 # "new": first payment of the week within the last 6 hours
FAST = 3600                    # "passes it on fast": forwards within an hour
SG_WINDOW = 3600               # scatter-gather and bipartite, as in features_shapes.py
MIDDLE_QUIET = 3 * 3600        # middle wallet "with no other activity" within +-3 hours
GROUP_WINDOWS = (300, 600)
ONCE_WINDOW = 3600
HOP_WINDOW, MAX_HOPS, LOOP_WINDOW, MAX_REACH = 900, 8, 21600, 200   # as features_shapes.py
MAX_ROUTE_WALLETS, MAX_ROUTES, MAX_ROUTE_LEN = 60, 50, 8
BIG = 7 * 86400.0
CACHE = common.DATA / "cache"
VERSION = "v2"    # v2: set loops sorted, so the clues no longer change with the hash seed


def _span(times, lo, hi):
    return np.searchsorted(times, lo, "left"), np.searchsorted(times, hi, "right")


class Touch:
    """Every payment touching each wallet, in time order (row positions in p)."""

    def __init__(self, src, dst, ts):
        w = np.concatenate([src, dst])
        i = np.concatenate([np.arange(len(src)), np.arange(len(src))])
        t = np.concatenate([ts, ts])
        order = np.lexsort((t, w))
        w, i, t = w[order], i[order], t[order]
        cuts = np.flatnonzero(w[1:] != w[:-1]) + 1
        starts, ends = np.r_[0, cuts], np.r_[cuts, len(w)]
        self.t = {w[s]: t[s:e] for s, e in zip(starts, ends)}
        self.i = {w[s]: i[s:e] for s, e in zip(starts, ends)}

    def count(self, wallet, lo, hi):
        a, b = _span(self.t[wallet], lo, hi)
        return b - a

    def rows(self, wallet, lo, hi):
        a, b = _span(self.t[wallet], lo, hi)
        return self.i[wallet][a:b]


def _cv(values):
    values = np.asarray(values, float)
    return float(values.std() / values.mean()) if len(values) > 1 and values.mean() > 0 else np.nan


# ------------------------------------------------------------------ fan-in and fan-out

def _fan_clues(p, hist, touch, hub_set, week_n, first_seen):
    src, dst = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy()
    ts = p.TIMESTAMP.to_numpy()
    n = len(p)
    cols = {}

    last_to_exit = np.full(n, np.nan)
    for i in range(n):
        if dst[i] in hub_set:
            continue
        h = hist[dst[i]]
        j = np.searchsorted(h.out_t, ts[i], "right")
        if j < len(h.out_t):
            exit_t = h.out_t[j]
            k = np.searchsorted(h.in_t, exit_t, "right") - 1     # last deposit at or before the exit
            if k >= 0:
                last_to_exit[i] = exit_t - h.in_t[k]
    cols["sig_fanin_last_deposit_to_exit"] = last_to_exit

    for W in WINDOWS:
        quiet, week_log, new = (np.full(n, np.nan) for _ in range(3))
        closeness = np.full(n, np.nan)
        fast, fwd_med = np.full(n, np.nan), np.full(n, np.nan)
        for i in range(n):
            t = ts[i]
            # fan-in, seen from this payment's receiver
            if dst[i] not in hub_set:
                hd = hist[dst[i]]
                a, b = _span(hd.in_t, t - W, t + W)
                senders = [s for s in set(hd.in_from[a:b]) if s not in hub_set]
                if senders:
                    aq, bq = _span(hd.in_t, t - QUIET, t + QUIET)
                    near_from = hd.in_from[aq:bq]
                    q = [touch.count(s, t - QUIET, t + QUIET) - int(np.sum(near_from == s)) <= 0 for s in senders]
                    quiet[i] = float(np.mean(q))
                    week_log[i] = float(np.median([np.log1p(week_n[s]) for s in senders]))
                    new[i] = float(np.mean([first_seen[s] >= t - NEW for s in senders]))
                a1, b1 = _span(hd.in_t, t - W, t)
                in_sum = hd.in_v[a1:b1].sum()
                a2, b2 = np.searchsorted(hd.out_t, t, "right"), np.searchsorted(hd.out_t, t + W, "right")
                out_sum = hd.out_v[a2:b2].sum()
                closeness[i] = min(in_sum, out_sum) / max(in_sum, out_sum) if max(in_sum, out_sum) > 0 else 0.0
            # fan-out, seen from this payment's sender
            if src[i] not in hub_set:
                hs = hist[src[i]]
                a, b = _span(hs.out_t, t - W, t + W)
                first_paid = {}
                for r, tr in zip(hs.out_to[a:b], hs.out_t[a:b]):
                    if r not in hub_set and r not in first_paid:
                        first_paid[r] = tr
                if first_paid:
                    waits = []
                    for r, tr in first_paid.items():
                        hr = hist[r]
                        j = np.searchsorted(hr.out_t, tr, "right")
                        waits.append(hr.out_t[j] - tr if j < len(hr.out_t) else BIG)
                    waits = np.asarray(waits, float)
                    fast[i] = float(np.mean(waits <= FAST))
                    fwd_med[i] = float(np.median(np.log1p(waits)))
        cols[f"sig_fanin_quiet_share_{W}"] = quiet
        cols[f"sig_fanin_sender_week_log_{W}"] = week_log
        cols[f"sig_fanin_new_share_{W}"] = new
        cols[f"sig_fanin_exit_closeness_{W}"] = closeness
        cols[f"sig_fanout_fast_share_{W}"] = fast
        cols[f"sig_fanout_fwd_median_log_{W}"] = fwd_med
    return cols


# ------------------------------------------------------------------ scatter-gather

def _scatter_gather(p, hist, touch, hub_set):
    """For the destination reached by the most middle wallets from this sender."""
    src, dst = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy()
    ts = p.TIMESTAMP.to_numpy()
    n, W = len(p), SG_WINDOW
    paths, balance, quiet = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    hold_med, hold_spread = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        s, d, t = src[i], dst[i], ts[i]
        if s in hub_set or d in hub_set:
            continue
        hd = hist[d]
        a, b = np.searchsorted(hd.out_t, t, "right"), np.searchsorted(hd.out_t, t + W, "right")
        finals = set(hd.out_to[a:b]) - hub_set - {s}
        if not finals:
            continue
        hs = hist[s]
        a2, b2 = _span(hs.out_t, t - W, t + W)
        paid = {}                                    # middle -> [first time, value, count]
        for m, tm, vm in zip(hs.out_to[a2:b2], hs.out_t[a2:b2], hs.out_v[a2:b2]):
            if m in hub_set:
                continue
            e = paid.setdefault(m, [tm, 0.0, 0])
            e[1] += vm
            e[2] += 1
        best = None
        for x in sorted(finals):                     # sorted: a set's order changes with the hash seed
            links = []
            for m, (tm, vm, cm) in paid.items():
                if m == x:
                    continue
                hm = hist[m]
                a3, b3 = np.searchsorted(hm.out_t, tm, "right"), np.searchsorted(hm.out_t, t + 2 * W, "right")
                hit = hm.out_to[a3:b3] == x
                if hit.any():
                    links.append((m, tm, vm, cm, hm.out_t[a3:b3][hit][0], hm.out_v[a3:b3][hit].sum(), int(hit.sum())))
            if len(links) >= 2 and (best is None or len(links) > len(best)):
                best = links
        if best is None:
            continue
        paths[i] = len(best)
        out_v = sum(l[2] for l in best)
        in_v = sum(l[5] for l in best)
        balance[i] = min(out_v, in_v) / max(out_v, in_v) if max(out_v, in_v) > 0 else 0.0
        holds = [l[4] - l[1] for l in best]
        hold_med[i] = float(np.median(holds))
        hold_spread[i] = _cv(holds)
        others = [touch.count(l[0], t - MIDDLE_QUIET, t + MIDDLE_QUIET) - l[3] - l[6] for l in best]
        quiet[i] = float(np.mean([o <= 0 for o in others]))
    return {"sig_sg_paths": paths, "sig_sg_value_balance": balance, "sig_sg_middle_quiet": quiet,
            "sig_sg_hold_median": hold_med, "sig_sg_hold_spread": hold_spread}


# ------------------------------------------------------------------ bipartite block

def _bipartite(p, hist, touch, hub_set):
    src, dst = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy()
    ts, val = p.TIMESTAMP.to_numpy(), p.VALUE.to_numpy()
    n, W = len(p), SG_WINDOW
    inside_share, pairs, spread = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        d, t = dst[i], ts[i]
        if d in hub_set:
            continue
        hd = hist[d]
        a, b = _span(hd.in_t, t - W, t + W)
        payers = set(hd.in_from[a:b]) - hub_set
        if len(payers) < 2:
            continue
        receivers = set()
        for m in payers:
            hm = hist[m]
            a2, b2 = _span(hm.out_t, t - W, t + W)
            receivers.update(r for r in hm.out_to[a2:b2] if r not in hub_set)
        if len(receivers) < 2:
            continue
        block = payers | receivers
        rows = set()
        for w in block:
            rows.update(touch.rows(w, t - W, t + W).tolist())
        rows = np.array(sorted(rows), dtype=np.int64)    # sorted: fixed order for the amount spread
        internal = rows[np.isin(src[rows], list(block)) & np.isin(dst[rows], list(block))]
        inside_share[i] = len(internal) / len(rows)
        cross = internal[np.isin(src[internal], list(payers)) & np.isin(dst[internal], list(receivers))]
        pairs[i] = len(set(zip(src[cross], dst[cross])))
        spread[i] = _cv(val[cross])
    return {"sig_bip_inside_share": inside_share, "sig_bip_pairs_active": pairs, "sig_bip_amount_spread": spread}


# ------------------------------------------------------------------ cycles: duration, hops, assets

def _search(hist, hub_set, asset, start, t0, direction, first_hops, target=None):
    """Time-respecting search (as features_shapes._reach) that also records each path's assets."""
    arrival = {start: (t0, first_hops, frozenset())}
    frontier = [(t0 * direction, start)]
    close = None
    while frontier:
        frontier.sort()
        key, wallet = frontier.pop(0)
        t, hops, assets = arrival[wallet]
        if hops >= MAX_HOPS:
            continue
        h = hist[wallet]
        if direction > 0:
            a, b = _span(h.out_t, t, t + HOP_WINDOW)
            times, peers, rows = h.out_t[a:b], h.out_to[a:b], h.out_i[a:b]
        else:
            a, b = _span(h.in_t, t - HOP_WINDOW, t)
            times, peers, rows = h.in_t[a:b], h.in_from[a:b], h.in_i[a:b]
        for tx, x, k in zip(times, peers, rows):
            if (tx - t) * direction <= 0 or x in hub_set or abs(tx - t0) > LOOP_WINDOW:
                continue
            path_assets = assets | {asset[k]}
            if target is not None and x == target:
                if close is None:
                    close = (tx, hops + 1, path_assets)
                continue
            prev = arrival.get(x)
            if prev is None or (tx - prev[0]) * direction < 0:
                arrival[x] = (tx, hops + 1, path_assets)
                frontier.append((tx * direction, x))
        if len(arrival) > MAX_REACH:
            break
    return arrival, close


def _loops(p, hist, hub_set, asset):
    src, dst, ts = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy(), p.TIMESTAMP.to_numpy()
    n = len(p)
    seconds, hops, n_assets = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        s, d, t0 = src[i], dst[i], ts[i]
        if s in hub_set or d in hub_set:
            continue
        ahead, close = _search(hist, hub_set, asset, d, t0, +1, 1, target=s)
        if close is not None:
            seconds[i], hops[i] = close[0] - t0, close[1]
            n_assets[i] = len(close[2] | {asset[i]})
            continue
        behind, _ = _search(hist, hub_set, asset, s, t0, -1, 0)
        best = None
        for z in sorted((set(ahead) & set(behind)) - {s, d}):     # sorted: ties keep the same loop every run
            span = ahead[z][0] - behind[z][0]
            if span > 0 and (best is None or span < best[0]):
                best = (span, ahead[z][1] + behind[z][1], ahead[z][2] | behind[z][2])
        if best is not None:
            seconds[i], hops[i] = best[0], best[1]
            n_assets[i] = len(best[2] | {asset[i]})
    return {"sig_loop_seconds": seconds, "sig_loop_total_hops": hops, "sig_loop_n_assets": n_assets}


# ------------------------------------------------------------------ wallet-group clues

def _count_routes(pairs, entry, exit_):
    """Distinct simple paths from entry to exit, up to MAX_ROUTE_LEN hops, capped."""
    adjacency = defaultdict(list)
    for a, b in pairs:
        adjacency[a].append(b)
    found, stack = 0, [(entry, frozenset([entry]))]
    while stack and found < MAX_ROUTES:
        node, seen = stack.pop()
        for nxt in adjacency[node]:
            if nxt == exit_:
                found += 1
            elif nxt not in seen and len(seen) < MAX_ROUTE_LEN:
                stack.append((nxt, seen | {nxt}))
    return min(found, MAX_ROUTES)


def _group_clues(p, touch, week_n, groups, prefix):
    """Per wallet group: middle wallets, hold times, outside traffic and routes."""
    src, dst = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy()
    ts, val = p.TIMESTAMP.to_numpy(), p.VALUE.to_numpy()
    n = len(p)
    names = ["middle_week_log", "middle_once_share", "hold_median", "hold_spread",
             "outside_in_log", "outside_out_log", "outside_balance", "closure", "routes"]
    out = {k: np.full(n, np.nan) for k in names}
    frame = pd.DataFrame({"g": groups, "i": np.arange(n)})
    for gid, rows in frame[frame.g >= 0].groupby("g").i:
        rows = rows.to_numpy()
        gs, gd, gt, gv = src[rows], dst[rows], ts[rows], val[rows]
        t0, t1 = gt.min(), gt.max()
        wallets = set(gs) | set(gd)
        middles = set(gs) & set(gd)
        res = {}
        if middles:
            inside = Counter(gs.tolist()) + Counter(gd.tolist())
            res["middle_week_log"] = float(np.median([np.log1p(week_n[m]) for m in middles]))
            res["middle_once_share"] = float(np.mean(
                [touch.count(m, t0 - 3600, t1 + 3600) == inside[m] for m in middles]))
            holds = []
            for m in sorted(middles):                # sorted: a set's order changes with the hash seed
                first_in = gt[gd == m].min()
                outs = gt[(gs == m) & (gt >= first_in)]
                if len(outs):
                    holds.append(outs.min() - first_in)
            if holds:
                res["hold_median"] = float(np.median(holds))
                res["hold_spread"] = _cv(holds)
        rowset = set(rows.tolist())
        external, ext_in, ext_out = set(), 0.0, 0.0
        for w in sorted(wallets):                    # sorted: fixed order for the money totals
            for k in touch.rows(w, t0, t1).tolist():
                if k in rowset or k in external:
                    continue
                external.add(k)
                if dst[k] in wallets and src[k] not in wallets:
                    ext_in += val[k]
                elif src[k] in wallets and dst[k] not in wallets:
                    ext_out += val[k]
        res["outside_in_log"] = float(np.log1p(ext_in))
        res["outside_out_log"] = float(np.log1p(ext_out))
        res["outside_balance"] = min(ext_in, ext_out) / max(ext_in, ext_out) if max(ext_in, ext_out) > 0 else np.nan
        res["closure"] = len(rows) / (len(rows) + len(external))
        if len(rows) >= 2 and len(wallets) <= MAX_ROUTE_WALLETS:
            net = defaultdict(float)
            for a, b_, v in zip(gs, gd, gv):
                net[a] -= v
                net[b_] += v
            entry, exit_ = min(net, key=net.get), max(net, key=net.get)
            res["routes"] = float(_count_routes(set(zip(gs, gd)), entry, exit_)) if entry != exit_ else 0.0
        for k, v in res.items():
            out[k][rows] = v
    return {f"{prefix}{k}": v for k, v in out.items()}


# ------------------------------------------------------------------ entry point

def add_features(payments: pd.DataFrame, week: str) -> pd.DataFrame:
    cache = CACHE / f"shape_signals_{VERSION}_{week}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    p = payments.sort_values(["TIMESTAMP", "TRANSACTION_HASH"]).reset_index(drop=True)
    src, dst, ts = p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy(), p.TIMESTAMP.to_numpy()
    hub_set = graph.hubs(p, week)
    hist = graph.histories(p)
    touch = Touch(src, dst, ts)
    everyone = pd.concat([pd.Series(ts, index=src), pd.Series(ts, index=dst)])
    week_n = everyone.groupby(level=0).size().to_dict()
    first_seen = everyone.groupby(level=0).min().to_dict()
    asset = load_rich(common.WEEKS[week]).set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH].asset.to_numpy()

    cols = {}
    cols.update(_fan_clues(p, hist, touch, hub_set, week_n, first_seen))
    cols.update(_scatter_gather(p, hist, touch, hub_set))
    cols.update(_bipartite(p, hist, touch, hub_set))
    cols.update(_loops(p, hist, hub_set, asset))
    for w in GROUP_WINDOWS:
        cols.update(_group_clues(p, touch, week_n, graph.wallet_groups(p, hub_set, w), f"sig_g{w}_"))
    pairs = pd.Series(list(zip(src, dst)))
    once = (pairs.map(pairs.value_counts()) == 1).to_numpy()
    groups = np.full(len(p), -1, dtype=np.int64)
    groups[once] = graph.wallet_groups(p[once].reset_index(drop=True), hub_set, ONCE_WINDOW)
    cols.update(_group_clues(p, touch, week_n, groups, "sig_gonce_"))

    out = pd.concat([p[["TRANSACTION_HASH"]], pd.DataFrame(cols)], axis=1)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


if __name__ == "__main__":
    import time
    for wk in ("train", "test"):
        t0 = time.time()
        f = add_features(common.payments(wk), wk)
        filled = f.drop(columns="TRANSACTION_HASH").notna().mean()
        print(f"{wk}: {f.shape[1] - 1} signal clues, {time.time() - t0:.0f}s; "
              f"share filled min {filled.min():.2f}, median {filled.median():.2f}")
