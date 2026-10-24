"""Group payments into candidate laundering motifs, and describe each motif's shape.

Why group at all: in the training labels every scenario is label-pure, so the real
unit of prediction is the motif, not the single payment. Scoring a payment on its
own throws away the thing that makes a shape a shape.

Why several time scales: a short window keeps a motif pure but cuts it into pieces,
while a long window recovers whole shape motifs but lets busy relay services merge
into one blob. Measured on train/: a shape scenario's largest piece covers 25% of
it at 120s, 50% at 300s and 100% at 600s, while at 600s relay motifs start merging
hundreds of scenarios together. So we build all three and let the model choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCALES = (120, 300, 600)
MAX_GRAPH_NODES = 60  # above this, skip the path and cycle search


class _DSU:
    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build(payments: pd.DataFrame, window: int) -> np.ndarray:
    """Link two payments that share a wallet and sit within `window` seconds.

    `payments` must be sorted by ts. Returns a motif id per row.
    """
    dsu = _DSU(len(payments))
    last: dict[str, tuple[int, int]] = {}
    for i, (a, b, t) in enumerate(zip(payments.src.to_numpy(),
                                      payments.dst.to_numpy(),
                                      payments.ts.to_numpy())):
        for wallet in (a, b):
            previous = last.get(wallet)
            if previous is not None and t - previous[1] <= window:
                dsu.union(i, previous[0])
            last[wallet] = (i, t)
    return np.array([dsu.find(i) for i in range(len(payments))])


def _longest_path(adjacency: list[list[int]], n: int) -> int:
    """Longest path by edge count (layered-stack depth); -1 if the motif has a cycle."""
    state = [0] * n  # 0 unseen, 1 on stack, 2 done
    best = [0] * n

    def walk(u: int) -> int:
        if state[u] == 1:
            raise ValueError("cycle")
        if state[u] == 2:
            return best[u]
        state[u] = 1
        longest = 0
        for v in adjacency[u]:
            longest = max(longest, 1 + walk(v))
        state[u] = 2
        best[u] = longest
        return longest

    try:
        return max((walk(u) for u in range(n)), default=0)
    except (ValueError, RecursionError):
        return -1


def _count_cycles(adjacency: list[list[int]], n: int, cap: int = 50) -> int:
    """Count simple directed cycles, stopping at `cap` (round-tripping signal)."""
    found = 0
    for start in range(n):
        stack = [(start, 1, frozenset([start]))]
        while stack and found < cap:
            node, length, seen = stack.pop()
            for nxt in adjacency[node]:
                if nxt == start and length > 1:
                    found += 1
                elif nxt > start and nxt not in seen and length < 8:
                    stack.append((nxt, length + 1, seen | {nxt}))
        if found >= cap:
            break
    return found


def _shape_of(src: np.ndarray, dst: np.ndarray, usd: np.ndarray) -> dict[str, float]:
    """Graph shape of one motif: the signals the training deck lists per typology."""
    wallets = pd.unique(np.concatenate([src, dst]))
    index = {w: i for i, w in enumerate(wallets)}
    n = len(wallets)
    si = np.fromiter((index[w] for w in src), dtype=np.int64, count=len(src))
    di = np.fromiter((index[w] for w in dst), dtype=np.int64, count=len(dst))

    out_deg = np.bincount(si, minlength=n)
    in_deg = np.bincount(di, minlength=n)
    out_val = np.bincount(si, weights=usd, minlength=n)
    in_val = np.bincount(di, weights=usd, minlength=n)

    both = (in_val > 0) & (out_val > 0)
    lo = np.minimum(in_val, out_val)[both]
    hi = np.maximum(in_val, out_val)[both]
    pass_through = lo / np.maximum(hi, 1e-9)

    pairs = set(zip(si.tolist(), di.tolist()))
    n_edges = len(pairs)
    middles = int(both.sum())

    feats = {
        "wallets": float(n),
        "edges": float(n_edges),
        "density": n_edges / max(n * (n - 1), 1),
        "max_in_deg": float(in_deg.max()),
        "max_out_deg": float(out_deg.max()),
        "fan_in_share": in_deg.max() / max(len(src), 1),
        "fan_out_share": out_deg.max() / max(len(src), 1),
        "n_sources": float(((in_deg == 0) & (out_deg > 0)).sum()),
        "n_sinks": float(((out_deg == 0) & (in_deg > 0)).sum()),
        "n_middles": float(middles),
        "middle_share": middles / max(n, 1),
        "pass_through_max": float(pass_through.max()) if pass_through.size else 0.0,
        "pass_through_mean": float(pass_through.mean()) if pass_through.size else 0.0,
        "payer_receiver_overlap": len(set(si.tolist()) & set(di.tolist())) / max(n, 1),
        "route_reuse": len(src) / max(n_edges, 1),
    }

    if n <= MAX_GRAPH_NODES:
        adjacency: list[list[int]] = [[] for _ in range(n)]
        for a, b in pairs:
            adjacency[a].append(b)
        feats["depth"] = float(_longest_path(adjacency, n))
        feats["n_cycles"] = float(_count_cycles(adjacency, n))
    else:
        feats["depth"] = -2.0
        feats["n_cycles"] = -2.0
    return feats


def describe(payments: pd.DataFrame, motif_id: np.ndarray, prefix: str) -> pd.DataFrame:
    """One row of motif features per motif id."""
    frame = payments.assign(_m=motif_id)
    rows: dict[int, dict[str, float]] = {}
    for mid, group in frame.groupby("_m", sort=False):
        times = np.sort(group.ts.to_numpy())
        gaps = np.diff(times) if len(times) > 1 else np.array([0.0])
        usd = group.usd.to_numpy()
        span = float(times[-1] - times[0])
        feats = {
            "n": float(len(group)),
            "span": span,
            "median_gap": float(np.median(gaps)),
            "min_gap": float(gaps.min()),
            "max_gap": float(gaps.max()),
            "rate_per_min": len(group) / max(span / 60, 1 / 60),
            "n_assets": float(group.asset.nunique()),
            "conversion_share": float(group.is_conversion.mean()),
            "bridged_share": float(group.is_bridged.mean()),
            "usd_total": float(usd.sum()),
            "usd_median": float(np.median(usd)),
            "usd_min": float(usd.min()),
            "usd_max": float(usd.max()),
            "usd_spread": float(usd.std() / max(usd.mean(), 1e-9)) if len(usd) > 1 else 0.0,
            "usd_range_ratio": float(usd.min() / max(usd.max(), 1e-9)),
        }
        feats.update(_shape_of(group.src.to_numpy(), group.dst.to_numpy(), usd))
        rows[mid] = feats
    return pd.DataFrame.from_dict(rows, orient="index").add_prefix(prefix)


def multiscale(payments: pd.DataFrame, scales=SCALES):
    """Motif features at every scale, broadcast back onto the payments."""
    blocks, ids = [], {}
    for window in scales:
        motif_id = build(payments, window)
        ids[window] = motif_id
        described = describe(payments, motif_id, prefix=f"m{window}_")
        blocks.append(described.loc[motif_id].reset_index(drop=True))
    return pd.concat(blocks, axis=1), ids
