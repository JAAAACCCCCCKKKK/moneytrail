"""Per-payment behaviour features plus multi-scale motif shape features.

Nothing here uses a wallet address, hash or absolute timestamp as a feature:
wallets and hashes are disjoint between train and test, so identity cannot
transfer. Tempo and amount features are also given in a scale-free form
(relative to the wallet's own week), because the organisers warn that the
test data has batch variations absent from training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import motifs
from .data import XRP_SENTINEL, leg_features, load_payments, load_rates

WINDOWS = (60, 180, 600, 3600)
BIG = 7 * 86400.0


def _by_wallet(wallets: np.ndarray, times: np.ndarray, values: np.ndarray):
    """wallet -> (sorted times, values in the same order)."""
    order = np.lexsort((times, wallets))
    w, t, v = wallets[order], times[order], values[order]
    cuts = np.flatnonzero(w[1:] != w[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(w)]
    return {w[s]: (t[s:e], v[s:e]) for s, e in zip(starts, ends)}


def _window_counts(table, wallets, times, window):
    out = np.zeros(len(wallets))
    for i, (w, t) in enumerate(zip(wallets, times)):
        entry = table.get(w)
        if entry is not None:
            ts = entry[0]
            out[i] = np.searchsorted(ts, t + window, "right") - np.searchsorted(ts, t - window, "left")
    return out


def _wallet_week(p: pd.DataFrame) -> pd.DataFrame:
    """Behaviour of each wallet over the whole window (the rules allow this)."""
    sent = p.groupby("src").agg(out_n=("usd", "size"), out_usd=("usd", "sum"),
                                out_med=("usd", "median"), out_peers=("dst", "nunique"))
    got = p.groupby("dst").agg(in_n=("usd", "size"), in_usd=("usd", "sum"),
                               in_med=("usd", "median"), in_peers=("src", "nunique"))
    week = sent.join(got, how="outer").fillna(0.0)
    week["total_n"] = week.out_n + week.in_n
    week["flow_ratio"] = np.minimum(week.in_usd, week.out_usd) / np.maximum(
        np.maximum(week.in_usd, week.out_usd), 1e-9)
    week["peers"] = week.out_peers + week.in_peers

    events = pd.concat([p[["src", "ts"]].rename(columns={"src": "w"}),
                        p[["dst", "ts"]].rename(columns={"dst": "w"})]).sort_values(["w", "ts"])
    gaps = events.groupby("w").ts.diff()
    week["median_gap"] = gaps.groupby(events.w).median().reindex(week.index).fillna(BIG)
    return week


def _conversion_spread(p: pd.DataFrame, directory: Path) -> np.ndarray:
    """What the sender really paid (leg 0) over what arrived, both in USD.

    Laundering often accepts a bad rate; honest users avoid one. Zero when the
    payment did not convert.
    """
    legs = leg_features(directory).set_index("TRANSACTION_HASH").reindex(p.TRANSACTION_HASH)
    rates = load_rates(directory)
    parts = legs.first_leg_asset.str.split("|", n=1, expand=True)
    token, symbol = parts[0].to_numpy(), parts[1].to_numpy()
    key = np.where(token == XRP_SENTINEL, "XRP", token.astype(str) + ":" + symbol.astype(str))
    hours = p.ts.to_numpy() // 3600 * 3600
    price = np.array([rates.get((h, k), np.nan) for h, k in zip(hours, key)], dtype="float64")
    paid = legs.first_leg_value.to_numpy() * price / 1_000_000
    spread = paid / np.maximum(p.usd.to_numpy(), 1e-9) - 1.0
    spread = np.where(p.is_conversion.to_numpy(), spread, 0.0)
    return np.nan_to_num(spread), legs.n_legs.to_numpy()


def payment_features(p: pd.DataFrame, directory: Path) -> pd.DataFrame:
    src, dst = p.src.to_numpy(), p.dst.to_numpy()
    t, usd = p.ts.to_numpy(), p.usd.to_numpy()
    f = pd.DataFrame(index=p.index)

    f["log_usd"] = np.log1p(usd)
    f["is_conversion"] = p.is_conversion.astype(float).to_numpy()
    f["is_bridged"] = p.is_bridged.astype(float).to_numpy()
    spread, n_legs = _conversion_spread(p, directory)
    f["conversion_spread"] = spread
    f["n_legs"] = n_legs

    # every payment touching a wallet, in either direction
    all_w = np.concatenate([src, dst])
    all_t = np.concatenate([t, t])
    touch = _by_wallet(all_w, all_t, np.concatenate([usd, usd]))
    sends = _by_wallet(src, t, usd)
    gets = _by_wallet(dst, t, usd)

    for w in WINDOWS:
        f[f"src_busy_{w}"] = _window_counts(touch, src, t, w)
        f[f"dst_busy_{w}"] = _window_counts(touch, dst, t, w)
        f[f"src_sends_{w}"] = _window_counts(sends, src, t, w)
        f[f"dst_gets_{w}"] = _window_counts(gets, dst, t, w)

    # pass-through timing: does the receiver move money on, and how fast and how much?
    to_fwd = np.full(len(p), BIG)
    fwd_ratio = np.zeros(len(p))
    since_in = np.full(len(p), BIG)
    for i in range(len(p)):
        out = sends.get(dst[i])
        if out is not None:
            ts, vs = out
            j = np.searchsorted(ts, t[i], "right")
            if j < len(ts):
                to_fwd[i] = ts[j] - t[i]
            k = np.searchsorted(ts, t[i] + 3600, "right")
            fwd_ratio[i] = vs[j:k].sum() / max(usd[i], 1e-9)
        inc = gets.get(src[i])
        if inc is not None:
            ts = inc[0]
            j = np.searchsorted(ts, t[i], "left")
            if j > 0:
                since_in[i] = t[i] - ts[j - 1]
    f["dst_seconds_to_forward"] = np.minimum(to_fwd, BIG)
    f["dst_forward_ratio_1h"] = np.minimum(fwd_ratio, 20.0)
    f["src_seconds_since_funded"] = np.minimum(since_in, BIG)

    # nearest neighbour in time on each wallet, absolute and relative to its own week
    def nearest(table, wallets):
        out = np.full(len(wallets), BIG)
        for i, (w, ti) in enumerate(zip(wallets, t)):
            ts = table[w][0]
            j = np.searchsorted(ts, ti, "left")
            k = np.searchsorted(ts, ti, "right")
            if k - j > 1:          # another payment on this wallet at the same second
                out[i] = 0.0
                continue
            cands = []
            if j > 0:
                cands.append(ti - ts[j - 1])
            if k < len(ts):
                cands.append(ts[k] - ti)
            if cands:
                out[i] = min(cands)
        return out

    week = _wallet_week(p)
    f["src_nearest_gap"] = nearest(touch, src)
    f["dst_nearest_gap"] = nearest(touch, dst)
    f["src_gap_vs_usual"] = f.src_nearest_gap / np.maximum(week.median_gap.reindex(src).to_numpy(), 1)
    f["dst_gap_vs_usual"] = f.dst_nearest_gap / np.maximum(week.median_gap.reindex(dst).to_numpy(), 1)

    # amount relative to what this wallet usually sends / receives
    f["usd_vs_src_usual"] = usd / np.maximum(week.out_med.reindex(src).to_numpy(), 1e-9)
    f["usd_vs_dst_usual"] = usd / np.maximum(week.in_med.reindex(dst).to_numpy(), 1e-9)

    for role, wallets in (("src", src), ("dst", dst)):
        wk = week.reindex(wallets)
        f[f"{role}_week_n"] = wk.total_n.to_numpy()
        f[f"{role}_week_peers"] = wk.peers.to_numpy()
        f[f"{role}_week_flow_ratio"] = wk.flow_ratio.to_numpy()
        f[f"{role}_week_median_gap"] = wk.median_gap.to_numpy()

    # the pair: how often does this sender pay this receiver over the week?
    pair_n = p.groupby(["src", "dst"]).ts.transform("size").to_numpy()
    f["pair_week_n"] = pair_n
    return f


def build_features(directory: str | Path, cache: Path | None = None):
    """Return (payments, feature matrix, motif ids per scale) for one split."""
    directory = Path(directory)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        stem = cache / directory.name
        if (stem.with_suffix(".features.parquet")).exists():
            p = pd.read_parquet(stem.with_suffix(".payments.parquet"))
            X = pd.read_parquet(stem.with_suffix(".features.parquet"))
            ids = pd.read_parquet(stem.with_suffix(".motifs.parquet"))
            return p, X, {int(c): ids[c].to_numpy() for c in ids.columns}

    p = load_payments(directory)
    per_payment = payment_features(p, directory)
    shape, ids = motifs.multiscale(p)
    X = pd.concat([per_payment.reset_index(drop=True), shape], axis=1).astype("float64")

    if cache is not None:
        p.to_parquet(stem.with_suffix(".payments.parquet"))
        X.to_parquet(stem.with_suffix(".features.parquet"))
        pd.DataFrame({str(k): v for k, v in ids.items()}).to_parquet(stem.with_suffix(".motifs.parquet"))
    return p, X, ids
