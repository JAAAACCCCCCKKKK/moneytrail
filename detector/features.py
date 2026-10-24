"""Per-transaction features for Decode & Detect. Input: base table from base.load()."""
import numpy as np, pandas as pd
from collections import defaultdict

WINDOWS = [300, 900, 1800, 3600, 7200, 4 * 3600, 8 * 3600, 24 * 3600]


def _idx(times_by, key, lo, hi):
    t = times_by.get(key)
    if t is None:
        return 0, 0
    return np.searchsorted(t, lo, "left"), np.searchsorted(t, hi, "right")


def make_features(tx):
    df = tx.sort_values(["TIMESTAMP", "TRANSACTION_INDEX"]).reset_index(drop=True)
    n = len(df)
    T = df.TIMESTAMP.to_numpy(np.int64)
    S = df.ACCOUNT.to_numpy(); R = df.DEST.to_numpy()
    V = df.DEL_USD.to_numpy(float)
    f = {"TRANSACTION_HASH": df.TRANSACTION_HASH.to_numpy()}

    # index structures: per account, sorted send / receive events
    send_rows = defaultdict(list); recv_rows = defaultdict(list)
    for i in range(n):
        send_rows[S[i]].append(i); recv_rows[R[i]].append(i)
    send_rows = {k: np.array(v) for k, v in send_rows.items()}
    recv_rows = {k: np.array(v) for k, v in recv_rows.items()}
    send_t = {k: T[v] for k, v in send_rows.items()}
    recv_t = {k: T[v] for k, v in recv_rows.items()}

    # ---- simple payment features
    f["log_usd"] = np.log1p(V)
    f["usd_round10"] = (np.abs(V - np.round(V, -1)) < 0.5).astype(float)
    f["is_conv"] = df.IS_CONV.to_numpy(float)
    f["n_legs"] = df.N_LEGS.to_numpy(float)
    f["conv_loss"] = df.CONV_LOSS.to_numpy(float)
    f["sm_slack"] = df.SM_SLACK.fillna(0).to_numpy(float)
    f["hour_of_day"] = (T % 86400) / 3600.0
    f["day"] = ((T - T.min()) // 86400).astype(float)
    f["sequence"] = df.SEQUENCE.to_numpy(float)

    # ---- account-level (whole week)
    acc = pd.DataFrame({"a": np.concatenate([S, R])}).a.unique()
    out_cnt = pd.Series(S).value_counts(); in_cnt = pd.Series(R).value_counts()
    out_usd = pd.Series(V, index=S).groupby(level=0).sum(); in_usd = pd.Series(V, index=R).groupby(level=0).sum()
    out_deg = pd.DataFrame({"s": S, "r": R}).groupby("s").r.nunique(); in_deg = pd.DataFrame({"s": S, "r": R}).groupby("r").s.nunique()
    med_out = pd.Series(V, index=S).groupby(level=0).median(); med_in = pd.Series(V, index=R).groupby(level=0).median()
    first_seen = pd.concat([pd.Series(T, index=S), pd.Series(T, index=R)]).groupby(level=0).min()
    seq_min = pd.Series(df.SEQUENCE.to_numpy(float), index=S).groupby(level=0).min()
    for role, A in (("s", S), ("r", R)):
        a = pd.Index(A)
        f[f"{role}_out_cnt"] = out_cnt.reindex(a).fillna(0).to_numpy(float)
        f[f"{role}_in_cnt"] = in_cnt.reindex(a).fillna(0).to_numpy(float)
        f[f"{role}_out_deg"] = out_deg.reindex(a).fillna(0).to_numpy(float)
        f[f"{role}_in_deg"] = in_deg.reindex(a).fillna(0).to_numpy(float)
        o = out_usd.reindex(a).fillna(0).to_numpy(float); i_ = in_usd.reindex(a).fillna(0).to_numpy(float)
        f[f"{role}_pass_through"] = np.clip(o / (i_ + 1e-9), 0, 5)
        f[f"{role}_log_flow"] = np.log1p(o + i_)
        f[f"{role}_seq_min"] = seq_min.reindex(a).fillna(-1).to_numpy(float)
        f[f"{role}_first_seen_h"] = (first_seen.reindex(a).to_numpy(float) - T.min()) / 3600
    f["usd_vs_s_med"] = np.log1p(V) - np.log1p(med_out.reindex(pd.Index(S)).to_numpy(float))
    f["usd_vs_r_med"] = np.log1p(V) - np.log1p(med_in.reindex(pd.Index(R)).to_numpy(float))

    # ---- pair history
    pair = pd.Series(1, index=pd.MultiIndex.from_arrays([S, R])).groupby(level=[0, 1]).sum()
    f["pair_cnt"] = pair.reindex(pd.MultiIndex.from_arrays([S, R])).to_numpy(float)
    f["pair_rev_cnt"] = pair.reindex(pd.MultiIndex.from_arrays([R, S])).fillna(0).to_numpy(float)
    pair_amt_std = pd.Series(np.log1p(V), index=pd.MultiIndex.from_arrays([S, R])).groupby(level=[0, 1]).std()
    f["pair_logamt_std"] = pair_amt_std.reindex(pd.MultiIndex.from_arrays([S, R])).fillna(-1).to_numpy(float)
    # previous payment of the same pair: gap
    gap = np.full(n, 1e7)
    last = {}
    for i in range(n):
        k = (S[i], R[i])
        if k in last:
            gap[i] = T[i] - last[k]
        last[k] = T[i]
    f["pair_prev_gap_log"] = np.log1p(gap)

    # ---- windowed activity around each payment
    for W in WINDOWS:
        tag = f"{W // 60}m"
        r_in, r_in_ds, s_out, s_out_ds = (np.zeros(n) for _ in range(4))
        r_fwd, r_fwd_usd, s_pre, s_pre_usd = (np.zeros(n) for _ in range(4))
        s_in_w, r_out_w = np.zeros(n), np.zeros(n)
        for i in range(n):
            t = T[i]
            lo, hi = _idx(recv_t, R[i], t - W, t + W)
            rows = recv_rows[R[i]][lo:hi]
            r_in[i] = len(rows); r_in_ds[i] = len(set(S[rows]))
            lo, hi = _idx(send_t, S[i], t - W, t + W)
            rows = send_rows[S[i]][lo:hi]
            s_out[i] = len(rows); s_out_ds[i] = len(set(R[rows]))
            # receiver sends on after t (within W)
            lo, hi = _idx(send_t, R[i], t, t + W)
            if R[i] in send_rows:
                rows = send_rows[R[i]][lo:hi]; r_fwd[i] = len(rows); r_fwd_usd[i] = V[rows].sum()
            # sender received before t (within W)
            lo, hi = _idx(recv_t, S[i], t - W, t)
            if S[i] in recv_rows:
                rows = recv_rows[S[i]][lo:hi]; s_pre[i] = len(rows); s_pre_usd[i] = V[rows].sum()
            if S[i] in recv_rows:
                lo, hi = _idx(recv_t, S[i], t - W, t + W); rows = recv_rows[S[i]][lo:hi]; s_in_w[i] = len(set(S[rows]))
            if R[i] in send_rows:
                lo, hi = _idx(send_t, R[i], t - W, t + W); rows = send_rows[R[i]][lo:hi]; r_out_w[i] = len(set(R[rows]))
        f[f"r_in_{tag}"] = r_in; f[f"r_in_ds_{tag}"] = r_in_ds
        f[f"s_out_{tag}"] = s_out; f[f"s_out_ds_{tag}"] = s_out_ds
        f[f"r_fwd_{tag}"] = r_fwd; f[f"r_fwd_ratio_{tag}"] = np.clip(r_fwd_usd / V, 0, 10)
        f[f"s_pre_{tag}"] = s_pre; f[f"s_pre_ratio_{tag}"] = np.clip(s_pre_usd / V, 0, 10)
        f[f"s_in_{tag}"] = s_in_w; f[f"r_out_{tag}"] = r_out_w
        # burstiness vs the account's own weekly rate
        span = 604800.0
        f[f"r_in_burst_{tag}"] = r_in / (f["r_in_cnt"] * 2 * W / span + 1e-3)
        f[f"s_out_burst_{tag}"] = s_out / (f["s_out_cnt"] * 2 * W / span + 1e-3)

    # ---- time to next forward / since last receive (uncapped, log)
    nxt = np.full(n, 1e7); prv = np.full(n, 1e7)
    for i in range(n):
        if R[i] in send_t:
            t = send_t[R[i]]; j = np.searchsorted(t, T[i], "right")
            if j < len(t): nxt[i] = t[j] - T[i]
        if S[i] in recv_t:
            t = recv_t[S[i]]; j = np.searchsorted(t, T[i], "left")
            if j > 0: prv[i] = T[i] - t[j - 1]
    f["r_next_send_log"] = np.log1p(nxt); f["s_prev_recv_log"] = np.log1p(prv)

    # ---- temporal motif: grow a time-respecting component around each payment
    # within a window: neighbours = payments sharing an account within +-W.
    for W in (900, 3 * 3600):
        tag = f"{W // 60}m"
        comp_tx, comp_acc, comp_cycle, comp_dur = (np.zeros(n) for _ in range(4))
        for i in range(n):
            seen = {i}; frontier = [i]; accs = {S[i], R[i]}
            t0 = T[i]
            while frontier and len(seen) < 60:
                j = frontier.pop()
                for a in (S[j], R[j]):
                    for rows_by, t_by in ((send_rows, send_t), (recv_rows, recv_t)):
                        if a not in rows_by: continue
                        lo, hi = _idx(t_by, a, T[j] - W, T[j] + W)
                        for k in rows_by[a][lo:hi]:
                            if k not in seen and abs(T[k] - t0) <= 4 * W:
                                seen.add(k); frontier.append(k); accs.add(S[k]); accs.add(R[k])
            rows = np.fromiter(seen, int)
            comp_tx[i] = len(rows); comp_acc[i] = len(accs)
            comp_dur[i] = T[rows].max() - T[rows].min()
            # cycle: some account in the component both sends and receives within it
            ss = set(S[rows]); rr = set(R[rows])
            comp_cycle[i] = len(ss & rr) / max(len(accs), 1)
        f[f"comp_tx_{tag}"] = comp_tx; f[f"comp_acc_{tag}"] = comp_acc
        f[f"comp_density_{tag}"] = comp_tx / np.maximum(comp_acc, 1)
        f[f"comp_through_{tag}"] = comp_cycle; f[f"comp_dur_{tag}"] = comp_dur

    # ---- does money come back to the sender within 3h (time-respecting path, <=6 hops)?
    back = np.zeros(n); back_hops = np.full(n, 9.0)
    H = 3 * 3600
    for i in range(n):
        target = S[i]; frontier = [(R[i], T[i], 1)]; visited = {R[i]}
        found = False
        while frontier and not found:
            nf = []
            for a, t, h in frontier:
                if a not in send_t: continue
                lo, hi = _idx(send_t, a, t, T[i] + H)
                for k in send_rows[a][lo:hi][:15]:
                    if R[k] == target:
                        back[i] = 1; back_hops[i] = h + 1; found = True; break
                    if h < 5 and R[k] not in visited:
                        visited.add(R[k]); nf.append((R[k], T[k], h + 1))
                if found: break
            frontier = nf[:200]
        pass
    f["returns_3h"] = back; f["return_hops"] = back_hops

    out = pd.DataFrame(f)
    for c in out.columns:
        if c != "TRANSACTION_HASH": out[c] = out[c].astype(float)
    return out


def gap_features(tx):
    """Gaps to the nearest other payments touching the sender or receiver (motif tightness)."""
    df = tx.sort_values(["TIMESTAMP", "TRANSACTION_INDEX"]).reset_index(drop=True)
    T = df.TIMESTAMP.to_numpy(np.int64); S = df.ACCOUNT.to_numpy(); R = df.DEST.to_numpy(); n = len(df)
    ev = defaultdict(list)
    for i in range(n):
        ev[S[i]].append(i); ev[R[i]].append(i)
    ev = {k: np.array(v) for k, v in ev.items()}; et = {k: T[v] for k, v in ev.items()}
    out = {"TRANSACTION_HASH": df.TRANSACTION_HASH.to_numpy()}
    g = np.full((n, 4), 1e6)
    gs = np.full((n, 2), 1e6); gr = np.full((n, 2), 1e6)
    for i in range(n):
        gaps = []
        for a, store in ((S[i], gs), (R[i], gr)):
            t = et[a]; j = np.searchsorted(t, T[i])
            lo, hi = max(0, j - 4), min(len(t), j + 5)
            d = np.abs(t[lo:hi] - T[i]); rows = ev[a][lo:hi]
            d = np.sort(d[rows != i]); gaps.extend(d.tolist())
            store[i, :min(2, len(d))] = d[:2]
        gaps = np.sort(gaps)[:4]
        g[i, :len(gaps)] = gaps
    for k in range(4): out[f"gap{k+1}_log"] = np.log1p(g[:, k])
    for k in range(2):
        out[f"s_gap{k+1}_log"] = np.log1p(gs[:, k]); out[f"r_gap{k+1}_log"] = np.log1p(gr[:, k])
    o = pd.DataFrame(out)
    # 2-hop tightness: mean of neighbours' nearest gap
    return o
