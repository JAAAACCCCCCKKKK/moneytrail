"""Gap clusters: link payments that share a wallet and happen within G seconds; describe each cluster."""
import numpy as np, pandas as pd

def gap_clusters(T, S, R, G):
    n = len(T); par = np.arange(n)
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    last = {}
    for i in range(n):
        for a in (S[i], R[i]):
            if a in last and T[i] - T[last[a]] <= G:
                par[find(i)] = find(last[a])
            last[a] = i
    return np.array([find(i) for i in range(n)])

def cluster_features(base, Gs=(120, 300, 600, 900)):
    df = base.sort_values(["TIMESTAMP", "TRANSACTION_INDEX"]).reset_index(drop=True)
    T = df.TIMESTAMP.to_numpy(np.int64); S = df.ACCOUNT.to_numpy(); R = df.DEST.to_numpy()
    V = df.DEL_USD.to_numpy(float); SV = df.SRC_USD.to_numpy(float)
    week_cnt = pd.concat([df.ACCOUNT, df.DEST]).value_counts()
    pair_cnt = df.groupby(["ACCOUNT", "DEST"]).size()
    pc = pair_cnt.reindex(pd.MultiIndex.from_arrays([S, R])).to_numpy(float)
    out = pd.DataFrame({"TRANSACTION_HASH": df.TRANSACTION_HASH})
    for G in Gs:
        c = gap_clusters(T, S, R, G)
        d = pd.DataFrame({"c": c, "T": T, "S": S, "R": R, "V": V, "SV": SV, "pc": pc, "conv": df.IS_CONV.to_numpy(),
                          "asset": df.DEL_ASSET.to_numpy()})
        rows = []
        for cid, g in d.groupby("c", sort=False):
            n = len(g); t = np.sort(g["T"].to_numpy())
            senders = set(g.S); receivers = set(g.R); accs = senders | receivers; mid = senders & receivers
            gaps = np.diff(t) if n > 1 else np.array([0.0])
            inn = g.groupby("R").V.sum(); outv = g.groupby("S").SV.sum()
            m = list(mid)
            cons = outv[m].sum() / inn[m].sum() if m else 0.0
            indeg = g.groupby("R").S.nunique().max(); outdeg = g.groupby("S").R.nunique().max()
            wk = week_cnt.reindex(list(accs)).to_numpy(float)
            rows.append((cid, n, len(accs), t[-1] - t[0], np.median(gaps), gaps.max(), len(mid) / len(accs),
                         cons, indeg, outdeg, g.V.std() / (g.V.mean() + 1e-9) if n > 1 else 0, g.conv.mean(),
                         g.asset.nunique(), (g.pc > 1).mean(), np.log1p(g.V.sum()), wk.mean(),
                         n / max(len(accs), 1), len(senders - receivers), len(receivers - senders)))
        cf = pd.DataFrame(rows, columns=["c", "n", "accs", "span", "gap_med", "gap_max", "mid_frac", "cons", "indeg", "outdeg",
                                         "cv", "conv", "assets", "repeat_pairs", "log_total", "acc_week", "density", "n_src", "n_sink"]).set_index("c")
        cf["span_per_tx"] = cf.span / cf.n
        f = cf.loc[c].reset_index(drop=True); f.columns = [f"cl{G}_{x}" for x in f.columns]
        out = pd.concat([out, f], axis=1)
    return out
