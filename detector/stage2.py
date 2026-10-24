"""Neighbour-score features: how suspicious are nearby payments (same wallets, nearby time)?"""
import numpy as np, pandas as pd
from collections import defaultdict
def neighbour_features(base, F, score, windows=(900, 3600, 4*3600, 12*3600)):
    d = base[["TRANSACTION_HASH","TIMESTAMP","ACCOUNT","DEST"]].merge(F[["TRANSACTION_HASH"]], on="TRANSACTION_HASH")
    d = d.set_index("TRANSACTION_HASH").loc[F.TRANSACTION_HASH].reset_index()
    T = d.TIMESTAMP.to_numpy(np.int64); S = d.ACCOUNT.to_numpy(); R = d.DEST.to_numpy(); P = np.asarray(score, float)
    rows = defaultdict(list)
    for i in range(len(d)):
        rows[S[i]].append(i); rows[R[i]].append(i)
    rows = {k: np.array(sorted(set(v), key=lambda j: T[j])) for k, v in rows.items()}
    tim = {k: T[v] for k, v in rows.items()}
    out = {"TRANSACTION_HASH": d.TRANSACTION_HASH.to_numpy()}
    for W in windows:
        mx, mean, cnt_hi, n = (np.zeros(len(d)) for _ in range(4))
        for i in range(len(d)):
            nb = []
            for a in (S[i], R[i]):
                lo = np.searchsorted(tim[a], T[i] - W, "left"); hi = np.searchsorted(tim[a], T[i] + W, "right")
                nb.append(rows[a][lo:hi])
            nb = np.unique(np.concatenate(nb)); nb = nb[nb != i]
            if len(nb):
                p = P[nb]; mx[i] = p.max(); mean[i] = p.mean(); cnt_hi[i] = (p > 0.3).sum(); n[i] = len(nb)
        tag = f"{W//60}m"
        out[f"nb_max_{tag}"] = mx; out[f"nb_mean_{tag}"] = mean; out[f"nb_hi_{tag}"] = cnt_hi
    out["self_score"] = P
    return pd.DataFrame(out)
