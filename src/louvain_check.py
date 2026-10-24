"""Idea 2: Louvain community clues, tested against the 67.8 gate. Self-contained: builds its own clues.

  python3 -m experiments.louvain_check                        # screen + honest test, fold seed 42
  python3 -m experiments.louvain_check --fold-seeds 42 7 2026 # three fold seeds (3x slower)
  python3 -m experiments.louvain_check --predict              # only if the honest test says keep

Louvain splits the network into communities: groups of wallets that pay each other a lot
and outsiders rarely. Unlike connected components (one stray payment merges two groups) it
looks at how dense the links are. graph.py found every ordinary wallet connects to every
other over a week, so it is run inside 1-hour and 6-hour blocks (each cut twice, offsets
averaged), with the 8 hubs set aside. Columns:
    comm_size_W      wallets in the sender's community
    same_comm_W      sender and receiver in the same community (0, 0.5 or 1 after averaging)
    comm_density_W   payments inside the sender's community / its wallets
    comm_outside_W   share of that community's payments crossing its edge (hubs count as outside)
A small, dense community fed from outside and paying out looks like a layering ring.
No labels; weights are payment counts; Louvain seed 42 on integer wallet ids, so a fresh
copy reproduces it exactly.

Honest test: the gate's recipe on main.py's folds, out-of-fold for both models, with and
without these clues, compared WITHOUT Violet's 10% share (i.e. against the 63.8 gate).
Keep rule: better in at least 4 of 5 folds on every fold seed. Recent +1 honest-test gains
lost 1.2-1.4 on the leaderboard, so treat a small gain with suspicion.

--predict writes two probability files for the unchanged blend:
  python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_signals_louvain_prob.csv \
      --violet predictions_violet_v2_fixed.csv --structure predictions_scheme_louvain_prob.csv \
      --out predictions_gate_v10_louvain.csv
"""
from __future__ import annotations

import argparse
import json
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from main import make_model, OUTPUTS
from src.predict_shapes_v2 import prepare
from src.predict_union import SHAPES, fit_seeds, violet_clues
from src.predict_union_signals import LEFT_OUT

WINDOWS = {"1h": 3600, "6h": 21600}
SEED = 42


# ---------------------------------------------------------------- the clues

def _encode(payments, week, hub_set):
    """Integer wallet ids (so results don't depend on Python's string hash seed), hub mask, times."""
    if hub_set is None:
        hub_set = graph.hubs(payments, week)
    wallets = np.array(sorted(set(payments.FROM_ADDRESS) | set(payments.TO_ADDRESS)))
    S = np.searchsorted(wallets, payments.FROM_ADDRESS.to_numpy())
    R = np.searchsorted(wallets, payments.TO_ADDRESS.to_numpy())
    T = payments.TIMESTAMP.to_numpy(np.int64)
    return S, R, np.isin(wallets, list(hub_set)), T - T.min()


def _block_pairs(S, R, hub, idx):
    """Payment counts per sender-receiver pair in one block, hubs and self-payments left out."""
    s, r = S[idx], R[idx]
    keep = ~hub[s] & ~hub[r] & (s != r)
    return pd.DataFrame({"s": s[keep], "r": r[keep]}).value_counts().sort_index()


def _windowed(S, R, hub, T, names, block_fn):
    """Run block_fn on 1h and 6h blocks, each cut at offset 0 and half a block, and average."""
    res = {}
    for name, W in WINDOWS.items():
        total = {k: np.zeros(len(T)) for k in names}
        count = {k: np.zeros(len(T)) for k in names}
        for offset in (0, W // 2):
            for idx in pd.Series(np.arange(len(T))).groupby((T + offset) // W).indices.values():
                for k, v in block_fn(S, R, hub, idx).items():
                    ok = ~np.isnan(v)
                    total[k][idx[ok]] += v[ok]
                    count[k][idx[ok]] += 1
        for k in names:
            res[f"{k}_{name}"] = np.where(count[k] > 0, total[k] / np.maximum(count[k], 1), np.nan)
    return res

LV_NAMES = ["comm_size", "same_comm", "comm_density", "comm_outside"]


def _lv_block(S, R, hub, idx):
    out = {k: np.full(len(idx), np.nan) for k in LV_NAMES}
    pairs = _block_pairs(S, R, hub, idx)
    if pairs.empty:
        return out
    U = nx.Graph()
    for (a, b), w in pairs.items():
        U.add_edge(a, b, weight=U[a][b]["weight"] + w if U.has_edge(a, b) else float(w))
    comm_of = {}
    for k, members in enumerate(nx.community.louvain_communities(U, weight="weight", seed=SEED)):
        for v in members:
            comm_of[v] = k
    k_max = max(comm_of.values()) + 1
    size = np.bincount(list(comm_of.values()), minlength=k_max).astype(float)
    inside, edge = np.zeros(k_max), np.zeros(k_max)
    s, r = S[idx], R[idx]
    for a, b in zip(s[s != r], r[s != r]):              # every payment in the block, hubs included
        ca, cb = comm_of.get(a), comm_of.get(b)
        if ca is not None and ca == cb:
            inside[ca] += 1
        else:
            if ca is not None:
                edge[ca] += 1
            if cb is not None:
                edge[cb] += 1
    for j, (a, b) in enumerate(zip(s, r)):
        ca = comm_of.get(a) if not hub[a] else None
        if ca is None:
            continue
        out["comm_size"][j] = size[ca]
        out["comm_density"][j] = inside[ca] / size[ca]
        out["comm_outside"][j] = edge[ca] / max(inside[ca] + edge[ca], 1.0)
        if not hub[b] and b in comm_of:
            out["same_comm"][j] = float(comm_of[b] == ca)
    return out


def add_features(payments, week, hub_set=None):
    """payments: clean table for one week. Returns TRANSACTION_HASH + comm_* columns, input order."""
    S, R, hub, T = _encode(payments, week, hub_set)
    res = pd.DataFrame({"TRANSACTION_HASH": payments.TRANSACTION_HASH.to_numpy()})
    for k, v in _windowed(S, R, hub, T, LV_NAMES, _lv_block).items():
        res[k] = v
    return res


# ---------------------------------------------------------------- the honest test

def tables():
    """Structure-model clues (with once-a-week groups); the combined model drops the once_ columns."""
    t0 = time.time()
    train, test, lab, X, Xt = prepare(once_groups=True, signals=True)
    X, Xt = X.drop(columns=list(LEFT_OUT)), Xt.drop(columns=list(LEFT_OUT))
    Us = pd.concat([X.reset_index(drop=True), violet_clues("train", train.TRANSACTION_HASH)], axis=1)
    Uts = pd.concat([Xt.reset_index(drop=True), violet_clues("test", test.TRANSACTION_HASH)], axis=1)[Us.columns]
    once = [c for c in Us.columns if c.startswith("once_")]
    Uc, Utc = Us.drop(columns=once), Uts.drop(columns=once)

    def net(week, hashes):
        N = add_features(common.payments(week), week)
        return N.set_index("TRANSACTION_HASH").loc[hashes].reset_index(drop=True).astype(float)

    N, Nt = net("train", train.TRANSACTION_HASH), net("test", test.TRANSACTION_HASH)
    print(f"clues: combined {Uc.shape[1]}, structure {Us.shape[1]} ({len(once)} once_), louvain {N.shape[1]}"
          f"  ({time.time() - t0:.0f}s)", flush=True)
    return train, test, lab, Uc, Utc, Us, Uts, N, Nt


def screen(N, y):
    """Each clue alone: average precision in its better direction (chance = share suspicious)."""
    print(f"\nEach clue alone (chance {100 * y.mean():.1f}):")
    rows = []
    for c in N.columns:
        v = N[c].fillna(N[c].min() - 1)
        up, down = aps(y, v), aps(y, -v)
        rows.append((c, 100 * max(up, down), "high" if up >= down else "low", N[c].notna().mean()))
    for c, ap, d, cov in sorted(rows, key=lambda r: -r[1]):
        print(f"  {c:24s} {ap:5.1f}  ({d} = suspicious, {cov:.0%} filled)")


def oof(U, y, folds):
    p = np.zeros(len(y))
    for tr, va in folds:
        p[va] = make_model("lgbm").fit(U.iloc[tr], y[tr]).predict_proba(U.iloc[va])[:, 1]
    return p


def gate(Uc, Us, y, ys, pay, folds):
    combined = graph.smooth_by_group(oof(Uc, y, folds), pay, "train")
    structure = oof(Us, ys, folds)
    return np.log(np.clip(combined, 1e-9, 1)) + np.log(np.clip(structure, 1e-9, 1))


def honest_test(train, lab, Uc, Us, N, fold_seeds):
    y = lab.is_suspicious.to_numpy()
    ys = lab.family.isin(SHAPES).to_numpy().astype(int)
    pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
    groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + pay.FROM_ADDRESS)
    Ucn, Usn = pd.concat([Uc, N], axis=1), pd.concat([Us, N], axis=1)
    summary = {}
    for seed in fold_seeds:
        t0 = time.time()
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Uc, y, groups))
        base, new = gate(Uc, Us, y, ys, pay, folds), gate(Ucn, Usn, y, ys, pay, folds)
        b = [100 * aps(y[va], base[va]) for _, va in folds]
        n = [100 * aps(y[va], new[va]) for _, va in folds]
        better = sum(x > z for x, z in zip(n, b))
        summary[seed] = dict(base=b, with_clues=n, better=better)
        print(f"\nfold seed {seed}: gate {np.mean(b):.1f} +/- {np.std(b):.1f} -> with louvain "
              f"{np.mean(n):.1f} +/- {np.std(n):.1f}  (gain {np.mean(n) - np.mean(b):+.2f}, "
              f"better in {better}/5)  [{time.time() - t0:.0f}s]")
        print("  per fold: " + ", ".join(f"{x:.1f}->{z:.1f}" for x, z in zip(b, n)), flush=True)

    model = make_model("lgbm").fit(Ucn, y)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=Ucn.columns)
    share = gain / gain.sum()
    print(f"\nThese clues' share of the combined model's gain: {share[N.columns].sum():.1%}; top 5:")
    for c, s in share[N.columns].sort_values(ascending=False).head(5).items():
        print(f"  {c:24s} {s:.2%}  (rank {int((share > s).sum()) + 1} of {len(share)})")

    keep = all(r["better"] >= 4 for r in summary.values()) and \
        all(np.mean(r["with_clues"]) > np.mean(r["base"]) for r in summary.values())
    print("\nDECISION: " + ("keep: try --predict and upload" if keep else "drop: log it in DECISIONS.md"))
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "louvain_check.json").write_text(json.dumps(
        {"fold_seeds": summary, "keep": keep, "louvain_gain_share": float(share[N.columns].sum())}, indent=2))


def predict(train, test, lab, Uc, Utc, Us, Uts, N, Nt):
    y = lab.is_suspicious.to_numpy()
    ys = lab.family.isin(SHAPES).to_numpy().astype(int)
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    combined = graph.smooth_by_group(
        fit_seeds(pd.concat([Uc, N], axis=1), y, pd.concat([Utc, Nt], axis=1), 1), test_pay, "test")
    structure = fit_seeds(pd.concat([Us, N], axis=1), ys, pd.concat([Uts, Nt], axis=1), 1)
    OUTPUTS.mkdir(exist_ok=True)
    for name, score in (("predictions_union_signals_louvain_prob.csv", combined),
                        ("predictions_scheme_louvain_prob.csv", structure)):
        pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / name, index=False)
        print(f"wrote {OUTPUTS / name}")
    print("next: python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_signals_louvain_prob.csv "
          "--violet predictions_violet_v2_fixed.csv --structure predictions_scheme_louvain_prob.csv "
          "--out predictions_gate_v10_louvain.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    t = tables()
    if args.predict:
        predict(*t)
        return
    train, test, lab, Uc, Utc, Us, Uts, N, Nt = t
    screen(N, lab.is_suspicious.to_numpy())
    honest_test(train, lab, Uc, Us, N, args.fold_seeds)


if __name__ == "__main__":
    main()
