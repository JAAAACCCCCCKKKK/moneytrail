"""Idea 1: PageRank clues, tested against the 67.8 gate. Self-contained: builds its own clues.

  python3 -m experiments.pagerank_check                        # screen + honest test, fold seed 42
  python3 -m experiments.pagerank_check --fold-seeds 42 7 2026 # three fold seeds (3x slower)
  python3 -m experiments.pagerank_check --predict              # only if the honest test says keep

PageRank: a coin wanders along payments (weighted by how many there are); a wallet's score
is the share of time the coin spends there, so wallets that money keeps flowing towards
score high. graph.py found every ordinary wallet connects to every other over a week, so it
is computed inside 1-hour and 6-hour blocks (each cut twice, offsets averaged), with the 8
hubs set aside. Columns (1 = average wallet in the block):
    pr_recv_W        receiver's PageRank
    prrev_send_W     sender's PageRank on the reversed graph: a big SOURCE of money
    pr_lift_recv_W   receiver's PageRank / its share of incoming payments ("more central than
                     its count suggests": a collector fed by relays)
    pr_recv_week, prrev_send_week   the same over the whole week (expected to be weak)
No labels; weights are payment counts, not USD (amounts drift between weeks); hub sides blank.

Honest test: the gate's recipe on main.py's folds, out-of-fold for both models, with and
without these clues, compared WITHOUT Violet's 10% share (i.e. against the 63.8 gate).
Keep rule: better in at least 4 of 5 folds on every fold seed. Recent +1 honest-test gains
lost 1.2-1.4 on the leaderboard, so treat a small gain with suspicion.

--predict writes two probability files for the unchanged blend:
  python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_signals_pagerank_prob.csv \
      --violet predictions_violet_v2_fixed.csv --structure predictions_scheme_pagerank_prob.csv \
      --out predictions_gate_v10_pagerank.csv
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

PR_NAMES = ["pr_recv", "prrev_send", "pr_lift_recv"]


def _pagerank(G):
    if G.number_of_edges() == 0:
        return {}
    try:
        return nx.pagerank(G, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        return {}


def _pr_block(S, R, hub, idx):
    out = {k: np.full(len(idx), np.nan) for k in PR_NAMES}
    pairs = _block_pairs(S, R, hub, idx)
    if pairs.empty:
        return out
    G = nx.DiGraph()
    G.add_weighted_edges_from((a, b, float(w)) for (a, b), w in pairs.items())
    n, total = G.number_of_nodes(), float(pairs.sum())
    pr, prr = _pagerank(G), _pagerank(G.reverse(copy=False))
    in_w = dict(G.in_degree(weight="weight"))
    for j, (a, b) in enumerate(zip(S[idx], R[idx])):
        if not hub[b] and b in pr:
            out["pr_recv"][j] = pr[b] * n
            out["pr_lift_recv"][j] = pr[b] * n / (1.0 + in_w.get(b, 0.0) * n / total)
        if not hub[a] and a in prr:
            out["prrev_send"][j] = prr[a] * n
    return out


def add_features(payments, week, hub_set=None):
    """payments: clean table for one week. Returns TRANSACTION_HASH + pr_* columns, input order."""
    S, R, hub, T = _encode(payments, week, hub_set)
    res = pd.DataFrame({"TRANSACTION_HASH": payments.TRANSACTION_HASH.to_numpy()})
    for k, v in _windowed(S, R, hub, T, PR_NAMES, _pr_block).items():
        res[k] = v
    whole = _pr_block(S, R, hub, np.arange(len(T)))
    res["pr_recv_week"], res["prrev_send_week"] = whole["pr_recv"], whole["prrev_send"]
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
    print(f"clues: combined {Uc.shape[1]}, structure {Us.shape[1]} ({len(once)} once_), pagerank {N.shape[1]}"
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
        print(f"\nfold seed {seed}: gate {np.mean(b):.1f} +/- {np.std(b):.1f} -> with pagerank "
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
    (OUTPUTS / "pagerank_check.json").write_text(json.dumps(
        {"fold_seeds": summary, "keep": keep, "pagerank_gain_share": float(share[N.columns].sum())}, indent=2))


def predict(train, test, lab, Uc, Utc, Us, Uts, N, Nt):
    y = lab.is_suspicious.to_numpy()
    ys = lab.family.isin(SHAPES).to_numpy().astype(int)
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    combined = graph.smooth_by_group(
        fit_seeds(pd.concat([Uc, N], axis=1), y, pd.concat([Utc, Nt], axis=1), 1), test_pay, "test")
    structure = fit_seeds(pd.concat([Us, N], axis=1), ys, pd.concat([Uts, Nt], axis=1), 1)
    OUTPUTS.mkdir(exist_ok=True)
    for name, score in (("predictions_union_signals_pagerank_prob.csv", combined),
                        ("predictions_scheme_pagerank_prob.csv", structure)):
        pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / name, index=False)
        print(f"wrote {OUTPUTS / name}")
    print("next: python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_signals_pagerank_prob.csv "
          "--violet predictions_violet_v2_fixed.csv --structure predictions_scheme_pagerank_prob.csv "
          "--out predictions_gate_v10_pagerank.csv")


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
