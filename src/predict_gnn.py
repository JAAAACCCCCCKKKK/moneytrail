"""A graph neural network over payments (Jack).

  python -m src.predict_gnn --cv     # honest test against the 67.1 recipe (main.py folds)
  python -m src.predict_gnn          # outputs/predictions_gnn.csv

Wallet identity carries no signal here (wallets are reused at random and differ between weeks),
but how payments connect in time does. So the nodes are payments, and two payments are linked
when they share a non-hub wallet within 2 hours (the candidate pairs of src.predict_linker), in
three relation types: money flowing in through the shared wallet (the earlier payment arrived
where the later one leaves from), flowing out, and siblings (same sender or same receiver). A
2-layer relational GraphSAGE with mean pooling, on top of the 527 clues of src.predict_two_q,
learns how to pool evidence from connected payments, instead of the fixed 2-10 minute wallet
groups used for smoothing. It is inductive: trained on the training week's graph, it scores the
test week's graph. Full-batch training on CPU, fixed seeds, single thread for determinism.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer
from torch import nn

import common
from main import SEED, OUTPUTS
from src.predict_linker import candidate_pairs
from src.predict_two_q import tables

HIDDEN, LAYERS, DROPOUT = 64, 2, 0.2
EPOCHS, LR, WEIGHT_DECAY, POS_WEIGHT = 60, 5e-3, 1e-4, 5.0

torch.set_num_threads(8)                    # fixed thread count: identical results on every run


def mean_operator(rows, cols, n):
    """Sparse matrix that averages each row's neighbours (rows with none stay zero)."""
    if len(rows) == 0:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (n, n)).coalesce()
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    vals = 1.0 / deg[rows]
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


def payment_graph(pay: pd.DataFrame, week: str):
    """Three averaging operators: upstream (money in), downstream (money out), siblings."""
    J, I, X = candidate_pairs(pay, week)
    flow = X.flow.to_numpy() == 1                # j received at the shared wallet, i sent from it
    n = len(pay)
    up = mean_operator(I[flow], J[flow], n)      # for the later payment: where its money came from
    down = mean_operator(J[flow], I[flow], n)    # for the earlier payment: where its money went
    sib_r, sib_c = np.r_[J[~flow], I[~flow]], np.r_[I[~flow], J[~flow]]
    return up, down, mean_operator(sib_r, sib_c, n)


class PaymentGNN(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.inp = nn.Linear(n_in, HIDDEN)
        self.layers = nn.ModuleList(nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(4)) for _ in range(LAYERS))
        self.out = nn.Linear(HIDDEN, 1)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, ops):
        h = self.drop(torch.relu(self.inp(x)))
        for own, *rel in self.layers:
            msg = own(h) + sum(lin(torch.sparse.mm(op, h)) for lin, op in zip(rel, ops))
            h = self.drop(torch.relu(msg)) + h        # residual
        return self.out(h).squeeze(-1)


def train_gnn(x, ops, y, mask, seed=SEED):
    torch.manual_seed(seed)
    model = PaymentGNN(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT))
    yt, mt = torch.tensor(y, dtype=torch.float32), torch.tensor(mask)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x, ops)[mt], yt[mt])
        loss.backward()
        opt.step()
    return model


def predict(model, x, ops):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, ops)).numpy().astype(float)


def node_features(U2: pd.DataFrame, Ut2: pd.DataFrame):
    """Every clue mapped to a normal shape on the training week; blanks become the middle (0)."""
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0).fit(U2.to_numpy(float))
    to = lambda F: torch.tensor(np.nan_to_num(qt.transform(F.to_numpy(float)), nan=0.0), dtype=torch.float32)
    return to(U2), to(Ut2)


def rank01(s):
    return pd.Series(s).rank(pct=True).to_numpy()


def cross_validate():
    """GNN alone and rank-blended with the 67.1 recipe, on main.py's folds, three fold seeds."""
    import graph
    from main import make_model
    from src.predict_two_q import NOT_SCHEME
    t0 = time.time()
    train, _, lab, U, _, U2, Ut2 = tables()
    y = lab.is_suspicious.to_numpy()
    scheme = (~lab.family.isin(NOT_SCHEME)).to_numpy().astype(int)
    pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
    x, _ = node_features(U2, Ut2)
    ops = payment_graph(pay, "train")
    groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + pay.FROM_ADDRESS)
    print(f"graph ready ({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for fs in (SEED, 7, 2026):
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(U, y, groups))
        g, a1, ps, pw = (np.zeros(len(y)) for _ in range(4))
        for tr, va in folds:
            mask = np.zeros(len(y), bool)
            mask[tr] = True
            g[va] = predict(train_gnn(x, ops, y, mask), x, ops)[va]
            trs = tr[scheme[tr] == 1]
            a1[va] = make_model("lgbm").fit(U.iloc[tr], y[tr]).predict_proba(U.iloc[va])[:, 1]
            ps[va] = make_model("lgbm").fit(U2.iloc[tr], scheme[tr]).predict_proba(U2.iloc[va])[:, 1]
            pw[va] = make_model("lgbm").fit(U2.iloc[trs], y[trs]).predict_proba(U2.iloc[va])[:, 1]
        recipe = 0.5 * rank01(graph.smooth_by_group(a1, pay, "train")) + 0.5 * rank01(ps * pw)
        base = [100 * aps(y[va], recipe[va]) for _, va in folds]
        top400 = lambda s: np.mean(y[np.argsort(-s)[:400]])
        for name, s in (("67.1 recipe", recipe), ("GNN alone", g),
                        ("recipe 75% + GNN 25%", 0.75 * recipe + 0.25 * rank01(g)),
                        ("recipe 50% + GNN 50%", 0.5 * recipe + 0.5 * rank01(g))):
            per = [100 * aps(y[va], s[va]) for _, va in folds]
            better = sum(a > b for a, b in zip(per, base))
            summary.setdefault(name, []).append((100 * aps(y, s), np.mean(per), better))
            print(f"  seed {fs} {name:22s} pooled {100 * aps(y, s):5.1f} folds {np.mean(per):.1f} "
                  f"better {better}/5 | top-400 precision {top400(s):.3f}", flush=True)
        print(f"--- fold seed {fs} done ({time.time() - t0:.0f}s)", flush=True)
    print("\nSUMMARY (3 fold seeds: pooled, fold mean, folds better than the 67.1 recipe of 15)")
    for k, v in summary.items():
        v = np.array(v)
        print(f"  {k:22s} {v[:, 0].mean():5.1f} {v[:, 1].mean():5.1f} {int(v[:, 2].sum()):2d}/15")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--out", default="predictions_gnn.csv", help="file name inside outputs/")
    args = parser.parse_args()
    if args.cv:
        cross_validate()
        return
    train, test, lab, _, _, U2, Ut2 = tables()
    y = lab.is_suspicious.to_numpy()
    x, xt = node_features(U2, Ut2)
    train_pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    model = train_gnn(x, payment_graph(train_pay, "train"), y, np.ones(len(y), bool))
    raw = predict(model, xt, payment_graph(test_pay, "test"))
    score = pd.Series(raw).rank(method="first").to_numpy() / len(raw)   # no ties
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(score):,} test scores to {OUTPUTS / args.out}")


if __name__ == "__main__":
    main()
