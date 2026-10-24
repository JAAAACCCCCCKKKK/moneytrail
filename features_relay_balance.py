"""Hard relay cases: does the relay wallet only pass on money it has already received? (Jack)

First principles: a launderer's relay wallet has no money of its own, so within its scheme its
running balance never goes negative; an honest relay service runs on its own float and can pay out
first. On the training week, with true scheme grouping, all 12 laundering relay schemes never go
negative against 29% of honest ones (AUC 0.80). But a scheme is only about 11% of its central
wallet's week, so the balance must be taken over that wallet's own scheme payments, not its week.

Step 1 (learned membership): for every non-hub wallet c and each of its payments k, a LightGBM
predicts whether k belongs to a relay scheme centred on c, from k's direction, how many of c's
payments go the same way within +-1h, whether opposite-direction payments follow 2-8 hours later
or came 2-8 hours earlier (collect, then disburse), the counterparty's activity, how often the pair
recurs, and k's amount against c's usual amount. Trained on the training week only (target from
scenario_id and family, never a clue); training-week rows get out-of-fold predictions, folds by wallet.
Step 2: c's running balance over its payments weighted by that membership, in time order.
Each payment gets these numbers for its sender and its receiver.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common
import graph

RELAY = ("aggregate_relay", "distributed_relay")
H = 3600
CLUES = ["into", "same_1h", "all_1h", "opp_after", "opp_before", "val_rel", "c_week", "other_week", "pair_n"]


def _rows(p: pd.DataFrame, week: str) -> pd.DataFrame:
    """One row per (non-hub wallet c, payment k touching c), with membership clues."""
    hubs = graph.hubs(p, week)
    T, A, B, V = p.TIMESTAMP.to_numpy(), p.FROM_ADDRESS.to_numpy(), p.TO_ADDRESS.to_numpy(), p.VALUE.to_numpy()
    week_n = pd.concat([p.FROM_ADDRESS, p.TO_ADDRESS]).value_counts()
    pairs = pd.Series([a + "|" + b if a < b else b + "|" + a for a, b in zip(A, B)])
    pair_n = pairs.map(pairs.value_counts()).to_numpy()
    idx: dict[str, list[int]] = {}
    for i, (a, b) in enumerate(zip(A, B)):
        if a not in hubs:
            idx.setdefault(a, []).append(i)
        if b not in hubs:
            idx.setdefault(b, []).append(i)
    out = []
    for c in sorted(idx):
        rows = np.array(idx[c])
        t = T[rows]
        into = (B[rows] == c).astype(int)
        t_in, t_out = t[into == 1], t[into == 0]

        def cnt(ts, lo, hi):
            return np.searchsorted(ts, t + hi, "right") - np.searchsorted(ts, t + lo, "left")

        other = np.where(into == 1, A[rows], B[rows])
        out.append(pd.DataFrame({
            "c": c, "k": rows, "t": t, "into": into, "value": V[rows],
            "same_1h": np.where(into == 1, cnt(t_in, -H, H), cnt(t_out, -H, H)) - 1,
            "all_1h": cnt(t, -H, H) - 1,
            "opp_after": np.where(into == 1, cnt(t_out, 2 * H, 8 * H), cnt(t_in, 2 * H, 8 * H)),
            "opp_before": np.where(into == 1, cnt(t_out, -8 * H, -2 * H), cnt(t_in, -8 * H, -2 * H)),
            "val_rel": np.log((V[rows] + 1e-9) / (np.median(V[rows]) + 1e-9)),
            "c_week": week_n[c], "other_week": week_n.loc[other].to_numpy(), "pair_n": pair_n[rows]}))
    return pd.concat(out, ignore_index=True)


def _target(R: pd.DataFrame, p: pd.DataFrame) -> np.ndarray:
    """1 if payment k belongs to a relay scenario whose central (most frequent) wallet is c."""
    lab = p[["TRANSACTION_HASH"]].merge(common.labels(), on="TRANSACTION_HASH", how="left")
    rel = np.flatnonzero(lab.family.isin(RELAY).to_numpy())
    s = pd.DataFrame({"sid": np.r_[lab.scenario_id.to_numpy()[rel], lab.scenario_id.to_numpy()[rel]],
                      "w": np.r_[p.FROM_ADDRESS.to_numpy()[rel], p.TO_ADDRESS.to_numpy()[rel]]})
    centre = s.groupby(["sid", "w"]).size().reset_index(name="n") \
        .sort_values(["sid", "n", "w"], ascending=[True, False, True]).drop_duplicates("sid").set_index("sid").w
    k_sid = lab.scenario_id.to_numpy()[R.k.to_numpy()]
    k_centre = pd.Series(k_sid).map(centre).to_numpy()
    return (k_centre == R.c.to_numpy()).astype(int)


def _model():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              random_state=42, n_jobs=8, verbose=-1)


def _balance(R: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Membership-weighted running balance of each wallet; then per payment, sender's and receiver's."""
    R = R.sort_values(["c", "t", "k"]).reset_index(drop=True)
    v = np.where(R.into == 1, R.value, -R.value) * R.member
    R = R.assign(v=v, run=pd.Series(v).groupby(R.c).cumsum().to_numpy())
    W = pd.DataFrame({
        "tin": R.v.clip(lower=0).groupby(R.c).sum(),
        "tout": (-R.v).clip(lower=0).groupby(R.c).sum(),
        "min_run": R.run.groupby(R.c).min(),
        "mass": R.member.groupby(R.c).sum(),
        "neg": ((R.run < -1e-9) * R.member).groupby(R.c).sum()})
    W["min_bal"] = W.min_run / W.tin.clip(lower=1e-9)
    W["out_in"] = W.tout / W.tin.clip(lower=1e-9)
    W["neg_share"] = W.neg / W.mass.clip(lower=1e-9)
    member = R.set_index(["c", "k"]).member
    cols = {"TRANSACTION_HASH": P.TRANSACTION_HASH.to_numpy()}
    for side, wal in (("snd", P.FROM_ADDRESS.to_numpy()), ("rcv", P.TO_ADDRESS.to_numpy())):
        w = W.reindex(wal)
        for name in ("min_bal", "neg_share", "out_in", "mass"):
            cols[f"relay_{side}_{name}"] = w[name].to_numpy()
        cols[f"relay_{side}_member"] = member.reindex(pd.MultiIndex.from_arrays([wal, np.arange(len(P))])).to_numpy()
    # the side the membership model thinks is the relay's centre, and that wallet's balance profile
    snd_c = np.nan_to_num(cols["relay_snd_member"], nan=-1) >= np.nan_to_num(cols["relay_rcv_member"], nan=-1)
    for name in ("min_bal", "neg_share", "out_in", "mass", "member"):
        cols[f"relay_ctr_{name}"] = np.where(snd_c, cols[f"relay_snd_{name}"], cols[f"relay_rcv_{name}"])
    return pd.DataFrame(cols)


def build(train_p: pd.DataFrame, test_p: pd.DataFrame):
    """Clue tables for both weeks, rows in each payments table's order (indexes must be 0..n-1)."""
    R = _rows(train_p, "train")
    y = _target(R, train_p)
    member = np.zeros(len(R))
    for tr, va in GroupKFold(5).split(R, y, R.c):
        member[va] = _model().fit(R[CLUES].iloc[tr], y[tr]).predict_proba(R[CLUES].iloc[va])[:, 1]
    R["member"] = member
    Rt = _rows(test_p, "test")
    Rt["member"] = _model().fit(R[CLUES], y).predict_proba(Rt[CLUES])[:, 1]
    return _balance(R, train_p), _balance(Rt, test_p), R, y
