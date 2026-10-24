"""Hard relays: learn to group each relay scheme, then score the whole scheme.

A hard relay runs three bursts across the week (days 0, 3 and 6) through the same one or two
middle wallets, with new outside wallets each burst. In each cohort of 100 look-alike relays the
guilty one always moves money forward (the middle receives, then pays on), while half the
innocent ones run backwards (pay out first, topped up after); and the guilty one passes on a
steadier share of the money in each burst. Both only show once the scheme is grouped as a whole,
and the middle wallets carry ~110 other payments each, so the grouping is learned:

1. Candidate links between payments: money into a wallet and money out of it within 6 hours,
   and repeats of the same sender-receiver pair (at most 6 in the week).
2. A LightGBM link model learns "same scheme?" on the training week (`scenario_id` is only the
   training target, never a clue). Honest test (5-fold, same folds as the gate): link AP 0.80
   (0.90 with conversion and same-asset clues, left out because they shift in the test week).
3. Around each repeating pair the model joins across days, gather each repeat's directly linked
   payments (one link away) as one burst. Per burst: money paid out of the pair's wallets to
   outside wallets / money paid in from outside, and whether it left after it arrived.
4. Multiplier on the gate, only where the structure model says "not a planted shape scheme"
   (probability < 0.3): backward-looking schemes x e^-4; forward ones x e^(4 (steadiness - 0.5)),
   steadiness = rank of how little the per-burst share varies.

Honest test (out-of-fold scores, same folds), gate + 10% Violet as in the 67.8 entry: 63.72 ->
63.97 (fold mean 64.28 -> 64.54, better in 5 of 5). Ceiling with the true scheme grouping: 66.4.
Training week: 19% of payments land in a relay scheme; test week: 8%.

  python3 -m src.relay_groups      # writes outputs/relay_multiplier.csv for the test week
  python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_prob.csv \\
      --violet predictions_violet_v2_fixed.csv --structure predictions_structure_prob.csv \\
      --relay relay_multiplier.csv --out predictions_gate_relay_v10.csv
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
from main import OUTPUTS
from src.data import load_payments

H = 6 * 3600          # hand-on window
SEED = 42


def candidates(p: pd.DataFrame):
    """Candidate links (i, j) and their clues. kind 0: payment i into a wallet, payment j out of
    it, within H either way; kind 1: the same sender and receiver again."""
    p = p.reset_index(drop=True)
    n = len(p); idx = np.arange(n)
    key = (p.src + ">" + p.dst).to_numpy()
    pn = pd.Series(key).map(pd.Series(key).value_counts()).to_numpy()
    deg = pd.concat([p.src, p.dst]).value_counts()
    a = pd.DataFrame({"i": idx, "w": p.dst, "ti": p.ts})
    b = pd.DataFrame({"j": idx, "w": p.src, "tj": p.ts})
    c = a.merge(b, on="w"); c = c[((c.tj - c.ti).abs() <= H) & (c.i != c.j)]
    c["kind"] = 0
    s = pd.DataFrame({"i": idx, "k": key}); s = s.merge(s.rename(columns={"i": "j"}), on="k")
    s = s[(s.i < s.j)]; s = s[pd.Series(pn[s.i.to_numpy()]).le(6).to_numpy()]
    s["kind"] = 1
    e = pd.concat([c[["i", "j", "kind"]], s[["i", "j", "kind"]]], ignore_index=True)
    i, j = e.i.to_numpy(), e.j.to_numpy()
    ts = p.ts.to_numpy(float); usd = p.usd.to_numpy(float)
    X = pd.DataFrame({"kind": e.kind.to_numpy()})
    X["dt_h"] = (ts[j] - ts[i]) / 3600; X["adt_h"] = X.dt_h.abs()
    X["lr"] = np.log(usd[j].clip(1e-6) / usd[i].clip(1e-6))
    X["usd_i"] = np.log1p(usd[i]); X["usd_j"] = np.log1p(usd[j])
    # no conversion or same-asset clues: conversions rise from 37% to 54% of repeating-pair payments
    # in the test week, and with them the link model found under half as many hand-on links there
    X["pn_i"] = pn[i]; X["pn_j"] = pn[j]
    span = pd.Series(ts).groupby(key).transform(lambda x: x.max() - x.min()).to_numpy() / 3600
    X["span_i"] = span[i]; X["span_j"] = span[j]
    for nm, col in [("src_i", p.src.to_numpy()[i]), ("dst_i", p.dst.to_numpy()[i]),
                    ("src_j", p.src.to_numpy()[j]), ("dst_j", p.dst.to_numpy()[j])]:
        X["deg_" + nm] = pd.Series(col).map(deg).to_numpy()
    # how crowded the choice is: candidates per payment, and this one's rank by amount and time
    e2 = pd.DataFrame({"i": i, "j": j, "k": e.kind.to_numpy(), "alr": X.lr.abs().to_numpy(), "adt": X.adt_h.to_numpy()})
    X["n_out_i"] = e2.groupby(["i", "k"]).j.transform("size").to_numpy()
    X["n_in_j"] = e2.groupby(["j", "k"]).i.transform("size").to_numpy()
    X["rk_lr_i"] = e2.groupby(["i", "k"]).alr.rank().to_numpy(); X["rk_lr_j"] = e2.groupby(["j", "k"]).alr.rank().to_numpy()
    X["rk_dt_i"] = e2.groupby(["i", "k"]).adt.rank().to_numpy(); X["rk_dt_j"] = e2.groupby(["j", "k"]).adt.rank().to_numpy()
    X["hod_i"] = (ts[i] % 86400) / 3600
    return i, j, X


def link_model(seed=SEED):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)


def anchored_schemes(p, i, j, prob, kind, thr_spine=0.5, thr_chain=0.5, hops=1):
    """Relay schemes built around repeating pairs the link model joins across days. Per payment:
    in_scheme, forward share and spread of the per-burst share passed on (best-anchored scheme)."""
    p = p.reset_index(drop=True); n = len(p)
    key = (p.src + ">" + p.dst).to_numpy()
    ts = p.ts.to_numpy(float); usd = p.usd.to_numpy(float); src = p.src.to_numpy(); dst = p.dst.to_numpy()
    ch = (kind == 0) & (prob >= thr_chain)
    nb = [[] for _ in range(n)]
    for a, b in zip(i[ch], j[ch]):
        nb[a].append(b); nb[b].append(a)
    sp = (kind == 1) & (prob >= thr_spine)
    by_key = pd.Series(np.arange(n)).groupby(key).apply(np.array).to_dict()
    best = np.full(n, -9.0); fwd_o = np.full(n, np.nan); cv_o = np.full(n, np.nan)
    for k in pd.unique(key[i[sp]]):
        occ = by_key[k]; mid = [src[occ[0]], dst[occ[0]]]
        bursts = []
        for o in occ:
            s = {o}; front = {o}
            for _ in range(hops):
                front = {y for x in front for y in nb[x]} - s; s |= front
            bursts.append(np.array(sorted(x for x in s if abs(ts[x] - ts[o]) <= 12 * 3600)))
        allp = np.unique(np.concatenate(bursts))
        pt = []; fw = []
        for b in bursts:
            ins = b[np.isin(dst[b], mid) & ~np.isin(src[b], mid)]
            outs = b[np.isin(src[b], mid) & ~np.isin(dst[b], mid)]
            if len(ins) and len(outs):
                pt.append(usd[outs].sum() / usd[ins].sum())
                fw.append(float(np.median(ts[outs]) > np.median(ts[ins])))
        if len(pt) < 2:
            continue
        cv = np.std(pt) / np.mean(pt); fwd = np.mean(fw)
        q = fwd - min(cv, 1)
        upd = allp[q > best[allp]]
        best[upd] = q; fwd_o[upd] = fwd; cv_o[upd] = cv
    return best > -9, fwd_o, cv_o


def multiplier(in_scheme, fwd, cv, struct_prob, back=4.0, boost=4.0, struct_max=0.3):
    """Gate multiplier: backward-looking relay schemes down, steady forward ones up, and only
    where the structure model doesn't see a planted shape scheme."""
    fwd = np.nan_to_num(fwd, nan=1.0)
    elig = in_scheme & (struct_prob < struct_max)
    forward = elig & (fwd >= 0.99)
    steady = 1 - pd.Series(np.where(forward, cv, np.nan)).rank(pct=True).to_numpy()
    z = np.where(forward, np.nan_to_num(steady, nan=0.0), 0.0)
    return np.where(elig & (fwd < 0.5), np.exp(-back), 1.0) * np.where(forward, np.exp(boost * (z - 0.5)), 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", default="predictions_structure_prob.csv",
                        help="the structure model's test-week probabilities, file inside outputs/")
    parser.add_argument("--out", default="relay_multiplier.csv", help="file name inside outputs/")
    args = parser.parse_args()

    tr = load_payments(common.WEEKS["train"]).reset_index(drop=True)
    sc = common.labels().set_index("TRANSACTION_HASH").scenario_id.loc[tr.TRANSACTION_HASH].to_numpy()
    i, j, X = candidates(tr)
    y = ((sc[i] == sc[j]) & pd.notna(sc[i])).astype(int)
    model = link_model().fit(X, y)

    te = load_payments(common.WEEKS["test"]).reset_index(drop=True)
    i, j, X = candidates(te)
    prob = model.predict_proba(X)[:, 1]
    in_scheme, fwd, cv = anchored_schemes(te, i, j, prob, X.kind.to_numpy())
    st = pd.read_csv(OUTPUTS / args.structure).set_index("TRANSACTION_HASH").score.loc[te.TRANSACTION_HASH].to_numpy()
    m = multiplier(in_scheme, fwd, cv, st)
    pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH, "score": m}).to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(te):,} relay multipliers to {OUTPUTS / args.out}: {in_scheme.mean():.1%} in a relay scheme, "
          f"{(m < 1).mean():.1%} pushed down, {(m > 1).mean():.1%} lifted")


if __name__ == "__main__":
    main()
