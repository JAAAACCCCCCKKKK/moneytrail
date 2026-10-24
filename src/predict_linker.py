"""A learned payment linker instead of fixed-window groups (Jack).

  python -m src.predict_linker            # outputs/predictions_linker.csv
  python -m src.blend --run3 predictions_linker.csv --violet predictions_violet_v2_fixed.csv \
      --out predictions_blend_linker_violet.csv      # built like the 67.1 entry

Every other grouping clue uses fixed windows (2-10 minutes, or once-a-week pairs at 30-60
minutes), and slow medium schemes fall apart in them. Here a LightGBM learns which payments
belong to the same planted scheme. Candidate pairs are payments sharing a non-hub wallet within
2 hours (every planted scheme is fully connected by them); `scenario_id` is the training target
only, never a clue. Payments are grouped through links scored at 0.5 or above; each group's
median gap gives the share of training schemes at that speed that are suspicious. Score:
P(scheme) x that share (P(suspicious | scheme) for ungrouped payments), rank-blended 50/50 with
the combined model, as in src.predict_two_q.

Honest test (DECISIONS.md, ~14:20; three fold seeds, pooled / fold mean): 62.3 / 63.9 against
the 67.1 recipe's 62.3 / 63.1, better in 10 of 15 folds; medium median rank about 2,640 -> 2,260.
"""
from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from main import SEED, make_model, OUTPUTS
from src.predict_two_q import NOT_SCHEME, rank01, tables

WINDOW = 7200       # candidate pairs: share a non-hub wallet within 2 hours
TAU = 0.5           # link payments whose pair the linker scores at this or above
GROUP_SIZE = (3, 40)
BINS = [0, 60, 120, 180, 240, 480, 960, 1920, 1e9]
ROLES = {"ss": 0, "sr": 1, "rs": 2, "rr": 3}   # earlier payment's role + later one's; "rs" = money flows through


def candidate_pairs(pay: pd.DataFrame, week: str):
    """Index pairs (earlier, later) sharing a non-hub wallet within WINDOW, and their clues."""
    hub_set = graph.hubs(pay, week)
    T, A, B, V = pay.TIMESTAMP.to_numpy(), pay.FROM_ADDRESS.to_numpy(), pay.TO_ADDRESS.to_numpy(), pay.VALUE.to_numpy()
    once = (pd.Series(list(zip(A, B))).map(pd.Series(list(zip(A, B))).value_counts()) == 1).to_numpy().astype(int)
    rows, recent = [], {}
    for i in np.argsort(T, kind="stable"):
        for role, w in (("s", A[i]), ("r", B[i])):
            if w in hub_set:
                continue
            lst = recent.setdefault(w, [])
            while lst and T[i] - lst[0][1] > WINDOW:
                lst.pop(0)
            rows.extend((j, i, T[i] - tj, rj + role) for j, tj, rj in lst)
            lst.append((i, T[i], role))
    P = pd.DataFrame(rows, columns=["j", "i", "dt", "roles"]).drop_duplicates(["j", "i"]).reset_index(drop=True)
    J, I = P.j.to_numpy(), P.i.to_numpy()
    degree = pd.concat([P.j, P.i]).value_counts()
    X = pd.DataFrame({"log_dt": np.log1p(P.dt), "flow": (P.roles == "rs").astype(int),
                      "once_j": once[J], "once_i": once[I], "val_ratio": np.log((V[I] + 1e-9) / (V[J] + 1e-9)),
                      "role": P.roles.map(ROLES), "deg_j": degree.loc[J].to_numpy(), "deg_i": degree.loc[I].to_numpy()})
    return J, I, X


def linker():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              random_state=SEED, n_jobs=-1, verbose=-1)


def link_groups(T, J, I, prob):
    """Size and median gap of each payment's group, linking pairs scored at TAU or above."""
    parent = np.arange(len(T))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keep = prob >= TAU
    for j, i in zip(J[keep], I[keep]):
        a, b = find(j), find(i)
        if a != b:
            parent[a] = b
    g = np.array([find(x) for x in range(len(T))])
    frame = pd.DataFrame({"g": g, "t": T})
    size = frame.g.map(frame.g.value_counts()).to_numpy()
    gap = frame.sort_values("t").groupby("g").t.agg(lambda t: np.median(np.diff(t)) if len(t) > 1 else np.nan)
    gap = gap.loc[g].to_numpy()
    grouped = np.isfinite(gap) & (size >= GROUP_SIZE[0]) & (size <= GROUP_SIZE[1])
    return grouped, np.digitize(np.nan_to_num(gap, nan=-1), BINS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="predictions_linker.csv", help="file name inside outputs/")
    args = parser.parse_args()

    train, test, lab, U, Ut, U2, Ut2 = tables()
    y = lab.is_suspicious.to_numpy()
    scheme = (~lab.family.isin(NOT_SCHEME)).to_numpy().astype(int)
    in_scheme = scheme == 1
    train_pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")

    single = make_model("lgbm").fit(U, y).predict_proba(Ut)[:, 1]          # = src.predict_union_signals
    single = graph.smooth_by_group(single, test_pay, "test")
    p_scheme = make_model("lgbm").fit(U2, scheme).predict_proba(Ut2)[:, 1]
    p_within = make_model("lgbm").fit(U2[in_scheme], y[in_scheme]).predict_proba(Ut2)[:, 1]

    # Linker on the training week. Out-of-fold link scores (by planted scheme) build the training
    # groups, so their speeds are read under the same conditions as the test week's.
    J, I, X = candidate_pairs(train_pay, "train")
    sid = lab.scenario_id.to_numpy()
    same = (pd.notna(sid[J]) & (sid[J] == sid[I])).astype(int)
    split_groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + train_pay.FROM_ADDRESS)
    fold_of = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(U, y, split_groups)):
        fold_of[va] = k
    prob = np.zeros(len(J))
    for k in range(5):
        fit = (fold_of[J] != k) & (fold_of[I] != k)
        score = (fold_of[J] == k) | (fold_of[I] == k)
        prob[score] = linker().fit(X[fit], same[fit]).predict_proba(X[score])[:, 1]
    grouped, speed = link_groups(train_pay.TIMESTAMP.to_numpy(), J, I, prob)
    rows = grouped & in_scheme
    table = pd.Series(y[rows]).groupby(speed[rows]).agg(["sum", "count"])
    share = (table["sum"] + 1) / (table["count"] + 4)          # suspicious share by speed, lightly smoothed

    # Linker on the test week.
    Jt, It, Xt = candidate_pairs(test_pay, "test")
    prob_t = linker().fit(X, same).predict_proba(Xt)[:, 1]
    grouped_t, speed_t = link_groups(test_pay.TIMESTAMP.to_numpy(), Jt, It, prob_t)
    within = np.where(grouped_t, pd.Series(speed_t).map(share).fillna(0.2).to_numpy(), p_within)

    blended = 0.5 * rank01(single) + 0.5 * rank01(p_scheme * within)
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)   # no ties
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(score):,} test scores to {OUTPUTS / args.out} "
          f"(linked pairs {int((prob_t >= TAU).sum()):,} of {len(prob_t):,}; grouped payments {grouped_t.mean():.0%})")


if __name__ == "__main__":
    main()
