"""Model the generator: scheme templates fitted on the true training schemes, applied to
groups rebuilt under the generator's grammar (Jack).

  python -m src.predict_generator --cv      # honest test vs the 67.1 recipe (main.py folds, 3 seeds)
  python -m src.predict_generator           # outputs/predictions_generator.csv

The data is synthetic and the labels are the generator's intent, so the best detector is the
generator's own likelihood ratio: how likely is this group under a laundering template against an
honest one of the same shape? Three parts:
  1. Grammar grouping. Payments are linked through the learned linker's links (src.predict_linker)
     in confidence order, but a merge is refused if it would break the grammar read off the training
     schemes: at most 16 payments, at most 10 hours. Done at three confidence levels, so each payment
     has three hypotheses about which scheme it belongs to.
  2. Template model. One row per scheme with the generator's parameters (src.motifs.describe: size,
     span, gaps, rate, wallets, edges, density, fan shares, sources, sinks, middles, pass-through,
     overlap, route reuse, depth, cycles; asset and conversion columns left out, as the README says
     they ignore the labels). A small LightGBM learns P(suspicious | parameters) from the true
     training schemes (scenario_id used only to form them) plus the rebuilt training groups that are
     at least 80% one scheme, so it also sees how rebuilt groups look.
  3. Score. P(scheme) from src.predict_two_q times the mean template posterior over the three
     hypotheses, rank-blended 50/50 with the combined model, exactly as the 67.1 recipe does with
     its learned second model.
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from main import SEED, make_model, OUTPUTS
from src import motifs as motif_lib
from src.data import load_payments as load_rich
from src.predict_linker import candidate_pairs, linker
from src.predict_two_q import NOT_SCHEME, rank01, tables

MAX_N, MAX_SPAN = 16, 36000
TAUS = (0.3, 0.5, 0.7)
MIN_PURITY, MIN_SCHEME_ROWS = 0.8, 3
DROP = ("n_assets", "conversion_share", "bridged_share")


def grammar_groups(T, J, I, prob, tau):
    """Union-find over links in confidence order; merges that break size or span limits are refused."""
    parent, size, lo, hi = np.arange(len(T)), np.ones(len(T), int), T.copy(), T.copy()

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in np.argsort(-prob, kind="stable"):
        if prob[k] < tau:
            break
        a, b = find(J[k]), find(I[k])
        if a == b or size[a] + size[b] > MAX_N or max(hi[a], hi[b]) - min(lo[a], lo[b]) > MAX_SPAN:
            continue
        parent[a] = b
        size[b] += size[a]
        lo[b], hi[b] = min(lo[a], lo[b]), max(hi[a], hi[b])
    return np.array([find(x) for x in range(len(T))])


def rich_table(pay: pd.DataFrame, week: str) -> pd.DataFrame:
    return load_rich(common.WEEKS[week]).set_index("TRANSACTION_HASH").loc[pay.TRANSACTION_HASH].reset_index()


def describe(rich: pd.DataFrame, g: np.ndarray) -> pd.DataFrame:
    D = motif_lib.describe(rich, g, prefix="")
    return D.drop(columns=[c for c in D.columns if c in DROP])


def template_model():
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=8, verbose=-1)


def link_probs_train(pay, U, y, groups):
    """Out-of-fold link probabilities on the training week (each fold's linker never sees its schemes)."""
    J, I, X = candidate_pairs(pay, "train")
    sid = common.labels().set_index("TRANSACTION_HASH").scenario_id.reindex(pay.TRANSACTION_HASH).to_numpy()
    same = (pd.notna(sid[J]) & (sid[J] == sid[I])).astype(int)
    fold_of = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(U, y, groups)):
        fold_of[va] = k
    prob = np.zeros(len(J))
    for k in range(5):
        fit = (fold_of[J] != k) & (fold_of[I] != k)
        sc = (fold_of[J] == k) | (fold_of[I] == k)
        prob[sc] = linker().fit(X[fit], same[fit]).predict_proba(X[sc])[:, 1]
    return J, I, X, same, prob


def hypotheses(T, J, I, prob, rich):
    """For each confidence level: group id per payment and that grouping's described table."""
    out = []
    for tau in TAUS:
        g = grammar_groups(T, J, I, prob, tau)
        out.append((g, describe(rich, g)))
    return out


def training_rows(lab, rich, hyps, scheme_mask):
    """Template-model training set: true schemes, plus rebuilt groups that are mostly one scheme."""
    sid = lab.scenario_id.to_numpy()
    true_g = pd.factorize(pd.Series(sid).fillna("none"))[0]
    rows = scheme_mask
    D_true = describe(rich[rows].reset_index(drop=True), true_g[rows])
    y_true = pd.Series(lab.is_suspicious.to_numpy()[rows]).groupby(true_g[rows]).first().loc[D_true.index].to_numpy()
    s_true = pd.Series(sid[rows]).groupby(true_g[rows]).first().loc[D_true.index].to_numpy()
    frames, ys, sids = [D_true], [y_true], [s_true]
    for g, D in hyps:
        df = pd.DataFrame({"g": g, "sid": sid, "y": lab.is_suspicious.to_numpy(), "scheme": scheme_mask})
        agg = df.groupby("g").agg(n=("g", "size"), n_scheme=("scheme", "sum"),
                                  top=("sid", lambda s: s.value_counts(dropna=True).iloc[0] if s.notna().any() else 0),
                                  top_sid=("sid", lambda s: s.value_counts(dropna=True).index[0] if s.notna().any() else None),
                                  y=("y", "max"))
        keep = agg[(agg.n_scheme >= MIN_SCHEME_ROWS) & (agg.top / agg.n >= MIN_PURITY)]
        frames.append(D.loc[keep.index]); ys.append(keep.y.to_numpy()); sids.append(keep.top_sid.to_numpy())
    return pd.concat(frames, ignore_index=True), np.concatenate(ys), np.concatenate(sids)


def posterior(model, hyps, n):
    """Mean over hypotheses of the template posterior, broadcast to payments."""
    acc = np.zeros(n)
    for g, D in hyps:
        p = pd.Series(model.predict_proba(D)[:, 1], index=D.index)
        acc += p.loc[g].to_numpy()
    return acc / len(hyps)


def cross_validate():
    t0 = time.time()
    train, _, lab, U, _, U2, _ = tables()
    y = lab.is_suspicious.to_numpy()
    scheme = (~lab.family.isin(NOT_SCHEME)).to_numpy().astype(int)
    pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
    T = pay.TIMESTAMP.to_numpy()
    groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + pay.FROM_ADDRESS)
    rich = rich_table(pay, "train")
    J, I, X, same, prob = link_probs_train(pay, U, y, groups)
    hyps = hypotheses(T, J, I, prob, rich)
    Dtr, ytr, str_ = training_rows(lab, rich, hyps, scheme == 1)
    print(f"template rows {len(Dtr):,} (true schemes {int((scheme == 1).sum() and len(set(lab.scenario_id[scheme == 1])))}), "
          f"suspicious {int(ytr.sum()):,} ({time.time() - t0:.0f}s)", flush=True)
    med, easy = (lab.difficulty == "medium").to_numpy(), (lab.difficulty == "easy").to_numpy()
    summary = {}

    def rep(name, s, folds, base=None):
        per = [100 * aps(y[va], s[va]) for _, va in folds]
        r = pd.Series(s).rank(ascending=False).to_numpy()
        nb = 0 if base is None else sum(a > b for a, b in zip(per, base))
        summary.setdefault(name, []).append((100 * aps(y, s), np.mean(per), nb))
        print(f"  {name:30s} pooled {100 * aps(y, s):5.1f} folds {np.mean(per):.1f}{'' if base is None else f' vs C {nb}/5'} "
              f"| easy top1250 {np.mean(r[easy] <= 1250):.2f} | med med-rank {np.median(r[med]):.0f} top2500 {np.mean(r[med] <= 2500):.2f} "
              f"| top400 prec {np.mean(y[np.argsort(-s)[:400]]):.3f}", flush=True)
        return per

    for fs in (SEED, 7, 2026):
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fs).split(U, y, groups))
        o1, ps, pw, post = (np.zeros(len(y)) for _ in range(4))
        for tr, va in folds:
            trs = tr[scheme[tr] == 1]
            o1[va] = make_model("lgbm").fit(U.iloc[tr], y[tr]).predict_proba(U.iloc[va])[:, 1]
            ps[va] = make_model("lgbm").fit(U2.iloc[tr], scheme[tr]).predict_proba(U2.iloc[va])[:, 1]
            pw[va] = make_model("lgbm").fit(U2.iloc[trs], y[trs]).predict_proba(U2.iloc[va])[:, 1]
            held = set(lab.scenario_id.to_numpy()[va]) - {None, np.nan}
            fit_rows = ~pd.Series(str_).isin(held).to_numpy()          # template rows from held-out schemes stay out
            post[va] = posterior(template_model().fit(Dtr[fit_rows], ytr[fit_rows]), hyps, len(y))[va]
        print(f"--- fold seed {fs} ({time.time() - t0:.0f}s)", flush=True)
        s1 = graph.smooth_by_group(o1, pay, "train")
        C = 0.5 * rank01(s1) + 0.5 * rank01(ps * pw)
        base = rep("C the 67.1 recipe", C, folds)
        rep("template posterior alone", post, folds, base)
        rep("G: s1 + P(scheme) x template", 0.5 * rank01(s1) + 0.5 * rank01(ps * post), folds, base)
        rep("G3: thirds s1, 2Q, template", (rank01(s1) + rank01(ps * pw) + rank01(ps * post)) / 3, folds, base)
        rep("G4: s1 + P(scheme) x mean(pw, template)", 0.5 * rank01(s1) + 0.5 * rank01(ps * (pw + post) / 2), folds, base)
    print("\nSUMMARY (3 seeds: pooled, fold mean, folds better than C of 15)")
    for k, v in summary.items():
        v = np.array(v)
        print(f"  {k:30s} {v[:, 0].mean():5.1f} {v[:, 1].mean():5.1f} {int(v[:, 2].sum()):2d}/15")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--mode", choices=["G", "G3", "G4"], default="G4")
    parser.add_argument("--out", default="predictions_generator.csv")
    args = parser.parse_args()
    if args.cv:
        cross_validate()
        return
    train, test, lab, U, Ut, U2, Ut2 = tables()
    y = lab.is_suspicious.to_numpy()
    scheme = (~lab.family.isin(NOT_SCHEME)).to_numpy().astype(int)
    pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
    test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
    groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + pay.FROM_ADDRESS)
    rich, rich_t = rich_table(pay, "train"), rich_table(test_pay, "test")
    J, I, X, same, prob = link_probs_train(pay, U, y, groups)
    hyps = hypotheses(pay.TIMESTAMP.to_numpy(), J, I, prob, rich)
    Dtr, ytr, _ = training_rows(lab, rich, hyps, scheme == 1)
    model = template_model().fit(Dtr, ytr)
    Jt, It, Xt = candidate_pairs(test_pay, "test")
    prob_t = linker().fit(X, same).predict_proba(Xt)[:, 1]
    post_t = posterior(model, hypotheses(test_pay.TIMESTAMP.to_numpy(), Jt, It, prob_t, rich_t), len(test_pay))
    in_scheme = scheme == 1
    single = graph.smooth_by_group(make_model("lgbm").fit(U, y).predict_proba(Ut)[:, 1], test_pay, "test")
    p_scheme = make_model("lgbm").fit(U2, scheme).predict_proba(Ut2)[:, 1]
    p_within = make_model("lgbm").fit(U2[in_scheme], y[in_scheme]).predict_proba(Ut2)[:, 1]
    if args.mode == "G":
        blended = 0.5 * rank01(single) + 0.5 * rank01(p_scheme * post_t)
    elif args.mode == "G3":
        blended = (rank01(single) + rank01(p_scheme * p_within) + rank01(p_scheme * post_t)) / 3
    else:
        blended = 0.5 * rank01(single) + 0.5 * rank01(p_scheme * (p_within + post_t) / 2)
    score = pd.Series(blended).rank(method="first").to_numpy() / len(blended)
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": score}).to_csv(OUTPUTS / args.out, index=False)
    print(f"wrote {len(score):,} test scores to {OUTPUTS / args.out} (mode {args.mode}, template rows {len(Dtr):,})")


if __name__ == "__main__":
    main()
