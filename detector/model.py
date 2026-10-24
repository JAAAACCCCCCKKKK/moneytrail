"""Two-stage model: gradient boosting on payment/wallet/graph features, then again with neighbour scores."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score as ap
from features import make_features, gap_features
from stage2 import neighbour_features

PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False)

def build(base):
    F = make_features(base)
    return F.merge(gap_features(base), on="TRANSACTION_HASH")

def fit_predict(Xtr, ytr, Xte, seeds):
    p = np.zeros(len(Xte))
    for s in seeds:
        p += HistGradientBoostingClassifier(random_state=s, **PARAMS).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return p / len(seeds)

def cross_validate(base, F, lab, seeds=(0,), folds_seed=0, report=True):
    """Out-of-fold scores for both stages, using the same folds (no leakage across stages)."""
    d = F.merge(lab[["TRANSACTION_HASH", "is_suspicious", "scenario_id", "difficulty"]], on="TRANSACTION_HASH")
    cols = [c for c in F.columns if c != "TRANSACTION_HASH"]
    X = d[cols].to_numpy(float); y = d.is_suspicious.to_numpy()
    g = d.scenario_id.fillna(d.TRANSACTION_HASH).to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=folds_seed).split(X, y, g))
    oof1 = np.zeros(len(d)); oof2 = np.zeros(len(d))
    for tr, te in folds:
        # stage 1 for the held-out fold, and inner OOF for the training part
        oof1[te] = fit_predict(X[tr], y[tr], X[te], seeds)
        inner = np.zeros(len(tr))
        for itr, ite in StratifiedGroupKFold(4, shuffle=True, random_state=1).split(X[tr], y[tr], g[tr]):
            inner[ite] = fit_predict(X[tr][itr], y[tr][itr], X[tr][ite], seeds)
        score = np.zeros(len(d)); score[tr] = inner; score[te] = oof1[te]
        N = neighbour_features(base, d[["TRANSACTION_HASH"]], score).drop(columns="TRANSACTION_HASH").to_numpy(float)
        X2 = np.hstack([X, N])
        oof2[te] = fit_predict(X2[tr], y[tr], X2[te], seeds)
    d["oof1"] = oof1; d["oof2"] = oof2
    if report:
        for name, o in (("stage1", oof1), ("stage2", oof2)):
            fa = [ap(y[te], o[te]) for _, te in folds]
            msg = f"{name}: AP {100*np.mean(fa):.2f} ± {100*np.std(fa):.2f} (overall {100*ap(y, o):.2f})"
            for dif in ("easy", "medium", "hard"):
                m = (d.difficulty == dif) | (y == 0); msg += f" | {dif} {100*ap(y[m], o[m]):.1f}"
            print(msg)
    return d

def train_and_score(base_tr, F_tr, lab, base_te, F_te, seeds=(0, 1, 2)):
    d = F_tr.merge(lab[["TRANSACTION_HASH", "is_suspicious", "scenario_id"]], on="TRANSACTION_HASH")
    cols = [c for c in F_tr.columns if c != "TRANSACTION_HASH"]
    X = d[cols].to_numpy(float); y = d.is_suspicious.to_numpy(); g = d.scenario_id.fillna(d.TRANSACTION_HASH).to_numpy()
    Xte = F_te[cols].to_numpy(float)
    # stage 1: OOF on train (for stage-2 training), full model on test
    oof = np.zeros(len(d))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        oof[te] = fit_predict(X[tr], y[tr], X[te], seeds)
    p1 = fit_predict(X, y, Xte, seeds)
    N_tr = neighbour_features(base_tr, d[["TRANSACTION_HASH"]], oof).drop(columns="TRANSACTION_HASH").to_numpy(float)
    N_te = neighbour_features(base_te, F_te[["TRANSACTION_HASH"]], p1).drop(columns="TRANSACTION_HASH").to_numpy(float)
    p2 = fit_predict(np.hstack([X, N_tr]), y, np.hstack([Xte, N_te]), seeds)
    return pd.DataFrame({"TRANSACTION_HASH": F_te.TRANSACTION_HASH, "score": p2})
