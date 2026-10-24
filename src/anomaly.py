"""Isolation Forest anomaly clues (unsupervised, never sees labels).

Unlike the other clue files, this runs AFTER main.py has joined everyone's
clue tables, because it scores how unusual each payment looks across all of
those clues at once. Call it once per week, on that week's rows only.

    from anomaly import add_anomaly_features
    anom = add_anomaly_features(feats_week, payments_week, hub_col="shape_sender_is_hub")
    feats_week = feats_week.merge(anom, on="TRANSACTION_HASH", how="left", validate="1:1")

Columns produced (higher = more unusual):
    anom_txn          how odd this payment's whole clue profile is
    anom_txn_nonhub   same, but "normal" is learned from non-hub payments only,
                      so exchanges/LPs don't define what ordinary looks like
    anom_sender       how odd the sender's week-long behaviour is
    anom_receiver     how odd the receiver's week-long behaviour is

Rules followed: no labels, no wallet IDs or hashes as inputs, fixed seed,
built identically for both weeks from that week's own data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Same assumption as features_wallet.py: change these to the clean-table names.
COLS = {"hash": "TRANSACTION_HASH", "sender": "SENDER", "receiver": "RECEIVER",
        "time": "DATETIME", "usd": "USD_VALUE"}

SEED = 42


def _prep(X):
    """Rank each column to 0..1 so heavy-tailed amounts and counts don't
    dominate the random splits; missing values get their own -0.1 slot."""
    X = X.astype(float).replace([np.inf, -np.inf], np.nan)
    return X.rank(pct=True).fillna(-0.1).to_numpy()


def _forest(X, fit_rows=None, seed=SEED):
    model = IsolationForest(n_estimators=300, max_samples=256, random_state=seed, n_jobs=-1)
    model.fit(X if fit_rows is None else X[fit_rows])
    return -model.score_samples(X)  # flip sign: higher = more anomalous


def _wallet_profiles(payments):
    """One row per wallet summarising its week (counts, value, speed)."""
    c = COLS
    p = payments[[c["sender"], c["receiver"], c["time"], c["usd"]]].copy()
    p.columns = ["s", "r", "t", "usd"]
    p["t"] = pd.to_datetime(p["t"])
    p["usd"] = pd.to_numeric(p["usd"], errors="coerce")

    out = p.groupby("s").agg(n_out=("r", "size"), cp_out=("r", "nunique"),
                             usd_out=("usd", "sum"), usd_out_med=("usd", "median"),
                             first_out=("t", "min"), last_out=("t", "max"))
    inn = p.groupby("r").agg(n_in=("s", "size"), cp_in=("s", "nunique"),
                             usd_in=("usd", "sum"), first_in=("t", "min"), last_in=("t", "max"))
    w = out.join(inn, how="outer")
    first = w[["first_out", "first_in"]].min(axis=1)
    last = w[["last_out", "last_in"]].max(axis=1)
    w["span_hours"] = (last - first).dt.total_seconds() / 3600
    w["in_to_out_hours"] = (w["first_out"] - w["first_in"]).dt.total_seconds() / 3600
    w["kept_ratio"] = np.where(w["usd_in"] > 0, w["usd_out"] / w["usd_in"], np.nan)
    w["cp_out_share"] = w["cp_out"] / w["n_out"]
    w["cp_in_share"] = w["cp_in"] / w["n_in"]
    return w.drop(columns=["first_out", "last_out", "first_in", "last_in"])


def add_anomaly_features(feats, payments, hub_col=None, seed=SEED):
    """feats: the joined clue table for ONE week (TRANSACTION_HASH + numeric
    clue columns). Drop any label or fold columns before calling.
    payments: that week's clean payments table (for the wallet profiles).
    hub_col: optional 0/1 column in feats marking hub payments.
    Returns TRANSACTION_HASH + anom_* columns, one row per transaction."""
    banned = {"label", "is_suspicious", "scenario_id", "fold", "group"}
    clue_cols = [col for col in feats.columns
                 if col != COLS["hash"] and col.lower() not in banned
                 and pd.api.types.is_numeric_dtype(feats[col])]
    X = _prep(feats[clue_cols])

    res = pd.DataFrame({"TRANSACTION_HASH": feats[COLS["hash"]].to_numpy()})
    res["anom_txn"] = _forest(X, seed=seed)
    if hub_col and hub_col in feats:
        non_hub = feats[hub_col].fillna(0).to_numpy() == 0
        res["anom_txn_nonhub"] = _forest(X, fit_rows=non_hub, seed=seed) if non_hub.sum() > 256 else np.nan

    prof = _wallet_profiles(payments)
    wallet_score = pd.Series(_forest(_prep(prof), seed=seed), index=prof.index)
    keyed = payments[[COLS["hash"], COLS["sender"], COLS["receiver"]]]
    keyed = keyed.set_index(COLS["hash"]).reindex(res["TRANSACTION_HASH"])
    res["anom_sender"] = keyed[COLS["sender"]].map(wallet_score).to_numpy()
    res["anom_receiver"] = keyed[COLS["receiver"]].map(wallet_score).to_numpy()

    assert len(res) == len(feats), "one row per transaction"
    return res
