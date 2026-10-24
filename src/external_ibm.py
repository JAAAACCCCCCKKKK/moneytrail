"""External helper clue: a model trained on IBM AMLworld HI-Small, used as ONE column.

Called by main.py like every other clue file:  add_features(payments, week)
Returns TRANSACTION_HASH + ext_ibm_score, one row per transaction, same order.

How it works
1. Load data/external/HI-Small_Trans.csv (download from Kaggle:
   ealtman2019/ibm-transactions-for-anti-money-laundering-aml; never commit it).
2. Take the first SLICE_DAYS days, map columns to our clean-table names,
   convert amounts to USD (rates estimated from IBM's own USD rows).
3. Build the SAME shared clues for IBM and for our payments (one function).
4. Keep every laundering row + a sample of normal rows (~3% suspicious),
   train a small LightGBM on IBM's labels. Trained once per run, cached.
5. Score our payments with it. The helper never sees our labels, so no leak.

Run `python3 external_ibm.py` on its own for a quick sanity check.
"""
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import features_wallet

IBM_PATH = Path(__file__).resolve().parent / "data" / "external" / "HI-Small_Trans.csv"
SLICE_DAYS = 7
NORMALS_PER_POSITIVE = 32  # about 3% suspicious, like our test week
SEED = 42
COLS = features_wallet.COLS  # our clean-table column names (one place to change)

IBM_COLUMNS = ["Timestamp", "FromBank", "FromAcct", "ToBank", "ToAcct",
               "AmtRecv", "CurRecv", "AmtPaid", "CurPaid", "Format", "Label"]

FEATURES = [
    "wallet_sender_age_hours", "wallet_receiver_age_hours",
    "wallet_mins_since_received", "wallet_out_vs_last_in",
    "ext_log_usd", "ext_pays_itself",
    "ext_sender_payments_week", "ext_receiver_payments_week",
    "ext_sender_burst_10m", "ext_sender_burst_1h",
    "ext_receiver_burst_10m", "ext_receiver_burst_1h", "ext_receiver_burst_24h",
]


def _prior_count(keys, seconds, window):
    """For each row: how many earlier rows (strictly earlier time) with the same key
    fall within `window` seconds before it. Vectorised, backward-looking only."""
    k = pd.factorize(keys)[0].astype(np.int64)
    t = seconds - seconds.min()
    big = int(t.max()) + 2 * window + 1          # keeps different keys apart
    order = np.lexsort((t, k))
    z = k[order] * big + t[order]
    left = np.searchsorted(z, z - window, side="left")
    first_same_time = np.searchsorted(z, z, side="left")
    out = np.empty(len(z), dtype=np.int64)
    out[order] = first_same_time - left
    return out


def _shared_features(payments):
    """Identical clue code for IBM and for our weeks. payments uses COLS names."""
    c = COLS
    wallet = features_wallet.add_features(payments, "external").reset_index(drop=True)

    time = pd.to_datetime(payments[c["time"]], utc=True).dt.tz_localize(None)
    df = pd.DataFrame({
        "sender": payments[c["sender"]].astype(str).to_numpy(),
        "receiver": payments[c["receiver"]].astype(str).to_numpy(),
        "usd": pd.to_numeric(payments[c["usd"]], errors="coerce").to_numpy(),
    })
    secs = (time.to_numpy().astype("datetime64[s]").astype(np.int64))

    df["ext_log_usd"] = np.log1p(df["usd"].clip(lower=0))
    df["ext_pays_itself"] = (df["sender"] == df["receiver"]).astype(int)
    df["ext_sender_payments_week"] = df.groupby("sender")["sender"].transform("size")
    df["ext_receiver_payments_week"] = df.groupby("receiver")["receiver"].transform("size")
    df["ext_sender_burst_10m"] = _prior_count(df["sender"], secs, 600)
    df["ext_sender_burst_1h"] = _prior_count(df["sender"], secs, 3600)
    df["ext_receiver_burst_10m"] = _prior_count(df["receiver"], secs, 600)
    df["ext_receiver_burst_1h"] = _prior_count(df["receiver"], secs, 3600)
    df["ext_receiver_burst_24h"] = _prior_count(df["receiver"], secs, 86400)

    for col in features_wallet.NEW_COLUMNS:
        df[col] = wallet[col].to_numpy()
    return df[FEATURES]


KAGGLE_HANDLE = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
KAGGLE_FILE = "HI-Small_Trans.csv"


def _ibm_csv_path():
    """Use data/external/HI-Small_Trans.csv if present; otherwise download just that
    one file with kagglehub (cached, so later runs don't download again)."""
    if IBM_PATH.exists():
        return IBM_PATH
    try:
        import kagglehub
    except ImportError as e:
        raise FileNotFoundError(
            f"{IBM_PATH} not found and kagglehub is not installed. Either run "
            "`python3 -m pip install kagglehub`, or download HI-Small_Trans.csv from "
            f"Kaggle ({KAGGLE_HANDLE}) into data/external/.") from e
    downloaded = Path(kagglehub.dataset_download(KAGGLE_HANDLE, path=KAGGLE_FILE))
    if downloaded.is_dir():                      # older kagglehub returns the folder
        downloaded = downloaded / KAGGLE_FILE
    print(f"IBM data from Kaggle cache: {downloaded}")
    return downloaded


def _load_ibm():
    raw = pd.read_csv(_ibm_csv_path())
    assert raw.shape[1] == 11, f"expected 11 IBM columns, found {raw.shape[1]}"
    raw.columns = IBM_COLUMNS
    raw["time"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")

    # Currency -> USD rates from IBM's own rows where one side is US Dollar.
    rates = {"US Dollar": 1.0}
    paid_usd = raw[(raw.CurPaid == "US Dollar") & (raw.CurRecv != "US Dollar") & (raw.AmtRecv > 0)]
    recv_usd = raw[(raw.CurRecv == "US Dollar") & (raw.CurPaid != "US Dollar") & (raw.AmtPaid > 0)]
    est = pd.concat([
        (paid_usd.AmtPaid / paid_usd.AmtRecv).groupby(paid_usd.CurRecv).median(),
        (recv_usd.AmtRecv / recv_usd.AmtPaid).groupby(recv_usd.CurPaid).median(),
    ]).groupby(level=0).median()
    rates.update(est.to_dict())

    start = raw["time"].min()
    week = raw[raw["time"] < start + pd.Timedelta(days=SLICE_DAYS)].reset_index(drop=True)

    c = COLS
    table = pd.DataFrame({
        c["hash"]: "ibm_" + week.index.astype(str),
        c["sender"]: week.FromBank.astype(str) + "_" + week.FromAcct.astype(str),
        c["receiver"]: week.ToBank.astype(str) + "_" + week.ToAcct.astype(str),
        c["time"]: week["time"],
        c["usd"]: week.AmtPaid * week.CurPaid.map(rates),
    })
    return table, week["Label"].astype(int).to_numpy(), rates


@lru_cache(maxsize=1)
def _helper():
    table, y, _ = _load_ibm()
    X = _shared_features(table)          # built on the FULL slice, so counts are real
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    neg = rng.choice(neg, size=min(len(neg), NORMALS_PER_POSITIVE * len(pos)), replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)
    model.fit(X.iloc[keep], y[keep])
    return model


def add_features(payments, week):
    """Score our payments with the IBM-trained helper. week is unused (same for both)."""
    X = _shared_features(payments)
    score = _helper().predict_proba(X)[:, 1]
    out = pd.DataFrame({"TRANSACTION_HASH": payments[COLS["hash"]].to_numpy(),
                        "ext_ibm_score": score})
    assert len(out) == len(payments), "output must have one row per input transaction"
    return out


if __name__ == "__main__":
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import train_test_split

    table, y, rates = _load_ibm()
    print(f"IBM slice: {len(table):,} payments, {y.sum():,} laundering ({y.mean():.3%})")
    print(f"USD rates found for {len(rates)} currencies; "
          f"{table[COLS['usd']].isna().mean():.1%} of rows have no USD value")
    X = _shared_features(table)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=50, random_state=SEED, verbose=-1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    print(f"Helper sanity check on IBM (random 20%): AP {average_precision_score(yte, p):.3f} "
          f"vs random {yte.mean():.4f}")
