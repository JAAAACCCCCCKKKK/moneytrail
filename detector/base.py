"""Clean base tables: one row per transaction with correct USD values (XRPL traps handled)."""
import json, numpy as np, pandas as pd
from pathlib import Path
XRP = "0x1111222211112222111122221111222211112222"

def asset_id(token, symbol):
    return np.where(token == XRP, "XRP", token.astype(str) + ":" + symbol.astype(str))

def _read(d, name, usecols):
    """Parquet if pyarrow is installed, else a CSV export (duckdb) next to it."""
    if (d / f"{name}.parquet").exists():
        try:
            return pd.read_parquet(d / f"{name}.parquet", columns=usecols)
        except ImportError:
            pass
    return pd.read_csv(d / f"{name}.csv.gz", usecols=usecols)

def load(d):
    d = Path(d)
    tx = _read(d, "transactions", usecols=["TRANSACTION_HASH","TIMESTAMP","LEDGER_INDEX","TRANSACTION_INDEX","ACCOUNT","SEQUENCE","DELIVERED_AMOUNT","EXTRA_TRANSACTION_FIELDS","SOURCE_FEE"])
    tr = _read(d, "transfers", usecols=["TRANSACTION_HASH","TRANSFER_TYPE","TRANSFER_INDEX","FROM_ADDRESS","TO_ADDRESS","TOKEN_ADDRESS","SYMBOL","VALUE"])
    rates = pd.read_csv(d/"market_rates.csv")
    rate = {(t, a): p / 1e6 for t, a, p in rates.itertuples(index=False)}
    tx["HOUR"] = tx.TIMESTAMP // 3600 * 3600
    # parse JSON fields
    ex = tx.EXTRA_TRANSACTION_FIELDS.map(json.loads)
    tx["DEST"] = ex.map(lambda e: e["Destination"])
    tx["HAS_DTAG"] = ex.map(lambda e: "DestinationTag" in e).astype(int)
    tx["DTAG"] = ex.map(lambda e: e.get("DestinationTag"))
    tx["HAS_SENDMAX"] = ex.map(lambda e: "SendMax" in e).astype(int)
    tx["HAS_PATHS"] = ex.map(lambda e: "Paths" in e).astype(int)
    tx["EXTRA_KEYS"] = ex.map(lambda e: ",".join(sorted(e)))
    def amt(a):
        if isinstance(a, str): return "XRP", int(a) / 1e6        # drops -> XRP
        return f"{a['issuer']}:{a['currency']}", float(a["value"])
    dl = tx.DELIVERED_AMOUNT.map(lambda s: amt(json.loads(s)))
    tx["DEL_ASSET"] = dl.map(lambda x: x[0]); tx["DEL_UNITS"] = dl.map(lambda x: x[1])
    sm = ex.map(lambda e: amt(e["SendMax"]) if "SendMax" in e else (None, np.nan))
    tx["SM_ASSET"] = sm.map(lambda x: x[0]); tx["SM_UNITS"] = sm.map(lambda x: x[1])
    tx["DEL_USD"] = [u * rate[(h, a)] for u, h, a in zip(tx.DEL_UNITS, tx.HOUR, tx.DEL_ASSET)]
    tx["SM_USD"] = [u * rate[(h, a)] if isinstance(a, str) else np.nan for u, h, a in zip(tx.SM_UNITS, tx.HOUR, tx.SM_ASSET)]
    # legs, valued in USD (VALUE is whole units for XRP and IOUs)
    tr["ASSET"] = asset_id(tr.TOKEN_ADDRESS, tr.SYMBOL)
    tr = tr.merge(tx[["TRANSACTION_HASH","HOUR"]], on="TRANSACTION_HASH")
    tr["USD"] = [v * rate[(h, a)] for v, h, a in zip(tr.VALUE, tr.HOUR, tr.ASSET)]
    tr = tr.sort_values(["TRANSACTION_HASH","TRANSFER_INDEX"])
    g = tr.groupby("TRANSACTION_HASH")
    legs = pd.DataFrame({
        "N_LEGS": g.size(),
        "N_ASSETS": g.ASSET.nunique(),
        "SRC_ASSET": g.ASSET.first(), "DST_ASSET": g.ASSET.last(),
        "SRC_USD": g.USD.first(), "DST_USD": g.USD.last(),
        "LP1": g.TO_ADDRESS.first(), "LEG_TYPE": g.TRANSFER_TYPE.first(),
    }).reset_index()
    tx = tx.merge(legs, on="TRANSACTION_HASH", how="left")
    tx["IS_CONV"] = (tx.N_LEGS > 1).astype(int)
    tx["LP1"] = np.where(tx.IS_CONV == 1, tx.LP1, None)
    tx["SELF_PAY"] = (tx.ACCOUNT == tx.DEST).astype(int)
    tx["CONV_LOSS"] = np.where(tx.IS_CONV == 1, 1 - tx.DST_USD / tx.SRC_USD, 0.0)
    tx["SM_SLACK"] = np.where(tx.HAS_SENDMAX == 1, tx.SM_USD / tx.SRC_USD - 1, np.nan)
    tx = tx.drop(columns=["EXTRA_TRANSACTION_FIELDS","DELIVERED_AMOUNT"])
    return tx, tr
