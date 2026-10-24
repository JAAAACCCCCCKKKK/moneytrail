"""What do the easy/medium suspicious payments our model still ranks low have in common,
and how do they differ from the innocent payments ranked above them?

EXPLORATION ONLY: uses labels, so it must never be imported by main.py or a clue file.

Needs OUT-OF-FOLD scores for the training week (each payment scored by a model that
never saw it, e.g. from the honest-test GroupKFold). Scores from a model trained on
the whole week make the misses look rarer than they really are.

Example:
  python3 analyse_misses.py --labels data/train/labels.csv --label-col label \
      --tier-col difficulty --oof data/oof_scores.parquet

Groups compared:
  missed  = suspicious, in the chosen tiers, ranked outside the top --top-frac
  rivals  = innocent payments ranked above the median missed payment
For every descriptor the script reports an AUC: 0.5 means no difference, above 0.5
means the missed payments have higher values, below 0.5 lower. Features far from
0.5 against the rivals are the best candidates for a new clue.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features_wallet import COLS, add_features, parse_time


def read_any(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def need(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f"{name} has no column(s) {missing}. It has: {list(df.columns)}")


def build_descriptors(df):
    """Simple per-payment descriptors, from this week's payments only (no labels)."""
    c = COLS
    t = parse_time(df[c["time"]])
    out = pd.DataFrame(index=df.index)
    out["amount"] = pd.to_numeric(df[c["usd"]], errors="coerce")
    out["amount_is_whole_number"] = (out["amount"] % 1 == 0).astype(float)
    out["hour_of_day"] = t.dt.hour
    out["day_of_week"] = t.dt.dayofweek
    out["pays_itself"] = (df[c["sender"]] == df[c["receiver"]]).astype(float)
    out["sender_payments_week"] = df.groupby(c["sender"])[c["hash"]].transform("size")
    out["receiver_payments_week"] = df.groupby(c["receiver"])[c["hash"]].transform("size")
    pair = [c["sender"], c["receiver"]]
    out["pair_payments_week"] = df.groupby(pair)[c["hash"]].transform("size")
    order = np.argsort(t.values, kind="mergesort")
    seq = df.iloc[order].groupby(pair).cumcount().to_numpy()
    first = pd.Series(seq == 0, index=df.index[order]).reindex(df.index)
    out["first_time_this_pair"] = first.astype(float)
    return out


def auc_between(a, b):
    """AUC of 'is in group a' using the value; NaNs dropped. None if not computable."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 20 or len(b) < 20:
        return None
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return roc_auc_score(y, np.r_[a.to_numpy(), b.to_numpy()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payments", default="data/clean/train_payments.parquet")
    ap.add_argument("--labels", required=True, help="training labels file (csv or parquet)")
    ap.add_argument("--label-col", default="label", help="0/1 suspicious column in the labels")
    ap.add_argument("--tier-col", default=None, help="difficulty column in the labels, if any")
    ap.add_argument("--tiers", nargs="+", default=["easy", "medium"])
    ap.add_argument("--oof", default="data/oof_scores.parquet", help="TRANSACTION_HASH + out-of-fold score")
    ap.add_argument("--score-col", default="score")
    ap.add_argument("--top-frac", type=float, default=0.01, help="share of the ranking counted as 'caught'")
    ap.add_argument("--out", default="miss_analysis.csv")
    a = ap.parse_args()

    h = COLS["hash"]
    pay, lab, oof = read_any(a.payments), read_any(a.labels), read_any(a.oof)
    need(pay, list(COLS.values()), "payments table")
    need(lab, [h, a.label_col] + ([a.tier_col] if a.tier_col else []), "labels file")
    need(oof, [h, a.score_col], "scores file")

    lab_cols = [h, a.label_col] + ([a.tier_col] if a.tier_col else [])
    df = pay.merge(lab[lab_cols], on=h, how="left").merge(
        oof[[h, a.score_col]].rename(columns={a.score_col: "score"}), on=h, how="left"
    )
    if len(df) != len(pay):
        sys.exit(f"merge changed the row count ({len(pay)} -> {len(df)}): the hash must be unique in every file")
    if df[a.label_col].isna().any() or df["score"].isna().any():
        print(f"WARNING: {df[a.label_col].isna().sum()} rows have no label and "
              f"{df['score'].isna().sum()} have no score; they are dropped from the groups.")
    df = df.dropna(subset=[a.label_col, "score"]).reset_index(drop=True)

    df["rank"] = df["score"].rank(ascending=False, method="first")
    cutoff = int(len(df) * a.top_frac)
    suspicious = df[a.label_col] == 1
    if a.tier_col:
        in_tier = df[a.tier_col].astype(str).str.lower().isin([t.lower() for t in a.tiers])
    else:
        in_tier = pd.Series(True, index=df.index)
        print("NOTE: no --tier-col given, so every suspicious payment is treated as easy/medium.")
    target = suspicious & in_tier
    missed = target & (df["rank"] > cutoff)
    if missed.sum() < 20:
        sys.exit(f"only {missed.sum()} missed payments: too few to compare. Check --label-col/--tier-col/--tiers.")
    rivals = (~suspicious) & (df["rank"] < df.loc[missed, "rank"].median())
    innocent = ~suspicious
    if rivals.sum() < 20:
        sys.exit(f"only {rivals.sum()} innocent payments are ranked above the typical miss, so there is "
                 f"nothing to compare against. The missed group probably includes payments the model "
                 f"already scores high: try a larger --top-frac.")

    print(f"{len(df):,} payments; top {a.top_frac:.0%} = first {cutoff:,} ranks")
    print(f"target suspicious (tiers {a.tiers}): {target.sum():,}; caught in the top: "
          f"{(target & ~missed).sum():,}; missed: {missed.sum():,}")
    print(f"rivals (innocent, ranked above the median miss at rank {df.loc[missed, 'rank'].median():,.0f}): "
          f"{rivals.sum():,}")

    feats = add_features(df, "train")
    assert (feats["TRANSACTION_HASH"].to_numpy() == df[h].to_numpy()).all()
    X = pd.concat(
        [build_descriptors(df), feats.drop(columns="TRANSACTION_HASH").set_index(df.index)], axis=1
    )

    rows = []
    for f in X.columns:
        vs_rivals = auc_between(X.loc[missed, f], X.loc[rivals, f])
        vs_innocent = auc_between(X.loc[missed, f], X.loc[innocent, f])
        if vs_rivals is None:
            continue
        rows.append({
            "feature": f,
            "auc_vs_rivals": round(vs_rivals, 3),
            "auc_vs_all_innocent": None if vs_innocent is None else round(vs_innocent, 3),
            "median_missed": X.loc[missed, f].median(),
            "median_rivals": X.loc[rivals, f].median(),
            "blank_missed": round(X.loc[missed, f].isna().mean(), 3),
            "blank_rivals": round(X.loc[rivals, f].isna().mean(), 3),
        })
    if not rows:
        sys.exit("no descriptor had enough non-blank values in both groups to compare.")
    res = pd.DataFrame(rows)
    res["distance_from_0.5"] = (res["auc_vs_rivals"] - 0.5).abs()
    res = res.sort_values("distance_from_0.5", ascending=False).drop(columns="distance_from_0.5")
    res.to_csv(a.out, index=False)
    print("\nWhat separates the missed payments from the innocent ones ranked above them:")
    print(res.to_string(index=False))
    print(f"\nSaved to {a.out}")


if __name__ == "__main__":
    main()
