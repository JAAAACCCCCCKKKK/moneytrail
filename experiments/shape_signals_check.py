"""Check the deck signals added in features_shape_signals.py (Jack).

1. Does each signal pick out its own typology? Average precision at finding the
   suspicious payments of that family among all payments (chance = their share), and
   among that family only, suspicious against honest look-alikes (chance about 0.5).
   Each signal is tried in both directions; the better one is reported.
2. Does adding them help? Honest test (main.py's folds, group smoothing, 0.75/0.25
   rank blend, as src.predict_shapes_v2) of candidate B with and without --signals,
   on three fold seeds.

Run:  python -m experiments.shape_signals_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from features_shape_signals import add_features
from main import make_model
from src.predict_shapes_v2 import column_sets, prepare

FAMILY_SIGNALS = {
    "fan_in": ["sig_fanin_quiet_share_600", "sig_fanin_new_share_600", "sig_fanin_sender_week_log_600",
               "sig_fanin_exit_closeness_600", "sig_fanin_last_deposit_to_exit"],
    "fan_out": ["sig_fanout_fast_share_600", "sig_fanout_fwd_median_log_600"],
    "fan_out_in": ["sig_sg_paths", "sig_sg_value_balance", "sig_sg_middle_quiet", "sig_sg_hold_spread"],
    "stack": ["sig_g600_middle_once_share", "sig_g600_middle_week_log", "sig_g600_hold_spread", "sig_g600_routes"],
    "bipartite": ["sig_bip_inside_share", "sig_bip_pairs_active", "sig_bip_amount_spread"],
    "cycle": ["sig_loop_seconds", "sig_loop_total_hops"],
    "multi_stage": ["sig_g600_routes", "sig_gonce_routes", "sig_g600_outside_balance"],
    "random": ["sig_g600_closure", "sig_gonce_closure", "sig_g600_outside_balance"],
}


def identification():
    f = add_features(common.payments("train"), "train").merge(common.labels(), on="TRANSACTION_HASH")
    rows = []
    for family, columns in FAMILY_SIGNALS.items():
        target = ((f.family == family) & (f.is_suspicious == 1)).to_numpy()
        same = (f.family == family).to_numpy()
        for c in columns:
            raw = f[c].to_numpy(float)
            best = None
            for sign in (1, -1):
                x = np.nan_to_num(sign * raw, nan=-1e12)        # blanks rank last
                score = (aps(target, x), aps(target[same], x[same]))
                if best is None or score[0] > best[0][0]:
                    best = (score, sign)
            (all_ap, within_ap), sign = best
            rows.append(dict(family=family, signal=c, direction="high" if sign > 0 else "low",
                             find_family=round(all_ap, 3), chance=round(target.mean(), 4),
                             susp_vs_lookalike=round(within_ap, 3), chance_within=round(target[same].mean(), 2),
                             filled=round(float(np.isfinite(raw).mean()), 3)))
    return pd.DataFrame(rows)


def honest_test(seeds=(42, 7, 2026)):
    results = {}
    for name, signals in (("B", False), ("B + signals", True)):
        train, _, lab, X, _ = prepare(drop_conversion=True, rank_amounts=True, signals=signals)
        y = lab.is_suspicious.to_numpy()
        pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
        groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + pay.FROM_ADDRESS)
        everything, typology = column_sets(X.columns)
        for seed in seeds:
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups))
            ranks = []
            for cols in (everything, typology):
                oof = np.zeros(len(y))
                for tr, va in folds:
                    oof[va] = make_model("lgbm").fit(X.iloc[tr][cols], y[tr]).predict_proba(X.iloc[va][cols])[:, 1]
                smoothed = graph.smooth_by_group(oof, pay, "train")
                ranks.append(pd.Series(smoothed).rank().to_numpy() / len(smoothed))
            s = 0.75 * ranks[0] + 0.25 * ranks[1]
            per = [100 * aps(y[va], s[va]) for _, va in folds]
            top5 = s >= np.sort(s)[::-1][int(len(s) * 0.05)]
            recall = {d: float(top5[(lab.difficulty == d).to_numpy()].mean()) for d in ("easy", "medium")}
            results[(name, seed)] = per
            print(f"  {name:12s} seed {seed:>4}: folds {np.mean(per):5.1f} +/- {np.std(per):3.1f}  "
                  f"all {100 * aps(y, s):5.1f}  top-5% recall easy {recall['easy']:.2f} medium {recall['medium']:.2f}  "
                  f"({X.shape[1]} clues)", flush=True)
    gains = [np.mean(results[("B + signals", s)]) - np.mean(results[("B", s)]) for s in seeds]
    wins = sum(a > b for s in seeds for a, b in zip(results[("B + signals", s)], results[("B", s)]))
    print(f"  gain over B per seed: {', '.join(f'{g:+.1f}' for g in gains)}; folds better {wins}/{5 * len(seeds)}")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print("1. Does each new signal find its typology?")
    print(identification().to_string(index=False))
    print("\n2. Honest test, candidate B with and without the new signals")
    honest_test()
