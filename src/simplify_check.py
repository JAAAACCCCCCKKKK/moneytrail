"""Which clues does the 67.8 gate really need? Family ablation, permutation importance,
duplicate removal and top-N pruning, all on the gate's honest test.

  python3 -m experiments.simplify_check                 # everything, fold seed 42 (~15-25 min)
  python3 -m experiments.simplify_check --skip-families # only importance, duplicates, top-N (~8 min)
  python3 -m experiments.simplify_check --fold-seeds 42 7 2026
  python3 -m experiments.simplify_check --predict-top 100   # probability files for a pruned gate

What it does (labels only as the training target and to score, never as clues):
  1. Baseline: the gate's recipe on main.py's folds, out of fold for both models.
       combined  = Jack's clues + deck signals + Violet's clues, LightGBM, group smoothing
       structure = same + once-a-week groups, target "part of a planted shape scheme"
       gate      = log(combined) + log(structure)
     Without Violet's 10% rank share (that needs her out-of-fold scores), i.e. the 63.8 gate.
  2. Family ablation: drop one family of clues from both models; the drop in the honest test
     is what that family is worth. Families are tested whole because near-duplicate clues
     (Jack's and Violet's versions of fan-in, time to forward...) cover for each other.
  3. Permutation importance: in each held-out fold, shuffle one clue and measure how much the
     combined model's average precision falls. No retraining, so it is quick.
  4. Duplicates: of every pair of clues with rank correlation above 0.95, keep the more important.
  5. Top-N: keep only the N most important clues (50, 100, 200) and rerun the gate.

Read the table: a family whose removal costs ~0 is dead weight; a pruned set that scores the
same as all clues is the better model (simpler, and less tied to quirks of the training week,
which matters given four honest-test gains today that lost on the leaderboard).
Outputs: printed table, outputs/simplify_check.json, outputs/clue_importance.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as aps
from sklearn.model_selection import StratifiedGroupKFold

import common
import graph
from main import make_model, OUTPUTS
from src.predict_shapes_v2 import prepare, CONVERSION_KEYS
from src.predict_union import SHAPES, fit_seeds, violet_clues
from src.predict_union_signals import LEFT_OUT

TOP_N = (50, 100, 200)
DUPLICATE_CORR = 0.95

# First matching rule wins. Printed at the start so the split can be checked.
FAMILIES = [
    ("conversion", lambda c: (not c.startswith("v_") and any(k in c for k in CONVERSION_KEYS))
     or (c.startswith("v_") and any(k in c.lower() for k in ("conv", "legs", "asset", "sm_slack")))),
    ("once-a-week groups", lambda c: c.startswith("once_")),
    ("Violet: pair history", lambda c: c.startswith("v_pair")),
    ("Violet: gaps and clusters", lambda c: c.startswith("v_") and re.search(r"gap|comp_|clust", c) is not None),
    ("Violet: activity windows", lambda c: c.startswith("v_") and re.search(r"_\d+$|burst|fwd|pre", c) is not None),
    ("Violet: wallet week", lambda c: c.startswith("v_")),
    ("Jack: deck signals", lambda c: c.startswith("sig_")),
    ("Jack: group shapes", lambda c: c.startswith(("shape_b_", "shape_g"))),
    ("Jack: typology detectors", lambda c: c.startswith("shape_")),
    ("Starter rules and run 3 basics", lambda c: True),
]


def family_of(columns):
    return {c: next(name for name, rule in FAMILIES if rule(c)) for c in columns}


def tables():
    t0 = time.time()
    train, test, lab, X, Xt = prepare(once_groups=True, signals=True)
    X, Xt = X.drop(columns=list(LEFT_OUT)), Xt.drop(columns=list(LEFT_OUT))
    Us = pd.concat([X.reset_index(drop=True), violet_clues("train", train.TRANSACTION_HASH)], axis=1)
    Uts = pd.concat([Xt.reset_index(drop=True), violet_clues("test", test.TRANSACTION_HASH)], axis=1)[Us.columns]
    print(f"clue tables built ({time.time() - t0:.0f}s)", flush=True)
    return train, test, lab, Us, Uts


class Gate:
    """The gate's honest test for any subset of clues, on fixed folds."""

    def __init__(self, train, lab, Us, fold_seeds):
        self.Us, self.once = Us, [c for c in Us.columns if c.startswith("once_")]
        self.y = lab.is_suspicious.to_numpy()
        self.ys = lab.family.isin(SHAPES).to_numpy().astype(int)
        self.pay = train[["TRANSACTION_HASH"]].merge(common.payments("train"), on="TRANSACTION_HASH")
        groups = np.where(lab.scenario_id.notna(), "s:" + lab.scenario_id.astype(str), "w:" + self.pay.FROM_ADDRESS)
        self.folds = {s: list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Us, self.y, groups))
                      for s in fold_seeds}

    def oof(self, cols, y, folds, keep_models=False):
        p, models = np.zeros(len(y)), []
        for tr, va in folds:
            m = make_model("lgbm").fit(self.Us.iloc[tr][cols], y[tr])
            p[va] = m.predict_proba(self.Us.iloc[va][cols])[:, 1]
            models.append(m)
        return (p, models) if keep_models else p

    def run(self, keep):
        """keep: the clues to use. The combined model never sees once_ clues; the structure model
        uses the kept clues plus whichever once_ clues are kept."""
        keep = [c for c in self.Us.columns if c in set(keep)]
        comb_cols = [c for c in keep if c not in self.once]
        out = {}
        for s, folds in self.folds.items():
            comb = graph.smooth_by_group(self.oof(comb_cols, self.y, folds), self.pay, "train")
            gate = np.log(np.clip(comb, 1e-9, 1)) + np.log(np.clip(self.oof(keep, self.ys, folds), 1e-9, 1))
            out[s] = [100 * aps(self.y[va], gate[va]) for _, va in folds]
        return out


def compare(name, res, base, n_clues, rows):
    b = np.concatenate([base[s] for s in base]); r = np.concatenate([res[s] for s in res])
    row = dict(config=name, clues=n_clues, honest=float(r.mean()), change=float(r.mean() - b.mean()),
               folds_better=int((r > b).sum()), folds=len(r))
    rows.append(row)
    print(f"  {name:38s} {n_clues:4d} clues  {row['honest']:5.1f}  ({row['change']:+.2f}, "
          f"better in {row['folds_better']}/{row['folds']})", flush=True)


def permutation_importance(gate, cols, seed):
    """Average fall in the combined model's held-out AP when one clue is shuffled."""
    folds = gate.folds[seed]
    _, models = gate.oof(cols, gate.y, folds, keep_models=True)
    rng = np.random.default_rng(0)
    drop = pd.Series(0.0, index=cols)
    for (tr, va), m in zip(folds, models):
        Xv, yv = gate.Us.iloc[va][cols].copy(), gate.y[va]
        base = aps(yv, m.predict_proba(Xv)[:, 1])
        for c in cols:
            keep = Xv[c].to_numpy().copy()
            Xv[c] = rng.permutation(keep)
            drop[c] += (base - aps(yv, m.predict_proba(Xv)[:, 1])) * 100 / len(folds)
            Xv[c] = keep
    return drop


def duplicates(U, importance):
    """Of each pair with |rank correlation| above DUPLICATE_CORR, the less important clue."""
    R = U.rank().fillna(-1)
    R = R.loc[:, R.std() > 0]
    C = np.abs(np.corrcoef(R.to_numpy().T))
    cols, drop = list(R.columns), set()
    order = sorted(range(len(cols)), key=lambda i: -importance.get(cols[i], 0))
    for a_i, a in enumerate(order):
        if cols[a] in drop:
            continue
        for b in order[a_i + 1:]:
            if cols[b] not in drop and C[a, b] > DUPLICATE_CORR:
                drop.add(cols[b])
    constant = [c for c in U.columns if c not in R.columns]
    return sorted(drop) + constant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--skip-families", action="store_true")
    parser.add_argument("--predict-top", type=int, default=None,
                        help="write probability files for a gate on the top-N clues (needs a previous run's "
                             "outputs/clue_importance.csv)")
    args = parser.parse_args()
    train, test, lab, Us, Uts = tables()
    fam = family_of(Us.columns)

    if args.predict_top:
        imp = pd.read_csv(OUTPUTS / "clue_importance.csv", index_col=0).importance
        keep = list(imp.sort_values(ascending=False).index[:args.predict_top]) + [c for c in Us.columns if c.startswith("once_")]
        keep = [c for c in Us.columns if c in set(keep)]
        comb = [c for c in keep if not c.startswith("once_")]
        y, ys = lab.is_suspicious.to_numpy(), lab.family.isin(SHAPES).to_numpy().astype(int)
        test_pay = test[["TRANSACTION_HASH"]].merge(common.payments("test"), on="TRANSACTION_HASH")
        p_comb = graph.smooth_by_group(fit_seeds(Us[comb], y, Uts[comb], 1), test_pay, "test")
        p_struct = fit_seeds(Us[keep], ys, Uts[keep], 1)
        OUTPUTS.mkdir(exist_ok=True)
        tag = f"top{args.predict_top}"
        for name, s in ((f"predictions_union_signals_{tag}_prob.csv", p_comb), (f"predictions_scheme_{tag}_prob.csv", p_struct)):
            pd.DataFrame({"TRANSACTION_HASH": test.TRANSACTION_HASH, "score": s}).to_csv(OUTPUTS / name, index=False)
            print(f"wrote {OUTPUTS / name}")
        print(f"next: python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_signals_{tag}_prob.csv "
              f"--violet predictions_violet_v2_fixed.csv --structure predictions_scheme_{tag}_prob.csv "
              f"--out predictions_gate_v10_{tag}.csv")
        return

    print("\nClue families:")
    for name, n in pd.Series(fam).value_counts().reindex([f[0] for f in FAMILIES]).dropna().items():
        print(f"  {name:32s} {int(n):4d}")

    gate, rows, t0 = Gate(train, lab, Us, args.fold_seeds), [], time.time()
    all_cols = list(Us.columns)
    print("\n1. Baseline")
    base = gate.run(all_cols)
    compare("all clues", base, base, len(all_cols), rows)
    print(f"   (one gate run took {time.time() - t0:.0f}s)")

    if not args.skip_families:
        print("\n2. Family ablation: drop one family (cost = what the family is worth)")
        for name, _ in FAMILIES:
            cols = [c for c in all_cols if fam[c] != name]
            if len(cols) < len(all_cols):
                compare(f"without {name}", gate.run(cols), base, len(cols), rows)

    print("\n3. Permutation importance (combined model, held-out folds)", flush=True)
    comb_cols = [c for c in all_cols if not c.startswith("once_")]
    imp = permutation_importance(gate, comb_cols, args.fold_seeds[0])
    by_family = imp.groupby(pd.Series(fam)).sum().sort_values(ascending=False)
    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame({"importance": imp, "family": pd.Series(fam)}).dropna().sort_values(
        "importance", ascending=False).to_csv(OUTPUTS / "clue_importance.csv")
    print(f"   clues that matter (AP falls > 0.05 when shuffled): {(imp > 0.05).sum()} of {len(imp)}; "
          f"no effect or harmful: {(imp <= 0).sum()}")
    print("   top 10:")
    for c, v in imp.sort_values(ascending=False).head(10).items():
        print(f"     {c:34s} {v:5.2f}  ({fam[c]})")
    print("   by family (sum):")
    for f_, v in by_family.items():
        print(f"     {f_:34s} {v:6.2f}")

    print("\n4. Duplicates and top-N pruning")
    dup = duplicates(Us[comb_cols], imp.to_dict())
    compare(f"without {len(dup)} near-duplicates", gate.run([c for c in all_cols if c not in set(dup)]), base,
            len(all_cols) - len(dup), rows)
    once = [c for c in all_cols if c.startswith("once_")]
    for n in TOP_N:
        top = list(imp.sort_values(ascending=False).index[:n])
        compare(f"top {n} + once-a-week groups", gate.run(top + once), base, len(set(top + once)), rows)

    table = pd.DataFrame(rows)
    simple = table[(table.change >= -0.2) & (table.folds_better >= table.folds * 0.4)].sort_values("clues")
    print("\nSummary (honest test of the gate without Violet's share; baseline is 'all clues'):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if len(simple) and simple.iloc[0].config != "all clues":
        s = simple.iloc[0]
        how = "--predict-top N" if s.config.startswith("top") else "a family drop (not wired to --predict yet)"
        print(f"\nSimplest set that holds the score: '{s.config}' ({s.clues} clues, {s.change:+.2f}). "
              f"To test it on the leaderboard: {how}.")
    else:
        print("\nNo smaller set holds the score: the clues are pulling their weight.")
    (OUTPUTS / "simplify_check.json").write_text(json.dumps(
        {"rows": rows, "families": by_family.to_dict(), "duplicates_dropped": dup}, indent=2))


if __name__ == "__main__":
    main()
