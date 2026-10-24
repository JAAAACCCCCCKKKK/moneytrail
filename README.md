# MoneyTrail

![MoneyTrail](docs/banner.jpeg)

Decode & Detect: The XRPL AI Hack, 24 October 2026. One suspicion score per test transaction on a
synthetic XRP Ledger week. Final entry: leaderboard **68.0** (100 × average precision),
`outputs/predictions_gate_relay_v10.csv`, commit `f071e47`. Method write-up: [`METHOD.md`](METHOD.md);
full decision record: [`DECISIONS.md`](DECISIONS.md) and [`SUBMISSIONS.md`](SUBMISSIONS.md).

**Team:** Alim Badmus (captain), Jack, Violet, Natasha.

## Quickstart

Put the organisers' data in `data/train/` and `data/test-data/` as downloaded, then from the
repository root:

```bash
pip install -r requirements.txt   # exact versions: requirements-lock.txt
python3 -m src.predict_union --probabilities --out predictions_union_prob.csv
python3 -m src.predict_union --target structure --probabilities --out predictions_structure_prob.csv
python3 detector/predict.py --train data/train --test data/test-data --out outputs/predictions_violet_v2_fixed.csv
python3 -m src.relay_groups
python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_prob.csv --violet predictions_violet_v2_fixed.csv --structure predictions_structure_prob.csv --relay relay_multiplier.csv --out predictions_gate_relay_v10.csv
```

The result is `outputs/predictions_gate_relay_v10.csv` (about 10 minutes from raw data on a laptop;
intermediate clues are cached in `data/cache/`). Every model is retrained on each run; seeds are
fixed (LightGBM 42; scikit-learn 0, 1, 2), and the output is identical under different Python hash
seeds.

## Deps

Python 3.12 or later (our runs used 3.14.7). Top-level packages in `requirements.txt`, pinned
versions in `requirements-lock.txt`: pandas, pyarrow, NumPy, scikit-learn, LightGBM, NetworkX
(XGBoost and SHAP only in dropped experiments). Pre-existing components and their licences are
listed in `METHOD.md`. Our code is released under the MIT licence (`LICENSE`).

## AI Usage Disclosure

Claude Code (Claude Opus 5 and Claude Opus 5.5, Anthropic) and Claude (Claude Opus 5.5, for
Violet's `detector/`) were used to write much of the code, run analyses and draft the logs and
write-up, with default settings. Details, prompts and limits: the "AI use" section of
`DECISIONS.md` and the "AI disclosure" section of `METHOD.md`.
