# MoneyTrail

![MoneyTrail](docs/banner.jpeg)

Team MoneyTrail's entry for Decode & Detect: The XRPL AI Hack (UKCBT x XRPL Commons, 24 October 2026).
Every test transaction gets a suspicion score; payments are flagged for review, never called criminal.
The data is synthetic, so these results say nothing about real accounts or the real XRP Ledger.

**Team:** Alim Badmus (captain), Natasha, Jack, Violet.
**Best leaderboard score:** 67.1 (commit `ed19b17`, `outputs/predictions_blend_two_q_violet.csv`).
What we tried and why, with every honest-test and leaderboard score: `DECISIONS.md` and `SUBMISSIONS.md`.

## Quickstart

1. Put the organisers' data under `data/` (it is not in this repo):

   ```
   data/train/        transactions.parquet, transfers.parquet, market_rates.csv, labels.csv
   data/test-data/    transactions.parquet, transfers.parquet, market_rates.csv
   ```

2. Install the dependencies (see Deps), then run, from the repo root:

   ```bash
   python detector/predict.py --train data/train --test data/test-data --out outputs/predictions_violet_v2_fixed.csv
   python -m src.predict_two_q
   python -m src.blend --run3 predictions_two_q.csv --violet predictions_violet_v2_fixed.csv --out predictions_blend_two_q_violet.csv
   ```

   The last file, `outputs/predictions_blend_two_q_violet.csv`, is the predictions CSV we upload.
   On a Windows laptop the three steps take about 20 minutes in total from a fresh clone (most of it building the clue tables); clue
   tables are cached in `data/cache/` (delete it after changing any clue code).

3. The run is deterministic: every random seed is fixed (LightGBM `random_state=42`; Violet's
   gradient boosting seeds 0, 1 and 2, and her out-of-fold split seed 0), and wallet sets are
   sorted before summing. Rerunning from a fresh clone gives byte-identical files.

**What the three steps do**

1. `detector/` (Violet): about 225 wallet, window, gap-cluster and motif clues per payment, scikit-learn
   gradient boosting, then a second stage with the scores of nearby payments.
2. `src/predict_two_q.py` (Jack, built on Alim's `src/predict_union_signals.py`): one LightGBM on Jack's
   shape and typology clues plus Violet's clues, smoothed over wallet groups; blended 50/50 by rank
   with two LightGBMs that answer "is this payment part of a planted scheme?" and "is that scheme
   suspicious?", the second also seeing slow-scheme groups built from once-a-week payment pairs.
3. `src/blend.py`: a 50/50 rank average of steps 1 and 2.

**Honest test:** 5-fold cross-validation on the training week, grouped by planted scenario so a scheme
never sits on both sides (`scenario_id` is used only to split). See `DECISIONS.md`.

## Deps

- Python 3.12 (we used 3.12.6).
- `python -m pip install -r requirements.txt`; exact versions in `requirements-lock.txt`. The ones the
  run uses: pandas 3.1.0, pyarrow 26.0.0, numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0, networkx 3.7.

**Not written by us:** the organisers' `payments.py` (one USD-valued edge per payment) and the twelve
rules of their starter notebook (copied into `notebook_features.py` and `features_starter.py`);
pandas, pyarrow, NumPy, scikit-learn, LightGBM and NetworkX (BSD, Apache 2.0 or MIT licences).
No external data, APIs or pre-trained models are used.

## AI Usage Disclosure

- **Claude Code** (Claude Opus 5 and Claude Opus 5.5, Anthropic), default settings, no custom
  temperature or sampling. Used interactively to inspect the data, measure the label structure, write
  most of `src/`, `graph.py`, the `features_*.py` clue files, `main.py`, `common.py`, `tools/` and the
  release workflow, run the honest tests and draft the logs. The reusable prompts are the skill files
  in `.claude/skills/`; the rest were the team's own messages in interactive sessions.
- **Claude** (Claude Opus 5.5, Anthropic), in a chat session directed by Violet, wrote `detector/`.
- **Before the event**, Claude Opus 5.5 via Claude Code wrote the skill files and the notes in `prep/`.
- **Reproducibility limits:** the AI conversations themselves can't be replayed exactly, but every
  line of code they produced is committed here and runs deterministically. The full list of files
  and what each AI tool did is in the "AI use" section of `DECISIONS.md`.

## Licence

MIT (see `LICENSE`).
