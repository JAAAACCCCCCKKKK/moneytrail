# Decision log

Team: MoneyTrail (Alim Badmus, captain; Natasha; Jack; Violet)
Honest test: the starter notebook uses a random 20% holdout. Our own honest test will be 5-fold GroupKFold grouped by `scenario_id` from `train/labels.csv` (used only to split, never as a feature), so each planted scheme stays on one side.

## Decisions

### ~10:00 — Baseline: organisers' starter notebook
- **Change:** ran `starter_notebook.ipynb` unchanged (logistic regression on 12 simple rules: payment size, how busy sender and receiver are, passing money on within an hour, 10-minute bursts, similar amounts forwarded).
- **Why:** a valid first upload and a number to beat.
- **Honest test:** 24.6 on the notebook's random 20% holdout (likely optimistic: schemes can sit on both sides of a random split).
- **Leaderboard:** 33.6 (commit 4d8970d, run at 09:34 UTC). Top 1%: precision 62%, recall 20%. ROC AUC 0.82. Caught in the top 1%: easy 30% of 800, medium 10% of 225, observable hard 0% of 200, ambiguous 0% of 25. The test week has 1,250 suspicious transactions (about 3.1%).
- **Decision:** kept as the baseline. Strongest rules: 10-minute bursts from the sender (+0.53) and into the receiver (+0.49); busy receivers count against suspicion (−0.90), the hub effect.

### ~10:50 — main.py, honest test, and LightGBM on the starter clues
- **Change:** built `main.py` (loads payments, adds every clue file, runs the honest test, trains, writes predictions) and moved the starter's 12 rules into `features_starter.py`.
- **Why:** one shared pipeline the team's clue files plug into, and a trustworthy way to judge each idea.
- **Honest test:** 5-fold StratifiedGroupKFold, grouped by `scenario_id` (normal payments without one grouped by sender). Starter logistic regression 30.4 ± 5.9; LightGBM 30.2 ± 6.2 on the same clues. Easy caught in top 1%: 26% vs 30%; medium 4% vs 3%; hard 0%.
- **Leaderboard:** not uploaded (no improvement).
- **Decision:** keep LightGBM as the default (it will use combinations of richer clues), but the gain has to come from new clues. The honest test (~30) is close to the leaderboard (33.6), so we trust it.

## AI use

- Claude Code (Claude Opus 5.5, Anthropic): ran the starter notebook, checked the submission file with our `submission-check` skill, and wrote these log entries, and helped write `main.py`, `common.py` and `features_starter.py`. Prompts: the skill files in `.claude/skills/`.
