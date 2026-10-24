# Decision log

Team: MoneyTrail (team list to be filled in before the final ZIP)
Honest test: 5-fold `GroupKFold` grouped by `scenario_id` from `train/labels.csv`, with each
unlabelled background transaction as its own group. Labels are pure per scenario, so any split
that lets one scenario's members fall on both sides inflates the score.
Team: MoneyTrail (Alim Badmus, captain; Natasha; Jack; Violet)
Honest test: the starter notebook uses a random 20% holdout. Our own honest test will be 5-fold GroupKFold grouped by `scenario_id` from `train/labels.csv` (used only to split, never as a feature), so each planted scheme stays on one side.

## Decisions

### ~10:00 — Baseline: organisers' starter notebook
- **Change:** ran `starter_notebook.ipynb` unchanged (logistic regression on 12 simple rules: payment size, how busy sender and receiver are, passing money on within an hour, 10-minute bursts, similar amounts forwarded).
- **Why:** a valid first upload and a number to beat.
- **Honest test:** 24.6 on the notebook's random 20% holdout (likely optimistic: schemes can sit on both sides of a random split).
- **Leaderboard:** 33.6 (commit 4d8970d, run at 09:34 UTC). Top 1%: precision 62%, recall 20%. ROC AUC 0.82. Caught in the top 1%: easy 30% of 800, medium 10% of 225, observable hard 0% of 200, ambiguous 0% of 25. The test week has 1,250 suspicious transactions (about 3.1%).
- **Decision:** kept as the baseline. Strongest rules: 10-minute bursts from the sender (+0.53) and into the receiver (+0.49); busy receivers count against suspicion (−0.90), the hub effect.
Times are London time (BST, UTC+1) and approximate for the first three entries.

### 10:00 — Dataset inspected, label structure mapped
- **Change:** no model change. Verified all 12 SHA-256 checksums in `checksums.json`, then
  mapped the label file and the three splits.
- **Why:** `labels.csv` carries `family`, `difficulty`, `scenario_id` and `cohort_id` as well as
  `is_suspicious`. None of those exist in the test data, but they are legitimate extra
  supervision and the only correct grouping key for validation.
- **Findings:** all 1,497 scenarios are label-pure, so the real unit of prediction is the
  scenario and not the transaction. Shape families are roughly half suspicious and half
  legitimate (fan_in 101/76, fan_out 92/75, cycle 78/113, bipartite 134/152), so shape alone is
  not the label; within every family the separator is tempo, with a median inter-payment gap
  near 60-106s for suspicious scenarios against 428-732s for legitimate ones. The 200 planted
  hard cases sit one per `cohort_id` inside the 20,000 `aggregate_relay` and `distributed_relay`
  rows; eleven scenario-level aggregates put them at a within-cohort percentile near 0.5, so
  none of them separates the hard cases. Wallets and hashes are disjoint across splits, so no
  identity feature can transfer. Conversion share shifts from 0.41 in train to 0.51 in test.
- **Honest test:** not measured (no model).
- **Decision:** kept as the basis for the plan. Predict per motif and broadcast to its
  transactions; group every split by `scenario_id`.

### ~10:50 — main.py, honest test, and LightGBM on the starter clues
- **Change:** built `main.py` (loads payments, adds every clue file, runs the honest test, trains, writes predictions) and moved the starter's 12 rules into `features_starter.py`.
- **Why:** one shared pipeline the team's clue files plug into, and a trustworthy way to judge each idea.
- **Honest test:** 5-fold StratifiedGroupKFold, grouped by `scenario_id` (normal payments without one grouped by sender). Starter logistic regression 30.4 ± 5.9; LightGBM 30.2 ± 6.2 on the same clues. Easy caught in top 1%: 26% vs 30%; medium 4% vs 3%; hard 0%.
- **Leaderboard:** not uploaded (no improvement).
- **Decision:** keep LightGBM as the default (it will use combinations of richer clues), but the gain has to come from new clues. The honest test (~30) is close to the leaderboard (33.6), so we trust it.
### 10:30 — Organisers' starter notebook run as the baseline
- **Change:** copied `starter_notebook.ipynb` and `payments.py` into the repo, pointed them at
  `data/train` and `data/test-data`, and executed the notebook unchanged otherwise. Its twelve
  features were copied verbatim into `notebook_features.py` so they can be scored again.
- **Why:** we need a valid submission file and a number to beat before adding anything.
- **Honest test:** the notebook's own held-back split reports 24.6. The same features and model
  under our grouped 5-fold score **29.9, per fold 30.9 +/- 7.3**. A stratified 5-fold scores
  30.8 +/- 2.1 on the same features.
- **Note:** the notebook's single random split is not just leaky, it is noisy, with only about
  250 positives in the held-back fifth. The grouped folds have a much wider spread (+/- 7.3
  against +/- 2.1), so single-split comparisons between ideas will not be trustworthy today.
  Every decision from here uses the grouped 5-fold mean and spread.
- **Recall inside the top 1% of grouped scores:** easy 0.270, medium 0.031, hard 0.013.
- **Reference ceiling:** ranking every easy and medium case perfectly while missing every hard
  one scores 84.0, so about 84 of the 100 points sit in easy plus medium.
- **Submission check:** `predictions.csv` passes the checker: 40,101 rows, 40,101 distinct
  scores, range 0.0023 to 0.9988, no ties, OK to upload.
- **Leaderboard:** uploaded by the team at 09:34 UTC as `moneytrail-run1-code.zip` (commit not
  recorded). Score **33.6**; inside the top 1%: precision 62%, recall 20%; ROC-AUC 0.82. Caught
  in the top 1%: easy 30% of 800, medium 10% of 225, observable hard 0% of 200, ambiguous 0% of
  25. The test difficulty mix is identical to train's.
- **Calibration:** grouped CV 29.9 against leaderboard 33.6, so the grouped CV is a slightly
  conservative but usable stand-in for the leaderboard.
- **Decision:** kept as the baseline to beat.

### ~10:35 — Motif grouping plus shape features, first probe
- **Change:** exploratory only, not in the repo yet. Grouped payments into candidate motifs with
  a union-find that links two payments sharing a wallet within T seconds, then computed
  scenario-level shape features (size, span, median and minimum gap, rate, wallet and edge
  counts, density, maximum in- and out-degree, pass-through ratio, simple-cycle count, longest
  path, source and sink counts, asset count, conversion and bridge share, USD total, median and
  spread, route reuse) and scored them with LightGBM under the same grouped 5-fold.
- **Why:** labels are pure per scenario, so features describing the whole motif should beat
  features describing one payment. The strongest single raw per-transaction feature is a count
  of payments involving the same wallet within 120s, at AP 0.125 against 0.031 prevalence.
- **Honest test:** 29.9 (baseline) -> **41.1** at T=300s. T=120s scores 36.7 and T=900s 29.4.
  A version without the graph features (no pass-through, cycles or longest path) scores 37.2 at
  T=300s, so the graph-shape features are worth about 4 points.
- **Limitation found:** the union-find recovers scenarios badly. A true scenario is split across
  about 13 motifs, and the median motif holds one payment, so the gain above comes mostly from
  tiny local neighbourhoods rather than real motif detection. Recall inside the top 1% for hard
  cases is 0.000.
- **Correction (10:40):** the fragmentation figure above was dominated by the 1,200 relay
  scenarios, which span about 133 hours by design. Measured on the 297 shape scenarios alone,
  the largest motif covers a median 25% of its scenario at 120s, 50% at 300s and 100% at 600s,
  while at 600s relay motifs start merging up to 361 scenarios. So the union-find works for
  shapes at the right scale; the fix is several scales at once, not a new grouping method.
- **Decision:** superseded by the next entry.

### 10:45 — Multi-scale motif shape model, v1
- **Change:** new code in `src/`. `src/data.py` loads one USD-valued edge per transaction and
  handles the README traps. `src/motifs.py` builds motifs at 120s, 300s and 600s and describes
  each one's shape: size, span, gaps, rate, wallet and edge counts, density, fan-in and fan-out
  share, sources, sinks, middle wallets, pass-through, payer-receiver overlap, route reuse,
  longest path, cycle count, assets, conversion share and USD spread. `src/features.py` adds
  per-payment behaviour: busy counts in 60s, 180s, 600s and 3600s windows, seconds to forward,
  share forwarded within an hour, seconds since the sender was funded, nearest-neighbour gap
  absolute and relative to the wallet's own week, amount relative to the wallet's usual amount,
  wallet week profile, pair frequency, conversion spread from leg 0, and leg count. 135
  features. LightGBM, seed 42, settings in `src/train.py`. Final score blends the raw score with
  the motif-mean score at each of the three scales, 0.25 each.
- **Why:** labels are pure per scenario, so motif-level shape plus tempo should beat isolated
  payment rules; smoothing lets one caught member lift its whole motif. Tempo and amount also
  appear relative to each wallet's own week, against the README's warning about batch
  variations in test.
- **Honest test:** 29.9 (baseline) -> **52.9** raw, **55.5** with three-scale smoothing, per
  fold 57.6 +/- 10.0. Other smoothings: 120s only 54.0, 300s only 54.4, 600s only 54.5.
- **Leakage check:** wallets repeat across scenario folds, so wallet-week features could act as
  fingerprints. Dropping the 13 wallet-week features costs 52.9 -> 48.8 raw under the scenario
  split. Under a wallet-disjoint split, where fingerprinting is impossible, the full set still
  wins, 63.8 against 62.1 raw on 8,175 rows, so those features carry real behavioural signal.
- **Test predictions:** `predictions_shapes_v1.csv` passes the checker (40,101 rows, all
  distinct). Rank correlation with the baseline's test scores is 0.50; their top-1% sets share
  127 of 401 transactions.
- **Leaderboard:** run 2, uploaded 09:59 UTC from commit bbf8675 (rerun on Alim's laptop: honest test 54.9, folds 57.3 ± 9.9, so it reproduces). Score **61.4** (run 1 was 33.6). Top 1%: precision 93%, recall 30%. Caught in the top 1%: easy 50% of 800, medium 0% of 225, observable hard 0%, ambiguous 0%. Note: the top 1% is only 401 transactions against 1,250 suspicious, so recall there can't exceed about 32%; at 93% precision it is nearly full of easy cases, which is why medium reads 0% even though the overall score rose.
- **Decision:** kept. Hard recall in the top 1% is still 0.00, as expected.

### ~11:00 — Label study: easy cases first, hard relays look ordinary payment by payment
- **Change:** analysis only, no model change.
- **Findings:** 98% of easy and medium suspicious payments are the first payment between that sender and receiver in the week, against 54% of innocent payments (innocent activity has recurring relationships). Easy cases forward fast (median receiver forward time about 20 minutes against the 60-minute cap) and come in short bursts. The hard cases are 102 suspicious `aggregate_relay` among 10,098 innocent and 98 suspicious `distributed_relay` among 9,702 innocent. Every per-payment clue tested separates them weakly (best about 0.18 on a 0-to-1 scale): wallet age, activity, counterparties, amounts, forwarding within 1, 6, 24 or 72 hours, repeat pairings. Every suspicious relay has a `cohort_id`.
- **Why it matters:** easy is 64% of the suspicious transactions and the baseline caught only 30% of them in the top 1%, so the quickest points are in easy and medium. Hard relays need group and neighbour clues, not single-payment ones.
- **Honest test:** not measured (analysis only).
- **Decision:** priority order for the team: easy and medium first (first-time pairing, tempo, bursts, shapes); hard cases from 14:30 with group-level and guilt-by-association clues.

## AI use

- Claude Code (Claude Opus 5.5, Anthropic): ran the starter notebook, checked the submission file with our `submission-check` skill, and wrote these log entries, and helped write `main.py`, `common.py` and `features_starter.py`. Prompts: the skill files in `.claude/skills/`.
- **Claude Code, Claude Opus 5 then Claude Opus 5.5, Anthropic.** Used to inspect the dataset,
  measure the label structure reported above, run the organisers' starter notebook, write
  `src/honest_eval.py`, `notebook_features.py`, `src/data.py`, `src/motifs.py`,
  `src/features.py` and `src/train.py`, and draft this log. Settings: default Claude Code configuration, no
  custom temperature or sampling parameters. Prompts: the conversation was interactive; the
  reusable prompts are the three skill files in `.claude/skills/`, which are kept in this repo.
- **Claude Opus 5.5, Anthropic, via Claude Code.** Used before the hackathon to write the three
  skill files in `.claude/skills/` and the background notes in `prep/`. Disclosed in
  `prep/SKILLS.md` and `prep/RESEARCH.md`.
- **Pre-existing components not written by us:** the organisers' `payments.py`,
  `starter_notebook.ipynb` and its twelve features (copied into `notebook_features.py`), plus
  pandas, pyarrow, scikit-learn, LightGBM and NetworkX.
