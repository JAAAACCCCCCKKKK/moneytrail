# MoneyTrail: method

Decode & Detect hackathon, 24 October 2026. Final entry: leaderboard score **68.0** (100 × average
precision on the 40,101 test transactions), file `outputs/predictions_gate_relay_v10.csv`, made by
the code at commit `f071e47`; a fresh copy of that commit with only the raw data recreates it byte
for byte.
The full day-by-day record, with every number quoted here, is in `DECISIONS.md` and `SUBMISSIONS.md`.

## Team

MoneyTrail: Alim Badmus (captain), Natasha, Jack, Violet. Commits also come from the GitHub account
`gdn6` (the external-data and anomaly experiments).

## Problem and data sources

Give every payment on a synthetic, XRP Ledger-style week a suspicion score, so that the planted
laundering payments rise to the top. Scored by average precision, which rewards the order of the
whole list, not a threshold.

- **Organiser data:** `train/` (40,131 labelled transactions, one week), `validation/` (unlabelled)
  and the test week (40,101 transactions). Labels give `is_suspicious` plus `family`, `difficulty`,
  `scenario_id` and `cohort_id`; we used the last four only to split the honest test, to report
  results, and (for `family`) as the training target of one model, never as clues.
- **External data:** the IBM Transactions for Anti Money Laundering (AMLworld) HI-Small file from
  Kaggle, tried as a helper model. It showed no signal on our data (its score alone: AP 2.9 against
  3.1 for random ranking; in our model 63.72 → 63.81 on the honest test, 3 of 5 folds) and is **not
  used in the final entry**. Details in the "External data" table of `DECISIONS.md`.

## How we handled the data

- **One row per transaction**, from the organisers' `payments.py`: one effective sender-to-receiver
  edge, valued in US dollars from `market_rates.csv` for the hour of the payment. The cost is the
  first executed transfer leg, the received amount comes from `DELIVERED_AMOUNT`, and an asset is
  issuer + currency (USD from two issuers are different assets). `SendMax` is a ceiling, not a cost.
- **No identity clues:** wallets, hashes and issuers differ between weeks, so no address or hash is
  ever a clue. The 8 liquidity-provider hubs per week are set aside when grouping payments.
- **Conversions:** about 41% of training payments are conversions against 51% in the test week, and
  the README says asset choices ignore the labels. We tested dropping the conversion clues (honest
  test unchanged, leaderboard 66.5 against 66.7) and kept them.
- **Reproducibility traps fixed:** two clue files added money up in the order of Python sets of
  wallet addresses, which changes every time Python starts (Violet's `detector/clusters.py`, and
  Jack's `features_shape_signals.py`, which is not in the final entry). Both now sort first; under
  different hash seeds every clue is identical.

## How we tested honestly

All 1,497 planted scenarios are label-pure, so the real unit is the scheme, not the payment. Our
honest test is a **5-fold `StratifiedGroupKFold`** on the training week, grouped by `scenario_id`,
with background payments grouped by sender (fixed seed 42). Every model in a blend is scored out of
fold, so no payment is ever scored by a model that saw its scheme. The organisers' random 20% split
scored the starter model 24.6, and our grouped test 29.9; the leaderboard gave 33.6, so the grouped
test is the conservative, usable one. We accepted a change only if it gained more than a point and
won in at least 4 of 5 folds, and we let the leaderboard decide when the honest test couldn't (the
test week has "batch variations absent from training").

## Features (412 clues in the final model)

- **Starter rules (12):** the organisers' notebook rules: payment size, how busy sender and receiver
  are, money passed on within an hour, 10-minute bursts, similar amounts forwarded.
- **Jack's shape and typology clues (175):** payments are joined into wallet groups (shared wallet
  within 2, 5 and 10 minutes) and each group's shape is described: size, span, gaps, fan-in and
  fan-out, middle wallets, money passed through, payer-receiver overlap, repeated pairs. One
  detector per laundering typology: fan-in and fan-out at 10 minutes to 6 hours, scatter-gather
  two-hop paths, bipartite density, chain depth, and a time-respecting forward and backward search
  that finds loops of 6-8 hops. *Why:* planted schemes are compact shapes run fast (a median of 60
  seconds between payments, against 572 seconds for their innocent look-alikes), and 98% of their
  payments are the first between that sender and receiver in the week, against 54% of innocent ones.
- **Violet's detector clues (225):** each wallet's week (counts, counterparties, pass-through,
  typical amount), pair history, activity windows from 5 minutes to 24 hours, gaps to the nearest
  payments, and gap clusters at 2, 5, 10 and 15 minutes. *Why:* the same speed and shape signal,
  measured relative to each wallet's own behaviour.

## Model and settings

1. **Combined model:** one LightGBM (400 trees, learning rate 0.05, 31 leaves, at least 50 rows per
   leaf, 80% row and column sampling, seed 42) on all 412 clues, target `is_suspicious`. Its scores
   are then smoothed over wallet groups (25% own score, 75% the mean of its groups at 2, 5 and 10
   minutes), because labels are all-or-nothing per scheme.
2. **Structure model:** the same LightGBM on the same clues, asked a different question: is this
   payment part of a planted shape scheme, **laundering or innocent look-alike**? (`family` is its
   training target only.) Guilty and look-alike schemes are hard to tell apart, but both are easy to
   tell from relays and background (structure AP 85.8 against 5.2 chance), which otherwise crowd
   the slow medium cases out of the top.
3. **The gate:** score = log(combined probability) + log(structure probability), i.e. the product
   of "suspicious?" and "planted scheme?". A low structure score can veto a relay or background
   payment, which a weighted average of ranks can't. Multiplying ranks instead of probabilities
   gains nothing (62.7).
4. **Violet's detector v2** (scikit-learn HistGradientBoosting, 400 iterations, learning rate 0.05,
   31 leaves, seeds 0, 1, 2, plus a second stage on neighbouring payments' scores) is rank-mixed in
   at 10%: final = 0.9 × rank(gate) + 0.1 × rank(Violet). On the leaderboard that 10% was worth
   +0.2 (67.6 without it).
5. **Relay multiplier** (`src/relay_groups.py`), for the hard relays: a relay scheme runs three
   bursts across the week through the same one or two middle wallets, and those wallets carry
   about 110 other payments each, so fixed time windows can't group it. A third LightGBM (300
   trees, 63 leaves, seed 42) learns "are these two payments in the same scheme?" on candidate
   pairs (money into a wallet and out of it within 6 hours; repeats of the same sender-receiver
   pair); `scenario_id` is its training target only. Around each repeating pair it joins across
   days, each repeat's linked payments form one burst. Where the structure model gives under 0.3,
   schemes whose money runs backwards (paid out first, topped up after) are multiplied by e^-4,
   and forward ones by e^(4 × (steadiness rank − 0.5)), steadiness being how little the share of
   money passed on varies between bursts. The multiplier is applied to the gate before the Violet
   mix. Link model AP 0.80 out of fold; conversion clues left out, because they shift between weeks.

## Results

| Version | Honest test (folds) | Leaderboard |
|---|---|---|
| Organisers' starter notebook | 29.9 (30.9 ± 7.3) | 33.6 |
| Jack's multi-scale motif model | 54.9 (57.3 ± 9.9) | 61.4 |
| + typology detectors, group smoothing | 56.8 (57.1 ± 5.9) | 62.0 |
| Violet's detector v2 alone | 57.4 (57.1 ± 7.2) | 65.1 |
| Combined model (both clue sets) + Violet, 50/50 | 60.6 (60.7 ± 5.9) | 66.4 |
| + structure model, rank-mixed at 0.4 | 62.5 (62.7 ± 5.6) | 66.7 |
| Gate: combined × structure, + 10% Violet | 63.7 (64.3 ± 5.2) | 67.8 |
| **Gate × relay multiplier, + 10% Violet (final)** | **64.0 (64.5 ± 5.2, better in 5 of 5)** | **68.0** |

The relay multiplier changes nothing in the top 1% (all 401 payments are the same as in the 67.8
file), so its gain is further down the list. Top 1% of the list (401 payments): precision 96%, recall 31%. Caught there: easy 50%
of 800 (the most 401 places can hold), medium 10% of 225, observable hard 0% of 200, ambiguous 0%
of 25. On the training week, even a perfect "planted scheme or not" step plus perfect grouping would
reach only about 67.3: the rest is built to be indistinguishable (next sections).

## What didn't work

- **Guilt by association over the whole week** (neighbour scores as clues): 54.9 → 54.6.
- **Stronger weight on medium cases, wider group windows** (run 4): +1.2 honest, −1.2 leaderboard.
- **Week-percentile amounts, dropping conversion clues, self-training on the test week:** flat on
  the honest test; 61.8 and 66.5 on the leaderboard.
- **Longer motif windows, once-a-week-pair groups, Natasha's wallet-age clues, a model trained
  only on planted schemes:** within the noise.
- **Changes that won on the honest test but lost on the leaderboard,** after the gate: Jack's deck
  signals and slow-scheme groups inside the gate (+0.5 honest, 66.4); gap-cluster smoothing of the
  gate (+1.2 honest, 66.6); averaging 5 seeds (67.7) or adding XGBoost (67.5).
- **Violet's share, chosen on the leaderboard:** since the leaderboard scores the whole test week,
  we probed the share directly in the last hour: 0% 67.6, 10% 67.8, 20% 68.0, 30% 68.0, 50% 67.3;
  with the relay multiplier, 20% at most 68.0, 30% 67.5, 50% 67.9. The two +0.2 gains don't add up,
  so we kept the relay version at 10%: it ties the best score and is the only one the honest test
  also supports (30% was −0.7 there).
- **Learned links for medium schemes:** the relay link model used to smooth the structure or
  combined score over linked payments, 36 settings: all at or below the baseline (best −0.01).
  Medium schemes' linked payments look as innocent as they do.
- **External IBM helper model** (no signal, above) and **Isolation Forest anomaly scores** (alone
  AP 27.5, a real signal without labels, but only +0.3 in the gate, 2-3 of 5 folds).

## Limitations

- **Hard relays (200 of 1,250 positives):** each laundering relay hides among 99 same-shaped
  innocent ones. We found that every guilty relay moves money forward in time (receive, then pass on
  less), while exactly half of its innocent siblings run backwards (pay out, then get topped up),
  and it passes on a steadier share of the money in each burst. Both only show over the whole
  week-long relay. The learned grouping above recovers part of it (+0.25 honest, +0.2 leaderboard);
  with the true grouping the same multiplier would reach 66.4 on the honest test, so better
  grouping is the clearest next step.
- **Medium cases and fast look-alikes:** medium schemes are easy shapes run slowly, and about 12% of
  innocent look-alike schemes are as fast as laundering; on every measure we tried they look the
  same. These are most of our false alarms near the top.
- **Honest test against the test week:** late changes of up to +1.2 on the honest test lost more than
  a point on the leaderboard, so small honest-test gains here are not reliable.
- **Synthetic data:** a planted-scheme generator, not real crime. A high score here does not show
  the method would work on real financial data, and a flag is a reason to look, not an accusation.

## Third-party and pre-existing components

- **Libraries** (versions pinned in `requirements-lock.txt`): pandas (BSD-3), NumPy (BSD-3), pyarrow
  (Apache-2.0), scikit-learn (BSD-3), LightGBM (MIT), NetworkX (BSD-3); XGBoost (Apache-2.0) and
  kagglehub (Apache-2.0) only in dropped experiments.
- **Organisers:** the data, `payments.py` and `starter_notebook.ipynb` (its 12 rules are copied into
  `notebook_features.py` and `features_starter.py`).
- **Our own pre-hackathon material:** the notes in `prep/` and the three Claude Code skill files in
  `.claude/skills/` (decision log, submission check, XRPL data traps), written before the event.
- **External data:** IBM AMLworld HI-Small (Kaggle), not in the repository or the ZIP.

Our code is released under the MIT licence (`LICENSE`).

## AI disclosure

- **Claude Code (Claude Opus 5 and Claude Opus 5.5, Anthropic)** was used throughout: inspecting the
  data and labels, writing much of the pipeline (`src/`, `main.py`, `common.py`, `graph.py`, the clue
  files, the gate and relay options in `src/blend.py`,
  `src/relay_groups.py`), the analyses quoted in `DECISIONS.md`, finding and fixing the
  two hash-seed bugs, checking submission files, and drafting `DECISIONS.md` and this write-up.
  Violet's `detector/` was written by Claude (Claude Opus 5.5, Anthropic) in a chat session under
  her direction. Settings: default configuration, no custom temperature or sampling.
- **Prompts:** the conversations were interactive; the reusable prompts are the skill files in
  `.claude/skills/`, kept in the repository. Every number in this write-up was produced by code in
  the repository, so reproducing the results does not depend on the AI tools.

## How to reproduce

Data in `data/train/` and `data/test-data/` as downloaded; `pip install -r requirements.txt` (exact
versions in `requirements-lock.txt`). From the repository root:

```bash
python3 -m src.predict_union --probabilities --out predictions_union_prob.csv
python3 -m src.predict_union --target structure --probabilities --out predictions_structure_prob.csv
python3 detector/predict.py --train data/train --test data/test-data --out outputs/predictions_violet_v2_fixed.csv
python3 -m src.relay_groups
python3 -m src.blend --gate --weight 0.1 --run3 predictions_union_prob.csv --violet predictions_violet_v2_fixed.csv --structure predictions_structure_prob.csv --relay relay_multiplier.csv --out predictions_gate_relay_v10.csv
```

The result is `outputs/predictions_gate_relay_v10.csv`. Leaving out `python3 -m src.relay_groups`
and `--relay relay_multiplier.csv` recreates the 67.8 file, `outputs/predictions_gate_v10.csv`. All models are retrained on each run, so there are
no model files to download; seeds are fixed (LightGBM 42; scikit-learn 0, 1, 2). A fresh clone
with only the raw data took 10 minutes on an Apple silicon laptop (24 GB of memory; under 4 GB
used) and recreated the file byte for byte under different Python hash seeds. Our runs used Python
3.14.7 with the pinned library versions; the repository README and the build check use Python 3.12.
