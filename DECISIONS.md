# Decision log

**Team:** MoneyTrail. Alim Badmus (captain), Natasha, Jack, Violet.
**Honest test:** 5-fold grouped cross-validation on `train/`, grouped by `scenario_id` from
`train/labels.csv` (used only to split, never as a feature), with each background transaction
as its own group. Labels are pure per scenario, so any split that lets one scenario's members
fall on both sides inflates the score.
**Times** are London time (BST, UTC+1) and approximate; leaderboard runs are also given in UTC.

## Where we are now (update this after every upload)

- **Best leaderboard score: 62.0** (run 3, commit `6641b38`). Top 1%: precision 91%, recall 29%.
- **Main pipeline: Jack's `src/`** (`python3 -m src.train`). Honest test 54.9 (folds 57.3 ± 9.9).
  `main.py` is kept only for comparison.
- **New clues go into the `src/` pipeline.** After changing features, delete `data/cache/`
  or the old cached features are reused.
- **Current best pipeline:** Jack's `python3 -m src.predict_shapes_v2` (typology detectors,
  group smoothing, rank blend), honest test 57.6 ± 6.1 on `main.py`'s folds.
- **Priority:** easy and medium cases first (about 84 of the 100 points live there; half the
  easy cases are still outside our top 1%). Hard relays from 14:30. Whole-week guilt by
  association was tried and dropped (11:45); the next hard-case idea is how a payment relates
  to its wallet's own earlier history.
- **Rules update (11:50): external datasets are allowed,** as long as we disclose each one: why
  we used it and whether it helped. Log every external dataset in the "External data" section
  below, including ones that didn't work.
- **Who's on what:** Jack, motif and shape clues; Natasha, longer-history wallet clues and
  studying the cases we still miss; Violet, XRPL asset and conversion clues, data checks,
  then the method write-up from 15:00; Alim, honest test, merging, uploads, this log.

## Decisions

### 10:00 — Dataset inspected, label structure mapped (Jack)
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

### 10:30 — Baseline: the organisers' starter notebook (run 1)
- **Change:** ran `starter_notebook.ipynb` unchanged (logistic regression on 12 simple rules:
  payment size, how busy sender and receiver are, passing money on within an hour, 10-minute
  bursts, similar amounts forwarded). Its twelve rules are copied verbatim into
  `notebook_features.py` and `features_starter.py`.
- **Why:** a valid first upload and a number to beat.
- **Honest test:** the notebook's own random 20% split reports 24.6. The same features and model
  under our grouped 5-fold score **29.9, per fold 30.9 ± 7.3** (a stratified, still leaky
  5-fold scores 30.8 ± 2.1). The single random split is noisy as well as leaky, with only about
  250 positives held back, so every decision from here uses the grouped 5-fold mean and spread.
- **Recall inside the top 1% of grouped scores:** easy 0.270, medium 0.031, hard 0.013.
- **Reference ceiling:** ranking every easy and medium case perfectly while missing every hard
  one scores 84.0, so about 84 of the 100 points sit in easy plus medium.
- **Leaderboard:** run 1, 09:34 UTC, commit `4d8970d`, `moneytrail-run1-code.zip`. Score
  **33.6**; top 1%: precision 62%, recall 20%; ROC-AUC 0.82. Caught in the top 1%: easy 30% of
  800, medium 10% of 225, observable hard 0% of 200, ambiguous 0% of 25. The test week has 1,250
  suspicious transactions (about 3.1%), the same difficulty mix as training.
- **Calibration:** grouped CV 29.9 against leaderboard 33.6, so the grouped CV is a slightly
  conservative but usable stand-in for the leaderboard.
- **Decision:** kept as the baseline. Strongest rules: 10-minute bursts from the sender (+0.53)
  and into the receiver (+0.49); busy receivers count against suspicion (−0.90), the hub effect.

### 10:35 — Motif grouping plus shape features, first probe (Jack)
- **Change:** exploratory only. Grouped payments into candidate motifs with a union-find that
  links two payments sharing a wallet within T seconds, then computed scenario-level shape
  features (size, span, median and minimum gap, rate, wallet and edge counts, density, maximum
  in- and out-degree, pass-through ratio, simple-cycle count, longest path, source and sink
  counts, asset count, conversion and bridge share, USD total, median and spread, route reuse)
  and scored them with LightGBM under the same grouped 5-fold.
- **Why:** labels are pure per scenario, so features describing the whole motif should beat
  features describing one payment. The strongest single raw per-transaction feature is a count
  of payments involving the same wallet within 120s, at AP 0.125 against 0.031 prevalence.
- **Honest test:** 29.9 (baseline) → **41.1** at T=300s. T=120s scores 36.7 and T=900s 29.4.
  Without the graph features (no pass-through, cycles or longest path) it scores 37.2 at
  T=300s, so the graph-shape features are worth about 4 points.
- **Limitation found:** a true scenario is split across about 13 motifs and the median motif
  holds one payment. Correction (10:40): that figure was dominated by the 1,200 relay
  scenarios, which span about 133 hours by design. On the 297 shape scenarios alone, the
  largest motif covers a median 25% of its scenario at 120s, 50% at 300s and 100% at 600s,
  while at 600s relay motifs start merging up to 361 scenarios. So the fix is several scales
  at once, not a new grouping method. Hard recall in the top 1%: 0.000.
- **Decision:** superseded by the next entry.

### 10:45 — Multi-scale motif shape model, v1 (Jack) (run 2)
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
- **Honest test:** 29.9 (baseline) → **52.9** raw, **55.5** with three-scale smoothing, per
  fold 57.6 ± 10.0. Other smoothings: 120s only 54.0, 300s only 54.4, 600s only 54.5. Rerun on
  Alim's laptop: 54.9, folds 57.3 ± 9.9, so it reproduces.
- **Leakage check:** wallets repeat across scenario folds, so wallet-week features could act as
  fingerprints. Dropping the 13 wallet-week features costs 52.9 → 48.8 raw under the scenario
  split. Under a wallet-disjoint split, where fingerprinting is impossible, the full set still
  wins, 63.8 against 62.1 raw on 8,175 rows, so those features carry real behavioural signal.
- **Test predictions:** pass the checker (40,101 rows, all distinct). Rank correlation with the
  baseline's test scores is 0.50; their top-1% sets share 127 of 401 transactions.
- **Leaderboard:** run 2, 09:59 UTC, commit `bbf8675`. Score **61.4** (run 1 was 33.6). Top 1%:
  precision 93%, recall 30%. Caught in the top 1%: easy 50% of 800, medium 0% of 225,
  observable hard 0%, ambiguous 0%. The top 1% is only 401 transactions against 1,250
  suspicious, so recall there can't exceed about 32%; at 93% precision it is nearly full of
  easy cases, which is why medium reads 0% even though the overall score rose.
- **Decision:** kept. This is the main pipeline. Hard recall in the top 1% is still 0.00, as
  expected.

### 10:50 — `main.py` and LightGBM on the starter clues (Alim)
- **Change:** built `main.py` (loads payments, adds clue files, runs the honest test, trains,
  writes predictions) and `common.py`.
- **Honest test:** 5-fold StratifiedGroupKFold, grouped by `scenario_id` (background grouped by
  sender). Starter logistic regression 30.4 ± 5.9; LightGBM 30.2 ± 6.2 on the same clues.
- **Leaderboard:** not uploaded (no improvement).
- **Decision:** a better model alone adds nothing on 12 simple clues; the gain must come from
  clues. Superseded by Jack's `src/` pipeline as the main pipeline; `main.py` kept for
  comparison only.

### 11:00 — Label study: easy cases first, hard relays look ordinary payment by payment (Alim)
- **Change:** analysis only, no model change.
- **Findings:** 98% of easy and medium suspicious payments are the first payment between that
  sender and receiver in the week, against 54% of innocent payments (innocent activity has
  recurring relationships). Easy cases forward fast (median receiver forward time about 20
  minutes against the 60-minute cap) and come in short bursts. The hard cases are 102
  suspicious `aggregate_relay` among 10,098 innocent and 98 suspicious `distributed_relay` among
  9,702 innocent. Every per-payment clue tested separates them weakly (best about 0.18 on a
  0-to-1 scale): wallet age, activity, counterparties, amounts, forwarding within 1, 6, 24 or
  72 hours, repeat pairings. Every suspicious relay has a `cohort_id`.
- **Honest test:** not measured (analysis only).
- **Decision:** priority order for the team: easy and medium first; hard cases from 14:30 with
  group-level and guilt-by-association clues.

### 11:10 — Team aligned on one pipeline (Alim)
- **Change:** no model change. Agreed Jack's `src/` pipeline is the one everyone builds on.
  Natasha and Violet add their clues as new feature files in `src/`, joined in by
  `build_features` (Jack wires in the hook). Jack's pipeline already has pair frequency,
  forwarding within an hour, seconds since funded and conversion spread from leg 0, so new
  clues should cover what's missing: longer histories, changing counterparties, wallet
  openers, asset hopping and issuer mixing.
- **Decision:** one pipeline, one honest test, uploads only from `src.train`.

### ~11:30 — graph.py and features_shapes.py: one detector per laundering typology (Jack)
- **Change:** `graph.py` builds each week's network, the hub list, time-windowed wallet groups,
  wallet histories, and `smooth_by_group` for `main.py`. `features_shapes.py` follows the clue-file
  format and adds 34 typology detectors on top of run 2's 141 behaviour and group clues (175
  `shape_*` columns): fan-in and fan-out at 10 min, 1 h and 6 h with amount spread;
  gather-scatter; scatter-gather two-hop paths; short loops; a time-respecting forward and
  backward search for long loops and layered depth; almost-whole chains; bipartite density; hub
  flags; first-time pair and pair frequency.
- **Hubs:** exactly 8 per week, the liquidity providers inside conversion legs (235-280
  counterparties, next wallet about 120). All their training payments are background. A
  "twice the median" peer threshold was dropped: it flagged 2 extra wallets in train and none in
  test, so the clue would differ by week.
- **Does each detector find its typology?** Average precision at picking out the suspicious
  payments of that family, against a chance level of 0.2-0.4%: on a loop 0.282 (cycle),
  fan-in at 10 min 0.222, two-hop paths 0.196 (fan-out-in), bipartite density 0.154, chain depth
  0.141, layered depth 0.111 (multi-stage), fan-out at 10 min 0.109. Layered stacks and random
  networks are not found by any single detector (0.005-0.013).
- **What didn't work first:** 2-3 hop loop checks found no cycles (0.002) because planted cycles
  run 6-8 hops and lose value on the way (77 USD out, 33 back). A forward-only search found loops
  only from the payment that starts one (0.105); adding a backward search and testing whether
  the two reach sets meet fixed it (0.282).
- **Honest test** (`main.py` folds, so not comparable with the 54.9 of run 2's own script):
  run 2's clues 53.8 raw, 56.5 smoothed; typology detectors alone 51.3 raw, 53.6 smoothed;
  everything 55.1 raw, 56.9 smoothed; rank blend 0.75 everything + 0.25 typology-only, smoothed,
  **57.6 +/- 6.1** (all folds together 57.1). Medium caught in the top 5%: 0.16 for run 2's clues,
  0.24 for the blend. The gain is about one point, inside the fold spread.
- **Test predictions:** `python -m src.predict_shapes_v2` writes
  `outputs/predictions_shapes_v2.csv`; it passes the checker (40,101 rows, all distinct). Rank
  correlation with run 2's test scores 0.865; top-401 overlap 340.
- **For Alim:** `main.py` has no group smoothing yet, and it was worth about 2.7 points here;
  `graph.smooth_by_group(scores, payments, week)` is ready to call on the test scores.
- **Leaderboard:** run 3, 10:39 UTC, commit `6641b38` (rerun on Alim's laptop gave a byte-identical
  file). Score **62.0** (run 2 was 61.4). Top 1%: precision 91%, recall 29%. Caught in the top 1%:
  easy 40%, medium 10%, observable hard 0%, ambiguous 0%.
- **Decision:** kept, now our best. The gain is small (0.6 points), matching the honest test's
  one point; medium moved into the top 1% (0% → 10%).

### ~11:45 — Guilt by association, two rounds, whole-week neighbours (Alim)
- **Change:** `src/stage2.py`. Round 1 is run 2's model; its scores become 12 neighbour clues per payment (mean, max and count of scores for money into the sender, the sender's other outgoing, the receiver's other incoming, and money out of the receiver, leaving the payment itself out). Round 2 retrains with them. Nested honest test on `src.train`'s folds: round 1 is retrained inside every fold, with an inner out-of-fold split, so no clue sees a held-out answer.
- **Why:** labels "follow the money", and the hard relays look ordinary payment by payment.
- **Honest test:** round 1 reproduces run 2 exactly (54.9, folds 57.3 ± 9.9). Round 2: **54.6**, folds 56.7 ± 10.6, better in only 2 of 5 folds. Medium in the top 1% 0.05 → 0.07 (noise-sized); hard still 0.00.
- **Leaderboard:** not uploaded.
- **Decision:** dropped (fails our rule: more than a point and at least 4 of 5 folds). Likely reason: the hard-relay wallets are busy innocent-looking wallets (about 60 payments a week), so whole-week neighbour averages are dominated by innocent traffic. Code kept in `src/stage2.py` for the record.

### ~11:50 — Rules update: external datasets allowed with disclosure (Alim)
- **Change:** no model change. The organisers confirmed on the day that external datasets may be
  used, provided we disclose each one: why we used it and whether it worked. This replaces
  the "organiser data only" rule from the training slides and the terms.
- **Still to respect:** the README still forbids external labels for this data, organiser
  files, generator seeds and test answers. Any dataset must be lawful to use, and raw datasets
  stay out of the repo and the code ZIP (we link to the source instead).
- **Decision:** every external dataset gets a row in the "External data" section below, kept
  up to date whether or not it helped, so the method write-up can report it honestly.

### ~11:50 — Where run 3 loses points: medium schemes are slow, and some honest ones are fast (Jack)
- **Change:** analysis, then one small change (run 4 candidate below).
- **Where the points are** (run 3 configuration, `main.py` folds, honest test 56.8): moving every
  easy case to the top would give 71.5; moving every medium case to the top would give 73.9. So
  the 225 medium cases are worth more than the 800 easy ones. Easy cases already sit in our top
  3% (median position top 1-2%); medium cases sit at a median of the top 10-22%.
- **Why medium ranks low:** medium schemes are the same shapes as easy ones run in slow motion.
  Scheme medians: easy, a gap of 60s between payments over 5 minutes; honest look-alikes, 572s
  over an hour; medium, **1,066s over 110 minutes**, often with 2 currencies against 1. They are
  slower than honest activity, so every tempo clue pushes them down, and our widest wallet-group
  window (10 minutes) covers only 25% of a medium scheme.
- **Tried for medium, no signal** (each scored alone against all honest payments, chance 0.006;
  against honest look-alike shapes, chance 0.17, every one stayed between 0.17 and 0.23): all 187
  existing clues (best 0.032); neighbourhood closure at 30, 60 and 120 minutes (share of traffic
  around the sender and receiver that stays in their small circle, circle size, currencies, pace);
  "out of character" amounts against the wallet's own week (FATF red flag 5) and quiet wallets;
  slow peel chains with currency hopping (XRPL deck); structuring under round limits (the
  amount distributions of medium and honest shapes match). Wider union-find windows recover
  medium schemes (100% at 30 minutes) but chain 57% of all payments into groups over 100.
- **Why some honest schemes are false alarms:** the top 3% holds 726 suspicious and 478 honest
  payments. All 17 honest shape schemes that move as fast as laundering (12% of honest
  look-alikes, median gap under 2 minutes) land in the top 3%; by tempo they cannot be told
  apart. The other 225 false alarms are background and relay payments.
- **Smoothing variants, no gain:** group weight 0.1 or 0.4 instead of 0.25 (within 0.1); group
  maximum (-2.7) or top-3 mean (-3.5) instead of the mean.
- **Run 4 candidate:** run 3 plus a 20-minute group window (covers 67% of a medium scheme before
  groups merge) and 6x training weight on medium suspicious rows (a training weight from the
  `difficulty` label, never a clue). Honest test against run 3, three fold seeds (fold mean, all
  folds together): seed 42 57.7 vs 56.5 (57.5 vs 55.8); seed 7 58.3 vs 58.1 (58.2 vs 57.0); seed
  2026 57.8 vs 57.5 (56.9 vs 56.5). Medium cases in the top 5%: 0.23 vs 0.20, 0.20 vs 0.16,
  0.20 vs 0.19. Small, but better on every seed.
- **Test predictions:** `python -m src.predict_shapes_v2 --windows 120 300 600 1200
  --medium-weight 6 --out predictions_shapes_v3.csv`; passes the checker; rank correlation with
  run 3 0.842, top-401 overlap 358. Run 3 itself still reproduces byte for byte with the defaults.
- **Leaderboard:** not uploaded.
- **Decision:** worth one upload to confirm; keep run 3 as the reference if it doesn't beat 62.0.

### ~12:10 — Hard relays: history and whole-scheme tests, nothing separates them (Alim)
- **Change:** analysis only. Tested per-payment history ideas on the 200 suspicious relay payments against their ~20,000 innocent look-alikes: amount equals the sum of the sender's last 2-12 incoming (same asset); amount is part of a split of an earlier receipt; amount equals an earlier receipt forwarded intact after an hour; days since the sender's previous payment. All at chance (AP 0.010 against 0.010 chance).
- **Structure found:** the hard cases are 12 schemes (6 aggregate, 6 distributed relays), about 14-16 payments each, spanning the whole week (~133 h) across 10-12 wallets. Each sits in a cohort of 100 same-shaped schemes, 1 laundering and 99 innocent. Comparing each laundering scheme with its 99 siblings on 18 scheme-level properties (wallets, sources, sinks, USD total and spread, gaps, timing, assets, conversions, round amounts, night and weekend share, repeat pairs), every average percentile sits between 0.30 and 0.74, and at most 2 of the 12 are extreme on any one property. This matches Jack's 10:00 finding.
- **Decision:** stop spending time on the hard relays. They look designed to be indistinguishable from what's observable (as the README warns, some hard cases have labels independent of observable features). The remaining points are in ranking easy and medium cases above innocent look-alikes, so Natasha moves to studying the easy and medium cases we still rank low.

### ~12:25 — Longer motif windows (30 min, 1 h, 2 h) on the `src.train` model (Alim)
- **Change:** experiment only (script not in the repo). Added Jack's motif features at 1,800s, 3,600s and 7,200s on top of run 2's 135 clues; also a variant where only quiet wallets (at most 15 payments in the week) can link payments.
- **Why:** medium schemes are slow (median gap 1,066s over 110 minutes), so the 2-10 minute motifs never see their shape.
- **Grouping:** plain long windows keep every medium scheme together but merge into huge groups (largest 9,184 payments at 30 minutes, 39,465 at 2 hours). The quiet-wallet variant links almost nothing (medium scheme wallets are busier than 15 payments a week).
- **Honest test** (`src.train` folds, same as run 2's 54.9): plain long windows **56.6**, folds 58.6 ± 8.9, better in 4 of 5 folds; with long-window smoothing too, 56.7 (folds 57.8 ± 7.8). Quiet-wallet variant 55.2. Medium caught in the top 1,250: 10% → 11%, so the gain is general, not the medium fix.
- **Leaderboard:** not uploaded.
- **Decision:** passes our rule (more than a point, 4 of 5 folds) on its own. Worth Jack trying the plain 30-minute, 1-hour and 2-hour motif features on top of his run 4 candidate under his folds; it won't solve the medium cases by itself.

## External data

Every dataset we use that didn't come from the organisers. Required by the organisers' rules
update at 11:50: say why we used it and whether it worked.

| Dataset | Source | Why we used it | How we used it | Did it help? (honest test before → after) | Kept? |
|---|---|---|---|---|---|
| None yet | | | | | |

Not counted as external data used in our entry: the public datasets we practised on before
the event (IBM AML transactions on Kaggle, a slice of the AWS public XRPL data). They were for
learning only and are not used by any of our code; see `prep/RESEARCH.md`.

### ~11:55 — Natasha's wallet clues, batch 1 (Natasha, tested by Alim)
- **Change:** `src/features_wallet.py`: sender and receiver age in the week, minutes since the sender last received money (strictly earlier), and this payment ÷ that last receipt. Tested through a temporary adapter because the file's column names (`SENDER`, `RECEIVER`, `DATETIME`, `USD_VALUE`) don't match the payments table (`FROM_ADDRESS`, `TO_ADDRESS`, `TIMESTAMP` in seconds, `VALUE`).
- **Honest test** (`main.py` folds, raw LightGBM, no smoothing): starter + shapes 54.8 ± 4.1 → with wallet clues 54.9 ± 4.6, better in 3 of 5 folds.
- **Decision:** not added. The pipeline already has the same ideas (`src_seconds_since_funded`, wallet week profiles, first-time pair). Natasha moves to clues the pipeline doesn't have: how a payment relates to its wallet's own earlier history, for the hard relays.

### ~11:50 — XRPL check: conversion clues removed from the shape model (Violet)
- **Change:** `experiments/xrpl_conversion_check.py` runs the `src.train` honest test with and
  without the 13 conversion and asset clues (`is_conversion`, `is_bridged`, `conversion_spread`,
  `n_legs`, and `n_assets`, `conversion_share`, `bridged_share` at 120s, 300s and 600s).
- **Why:** the README says asset and conversion choices ignore the labels, conversion share is
  about equal for suspicious and innocent payments (43% vs 45%, median loss vs market 0.37% for
  both), and conversion share shifts from 41% in train to 51% in test. Clues with no signal
  that shift between weeks can only hurt on the test week.
- **Honest test:** with 54.9 (folds 57.3 +/- 9.9) → without **55.6** (folds 57.5 +/- 10.2).
  Raw 52.5 → 53.4. All six smoothing variants improve. Top-1% recall: medium 0.05 → 0.07,
  easy 0.41 → 0.39. The gain is inside the fold spread, but the direction is consistent.
- **Leaderboard:** not uploaded yet.
- **Decision:** drop the conversion clues from the shape model: no loss, and less exposure to the
  train-to-test conversion shift. Other XRPL clues checked and found not applicable here: the data
  has no self-payments, destination tags, memos or partial payments, and no wallet is opened
  during the week (holdings exist before the window), so "who opened this wallet" can't be
  computed.

## AI use

- **Claude Code (Claude Opus 5 and Claude Opus 5.5, Anthropic).** Used to inspect the dataset,
  measure the label structure reported above, run the organisers' starter notebook, check
  submission files with our `submission-check` skill, write `src/honest_eval.py`,
  `notebook_features.py`, `src/data.py`, `src/motifs.py`, `src/features.py`, `src/train.py`,
  `main.py`, `common.py`, `features_starter.py`, `graph.py`, `features_shapes.py`,
  `src/predict_shapes_v2.py`, `tools/track_outputs.py`, `tools/check_zip_limits.py` and the
  release workflow `.github/workflows/release.yml`, and draft this log. Settings: default Claude
  Code configuration, no custom temperature or sampling parameters. Prompts: the conversation
  was interactive; the reusable prompts are the three skill files in `.claude/skills/`, which
  are kept in this repo.
- **Before the hackathon:** Claude Opus 5.5 via Claude Code wrote the three skill files in
  `.claude/skills/` and the background notes in `prep/`. Disclosed in `prep/SKILLS.md` and
  `prep/RESEARCH.md`.
- **Pre-existing components not written by us:** the organisers' `payments.py`,
  `starter_notebook.ipynb` and its twelve rules (copied into `notebook_features.py` and
  `features_starter.py`), plus pandas, pyarrow, NumPy, scikit-learn, LightGBM and NetworkX.
