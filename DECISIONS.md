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
