# MoneyTrail: method write-up

Decode & Detect: The XRPL AI Hack, 24 October 2026. Team MoneyTrail: Alim Badmus (captain), Natasha, Jack, Violet.
Final entry: `outputs/predictions_blend_generator_G_violet.csv` (predictions from commit `5b83e62`, code at the commit this ZIP was built from), leaderboard **67** (best single run 67.1).
All numbers below come from `DECISIONS.md` and `SUBMISSIONS.md`. This is synthetic data; nothing here is evidence
about real accounts or the real XRP Ledger, and we say "flagged for review", never "criminal".

## 1. Problem and data

Rank 40,101 test transactions so the planted laundering ones come first; score = 100 x average precision.
Files: `transactions.parquet`, `transfers.parquet`, `market_rates.csv`, `labels.csv` (training only). We used
the organisers' `payments.py` for one USD-valued edge per payment (first leg for cost, delivered amount for
what arrived, hourly reference price, assets as issuer + symbol). Data checks found no self-payments,
destination tags, memos or partial payments; every wallet exists before the week; 8 liquidity providers are
the only hubs. No external data.

**What the labels are.** Every planted scenario is entirely suspicious or entirely honest (1,497 scenarios,
no exceptions), so the unit of prediction is the scheme, not the payment. Each of the nine shape families
(fan-in, fan-out, scatter-gather, bipartite, cycle, stack, chain, multi-stage, random) is about half
suspicious and half honest look-alikes; the separator inside a family is **speed**: suspicious schemes run
at a median 60-106 s between payments, honest ones of the same shape at 428-732 s. The 200 "observable hard"
cases are 12 relay schemes hidden one per cohort among 99 near-identical honest relay services.

## 2. Approach

**Honest test.** 5-fold cross-validation on the training week grouped by `scenario_id` (used only to split),
each background payment its own group; reported as the mean over folds and the score on all folds pooled,
usually over three fold seeds. We kept a change only if it gained more than 1.5 points in most folds, and
we logged every attempt, including the failures (`DECISIONS.md`).

**Clues (features), about 530 per payment.**
- Jack: wallet groups (payments sharing a non-hub wallet within 2, 5 and 10 minutes) described as motifs:
  size, span, gaps, rate, wallets, edges, density, fan shares, sources, sinks, middle wallets,
  pass-through, payer-receiver overlap, route reuse, depth, cycles; typology detectors for each of the
  deck's eight shapes (time-respecting loop search forwards and backwards, two-hop paths, bipartite
  density, layered depth, almost-whole chains); the deck's remaining "signals to compute" (quiet or new
  fan-in senders, recipients' forwarding speed, value out vs in, hold times, middles used once, routes,
  group closure); slow-scheme groups linked only through once-a-week sender-receiver pairs (96% of planted
  scheme payments are once-a-week pairs, against 28% of background).
- Violet: wallet history, pair history, activity windows from 5 minutes to 24 hours, gaps to the nearest
  payments, time-respecting motifs and gap clusters at 2-15 minutes, in a two-stage scikit-learn model whose
  second stage adds the scores of nearby payments.
- Natasha: wallet-age and last-receipt clues (tested; overlapped with existing clues, not kept).
- XRPL clues (conversion, bridging, legs, rate loss) were checked and dropped: the README says asset choices
  ignore the labels, and conversion share shifts from 41% (train) to 51% (test).

**Models.** LightGBM (fixed seeds). The final score is a 50/50 rank blend of (a) Violet's two-stage model
and (b) Jack's "two questions with a generator model": one LightGBM on all clues, smoothed over wallet
groups; plus the product of a scheme model ("is this payment part of a planted shape scheme, suspicious or
honest?", `family` as a training target only) and a template model fitted on the 272 true training
schemes' whole-scheme parameters, applied to groups rebuilt under the generator's grammar (a learned
payment linker's links, merged in confidence order, refusing merges above 16 payments or 10 hours).
Rationale: a medium (slow) scheme looks like an isolated background payment in a 2-10 minute window; it is
recognisable only as a whole scheme, and the generator's own parameters are the right description of one.

**Reproducibility.** Every step is seeded and sorted; two runs under different Python hash seeds and a
fresh clone with raw data only give byte-identical files (about 20-30 minutes on a laptop). One bug found
and fixed on the day: a clue file iterated over Python sets when breaking ties, so our 66.7 and 67.1 uploads
used one random draw; the fix is in the final code.

## 3. Results

| Step | Honest test | Leaderboard |
|---|---|---|
| Organisers' starter notebook | 29.9 | 33.6 |
| Multi-scale motif model | 54.9 | 61.4 |
| + typology detectors, group smoothing | 57.6 | 62.0 |
| Violet's two-stage detector alone | 57.4 | 65.1 |
| One model on both clue sets, blended with Violet | 60.6 | 66.4 |
| + deck signals | 60.4 | 66.7 |
| + two questions with slow-scheme groups | 62.3 | 67.1 |
| + generator model (final) | 63.1 | 67 |

Top 1% (402 payments): precision about 93-95%, recall about 30% (the top 1% can hold at most half of the
800 easy cases). Where the remaining points are: easy cases contribute about 52 of our points and are near
their limit (17 of 140 honest look-alike schemes are as fast as laundering); medium contributes about 6
(they sit at a median rank near 2,100 after our changes); hard relays about 2.4.

**Case studies (training week, flagged for review).** (1) A cycle: 6-8 hops returning 33 USD of 77 sent
within minutes, found by the backward-and-forward loop search (per-family AP 0.61 vs chance 0.004).
(2) A slow stack of 12 payments over 2 hours, invisible to 10-minute windows, held together by once-a-week
pairs and ranked by the template model. (3) A hard relay: the laundering relay wallet never pays out money
it hasn't received in the scheme (running balance never negative, 12 of 12 laundering relays vs 29% of
honest ones, AUC 0.80 with true grouping), while honest relays pay from their own float.

**What we deliberately don't flag.** The 8 liquidity providers (all background) are excluded from every
group and search; recurring sender-receiver pairs (66% of background) get low scores; exchange-like
gather-scatter hubs are described, not flagged.

## 4. What didn't work, limits, next steps

Didn't carry to the test week: medium up-weighting, week-percentile amounts, test-week self-training
(+0.7 honest, level on the leaderboard), a payment GNN (57 alone; +0.4 at a 25% share), a learned payment
linker alone, and the relay-balance finding as a per-payment clue (the signal fades from AUC 0.80 at scheme
level to 0.59 per payment because we cannot yet reconstruct each relay wallet's episodes). The 25
ambiguous cases are random by design. **Limits:** our honest test only sees the training week, and gains of
1-2 points there repeatedly shrank to 0-0.5 on the leaderboard. **Next:** full template matching for all
nine shapes with iterative grouping-and-scoring; episode reconstruction for relay wallets, then the
running-balance test; data augmentation (copies of schemes with new wallets and times, started at 15:40).

## 5. Pre-existing components, AI use, licence

Not written by us: the organisers' `payments.py` and the starter notebook's twelve rules; pandas, pyarrow,
NumPy, scikit-learn, LightGBM, NetworkX, PyTorch (BSD/Apache/MIT). AI: Claude Code (Claude Opus 5.5 and
Claude Fable 5.1, Anthropic; default settings) wrote most of the code and the logs interactively under the
team's direction, and Claude (Opus 5.5) wrote `detector/` in a chat session directed by Violet; reusable
prompts are the skill files in `.claude/skills/`; the pre-event notes in `prep/` are disclosed there. The
AI conversations cannot be replayed, but every line of code is committed and runs deterministically.
Licence: MIT. Repository: https://github.com/JAAAACCCCCCKKKK/moneytrail (public before the deadline).
