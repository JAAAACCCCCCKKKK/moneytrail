# MoneyTrail: pre-hackathon research notes

> **Disclosure.** Written before the hackathon (23 October 2026) as background
> reading for the team. Research notes only: no code, no data, no model. Prepared
> with help from Claude (Claude Opus 5.5, Anthropic, via Claude Code), using the
> public sources listed at the end. All the competition work is done on the day.

## Contents

1. [The challenge in brief](#1-the-challenge-in-brief)
2. [Money laundering basics](#2-money-laundering-basics)
3. [The eight laundering shapes](#3-the-eight-laundering-shapes)
4. [Red flags you can measure](#4-red-flags-you-can-measure)
5. [What IBM's AMLworld paper found](#5-what-ibms-amlworld-paper-found)
6. [XRP Ledger basics](#6-xrp-ledger-basics)
7. [Data traps](#7-data-traps)
8. [Scoring and testing honestly](#8-scoring-and-testing-honestly)
9. [What we learned from practice runs](#9-what-we-learned-from-practice-runs)
10. [Ideas to test on the day](#10-ideas-to-test-on-the-day)
11. [Sources](#11-sources)

---

## 1. The challenge in brief

Our own notes from the event page, the terms and the pre-hackathon training.

- **Task:** give every test transaction a suspicion score from 0 to 1, so the
  suspicious ones rank at the top. The data is a fictional ledger in the real XRPL
  (AWS/SonarX) format: about 40,000 labelled training transactions and 40,000 test
  transactions, each covering 7 days.
- **Score:** 100 × average precision on all test transactions. Only the order
  counts. Tied scores are ranked as one block, so use continuous scores. Top-1%
  precision and recall and ROC-AUC are shown but don't affect the ranking.
- **Files:** `transactions.parquet`, `transfers.parquet` (1 to 3 money legs per
  transaction), `blocks.parquet`, `market_rates.csv` (hourly USD price per asset),
  `labels.csv` (training only). A README, `payments.py` and `baseline.py` are supplied.
- **Difficulty:** suspicious cases are about 64% easy, 18% medium, 18% hard. Hard
  cases span many payments and changing counterparties. The test week has
  variations that aren't in training, and some cases are deliberately ambiguous.
- **Submission:** a CSV `TRANSACTION_HASH,score` with every test transaction once
  (max 12 MiB), plus a code ZIP (max 8 MiB, no installed packages, no secrets)
  with the run command, seeds, dependencies, team list, commit and a method
  write-up (under 10 slides or 4 pages). 50 uploads per team per day; the
  leaderboard shows the best run. Deadline 16:30.
- **Rules that matter:** organiser data only; any tools; AI allowed if disclosed
  (model, provider, prompts, settings); the judged work is built on the day;
  results must be reproducible; open-source licence by the deadline.

## 2. Money laundering basics

- **Fraud** has a victim and a loss ("should this payment happen at all?").
  **Laundering** moves criminal money until it looks clean ("where did this money
  really come from?"). This challenge is about laundering.
- **Three stages:** *placement* (dirty money gets in), *layering* (it's moved
  around to break the trail), *integration* (it comes back out looking clean). On
  a ledger you mostly see layering: quick hops, splits, merges and conversions.
- **Money mule:** someone whose account is used to move the money, often a fresh
  wallet used once or twice.
- **What an AML team does:** know the customer, monitor payments, investigate
  alerts, report, keep records. Analysts can only review so many alerts a day, so
  a good model puts the real cases at the top of their queue. That is exactly what
  average precision rewards.

## 3. The eight laundering shapes

The pattern families come from IBM's AMLSim and AMLworld datasets. Every one of
them also happens in normal business, so context separates them.

| Shape | What happens | Signals to compute | Harmless look-alike |
|---|---|---|---|
| **Fan-in** | Many wallets (often mules) pay one collector, which moves the total on | Distinct senders in a short window; senders that are new or do nothing else; time from last deposit to exit; exit amount close to the sum in | Shop, fundraiser, exchange deposit address |
| **Fan-out** | One wallet splits a large sum into many similar payments | Distinct recipients in a short window; how similar the amounts are; recipients that pass it on fast; a source funded just before | Payroll, refunds, giveaways, exchange withdrawals |
| **Scatter-gather** | Money splits across middlemen, then meets again at one destination | Several 2-hop paths between the same pair; value out ≈ value in; middle wallets with no other activity; similar short hold times | A company moving money between its own wallets |
| **Gather-scatter** | A hub collects from many and pays out to many others | Many senders and receivers; little balance kept; short time from in to out; senders and receivers that never overlap | Exchange hot wallet, payment processor, payroll provider |
| **Cycle** | Money goes round several wallets and comes home, sometimes changing currency on the way | A path back to the start within a time window; number of hops and time taken; value returned vs sent; assets changing along the way | Arbitrage, market making, refunds, moves between a firm's own wallets |
| **Layered stack** | Rows of middlemen between sources and destinations | Depth of the longest chain; middle wallets used once; value kept almost whole at each layer; similar delays | Payment processors, groups of companies paying each other |
| **Bipartite** | One group of wallets pays another, many to many, keeping each pair small | Dense blocks of payments between two sets; few contacts outside the block; many pairs active at once; similar amounts | Marketplaces, a company paying suppliers |
| **Irregular network** | Many partial routes with splits, merges and returns, and no clean template | Size and density of the connected group; number of distinct routes between two wallets; value into vs out of the group; how tightly packed in time | Ordinary peer-to-peer life |

**From the training:** gather-scatter matters most because it looks the most
legitimate; cycles are often currency round trips; irregular networks are the
hardest and are where machine learning beats fixed rules.

## 4. Red flags you can measure

From FATF's virtual-asset red flags (2020) and the training's "behaviour that
raises a flag". Many FATF flags (IP addresses, ID checks, countries) aren't in
ledger data; these are the ones you can compute from transactions.

| Red flag | How it might show up in the data |
|---|---|
| Rapid pass-through | Money leaves soon after it arrives and little is kept: time from receiving to sending, money out ÷ money in |
| Hops with no purpose | Long chains through wallets that do nothing else |
| Splitting / structuring | One amount broken into many smaller ones, or amounts just under round limits |
| Currency hopping | Converting through several assets; conversions straight after receiving |
| Round-tripping | Funds return to where they started, maybe in another asset |
| Converting at a loss | A much worse exchange rate than the hourly market price, which honest users avoid |
| New or dormant accounts | Large activity soon after a wallet is opened, or after a long silence |
| Accumulation | Many small payments from unrelated wallets, then sent on in one go |
| Patterns over time | A flow that only makes sense across the whole week |

FATF's golden rule: one red flag is not proof. Suspicion comes from several flags
together with no sensible business reason.

## 5. What IBM's AMLworld paper found

Altman et al. (NeurIPS 2023) built a simulated world of people, companies and
banks with criminals laundering money, and released the data on Kaggle.

- **Most laundering isn't one of the planted shapes.** In their largest
  low-laundering dataset, only about 1 in 5 laundering transactions came from the
  eight patterns. The rest were disguised as normal business (payroll, supplies).
- **The label follows the money.** If dirty money is passed on, the later payments
  are labelled too. "Did this wallet recently receive suspicious money?" is a
  strong signal.
- **Trees with graph features matched or beat graph neural networks.** XGBoost or
  LightGBM with graph features reached about 63% F1 on their small dataset, against
  about 57% for their best GNN.
- **Seeing the whole network helps a lot.** Banks that shared the full graph caught
  far more than banks seeing only their own customers.
- **How they tested:** a time-based 60/20/20 split, graph features built from
  earlier transactions only, and no raw account IDs as features.

## 6. XRP Ledger basics

- **Ledger and validators:** independent servers agree on each new ledger page
  every few seconds; 80% agreement makes it final. Every transaction burns a tiny
  fee (10 to 15 drops).
- **Accounts:** an address starting with `r`, no name. A new account exists only
  once someone funds it with XRP (the reserve is currently 1 XRP), so every wallet
  has an opener. One wallet opening many fresh wallets can be a boss setting up
  mules.
- **Destination tags:** an exchange uses one address for many customers and tells
  them apart by tag, like flat numbers in one building.
- **Tokens:** anyone can issue a token (an IOU). You need a trust line to hold one.
  USD from two issuers is two different assets. Long currency codes are stored as
  40 hex characters.
- **The DEX:** a built-in exchange with an order book (bids and asks). Pathfinding
  finds the cheapest route between currencies, sometimes via a stopover
  (auto-bridging uses XRP itself). A conversion is one transaction with 2 or 3
  transfer legs through liquidity providers.
- **Partial payments:** a payment can ask for a huge `Amount` and deliver far less.
  Only the delivered amount counts.
- **Other ways to move money:** escrow (held until a date or condition), checks
  (the receiver chooses when to cash), payment channels (many small payments
  settled once), multi-signing (several keys needed).

## 7. Data traps

From the training, plus what we saw in a real slice of XRPL data (AWS public
blockchain data, 12 October 2026).

1. **Score transactions, not transfers.** Join on `TRANSACTION_HASH`; one row per
   transaction in the CSV. Real transactions had from 1 to 26 legs.
2. **XRP comes in drops.** For XRP, `SOURCE_VALUE` is in drops (1 XRP = 1,000,000);
   `VALUE` is in whole units.
3. **An asset is issuer + symbol.** Group by `TOKEN_ADDRESS` and `SYMBOL`. In 12
   real minutes, USD and USDC each had two issuers. Decode hex codes before reading
   them.
4. **SendMax is a ceiling, not the cost.** The first transfer leg shows what was
   really paid. In the real slice, 31% of payments delivered less than their
   `Amount`; one asked for a 100-digit number and delivered 0.000001.
5. **Compare value in USD** with `market_rates.csv`, using the hour of the payment.
6. **Some fields are JSON text:** `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT`
   (JSON for tokens, a plain drops string for XRP).

Also seen in real data:

- About 36% of payments were wallets paying themselves: currency swaps, not loops
  to worry about.
- The busiest receivers had 100+ senders in 12 minutes and used no destination
  tags, so not every hub is a tagged exchange.
- Some real transfer legs have `TRANSFER_INDEX` -999 and look like summary rows.
  Check the README before counting them, or value is double-counted.

## 8. Scoring and testing honestly

- **Average precision** walks down your ranked list and, each time it reaches a
  suspicious transaction, asks what share of the list so far is suspicious. It
  averages those. Random guessing scores about the share of suspicious
  transactions; a perfect ranking scores 1. One normal transaction ranked very high
  hurts a lot.
- **Test like the real test.** Use 5-fold cross-validation, grouped by connected
  wallets (`GroupKFold`), so one laundering scenario never sits on both sides of a
  split. A single split on 40,000 rows is noisy; look at the mean and the spread.
- **Adversarial validation:** train a quick model to tell training rows from test
  rows. If it can, use the most test-like training rows as the check set.
- **Avoid overfitting:** behaviour features (ratios, timings, counts) rather than
  wallet addresses or exact amounts; modest models (fewer leaves, more rows per
  leaf, early stopping); little hyperparameter tuning; a blend of a few seeds;
  keep a change only if it helps in most folds.
- **Warning signs:** a near-perfect score from one feature usually means a leak.
  Shuffling the labels should drop the score to random. Guilt-by-association
  features must use out-of-fold scores.
- **The leaderboard scores the whole test set** and keeps the best run, so it's
  the truest signal. Use cross-validation for everyday decisions and the
  leaderboard to confirm big ones. Commit before every upload so the best run can
  be reproduced.

## 9. What we learned from practice runs

All on synthetic practice data, so the numbers only show the process.

- **A simple fake dataset (97,000 payments):** average precision went from 0.13
  with raw columns to 0.98 with account, shape, XRPL and neighbour features. Far
  too easy to be realistic, but the order of steps worked.
- **IBM's HI-Small dataset (5 million transactions):** raw columns 0.06 → account
  behaviour 0.30 → shapes 0.51 → payment details 0.57 → guilt by association
  0.61 → blend 0.63 (best F1 0.67, in line with the paper). Payment format alone
  was a huge clue there, found just by looking at the data first.
- **A practice competition in the XRPL format (two separate weeks):** a simple
  baseline scored 89 on its own holdout but 77 on the different test week ("test
  is not train"). It caught 91% of easy cases and only 16% of hard ones. Its
  biggest mistake was flagging ordinary peer-to-peer payments.
- **Strongest clues there:** brand-new, quiet receiving wallets; money moving on
  within about an hour; one wallet opening many others; conversions at a worse
  rate. Raw fan-in counts mostly found busy hubs (exchanges, shops) and needed
  context to be useful.

## 10. Ideas to test on the day

Ideas only, to be built and checked against the honest test during the hackathon.

- **Wallet history:** age when paying, activity count, usual amount, balance
  kept vs passed on, days dormant.
- **Shapes:** distinct senders and receivers in 6 and 24 hours; time from receiving
  to sending; money out ÷ money in; 2-hop paths between pairs; short cycles with
  value returned and asset changes; chain depth.
- **XRPL details:** wallet opener and how many wallets it opened; issuer + symbol;
  rate vs the hourly market price; destination tag use; liquidity providers and
  other hubs marked as context.
- **Guilt by association:** how suspicious the money a wallet received was, and
  what its receivers do next (out-of-fold first-round scores).
- **Group level, for hard cases:** connected-component size, density, value in vs
  out, time packing.

## 11. Sources

- Decode & Detect event page: <https://luma.com/71nckopa>
- Participant terms: <https://hackathon.ukcbt.org/terms/2026-10-21>
- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models* (NeurIPS 2023): <https://arxiv.org/abs/2306.16424>
- IBM AML transactions dataset (Kaggle): <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
- Egressy et al., *Provably Powerful Graph Neural Networks for Directed Multigraphs*: <https://arxiv.org/abs/2306.11586> and code: <https://github.com/IBM/Multi-GNN>
- Weber et al., the Elliptic Bitcoin dataset paper: <https://arxiv.org/abs/1908.02591>
- Deprez et al., *Network Analytics for Anti-Money Laundering: a systematic literature review and experimental evaluation*: <https://arxiv.org/abs/2405.19383>
- FATF, *Virtual Assets Red Flag Indicators of Money Laundering and Terrorist Financing* (2020): <https://www.fatf-gafi.org/en/publications/Methodsandtrends/Virtual-assets-red-flag-indicators.html>
- XRP Ledger docs: Payment <https://xrpl.org/docs/references/protocol/transactions/types/payment>, cross-currency payments <https://xrpl.org/docs/concepts/payment-types/cross-currency-payments>, partial payments <https://xrpl.org/docs/concepts/payment-types/partial-payments>, reserves <https://xrpl.org/docs/concepts/accounts/reserves>
- XRPL Learning Portal, *Intro to the XRPL*: <https://learn.xrpl.org/course/intro-to-the-xrpl/>
- AWS public blockchain data (XRPL, SonarX schema): <https://registry.opendata.aws/aws-public-blockchain>
- NVIDIA, *The Kaggle Grandmasters Playbook*: <https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/>
- scikit-learn, precision-recall and average precision: <https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics>
