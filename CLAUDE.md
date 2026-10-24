# CLAUDE.md — Decode & Detect: The XRPL AI Hack

Project context for Claude Code. Read this before every task.
Source: official UKCBT / XRPL Commons training decks (23 Oct 2026) and the day-of briefing slides (24 Oct 2026).
If anything here conflicts with the organiser's `README.md` in the data package or with the portal, **the organiser's material wins** — say so and follow it.

---

## 1. The task

- Synthetic, fictional ledger in the real **AWS/SonarX XRP Ledger schema** (not real accounts or crime). Pattern families come from IBM AMLworld (Altman et al., NeurIPS 2023).
- **40,131 labelled training transactions** and **40,101 unlabelled test transactions**, each split covering **7 days**.
- Output: **one suspicion score per test transaction** (`TRANSACTION_HASH`), higher = more suspicious.
- Suspicious cases by difficulty: **easy 64%, medium 18%, hard 18%**. Hard cases span many payments and changing counterparties, so look at whole flows, not single amounts.
- **Test is not train**: the test set has variations absent from training, and a few cases are deliberately ambiguous. Prefer general behavioural features over train-specific quirks.
- Lots of normal life is mixed in: shopping, peer payments, exchange and payroll hubs, legitimate treasury flows. Both sides use currency conversions.
- The data is mostly about **layering** (money moving between wallets). Nothing outside the 7-day window is visible.

## 2. Data files

| File | Contents |
|---|---|
| `transactions.parquet` | One row per transaction: sender, fee, payment details as JSON |
| `transfers.parquet` | Money legs: 1–3 rows per transaction |
| `blocks.parquet` | Ledger records: index, hash, close time |
| `market_rates.csv` | Hourly reference USD price for each asset |
| `labels.csv` | Training split only: which transactions are suspicious |

Supplied by the organisers (pre-existing components, credit them):
- `payments.py`: builds one edge per payment (who paid whom, how much in USD, when).
- `baseline.py`: produces a valid submission file and a score to beat.
- `README.md`: full field documentation. Read it before writing joins.

## 3. Six data traps (check whenever touching amounts, assets or joins)

1. **Score transactions, not transfers.** Join on `TRANSACTION_HASH`; exactly one output row per transaction.
2. **XRP comes in drops.** For XRP rows, `SOURCE_VALUE` is in drops: 1 XRP = 1,000,000 drops.
3. **An asset is address + symbol.** Group by `TOKEN_ADDRESS` and `SYMBOL`. USD from two issuers is two different assets.
4. **SendMax is a ceiling, not the cost.** The first transfer leg shows what was really paid.
5. **Compare value in USD** using `market_rates.csv`, taking the hour that contains the payment.
6. **Some fields are JSON text.** Parse `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT`.

XRPL rules the data keeps:
- A wallet is an `r…` address with no name. Judge behaviour, not identity.
- USD, GBP and EUR tokens are IOUs from an issuer.
- A currency conversion is one transaction via 1–2 liquidity providers, so it has 2–3 transfer rows.
- Every transaction burns 10–15 drops of XRP as a fee, even token payments.
- `DELIVERED_AMOUNT` is what actually arrived.
- One address is not one person: exchanges and hubs serve many customers. Dust payments can dominate a raw graph. Short time windows cut multi-day patterns in half.

## 4. Eight laundering shapes: signals and legitimate look-alikes

| Shape | Signals to compute | Legit look-alike |
|---|---|---|
| Fan-in | Distinct senders into one wallet in a short window; senders that are new or do nothing else; time from last deposit to exit; exit ≈ sum of deposits | Shop, fundraiser, exchange deposit address |
| Fan-out | Distinct recipients per sender in a short window; similarity of payout amounts; recipients that forward quickly; source funded just before | Payroll, refunds, giveaways, exchange withdrawals |
| Scatter-gather | Several two-hop paths between the same pair; value out of source ≈ value into destination; middle wallets with no other activity; short, similar hold times | Company moving between own accounts, exchange rebalancing |
| Gather-scatter | Many senders and receivers on one wallet; little balance kept vs flow-through; short in→out time; senders and receivers never overlap | Exchange hot wallet, payment processor, payroll provider |
| Cycle | Path returning to its start within a window; hop count and duration; value returned vs sent; assets changing along the way | Arbitrage, market making, refunds, own-wallet moves |
| Layered stack | Depth (longest followable chain); middle wallets used once then abandoned; value nearly whole per layer; similar delays per layer | Payment processors, correspondent chains, company groups |
| Bipartite | Dense payment blocks between two sets of wallets; few contacts outside the block; many pairs active in the same window; similar amounts across pairs | Marketplace buyers/sellers, supplier payments |
| Irregular network | Size and density of the connected group; number of distinct routes between two wallets; value into vs out of the group; how tightly packed in time | Ordinary peer-to-peer life |

Other flag behaviours: rapid pass-through (leaves soon, little kept), hops with no purpose, splitting, currency hopping, round-tripping, flows that only make sense over the whole week.
A pattern is not proof: exchanges, payroll, merchants and treasury moves make the same shapes. Context separates them.

## 5. Scoring

- **Score = 100 × Average Precision** on the organisers' private labels for all 40,101 test transactions.
- Only the order counts; scores need not be calibrated probabilities.
- **Avoid ties**: equal scores are grouped as one block. Always output continuous scores, never just 0/1.
- Diagnostics shown but not scored: ROC-AUC, and precision / recall / F1 for the top 1% (402 transactions).
- Tie-break: organisers pick a fair resolution. Top three eligible, complete and verified entries with sufficient documentation win. The leaderboard stays provisional until verification.

## 6. Submission

Portal: https://hackathon.ukcbt.org/ (sign in with the team's organiser-issued access code; download data and the submission template there).

1. Upload the **code ZIP (≤ 8 MiB)** and the **predictions CSV (≤ 12 MiB)** together.
2. Check that the portal accepts the files and returns a score.
3. Review the score and leaderboard; predictions can be updated before the deadline.
4. Complete the **supporting-material submission** following the portal instructions.

A predictions CSV alone is not a complete final submission. **Everything closes at 16:30 London.** Limit: 50 uploads per team per UTC day. Problems → contact Narmin Rahimli promptly.

**Predictions CSV**
```
TRANSACTION_HASH,score
```
Validate before every upload:
- [ ] Exactly 40,101 rows; every test transaction appears exactly once
- [ ] No missing, duplicate or additional transaction IDs
- [ ] Every score is finite and between 0 and 1
- [ ] Column names and format exactly match the portal template
- [ ] One row per transaction, even when a transaction has multiple transfers

**Code ZIP** must contain: code, exact run command, seeds, dependencies, team list, commit or version, results, and a method write-up (under 10 slides or 4 pages). No secrets, no installed packages (no venv / node_modules).

## 7. Rules (full terms: https://hackathon.ukcbt.org/terms/2026-10-21)

- **Organiser data only.** Use the supplied train/test files. Public documentation, research and reference data may support the work. No outside transaction data without approval, so no live XRPL / XRPScan / Bithomp data as features.
- Disclose external data sources, APIs, models and services.
- Any language, framework or model is permitted (rules, statistics, ML, graph algorithms, LLM-based approaches). Libraries and pre-trained models are fine.
- **Built on the day.** The judged work is done during the hackathon.
- **Identify pre-existing components** (libraries, `payments.py`, `baseline.py`) and explain what the team developed.
- **LLM use must be disclosed**: model/provider, relevant prompts, settings, and reproducibility limitations.
- **Reproducible**: exact run command, seeds, dependencies. If organisers cannot reproduce the predictions, the entry may be retracted. The team must understand its own solution.
- Do not access private test labels, organiser-only files or other teams' work. Do not fabricate results. Preserve third-party licences and attribution.
- Do not present synthetic results as evidence of wrongdoing by real people or accounts.
- **Open source**: the team keeps ownership, but original submission materials must be released under an approved open-source licence permitting commercial reuse, whether or not the entry wins. (MIT or Apache-2.0 are common choices that permit commercial reuse; confirm against the approved list in the terms.)

## 8. Team and working conventions

- Team of four: alim (captain), Jack, Vi, Nat. Proposed split:
  - **P1 wallet history** (prefix `w_`): wallet age, payment frequency, usual amounts, time to forward, share of funds kept, out-of-character activity.
  - **P2 laundering shapes** (prefix `s_`): fan-in/out, cycles, scatter/gather, layer depth, dense bipartite blocks. Start with clear rule-like features; let the model learn combinations.
  - **P3 XRPL-specific clues** (prefix `x_`): currency hopping, conversion legs, SendMax vs delivered, fees, destination tags, issuer differences.
  - **P4 pipeline & model**: load/join, USD conversion, edges, time-based holdout + local AP scorer, merge features, train, generate CSV, package ZIP.
- Every feature module returns **one DataFrame keyed by `TRANSACTION_HASH`** (or by account, joined back to each transaction), with columns prefixed by owner.
- Timeline: first hour = shared pipeline, run `baseline.py`, **first upload to prove the portal works**; others study the labels. Then hourly: merge, retrain, check local AP, upload if better. ~15:00 start write-up and pick 2–3 flagged case studies. **16:00 feature freeze.** Aim to be fully submitted by 16:15.

## 9. Modelling approach (team plan, not an organiser requirement)

- Gradient-boosted trees (LightGBM / XGBoost) on account- and flow-level features joined to each transaction. Avoid spending the day on a GNN.
- **Time-based validation**: train on earlier days of the training week, validate on later days. Compute AP locally before spending uploads.
- IsolationForest only as a sanity check (labels exist).
- Fix all random seeds. Use SHAP or per-feature-group ablation to explain why the model works.
- Evidence for the write-up: 2–3 flagged flows with transaction hashes, synthetic injection (plant a fake peel chain or fan-out and check it is found), stability across time windows, and what is deliberately not flagged (hubs, payroll) and why.

## 10. Write-up and language

- Structure: (1) problem & data: target behaviour, files used, cleaning; (2) approach & evidence: features, rules, model in plain words plus 1–2 flagged cases; (3) limits & next steps: what more labels, time or data would change; honest scores.
- Say "**flagged for review**" and show the evidence. Never call an address criminal or try to identify a person.
- Caveat: this is fictional data; performance here does not establish performance on real financial crime or the real XRP Ledger.

## 11. Instructions for Claude Code

- Keep `AI_USAGE.md` current: model and provider, key prompts, settings, reproducibility limits, and which code is pre-existing vs team-written.
- Keep `README.md` current with the exact run command, Python version, pinned dependencies and seeds.
- Run the CSV checklist in section 6 before producing any submission file.
- Check the six traps in section 3 whenever writing joins, amount handling or asset grouping.
- Don't add outside data sources. Don't commit secrets or the access code.
- Be concise; the team is on a deadline.
