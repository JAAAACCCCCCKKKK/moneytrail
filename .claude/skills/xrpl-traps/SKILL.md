---
name: xrpl-traps
description: Checklist of the data traps in the Decode & Detect XRPL AML hackathon data (AWS/SonarX XRP Ledger layout - transactions.parquet, transfers.parquet, blocks.parquet, market_rates.csv, labels.csv). Use this whenever writing, reviewing or debugging code that loads, joins, parses, converts or values this data - TRANSACTION_HASH joins, SOURCE_VALUE drops, TOKEN_ADDRESS + SYMBOL assets, SendMax, DELIVERED_AMOUNT, EXTRA_TRANSACTION_FIELDS JSON, USD prices, transfer legs - and whenever amounts, counts or scores look surprisingly large, small or perfect, even if the user doesn't mention traps.
---

# XRPL data traps

Most early mistakes on this data are not modelling mistakes. They are loading
mistakes that quietly produce wrong numbers: values a million times too big,
two currencies merged into one, or one row per leg instead of per transaction.
A model trained on wrong numbers can still score well on the training week and
then collapse on the test week. Check every piece of data-handling code against
the list below.

**The data README wins.** If the organisers' README says something different from
this list, follow the README and tell the user about the difference.

## The checklist

1. **Score transactions, not transfers.** One transaction has 1 to 3 transfer legs
   (more on the real ledger). Join on `TRANSACTION_HASH`. The output needs exactly
   one row per test transaction. Check: row counts before and after every join; a
   join that multiplies rows needs a reason.

2. **XRP comes in drops.** For XRP rows, `SOURCE_VALUE` is in drops:
   1 XRP = 1,000,000 drops. `VALUE` is already in whole units. XRP strings in
   `Amount` or `DELIVERED_AMOUNT` are drops too. XRP has a placeholder
   `TOKEN_ADDRESS` (`0x1111…`) because it has no issuer.

3. **An asset is issuer + symbol.** Group by `TOKEN_ADDRESS` and `SYMBOL`
   together. USD from two issuers is two different assets with different prices.
   Codes longer than 3 letters are stored as 40 hex characters: decode them with
   `bytes.fromhex(code).rstrip(b"\x00").decode()` before reading them.

4. **SendMax is a ceiling, not the cost.** What was really paid is the first
   transfer leg (`TRANSFER_INDEX` 0). Partial payments (`FLAGS` includes 131072)
   can ask for absurd amounts in `Amount` and deliver almost nothing: use
   `DELIVERED_AMOUNT` or the legs, never `Amount` or `SendMax`, for value.

5. **Compare value in USD.** Use `market_rates.csv`, matching the asset and the
   hour that contains the payment. Raw amounts in different assets aren't
   comparable.

6. **Some fields are JSON text.** Parse `EXTRA_TRANSACTION_FIELDS` (holds
   `Destination`, `DestinationTag`, `Amount`, `SendMax`, `DeliverMin`, `InvoiceID`)
   and `DELIVERED_AMOUNT` (a JSON object for tokens, a plain drops string for XRP).
   Handle both shapes.

## Things that look suspicious but are normal

- **Payments to yourself** are usually currency swaps through the DEX. They form
  loops but are routine.
- **Liquidity providers** appear as huge hubs inside conversions (the middle legs).
  Exchanges, payment processors and payroll wallets are hubs too. Being busy
  isn't a signal on its own; compare a wallet with its own history and context.
- **Fees** are 10 to 15 drops and are burned, so a forwarded amount can be a few
  drops lower than what arrived.
- On the real ledger some transfer legs have `TRANSFER_INDEX` -999 (they look like
  summary rows). If the challenge data has them, check the README before counting
  them as money movements, or value will be double-counted.

## Leakage and rules checks

- Outside datasets are allowed (organisers' update on the day) as long as each
  one is disclosed: why it was used and whether it helped, logged in the
  "External data" section of `DECISIONS.md`. Flag any code that pulls in outside
  data that isn't logged there. Still forbidden: external labels for this data,
  organiser files, generator seeds and test answers.
- Never use wallet addresses, transaction hashes or exact timestamps as model
  features: the test week has different wallets and new variations.
- Any feature built from labels (for example "how suspicious are this wallet's
  neighbours") must use out-of-fold scores so no row sees its own label.
- A single feature with a near-perfect honest-test score almost always means a
  leak. Investigate before celebrating.

## How to report a review

When reviewing code, list each of the six traps with ✓ (handled), ✗ (wrong) or
– (not relevant to this code), a line of evidence each, then the fix for every ✗.
Run a quick check on the data where possible (row counts, a sample value
converted by hand) rather than reasoning from the code alone.
