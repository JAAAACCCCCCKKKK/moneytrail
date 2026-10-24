# Data dictionary: what every column means

> Written before the hackathon from the training slides and a real slice of public
> XRP Ledger data in the same layout (AWS/SonarX). **The organisers' README wins:**
> if a column on the day is named or works differently, follow the README and
> update this page.

## The files

| File | What one row is | Notes |
|---|---|---|
| `transactions.parquet` | One transaction | Who sent it, the fee, and its details as JSON text |
| `transfers.parquet` | One money **leg** inside a transaction | 1 to 3 legs per transaction. Join to transactions on `TRANSACTION_HASH` |
| `blocks.parquet` | One ledger page | The ledger closes a new page every few seconds |
| `market_rates.csv` | One asset's US-dollar price for one hour | Use the hour that contains the payment |
| `labels.csv` | One training transaction's answer | Training week only. Says whether it's suspicious |

## transactions.parquet

| Column | Plain meaning | Watch out for |
|---|---|---|
| `TRANSACTION_HASH` | The transaction's unique ID, like a receipt number | The key for joining files, matching labels and the submission |
| `ACCOUNT` | The wallet that **sent** the transaction | The receiver isn't here; it's inside `EXTRA_TRANSACTION_FIELDS` |
| `TRANSACTION_TYPE` | What kind of transaction it is, for example `Payment` | The real ledger also has trades, checks, escrows and more |
| `DATETIME` | When it happened, as a date and time (UTC) | |
| `TIMESTAMP` | The same moment as a number | |
| `LEDGER_INDEX` | Which ledger page it's on | Higher means later |
| `LEDGER_HASH` | The unique ID of that ledger page | |
| `TRANSACTION_INDEX` | Its position within the ledger page | |
| `SOURCE_FEE` | The fee, in **drops** | Usually 10 to 15 drops. Burned, not paid to anyone |
| `FEE` | The same fee in whole XRP | |
| `SEQUENCE` | The sender's running transaction count | |
| `FLAGS` | Settings switched on for the transaction | 131072 means a **partial payment** |
| `TRANSACTION_RESULT` | Whether it worked, for example `tesSUCCESS` | |
| `SUCCESS` | True or false: did it go through? | |
| `DELIVERED_AMOUNT` | What **actually arrived** at the receiver | JSON text for tokens, a plain number of **drops** for XRP |
| `EXTRA_TRANSACTION_FIELDS` | The payment's details, as JSON text | Has to be unpacked. See the next table |

### Inside `EXTRA_TRANSACTION_FIELDS`

| Field | Plain meaning | Watch out for |
|---|---|---|
| `Destination` | The wallet that **receives** the payment | |
| `Amount` | What the sender *asked* to deliver | Not always what arrived: partial payments can ask for absurd amounts |
| `DeliverMax` | A copy of `Amount` | |
| `SendMax` | The **most** the sender was willing to pay | A ceiling, not the real cost. The first transfer leg shows what was really paid |
| `DeliverMin` | The least that must arrive, for partial payments | |
| `DestinationTag` | A number saying which customer at the receiver the money is for | Exchanges use one per customer, like flat numbers in a building |
| `InvoiceID` | A reference number, like an invoice number | Often used by payment processors |
| `Paths` | The route the ledger used for a currency conversion | |

## transfers.parquet

| Column | Plain meaning | Watch out for |
|---|---|---|
| `TRANSACTION_HASH` | Which transaction this leg belongs to | Several legs can share one |
| `TRANSFER_INDEX` | The leg's order: 0 is the first | Leg 0 shows what the sender really paid. On the real ledger, -999 looks like a summary row: check the README before counting it |
| `TRANSFER_TYPE` | What kind of movement, for example `direct_payment` or `path_payment` | `path_payment` usually means a currency conversion |
| `FROM_ADDRESS` | Where this leg's money came from | In a conversion, a middle leg can come from a liquidity provider |
| `TO_ADDRESS` | Where this leg's money went | |
| `SYMBOL` | The currency code, for example `XRP`, `USD` | Codes longer than 3 letters are stored as 40 hex characters and need decoding |
| `TOKEN_ADDRESS` | The **issuer** of the currency | XRP has a placeholder (`0x1111…`). USD from two issuers is two different assets |
| `SOURCE_VALUE` | How much moved, in the currency's raw units | **For XRP this is drops** (1 XRP = 1,000,000 drops). For tokens it's the normal amount |
| `VALUE` | How much moved, in whole units | Already converted for XRP |

## blocks.parquet

| Column | Plain meaning |
|---|---|
| `LEDGER_INDEX` | The ledger page number |
| `LEDGER_HASH` | The page's unique ID |
| `DATETIME` / `TIMESTAMP` | When the page closed |
| `CLOSE_TIME` | The close time in the ledger's own clock (seconds since 1 January 2000) |

## market_rates.csv

| Column | Plain meaning | Watch out for |
|---|---|---|
| The hour | Which hour the price is for | Match each payment to the hour that contains it |
| The asset (issuer + symbol) | Which currency | Match on both issuer and symbol |
| The USD price | What one unit was worth in US dollars that hour | Use it to put every amount in dollars before comparing |

The exact column names for this file weren't shown in the training: check the README.

## labels.csv

| Column | Plain meaning |
|---|---|
| `TRANSACTION_HASH` | Which training transaction |
| The label | Whether it's suspicious (1) or not (0). Check the README for the column name |

## Glossary

| Term | Plain meaning |
|---|---|
| **Wallet / account / address** | An account on the ledger, a string starting with `r`. No names, so judge by behaviour |
| **Drop** | The smallest unit of XRP. 1,000,000 drops = 1 XRP |
| **Token / IOU** | A currency issued by someone, like a voucher. Only worth what you trust the issuer |
| **Issuer** | The wallet that created a token. Same name, different issuer = different asset |
| **Trust line** | A wallet saying it's willing to hold a particular issuer's token |
| **Conversion** | One transaction where the sender pays one currency and the receiver gets another |
| **Liquidity provider** | A wallet in the middle of conversions that swaps one currency for another. Shows up as a huge, normal hub |
| **DEX** | The ledger's built-in marketplace for swapping currencies |
| **Partial payment** | A payment allowed to deliver less than it asked for. Only the delivered amount counts |
| **Destination tag** | A number telling a shared wallet (like an exchange) which customer the money is for |
| **Hub** | A very busy wallet, like an exchange, processor or liquidity provider. Busy isn't suspicious by itself |
| **Paying yourself** | Sender and receiver are the same wallet, usually a currency swap. Normal |
| **Leg** | One movement of money inside a transaction. A conversion has 2 or 3 |
