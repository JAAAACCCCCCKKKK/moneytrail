# MoneyTrail detector (detector/)

Two-stage gradient-boosting detector. This is the code behind `predictions_v1.csv` and `predictions_v2.csv`.

## Run

```bash
pip install pandas pyarrow scikit-learn numpy        # Python 3.12 or 3.13
cd detector
python3 predict.py --train ../data/train --test ../data/test-data --out predictions.csv
python3 run_cv.py  --train ../data/train              # honest test (optional, ~15 min)
```

The code needs only pandas, numpy, scikit-learn and pyarrow (no LightGBM). Fixed seeds (0, 1, 2) make it deterministic. It takes about 10 minutes on a laptop and needs under 4 GB of RAM. If pyarrow is missing, it reads `transactions.csv.gz` / `transfers.csv.gz` exports of the parquet files instead.

## Method

1. **Clean data (`base.py`):** one row per transaction. XRP drops are converted to XRP, an asset is issuer + currency, the delivered amount comes from `DELIVERED_AMOUNT`, the cost is the first transfer leg, and everything is valued in USD with the hourly `market_rates.csv`.
2. **Clues (`features.py`, `clusters.py`), about 225 per transaction:**
   - wallet history over the week (counts, distinct counterparties, pass-through, flow, typical amount);
   - pair history (how often this pair pays, reverse payments, time since the pair's last payment);
   - activity windows from 5 minutes to 24 hours (distinct senders into the receiver, distinct receivers from the sender, forwarding by the receiver, money the sender received just before, bursts relative to the wallet's own rate);
   - gaps to the nearest payments touching either wallet (how tight the pattern is);
   - time-respecting motifs (component size and duration, money returning to the sender within 3 hours);
   - **gap clusters** at 2, 5, 10 and 15 minutes: payments sharing a wallet within G seconds are joined, and each cluster is described by size, wallets, span, median gap, fan-in and fan-out, value kept through intermediaries, amount spread and repeated pairs.
3. **Model (`model.py`):** sklearn HistGradientBoosting, averaged over 3 seeds. A second stage adds the max, mean and count of high stage-1 scores among nearby payments sharing a wallet (15 minutes to 12 hours), because labels are given per scenario. Stage-2 training uses out-of-fold stage-1 scores only.
4. **Honest test:** 5-fold StratifiedGroupKFold grouped by `scenario_id`. `scenario_id`, `family`, `difficulty` and `cohort_id` are used only to split and report, never as features. Wallet addresses and hashes are never features.

| Version | Honest-test AP | Easy | Medium | Hard |
|---|---|---|---|---|
| v1 (payment, wallet, window and motif clues, two stages) | 58.0 ± 9.5 | 75.7 | 4.9 | 0.6 |
| v2 (+ gap clusters) | 58.9 ± 8.0 | 76.9 | 5.8 | 0.6 |

## Findings

- XRPL conversion clues (conversion or not, loss vs the market price, number of legs) don't separate suspicious payments from normal ones: conversions are 45% vs 43%, and the median loss is 0.37% for both. The README confirms that asset choice ignores labels. The data has no self-payments, destination tags, memos or partial payments.
- Each planted shape has look-alike legitimate scenarios. Timing separates them: easy scenarios finish in about 5 minutes, decoys usually take 1 to 2 hours. About 15% of decoys are just as fast, and those are the main false positives.
- Hard relay cases are about 1 suspicious scenario per 100 near-identical legitimate ones in each cohort. Medium cases look like decoys, only slower. Both stay close to random.

## Not written by us

pandas, numpy, scikit-learn and pyarrow (BSD/Apache), plus `payments.py` and the data (organisers). AI assistance: Claude (Claude Opus 5.5, Anthropic) wrote this code and README in a chat session, under the direction of Violet. Prompts: the team members' messages in that session.
