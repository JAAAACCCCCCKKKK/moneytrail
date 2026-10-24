# Outputs ledger

Every predictions file we have produced, whether or not it was uploaded. The CSVs are
committed in `outputs/` but left out of the release code ZIP (see `.gitattributes`). Update it with
`python tools/track_outputs.py` after writing a file. Rows are keyed by SHA-256 with line
endings ignored, so the same file has the same key on Windows and macOS, and a
regenerated file gets a new row. "Check" is the `submission-check` result against all
40,101 test transactions. Leaderboard scores and the commit behind each upload live in
`SUBMISSIONS.md`.

| File | Written (UTC) | Rows | Distinct scores | SHA-256 | Check | Status | Notes |
|---|---|---|---|---|---|---|---|
| `predictions.csv` | 2026-10-24 09:32 | 40,101 | 40,101 | 7fecd66cad56 | OK | present | Organisers' starter notebook (run 1 code). Written 2 minutes before the run 1 upload (33.6), so probably the uploaded file; not verified. |
| `predictions_shapes_v1.csv` | 2026-10-24 09:44 | 40,101 | 40,101 | 54d759abe2a3 | OK | present | Multi-scale motif model, `python -m src.train --out ...`, from code not yet committed at the time. Run 2 (61.4) was uploaded from Alim's rerun at bbf8675, which may differ from this file. |
| `predictions_shapes_v2.csv` | 2026-10-24 10:14 | 40,101 | 40,101 | a38c9e4dde78 | OK | present | Run 3: shape clues + typology detectors, group smoothing, 0.75/0.25 rank blend. `python -m src.predict_shapes_v2` (defaults; reproduces byte for byte). Honest test 57.6 +/- 6.1. A run scoring 62.0 was uploaded at 10:39 UTC, most likely this file; not yet recorded in SUBMISSIONS.md. |
| `predictions_shapes_v3.csv` | 2026-10-24 10:47 | 40,101 | 40,101 | c127b5d9f9d4 | OK | present | Run 4, uploaded as 'TEST 3' at ~11:30 UTC: run 3 + 20-minute group window + 6x medium weight. Leaderboard 60.8 (run 3: 62.0), although the honest test was +1.2. Dropped. |
| `predictions_violet_v2.csv` | 2026-10-24 11:00 | 40,101 | 40,101 | bac3c6e63949 | OK | present | Violet's two-stage detector v2 (`detector/`, `python3 predict.py`): sklearn gradient boosting on ~225 clues incl. gap clusters, then neighbour-score stage. Honest test 58.9 +/- 8.0 (own grouped folds). Not uploaded. Known issue: a fresh rerun differs slightly; being fixed. |
| `predictions_blend_violetv2_jackv3.csv` | 2026-10-24 11:00 | 40,101 | 40,101 | e7be1321a1af | OK | present | Rank average (50/50) of predictions_violet_v2.csv and predictions_shapes_v3.csv. Spearman between the two 0.49; top-400 overlap 294. No honest test (needs both models on the same folds). Not uploaded. |
| `predictions_shift_a.csv` | 2026-10-24 11:02 | 40,101 | 40,101 | 8ee2d66152bd | OK | present | Candidate A: run 3 without the 13 conversion and asset clues (no label signal per the README; conversion share 41% train vs 51% test). `python -m src.predict_shapes_v2 --drop-conversion --out predictions_shift_a.csv`. Honest test vs run 3: +0.3, -0.0, -0.2 on three seeds. Not uploaded. |
| `predictions_shift_b.csv` | 2026-10-24 11:04 | 40,101 | 40,101 | 4f68835db639 | OK | present | Candidate B: A + every amount-based clue as a percentile within its own week (test amounts ~12% lower, amount spreads 0.58 -> 0.69). `--drop-conversion --week-rank`. Honest test vs run 3: +0.4, +0.7, -0.1; 10 of 15 folds better. Not uploaded. |
| `predictions_shift_d.csv` | 2026-10-24 11:04 | 40,101 | 40,101 | aa5676d3ad58 | OK | present | Candidate D: B + one round of self-training on the test week (own top 2% as suspicious, bottom 50% as innocent, half weight). `--drop-conversion --week-rank --self-train`. Simulated on held-out folds: +0.7, +1.3; 7 of 10 folds better. Top-401 overlap with run 3: 291. Not uploaded. |
