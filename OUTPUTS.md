# Outputs ledger

Every predictions file we have produced, whether or not it was uploaded. The CSVs are
committed in `outputs/` but left out of the release code ZIP (see `.gitattributes`). Update it with
`python tools/track_outputs.py` after writing a file. Rows are keyed by SHA-256, so a
regenerated file gets a new row. "Check" is the `submission-check` result against all
40,101 test transactions. Leaderboard scores and the commit behind each upload live in
`SUBMISSIONS.md`.

| File | Written (UTC) | Rows | Distinct scores | SHA-256 | Check | Status | Notes |
|---|---|---|---|---|---|---|---|
| `predictions.csv` | 2026-10-24 09:32 | 40,101 | 40,101 | d432a2fe5b8a | OK | present | Organisers' starter notebook (run 1 code). Written 2 minutes before the run 1 upload (33.6), so probably the uploaded file; not verified. |
| `predictions_shapes_v1.csv` | 2026-10-24 09:44 | 40,101 | 40,101 | e56cfeea27ef | OK | present | Multi-scale motif model, `python -m src.train --out ...`, from code not yet committed at the time. Run 2 (61.4) was uploaded from Alim's rerun at bbf8675, which may differ from this file. |
| `predictions_shapes_v2.csv` | 2026-10-24 10:14 | 40,101 | 40,101 | d35d85074a92 | OK | present | Run 3: shape clues + typology detectors, group smoothing, 0.75/0.25 rank blend. `python -m src.predict_shapes_v2` (defaults; reproduces byte for byte). Honest test 57.6 +/- 6.1. A run scoring 62.0 was uploaded at 10:39 UTC, most likely this file; not yet recorded in SUBMISSIONS.md. |
| `predictions_shapes_v3.csv` | 2026-10-24 10:47 | 40,101 | 40,101 | 0c791cd61ca6 | OK | present | Run 4 candidate: run 3 plus a 20-minute group window and 6x training weight on medium cases. `python -m src.predict_shapes_v2 --windows 120 300 600 1200 --medium-weight 6 --out predictions_shapes_v3.csv`. Beats run 3 on three fold seeds by +1.2, +0.2, +0.3. Not uploaded. |
