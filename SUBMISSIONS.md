# Submissions log

| Time | Commit | Honest-test AP | Leaderboard | What changed |
|---|---|---|---|---|
| 09:34 UTC | 4d8970d | 24.6 (organisers' random 20% holdout) | **33.6** (top 1%: precision 62%, recall 20%; ROC AUC 0.82) | Run 1: the organisers' starter notebook, unchanged (logistic regression, 12 simple rules) |
| 09:59 UTC | bbf8675 | 54.9 (grouped 5-fold; folds 57.3 ± 9.9) | **61.4** (top 1%: precision 93%, recall 30%) | Run 2: Jack's multi-scale motif shape model (135 clues, LightGBM, three-scale motif smoothing) |
| 10:39 UTC | 6641b38 | 57.6 ± 6.1 (main.py folds) | **62.0** (top 1%: precision 91%, recall 29%) | Run 3: Jack's typology detectors + group smoothing + rank blend (`src.predict_shapes_v2`); rerun byte-identical |
