# Submissions log

| Time | Commit | Honest-test AP | Leaderboard | What changed |
|---|---|---|---|---|
| 09:34 UTC | 4d8970d | 24.6 (organisers' random 20% holdout) | **33.6** (top 1%: precision 62%, recall 20%; ROC AUC 0.82) | Run 1: the organisers' starter notebook, unchanged (logistic regression, 12 simple rules) |
| 09:59 UTC | bbf8675 | 54.9 (grouped 5-fold; folds 57.3 ± 9.9) | **61.4** (top 1%: precision 93%, recall 30%) | Run 2: Jack's multi-scale motif shape model (135 clues, LightGBM, three-scale motif smoothing) |
| 10:39 UTC | 6641b38 | 57.6 ± 6.1 (main.py folds) | **62.0** (top 1%: precision 91%, recall 29%) | Run 3: Jack's typology detectors + group smoothing + rank blend (`src.predict_shapes_v2`); rerun byte-identical |
| ~11:30 UTC | a265fa9 | 57.7 vs run 3's 56.5 (main.py folds, seed 42) | **60.8** (top 1%: precision 89%, recall 29%) | Run 4, uploaded by Jack as "TEST 3": run 3 + 20-minute group window + 6x medium training weight. Worse than run 3; dropped |
| ? | ? | ? | ~62 and ~61.5 (from the team page chart) | Runs 5 and 6: not logged yet. Whoever uploaded them, please fill in what they were |
| ~12:15 BST | 69ff81b | 57.4 (Violet's own folds 58.4 ± 7.2; on main.py folds 57.1 ± 7.2) | **65.1** (top 1%: easy 40%, medium 0%) | Run 7: Violet's detector v2 alone (`outputs/predictions_violet_v2.csv`, `python3 detector/predict.py`). New best |
| ? | ? | ? | ~61.6 (from the team page chart) | Attempt 8: not logged yet. Whoever uploaded it, please fill in what it was |
| ~12:25 BST | b3b0d5a | 58.6 (blend of out-of-fold scores; folds 58.9 ± 6.3) | **65.5** (top 1%: precision 92%, recall 30%; easy 50%, medium 0%) | Attempt 9 ("Run 9"): 50/50 rank blend of run 3 and Violet v2 (`python3 -m src.blend`). New best, +0.4 on Violet alone |
| 11:29 UTC | 9e2b940 | vs run 3: +0.4, +0.7, -0.1 on three fold seeds (10 of 15 folds better) | **61.8** (top 1%: precision 90%, recall 29%; easy 40%, medium 10%) | Attempt 10: Jack's candidate B alone (`python3 -m src.predict_shapes_v2 --drop-conversion --week-rank`). Below run 3's 62.0; dropped |
