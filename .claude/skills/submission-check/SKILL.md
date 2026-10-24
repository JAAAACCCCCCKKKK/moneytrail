---
name: submission-check
description: Pre-upload checklist for the Decode & Detect XRPL AML hackathon. Validates the predictions CSV (TRANSACTION_HASH,score; every test transaction once; scores 0-1; no ties; under 12 MiB), makes sure the code is committed so the run can be reproduced, logs the upload, and checks the final code ZIP (under 8 MiB, no installed packages, no secrets, run instructions present). Use this whenever the user is about to upload, submit or "send" predictions to the leaderboard, asks whether a submission or CSV is valid, wants to log a leaderboard score, or is packaging or zipping the code for the final entry, even if they don't say "check".
---

# Submission check

The leaderboard keeps a team's best run, and the organisers re-run that exact
version before confirming a prize. So every upload must be a valid file *and* a
committed, reproducible version of the code. A rejected or unreproducible run can
cost first place after it has been won. This checklist prevents that.

These rules come from the participant terms (hackathon.ukcbt.org/terms) and the
pre-hackathon training. If the organisers announce something different on the
day, follow what they announce and tell the user what changed.

## Before every upload

1. **Find the test transaction hashes.** The checker compares the CSV against the
   full list of test transactions. Look in the data README for which file holds
   them (on the practice data and probably on the day: the test split's
   `transactions.parquet`). Score transactions, never transfer legs.

2. **Run the checker** (it ships with this skill):

   ```bash
   python3 <this-skill-dir>/scripts/check_submission.py csv submission.csv --test-file <path/to/test/transactions.parquet>
   ```

   Add `--expected-rows 40101` on the day as a second check. Fix every PROBLEM
   before uploading. Treat WARNINGs (ties, very few distinct scores) seriously:
   tied scores are ranked as one block and lower average precision.

3. **Commit the code that produced this file.** Run `git status`. If anything that
   affects the predictions is uncommitted, commit it now with a message that says
   what changed and the honest-test score, for example
   `fan-in features: CV AP 0.71 ± 0.04`. Note the short commit hash. Don't commit
   data files, virtual environments or secrets.

4. **Log the upload** by appending a row to `SUBMISSIONS.md` in the repo root
   (create it with this header if missing):

   ```markdown
   | Time | Commit | Honest-test AP | Leaderboard | What changed |
   |---|---|---|---|---|
   ```

   Leave "Leaderboard" blank until the user reports the score, then fill it in.
   This table is how the team finds which commit produced its best run.

5. **The user uploads.** Remind them: hackathon.ukcbt.org, with the team's access
   code. Don't upload or handle the access code yourself. There are 50 uploads per
   team per UTC day.

## The final entry (aim to finish by 16:15)

1. Run the CSV check above on the final predictions.
2. Confirm a fresh copy of the repo reproduces the CSV: clone into a temporary
   folder, install from the requirements file, run the documented command, and
   compare the output with the uploaded file.
3. Build the code ZIP from the repo, then check it:

   ```bash
   python3 <this-skill-dir>/scripts/check_submission.py zip code.zip
   ```

   At most 8 MiB, with no installed packages, and no private keys, passwords,
   API keys or data the rules don't allow us to share. The terms ask for all of
   these, so go through the list with the user:
   - the code or notebook, plus any prompts used (the skill files count)
   - the trained model files if we're allowed to release them, or instructions to recreate them
   - the exact run command, random seeds, dependencies and computing requirements
   - the team name, the captain and every member
   - the commit (version) being submitted, and the results
   - a method write-up of at most 10 slides or 4 pages, covering the data
     sources, method, evaluation and limitations
   - a list of every third-party or pre-existing component (libraries, models,
     services, our pre-hackathon notes and skills) with its source and licence
   - any external resources, APIs or datasets used (for each dataset: why, and whether it helped)
   - for AI tools: the model, provider, prompts, settings and any limits on
     reproducing that part
4. Make sure the repo is public by the deadline, has an approved open-source
   licence (for example MIT or BSD 3-Clause) with its copyright notice, and that
   every team member has accepted the terms. If the `decision-log` skill is in
   use, its `METHOD.md` covers the write-up and disclosures.
