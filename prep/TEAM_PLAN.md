# MoneyTrail: team plan for the day

Read this before 10:30. If anything here disagrees with what the organisers say
on the day, the organisers win.

## Who does what

| Person | Role | Owns |
|---|---|---|
| **Alim** | Lead | Preparing the data, the honest test, the main model script (`main.py`), combining everyone's work, every upload, the decision log |
| **Natasha** | Wallet history | `features_wallet.py`: clues about how each wallet behaves over time |
| **Jack** | Graph and shapes | `graph.py` and `features_shapes.py`: the network of wallets and the laundering shapes |
| **Violet** | XRPL details and write-up | Checking the data is read correctly, `features_xrpl.py`, and the method document |

## Before we arrive

- Everyone: you must have filled in your own registration form, accepted the terms and attended Friday's training. Check in by **10:30** at the latest.
- Everyone: download the repo fresh and install the tools from inside the folder:

  ```bash
  python3 -m pip install -r requirements.txt
  ```

- Everyone: open Claude Code inside the `moneytrail` folder once. Our three skills (`submission-check`, `xrpl-traps`, `decision-log`) load automatically. `prep/SKILLS.md` explains them.
- Read `prep/RESEARCH.md` (background) and `prep/DATA_DICTIONARY.md` (what every column means).
- Bring a laptop, charger, student ID and photo ID. If Claude Code or the Wi-Fi stops working, switch to a phone hotspot straight away.

## How we work together

- **Each person writes their clues in their own file.** Nobody edits anyone else's file. If you need a change in someone else's file, ask them.
- **Every clue file has one function** that takes the clean payments table and returns a table with `TRANSACTION_HASH` plus your new columns, one row per transaction.
- **Everyone commits and pushes at least every hour** (and pulls before pushing).
- **Only Alim uploads.** We share one access code and 50 uploads a day.
- **Every upload follows the same routine: Check → Commit → Upload → Log.** The `submission-check` skill does this.
- **Every idea goes in the decision log,** including the ones that didn't work.
- **Raw data never goes on GitHub.** Our `.gitignore` already blocks it.

## 09:30–10:30: arrival and opening talk

- Register the team at the venue. Alim keeps the team's submission code.
- During the opening talk, Alim writes down the answers and posts them in the group chat. Ask about anything not covered:
  - Is the test week later in time than the training week, with different wallets?
  - Does a "suspicious" label follow the money on to the next wallets?
  - Can we use the unlabelled test transactions when building the graph?
  - How long is the pitch, does it affect the ranking, and what's the tie-break?
  - Is it fine that we prepared research notes and checklists beforehand, all disclosed?

## 10:30–11:15: the foundation

**Alim**
1. Put the organisers' data in a folder called `data` inside the repo.
2. Run the organisers' `payments.py` on the training week and the test week, and save the two clean tables. If it doesn't save them itself, write a short `prepare_data.py` that does, so everyone gets identical tables with one command.
3. Run the organisers' `baseline.py`, then **Check → Commit → Upload → Log**. Aim to be on the leaderboard by about 11:00.
4. Start `DECISIONS.md` with the decision log and record the baseline score.

**Violet**
1. Read the organisers' README carefully. Note anything that differs from our data dictionary (column names, `-999` money legs, how the test week was chosen).
2. Check the clean tables with the `xrpl-traps` skill. Check three or four payments by hand, for example turning an XRP amount from drops into dollars yourself.
3. Post a short "things to know about this data" message in the group chat.

**Jack**
1. Read the README, then build `graph.py` on top of the clean tables:
   - the network of wallets for each week separately
   - a list of hub wallets (exchanges, liquidity providers and other very busy wallets)
   - groups of connected wallets, worked out after setting the hubs aside
   - each wallet's money in and money out, in time order
2. Give Alim the wallet groups by about 11:00: the honest test needs them.

**Natasha**
1. Read the README, then compare suspicious and normal payments in the clean tables: wallet age, how busy wallets are, how fast money moves on, typical amounts.
2. Draft the list of wallet-history clues you'll build, starting with the biggest differences.

## 11:00–11:15: five-minute huddle

- Everyone shares the one or two biggest differences they found between suspicious and normal payments. These become the first clues.
- Agree the clue-file format (see "How we work together").
- Alim confirms the honest test: 5 parts, keeping Jack's wallet groups together, scored with average precision.

## 11:15–14:30: building clues

**Natasha** (`features_wallet.py`)
- Wallet age when it pays, and how many payments it makes that week
- Its usual payment size, and how unusual this payment is compared with that
- How long since it last received money, and how much of that it's now passing on
- How much it keeps compared with how much flows through it
- How long it was quiet before suddenly becoming active

**Jack** (`features_shapes.py`)
- Fan-in and fan-out: distinct senders and receivers in 6- and 24-hour windows, with hubs marked
- Gather-scatter (the host's "most important" shape): many in and many out, little kept, money passing through quickly, senders and receivers that never overlap
- Scatter-gather: several two-step paths between the same pair of wallets
- Cycles: money returning to where it started in 2 to 4 steps, how much came back, whether the currency changed. Keep the search small (hubs set aside, a time window, tested on a sample first)
- Chains: how long a line of middlemen the money passes through

**Violet** (`features_xrpl.py`)
- How much value a currency conversion lost compared with that hour's market price
- Who opened the wallet, and how many wallets that opener has opened
- How many currencies the wallet hops between
- Destination tag habits (share of payments with tags, how many different tags)
- Markers for harmless hubs, plus partial-payment and pay-yourself flags

**Alim** (`main.py` and the decision log)
- The main script: load the clean tables, add everyone's columns, run the honest test, train the model, write the predictions
- Log every result

**Every hour (12:00, 13:00, 14:00):**
1. Everyone commits and pushes.
2. Alim pulls, runs the honest test with the new clues, and keeps only the ones that help in most of the five parts.
3. If the score improved: **Check → Commit → Upload → Log.**
4. Alim posts the new score in the chat.

Lunch at our laptops around 13:00.

## 14:30–15:30: the hard cases

- **Jack:** clues about whole groups of wallets: size, how tightly connected, money in versus out, how bunched in time.
- **Alim:** guilt by association, built from out-of-fold model scores, never from the real labels. Then combine 2 or 3 models.
- **Natasha:** help Jack with group clues, or check whether the test week looks different from the training week.
- **Violet:** finish your best clues, then switch to the write-up at 15:00.

## 15:00–16:15: freeze and hand in

- **15:00, Violet:** turn `DECISIONS.md` into `METHOD.md` with the decision log skill (at most 4 pages, every required section).
- **15:30, everyone: no new ideas.**
- **Alim:** fresh copy of the repo, run everything from scratch, check it gives the same predictions. Final submission check on the CSV.
- **Natasha and Jack:** fill in the README's Quickstart (exact commands), Deps (what to install) and AI Usage Disclosure (Claude Code, Claude Opus 5.5 by Anthropic, with the skill files in `.claude/skills` as our prompts). Build the code ZIP; Alim checks it with the submission check.
- **Jack:** make the repo public.
- **16:15, Alim: final upload.** Not at 16:29.

## After 16:30: demos and the pitch

- Violet turns the method document into 4 or 5 simple slides if there's a pitch.
- Suggested split: Alim gives the overview and results; Jack or Natasha explains one clue that worked and why.
- The story in four lines: the problem, what we built, what worked and why, the limits.

## Rules to keep in our heads all day

1. **Check → Commit → Upload → Log** for every upload.
2. Trust the honest test for everyday decisions, the leaderboard for big ones.
3. A score that jumps suspiciously high is a leak until proven otherwise.
4. The easy cases come quickly. First place is won on the hard ones.
5. Say "laundering", not "fraud". They're different crimes.
