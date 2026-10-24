# MoneyTrail: team plan for the day

Read this before 10:30. If anything here disagrees with what the organisers say
on the day, the organisers win.

**Contents**

1. [The team at a glance](#1-the-team-at-a-glance)
2. [Before we arrive](#2-before-we-arrive)
3. [How we work together](#3-how-we-work-together)
4. [The day, hour by hour](#4-the-day-hour-by-hour)
5. [Alim: lead](#5-alim-lead)
6. [Natasha: wallet history](#6-natasha-wallet-history)
7. [Jack: graph and shapes](#7-jack-graph-and-shapes)
8. [Violet: XRPL details and the write-up](#8-violet-xrpl-details-and-the-write-up)
9. [When things go wrong](#9-when-things-go-wrong)
10. [Rules to keep in our heads](#10-rules-to-keep-in-our-heads)

---

## 1. The team at a glance

| Person | Role | Builds | Hands to |
|---|---|---|---|
| **Alim** | Lead | Clean data tables, the honest test, `main.py`, every upload, `DECISIONS.md` | Clean tables to everyone; scores to everyone |
| **Natasha** | Wallet history | `features_wallet.py` | Her clue columns to Alim |
| **Jack** | Graph and shapes | `graph.py`, `features_shapes.py` | Wallet groups and hub list to Alim and Violet; his clue columns to Alim |
| **Violet** | XRPL details and write-up | Data checks, `features_xrpl.py`, `METHOD.md` | "Things to know" to everyone; her clue columns to Alim; the method document |

**The one idea behind everything:** our score depends on putting truly suspicious
transactions at the top of a ranked list. The model (LightGBM) is the same for
every team. What wins is the quality of the **clues** (extra columns about each
payment) we feed it, and testing them honestly.

---

## 2. Before we arrive

- **Eligibility:** everyone must have filled in their own registration form, accepted the terms and attended Friday's training. Everyone checks in individually by **10:30** at the latest.
- **Repo access:** Jack adds Natasha, Violet and Alim as collaborators on GitHub, so everyone can push.
- **Download the repo fresh** and install the tools from inside the folder:

  ```bash
  python3 -m pip install -r requirements.txt
  ```

- **Claude Code:** open it inside the `moneytrail` folder once. Our three skills (`submission-check`, `xrpl-traps`, `decision-log`) load automatically. `prep/SKILLS.md` explains them.
- **Read** `prep/RESEARCH.md` (background), `prep/DATA_DICTIONARY.md` (what every column means) and your own section of this file.
- **Bring** a laptop, charger, student ID and photo ID. Charge your phone: it's the backup internet.
- **Agree in the group chat:** the prize split (equal unless we all agree otherwise in writing) and who presents if there's a pitch.

---

## 3. How we work together

**Files and ownership**

| File | Owner | What it contains |
|---|---|---|
| `prepare_data.py` | Alim | Runs `payments.py` on both weeks and saves the clean tables |
| `data/` | nobody (never committed) | The organisers' raw data and our clean tables |
| `graph.py` | Jack | The network, hubs, wallet groups, wallet histories |
| `features_wallet.py` | Natasha | Wallet-history clues |
| `features_shapes.py` | Jack | Shape clues |
| `features_xrpl.py` | Violet | XRPL-detail clues |
| `main.py` | Alim | Combines everything, honest test, model, predictions |
| `DECISIONS.md` | Alim (everyone adds) | The diary of what we tried |
| `SUBMISSIONS.md` | Alim | Which commit produced which upload and score |
| `METHOD.md` | Violet | The method document for the judges |

**The clue-file format, so everything slots together.** Every clue file has one
function with this shape:

```python
def add_features(payments, week):
    """payments: the clean table for one week (one row per transaction).
    week: "train" or "test".
    Returns a table with TRANSACTION_HASH plus our new columns, one row per transaction."""
```

- Every transaction must appear exactly once in what you return, even if a clue is blank for it. Leave blanks as missing values; LightGBM copes with them.
- Name columns clearly with your area first, for example `wallet_age_hours`, `shape_fan_in_6h`, `xrpl_rate_loss`.
- Build every clue **the same way for both weeks**, and only from that week's own payments.
- Never use the labels inside a clue file. Labels are only used by Alim's `main.py`.

**Saving and sharing**

- Commit and push at least once an hour, and always pull before you push.
- Never edit someone else's file. If you need a change, ask them.
- Raw data never goes on GitHub. Our `.gitignore` already blocks it.
- Only Alim uploads, using the routine **Check → Commit → Upload → Log** (the `submission-check` skill).
- Every idea goes in `DECISIONS.md`, including ones that didn't work. Tell Alim and he logs it, or ask Claude to "add this to the decision log".

**Talking**

- One group chat for everything. Post scores, findings and problems there.
- If you're stuck for more than 15 minutes, say so. Being stuck quietly is the most expensive thing on the day.

---

## 4. The day, hour by hour

| Time | Alim | Natasha | Jack | Violet |
|---|---|---|---|---|
| 09:30 | Register the team, keep the access code | Check in | Check in | Check in |
| 10:00 | Write down the opening talk's answers, post them | Listen | Listen | Listen, note anything about the data |
| 10:30 | Data into `data/`, run `payments.py`, save clean tables | Read README | Read README | Read README closely |
| 10:45 | Run `baseline.py`, first upload | Explore wallet behaviour | Start `graph.py` | Check the clean tables against the traps |
| 11:00 | Start `DECISIONS.md` | Draft clue list | Hubs and wallet groups to Alim | Post "things to know" |
| 11:00–11:15 | **Huddle: everyone shares their top findings** | | | |
| 11:15 | Honest test, `main.py` | Build wallet clues | Build shape clues | Build XRPL clues |
| 12:00 | **Merge, test, upload if better** | Push | Push | Push |
| 13:00 | **Merge, test, upload if better.** Lunch at laptops | Push | Push | Push |
| 14:00 | **Merge, test, upload if better** | Push | Push | Push |
| 14:30 | Guilt by association, blend models | Help Jack or check week differences | Group-level clues | Finish best clues |
| 15:00 | Carry on | Carry on | Carry on | Start `METHOD.md` |
| 15:30 | **Freeze.** Fresh-copy rerun, final checks | README sections, code ZIP | README sections, make repo public | Finish `METHOD.md` |
| 16:15 | **Final upload** | | | Check `METHOD.md` matches the final run |
| 16:30+ | Pitch: overview and results | Pitch: one clue | Pitch: one clue | Slides |

---

## 5. Alim: lead

**Your job in one sentence:** make sure the team always has clean data, an honest
way to test ideas, a working model and a valid upload, and that nothing gets lost.

### 10:30–11:15: foundation

1. **Get the data in.** Download the organisers' package and put it in `moneytrail/data/`. Run `git status` and check none of it shows up as something to commit.
2. **Make the clean tables.** Run the organisers' `payments.py` on the training week and the test week. If it doesn't save files itself, write `prepare_data.py` (10 lines) that runs it on both weeks and saves `data/clean/train_payments.parquet` and `data/clean/test_payments.parquet`. Commit `prepare_data.py` and tell everyone the one command to run.
3. **Check the counts.** The test table should have one row per test transaction (about 40,101). Post the numbers in the chat.
4. **First upload.** Run the organisers' `baseline.py`, then say "run the submission check". Commit, upload on hackathon.ukcbt.org with the access code, and log the score. Target: on the leaderboard by **11:00**.
5. **Start the diary.** Ask Claude to start `DECISIONS.md` with the decision log skill, and record the baseline score and how we're testing.

### 11:15–14:30: the honest test and `main.py`

1. **The honest test.** Split the training week into 5 parts with `GroupKFold`, using Jack's wallet groups so connected wallets stay together. For each part: learn from the other four, score the hidden one with average precision. Report the **average and the spread** of the five scores.
2. **`main.py`** does, in order: load the clean tables → call each person's `add_features` for both weeks → join everything on `TRANSACTION_HASH` → run the honest test → train on the whole training week → score the test week → write `submission.csv`, built from the **full list of test transactions** so none go missing.
3. **Model settings to start with:** LightGBM with about 300 trees, learning rate 0.05, 31 leaves, at least 50 rows per leaf, fixed random seeds. Don't spend time tuning; spend it on clues.
4. **Every hour (12:00, 13:00, 14:00):**
   - Pull everyone's work and run the honest test with the new clues.
   - Keep a clue only if it helps in **most of the five parts**, not just on average.
   - If the score improved: **Check → Commit → Upload → Log.**
   - Post the new honest score and leaderboard score in the chat, and say which clues helped.
5. **Quick checks Claude can do for you:** "check this clue for leaks" (uses `xrpl-traps`), and "does this feature help in most folds?".

### 14:30–15:30: hard cases

1. **Guilt by association, done safely:**
   - Round 1: train without it, and give every training payment an **out-of-fold** score (from a model that never saw that payment). Score the test week with a model trained on all of training.
   - Build clues from those scores: the average and highest score of payments **into** the sender, and of payments **out of** the receiver.
   - Round 2: train again with those clues added.
   - Never build this from the real labels.
2. **Blend:** train 2 or 3 versions (different seeds or leaf counts), turn each one's scores into ranks, and average them. Keep it only if the honest test agrees.

### 15:30–16:15: freeze and hand in

1. 15:30: announce the freeze. No new clues.
2. Clone a fresh copy of the repo into a new folder, install from `requirements.txt`, run the exact command, and check the predictions match the best upload. If they don't, see section 9.
3. Run the submission check on the final CSV and on the code ZIP.
4. Final upload by **16:15**. Fill in `SUBMISSIONS.md`.

### Watch out for

- Uploading without committing first.
- A join that silently drops test transactions.
- Chasing tiny leaderboard changes. A big jump on the honest test is a leak until proven otherwise.

---

## 6. Natasha: wallet history

**Your job in one sentence:** describe how each wallet normally behaves, so the
model can spot wallets acting like mules or pass-through accounts.

**You start with:** the clean payments table for each week (sender, receiver, time,
amounts in USD). **You produce:** `features_wallet.py` with `add_features`.

### 10:30–11:15: explore

1. Read the README, then load the clean training table and the labels.
2. Compare suspicious and normal payments, ideally by asking Claude to "compare suspicious and normal payments on these columns". Look at:
   - how old the sender and receiver are (time since each first appears in the week)
   - how many payments each makes that week
   - how big the payment is compared with what that wallet usually sends
   - how long since the sender last received money
3. Pick the **two biggest differences** to share at the 11:00 huddle. In practice the strongest were brand-new quiet receiving wallets and money moving on within about an hour.

### 11:15–14:30: build the clues

Build them one at a time, push, and tell Alim after each batch so he can test it.

| Column | Plain meaning | How to work it out |
|---|---|---|
| `wallet_sender_age_hours` | How long the sender has existed in this week | Payment time minus the first time the sender appears (as sender or receiver) |
| `wallet_receiver_age_hours` | Same for the receiver | |
| `wallet_sender_payments_week` | How busy the sender is | Count of payments the sender makes in the week |
| `wallet_receiver_payments_week` | How busy the receiver is | Count of payments into the receiver |
| `wallet_amount_vs_usual` | Is this payment unusually big for this wallet? | This payment's USD value ÷ the sender's median payment |
| `wallet_mins_since_received` | Pass-through speed | Minutes since the sender last received money (look backwards in time only) |
| `wallet_out_vs_last_in` | How much of what arrived is being passed on | This payment's USD ÷ the last amount the sender received |
| `wallet_kept_ratio` | Does the wallet keep money or pass it all on? | Total USD out ÷ total USD in for the sender over the week |
| `wallet_quiet_before_hours` | Was it dormant, then suddenly active? | Hours since the sender's previous payment |
| `wallet_burst_24h` | A sudden flurry | Payments the sender made in the 24 hours before this one |

**Done means:** `add_features(payments, week)` returns one row per transaction with
these columns, for both weeks, and Alim's honest test runs with it.

### 14:30–15:30

- Help Jack with group-level clues, or
- Run the "is the test week different?" check: train a quick model to tell training payments from test payments. If it finds big differences, tell Alim which columns differ most; those clues may not carry over.

### 15:30–16:15

- With Jack, fill in the README: **Quickstart** (the exact commands to reproduce our predictions) and **Deps** (`python3 -m pip install -r requirements.txt`, Python version). Build the code ZIP for Alim to check.

### Watch out for

- "Looking forward" by accident. For "time since last received", only use earlier payments.
- Busy hubs (exchanges) dominate counts. That's fine: Violet's hub markers give the model the context.
- Dividing by zero for brand-new wallets: leave those as missing rather than infinity.

---

## 7. Jack: graph and shapes

**Your job in one sentence:** build the map of who paid whom, then turn the
laundering shapes from the training into clues.

**You start with:** the clean payments table for each week. **You produce:**
`graph.py`, then `features_shapes.py` with `add_features`.

### 10:30–11:15: `graph.py` (the foundation everyone needs)

The graph is the payments table seen as a map: wallets are the dots, payments are
one-way arrows with an amount and a time. Build it **separately for each week**.
`networkx` handles 40,000 payments in seconds.

`graph.py` should provide:

1. **The network for a week**, built from the clean payments table.
2. **A hub list:** the busiest wallets, for example the top 0.5% by number of different counterparties. These are exchanges, liquidity providers and processors. Post the top few in the chat so Violet can sanity-check them.
3. **Wallet groups:** after setting hubs aside, which wallets connect to which (connected components). Every payment gets a group number. **Alim needs this by about 11:00** for the honest test.
4. **Wallet histories:** each wallet's money in and money out, in time order, so shape and wallet clues are quick to build.

Without setting hubs aside first, nearly everything joins into one giant group,
and loops and paths run through exchanges, which means nothing.

### 11:15–14:30: shape clues

Do the quick ones first and push them early. Tell Alim after each batch.

| Column | Shape | How to work it out |
|---|---|---|
| `shape_fan_in_6h`, `shape_fan_in_24h` | Fan-in | Distinct senders into the receiver in the same 6-hour and 24-hour windows |
| `shape_fan_out_6h`, `shape_fan_out_24h` | Fan-out | Distinct receivers from the sender in the same windows |
| `shape_payout_similarity` | Fan-out | How similar the sender's payment sizes are in that window (spread ÷ average) |
| `shape_receiver_is_hub`, `shape_sender_is_hub` | Context | Whether either side is on the hub list |
| `shape_gather_scatter` | Gather-scatter | For the receiver: distinct senders × distinct receivers, plus whether its senders and receivers overlap |
| `shape_hub_pass_minutes` | Gather-scatter | Typical minutes between money arriving at the receiver and leaving it |
| `shape_two_hop_paths` | Scatter-gather | Number of different middle wallets linking this sender to the same final destination |
| `shape_returns_2hop`, `shape_returns_3hop` | Cycle | Does money from this payment come back to the sender in 2 or 3 steps? |
| `shape_cycle_value_back` | Cycle | Value that came back ÷ value sent |
| `shape_chain_depth` | Layered stack | How many hops the money travels onward, staying almost whole |

**Keeping cycle searches fast:** only look for loops of 2 to 4 steps, set hubs
aside, only count loops within 24 to 48 hours, and test on a sample of a few
thousand payments first. If it's still slow after 15 minutes, park it and move on.

**Done means:** `add_features(payments, week)` returns one row per transaction with
these columns for both weeks, and runs in a few minutes at most.

### 14:30–15:30: group-level clues (for the hard cases)

| Column | How to work it out |
|---|---|
| `group_size` | Number of wallets in the payment's wallet group |
| `group_density` | Payments within the group ÷ wallets in the group |
| `group_value_in_vs_out` | USD entering the group from outside ÷ USD leaving it |
| `group_time_span_hours` | Time between the group's first and last payment |
| `group_distinct_routes` | Different routes between the group's entry and exit wallets |

### 15:30–16:15

- With Natasha, fill in the README Quickstart and Deps sections.
- **Make the repo public** (you own it) once Alim confirms the final upload.
- Optional, for the pitch: a small picture of one suspicious group next to one exchange hub, drawn with `networkx` and `matplotlib`.

### Watch out for

- Letting hubs into loop and path searches.
- Building the test week's graph from the training week, or the reverse. One graph per week.
- Whether we may use the unlabelled test transactions in the test-week graph: Alim asks at 10:00.

---

## 8. Violet: XRPL details and the write-up

**Your job in one sentence:** keep the data honest, add the clues only XRPL
knowledge gives, and tell our story to the judges.

### 10:30–11:15: data guardian

1. Read the organisers' README line by line, with `prep/DATA_DICTIONARY.md` open. Note anything that differs: column names, the `-999` money legs, how the test week was chosen, how `labels.csv` is laid out.
2. When Alim's clean tables exist, ask Claude to check them with the `xrpl-traps` skill. Then check three or four payments by hand:
   - an XRP payment: drops ÷ 1,000,000 = XRP, then × that hour's price = USD
   - a conversion: does "sent" come from the first money leg, not SendMax?
   - a partial payment: is the delivered amount used, not `Amount`?
   - two tokens with the same symbol but different issuers: are they kept separate?
3. Post "things to know about this data" in the chat, for example "two different USDs", "a third of payments are partial payments", "liquidity providers are wallets X and Y".

### 11:15–14:30: XRPL clues (`features_xrpl.py`)

| Column | Plain meaning | Why it might catch launderers |
|---|---|---|
| `xrpl_rate_loss` | Value lost in a conversion compared with that hour's market price | Honest people want a good price; launderers accept bad rates. Use the **first money leg** for what was paid |
| `xrpl_is_conversion` | Did the currency change? | Context for the line above |
| `xrpl_opener` | Which wallet sent this wallet its first XRP | Bosses open mule wallets |
| `xrpl_opener_count` | How many wallets that opener opened this week | Many fresh wallets from one opener is a red flag, unless it's an exchange |
| `xrpl_currencies_used` | How many different assets (issuer + symbol) the wallet touches | Currency hopping blurs the trail |
| `xrpl_convert_after_receive_mins` | Minutes between receiving and converting | Converting straight away is a layering habit |
| `xrpl_tag_share` | Share of the receiver's incoming payments with a destination tag | Real exchanges use tags |
| `xrpl_distinct_tags` | Number of different tags the receiver gets | Many tags looks like a real exchange with real customers |
| `xrpl_is_partial` | Is this a partial payment? | Mostly spam or bots; lets the model ignore them |
| `xrpl_pays_itself` | Sender and receiver are the same wallet | Usually a normal currency swap |
| `xrpl_is_liquidity_provider` | Is either side a liquidity provider? | Stops the model flagging normal conversion hubs |

**Done means:** `add_features(payments, week)` returns one row per transaction with
these columns for both weeks.

### 15:00–16:15: the method document

1. Ask Claude: "use the decision log skill to write METHOD.md from DECISIONS.md".
2. Check every section the terms require is there: team (name, captain, members); problem and data sources; how we handled the data; how we tested; features with why each works; model and settings; results with the final commit; what didn't work; limitations; third-party and pre-existing components with licences (including our prep notes and skills); AI disclosure (Claude Code, Claude Opus 5.5, Anthropic, with the skill files as prompts); how to reproduce.
3. Keep it to **4 pages at most**. Check every number against `DECISIONS.md` and `SUBMISSIONS.md`; never guess a number.
4. At 16:15, check the final commit and score in `METHOD.md` match the final upload.

### After 16:30

- If there's a pitch, turn `METHOD.md` into 4 or 5 slides: the problem, our approach, the clues that worked and why, results, limits.

### Watch out for

- Using SendMax or `Amount` as "what was paid". Always the first leg or the delivered amount.
- Matching prices on symbol alone. Match on issuer **and** symbol, and on the hour.
- Writing "fraud" in the document. It's **laundering**.

---

## 9. When things go wrong

| Problem | What to do |
|---|---|
| Claude Code or the Wi-Fi won't connect | Switch to a phone hotspot straight away. If it's a VPN, turn it off |
| Someone's code has run for 15+ minutes | Stop it, make it smaller (fewer steps, hubs set aside, a sample), and move on if still stuck |
| Honest test jumps to something like 0.99 | Treat it as a leak. Check the clue doesn't use labels, IDs or the hidden part's answers. Shuffle the labels: the score should collapse |
| Honest test up, leaderboard down | Revert the clues (our best upload still stands). Check whether the gain was in most folds and whether the clues are tied to the training week |
| Submission check says transactions are missing | Rebuild the CSV from the full list of test transactions and score the missing ones. Never upload a file the check rejects; our best upload still counts |
| Fresh-copy rerun gives a different score | Check random seeds are fixed, nothing important is only on one laptop, and package versions match `requirements.txt`. Fix, commit, rerun |
| Two people edited the same file | Stop, talk, keep one version. This is why we each own our own files |

---

## 10. Rules to keep in our heads

1. **Check → Commit → Upload → Log** for every upload. ("Can Cats Use Laptops?")
2. Trust the honest test for everyday decisions, and the leaderboard for big ones.
3. A score that jumps suspiciously high is a leak until proven otherwise.
4. The easy cases come quickly. First place is won on the medium and hard ones.
5. Build the judged work today. Our prep notes and skills are disclosed; nothing else from before the event goes in.
6. Say "laundering", not "fraud". They're different crimes.
