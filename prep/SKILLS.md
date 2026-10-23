# Our Claude Code skills: what they are and when to use them

> **Disclosure.** These three skills were written before the hackathon
> (23 October 2026) with help from Claude (Claude Opus 5.5, Anthropic, via Claude
> Code). They are checklists and one file-format checker. They contain no
> features, models or competition data. The skill files themselves are our
> prompts, so they are kept in this repo as the rules ask.

## What a skill is

A skill is a set of written instructions that Claude follows whenever a certain
kind of task comes up. Think of it as a checklist taped to the wall: whenever
we're about to do that task, Claude reads the checklist first, so nothing
important gets forgotten when we're tired or rushing.

None of these skills does the competition work for us. They stop us making
costly mistakes along the way.

## How to use them

The skills live in `.claude/skills/` in this repo. If you open Claude Code inside
the `moneytrail` folder, it picks them up automatically, with nothing to install.

Claude should use the right skill by itself when the moment comes, but you can
always ask for one by name, for example by typing `/submission-check`, or just
saying "run the submission check".

---

## 1. Submission check: "Is this safe to upload?"

**What it does.** Before we upload predictions to the leaderboard, Claude checks
that the file is exactly what the organisers asked for: the right two columns
(`TRANSACTION_HASH,score`), every test transaction included exactly once, all
scores between 0 and 1, no tied scores, and under 12 MiB. It then makes sure our
code is saved to GitHub at that exact moment, and adds a line to `SUBMISSIONS.md`
saying which saved version produced that upload.

At the end of the day it also checks our final code ZIP: that it's under 8 MiB,
has no passwords, keys or installed software in it, and includes the run
instructions, the dependency list, the licence and the method write-up.

**Why it matters.** The leaderboard keeps our best upload, and the organisers
re-run that exact version before confirming any prize. If the file breaks a rule,
or we can't show which version of the code made it, we could lose first place
after winning it.

**When to use it:**
- every single time, just before uploading to the leaderboard (including the very first baseline upload)
- whenever someone asks "is our submission file OK?"
- when the leaderboard score comes back, to record it next to the right version
- at about 15:30, when packaging the final code ZIP
- one last time on the final file before the 16:15 upload

**Running the checker yourself** (from the repo folder):

```bash
python3 .claude/skills/submission-check/scripts/check_submission.py csv submission.csv --test-file <path to the test transactions file> --expected-rows 40101
```

```bash
python3 .claude/skills/submission-check/scripts/check_submission.py zip code.zip
```

"OK to upload" means you're good. "NOT READY" lists exactly what to fix.

---

## 2. XRPL traps: "Is the data being read correctly?"

**What it does.** The data has some sneaky features that can silently give wrong
numbers. Whenever someone writes or changes code that reads, joins or values the
data, Claude checks it against the known traps:

1. We score transactions, not the individual money "legs" inside them.
2. XRP amounts can be written in "drops", a million times smaller than one XRP.
3. A currency is its issuer plus its symbol: two different "USD"s are two different things.
4. "SendMax" is only a ceiling. What was really paid is in the first money leg.
5. Values in different currencies must be turned into US dollars using the hourly price list.
6. Some columns are JSON text that has to be unpacked first.

It also warns about things that look suspicious but are normal, like people
paying themselves to swap currencies, or busy hubs such as exchanges.

**Why it matters.** A model built on quietly wrong numbers can look great on the
training week and then fall apart on the test week. These are the mistakes the
organisers warned us about in the training.

**When to use it:**
- in the first hour, when writing the code that loads and joins the files
- any time someone changes how amounts, currencies or prices are worked out
- whenever a number looks surprisingly huge, tiny or perfect, for example a score of 0.99 from one simple feature
- before trusting a new feature that uses amounts or currencies

---

## 3. Decision log: "Write down what we tried and why."

**What it does.** Every time the team tries an idea, Claude adds a short entry to
a diary file called `DECISIONS.md`: what we changed, why we thought it would help,
our honest-test score before and after, the leaderboard score if we uploaded it,
and whether we kept or dropped the idea. It also keeps a list of where AI was
used, which the rules require us to disclose.

At the end of the day, Claude turns that diary into our method write-up
(`METHOD.md`, at most 4 pages), using only the real numbers we recorded.

**Why it matters.** The judges want to know *why* our model works, not only its
score. Writing a note as we go takes seconds. Trying to remember everything at
4pm, under pressure, is where teams forget things or report the wrong numbers.
The diary also stops us accidentally trying the same failed idea twice.

**When to use it:**
- right after setting up our honest test (to record how we split the data)
- every time a new feature, setting or model is tried, whether it worked or not
- every time we get a new honest-test score or leaderboard score
- whenever we use an AI tool for a piece of the work
- at about 15:30, to write the method write-up and the AI disclosure

---

## The day at a glance

| Time | Skill to use |
|---|---|
| 10:30, setting up and loading the data | **XRPL traps**, then **submission check** before uploading the baseline |
| 11:00, setting up the honest test | **Decision log** (record how we test) |
| 11:30 to 14:30, feature rounds | **XRPL traps** for any data code, **decision log** after every idea, **submission check** before every upload |
| 15:30, freeze | **Submission check** on the final file and ZIP, **decision log** to write `METHOD.md` |
| 16:15, final upload | **Submission check** one last time |
