---
name: decision-log
description: Keeps the team's decision log for the Decode & Detect XRPL AML hackathon and turns it into the method write-up. Use this whenever the team tries an idea or changes features, validation, model or settings, gets a new honest-test (cross-validation) or leaderboard score, decides to keep or drop something, uses an AI tool for part of the work, or asks for the method write-up, report, slides or AI disclosure - even if they don't mention a log.
---

# Decision log

The judges want to know *why* a model works, not only its score, and the rules
require disclosing AI use (model, provider, prompts, settings). Writing this down
as you go takes seconds; reconstructing it at 16:00 under pressure is where teams
lose marks or misreport results. The log also stops the team repeating ideas that
already failed.

## Logging (all day)

Keep `DECISIONS.md` in the repo root. Create it from this skeleton if missing:

```markdown
# Decision log

Team: <names>   Honest test: <how it is split, e.g. 5-fold GroupKFold by connected wallets>

## Decisions

## AI use
```

Append one entry under **Decisions** each time something is tried:

```markdown
### HH:MM — <short title>
- **Change:** what was added, removed or changed
- **Why:** the reasoning, ideally linked to a laundering signal (fan-in, pass-through, cycle...) or a data trap
- **Honest test:** AP before → after (mean ± spread across folds)
- **Leaderboard:** score and commit, if it was uploaded
- **Decision:** kept / dropped / needs more work, and why
```

Record what was actually measured. If a number isn't known yet, write "not
measured" rather than estimating, because the write-up will quote these numbers
as results. Log failed ideas too: "what didn't work" is part of a good method.

Under **AI use**, add a line whenever an AI tool did part of the work: the tool,
model and provider (for example "Claude Code, Claude Opus 5.5, Anthropic"), what
it was used for, and where its prompts live (for example the skill files in
`.claude/skills/`). Keep the actual prompts or skill files in the repo, since the
rules ask for them.

## The method write-up (from about 15:30)

Write `METHOD.md` from `DECISIONS.md`, at most 4 pages (or turn it into at most 10
slides if the team prefers). Use only facts and numbers from the log and the
code; where something is missing, ask the team rather than inventing it.

```markdown
# <Team name>: method

## Problem and data
## How we handled the data (traps, USD values, one row per transaction)
## How we tested honestly (the split, and why)
## Features, grouped, each with why it should work
## Model and settings (including seeds)
## Results (honest test and leaderboard, with the final commit)
## What didn't work
## Limitations (false positives, synthetic data, ambiguous cases)
## Pre-existing components and AI disclosure (libraries, templates, AI tools: model, provider, prompts, settings)
## How to reproduce (exact command)
```

Keep the language plain and the claims modest: this is synthetic data, and a high
score here doesn't prove a method works on real financial crime.
