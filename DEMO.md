# Demo script — 3 minutes

Rubric this is written against: **completed the loop**, technical implementation,
innovation, impact, presentation.

> Written for the 2026-09-11 submission. The narration has since been corrected:
> a controlled run on 2026-09-17 found **no significant advantage** for the
> bandit over picking strategies at random, so the lines that claimed a learned
> per-arm preference are gone. What is demonstrated — the loop, the verifiable
> reward, self-written lessons, attribution by construction, the real commit — is
> still the whole demo, and it is all you need. See
> [What it actually learned](README.md#what-it-actually-learned-and-what-it-did-not).

Have two terminals open in `cleanroom/` with the venv active, plus a browser tab
on `https://github.com/lambdabypi/cleanroom`.

---

## 0:00–0:20 — The problem (say, don't type)

> "Building a dataset from the web means writing a parser per source, and they
> break silently when pages change. Cleanroom is an agent that *learns* to
> extract instead of being told how — and every attempt gets a verifiable score."

Show `cleanroom doctor` on screen: You.com, the code writer, Daytona, One, all green.

```
cleanroom doctor
```

## 0:20–1:10 — The learning loop, live

```
cleanroom run -n 3 --greedy
```

**Use `--greedy`.** Without it the bandit explores, which means it deliberately
samples a strategy that may not fit and you get a `0.00` on camera. Greedy takes
the current best-posterior arm in each bucket instead, so episodes are more
likely to land at 0.9+. Frame it as what it is: *"greedy mode shows you the
policy it currently holds, with exploration switched off."*

Do **not** say it picks the right strategy every time — it doesn't, and a `0.00`
can still appear. If one does, that is a better talking point than a clean run:
the score is real, it came from code that actually executed, and the next slide
is the lesson the agent wrote about it.

Only one run at a time: Groq's free tier is rate-limited per minute, and two
concurrent runs will make each other wait.

Narrate while it scrolls:

- "You.com Search returns the page **with its markdown inline** — one call."
- "It classifies the page shape, recalls what failed on similar pages, and a
  **contextual bandit** picks one of five extraction strategies."
- "Groq writes a throwaway `extract()` function for that strategy."
- "**That code runs in a Daytona sandbox** — and the fraction of rows that pass
  schema validation *is* the reward. Automatic and verifiable: every episode
  scores itself against hundreds of individual row checks, with no human in the
  loop."
- Point at a low score: "that one scored badly, so it repairs and retries."

## 1:10–1:50 — What it recorded about itself

```
cleanroom report
```

- Point at the per-bucket posterior table: "one posterior per page shape per
  strategy, updated from the sandbox score, persisted as readable JSON after
  every episode." That is the mechanism, and it is what the table shows.
- **Do not claim a winner.** Three runs produced three different best arms from
  2–9 pulls each, and pooled over a 48-episode control `table_parse` is the
  *worst* of the five. If asked whether it works: *"the control says not
  measurably yet, and we can tell you exactly how many episodes it would take to
  find out — 214 per arm."* That answer is stronger than a posterior table,
  because most projects cannot answer it at all.
- Point at a stored lesson, read it aloud — **this is the money shot**:
  > *"heading_sections on thundercompute.com parsed cleanly but found zero rows —
  > the record boundary assumption is wrong for this layout."*
  "It wrote that itself, and the next attempt on a similar page gets it back."

```
cleanroom curve
```

Open the PNG. "Reward per episode on top; underneath, exploration collapsing onto
the learned policy — that second panel matters because reward isn't comparable
across page shapes."

## 1:50–2:20 — Learning to be *cheap*, and observability

```
cleanroom costs
```

- "Every external call is ledgered: latency, failure rate, spend per component."
- "A **second bandit** chooses how much to spend per page shape — document budget
  and repair allowance — scored on reward *minus* a cost penalty." Stop there.
  Its winner also reverses between runs, and because the score is cost-penalised
  a cheap profile winning says nothing about quality. Claiming "same quality for
  7x fewer tokens" is the one inference this design cannot support.
- "In-episode spend for a 24-episode run: **$0.0006**, measured, not estimated.
  We also measured that You.com's Agents API is $15/call — $120 for eight calls —
  so code generation deliberately runs on a free provider while retrieval stays
  on You.com at $0.005. `auto` will never select the paid writer."

## 2:20–2:50 — Closing the loop (the criterion listed first)

```
cleanroom run -n 2 --publish
```

Then switch to the browser and **refresh the repo**:

> "The agent just committed `data/gpu_cloud_pricing.csv` and its provenance
> manifest to a real GitHub repo — through One, using One's four-tool loop:
> discover the action, read its schema, then execute. That's the agent changing a
> real system, not just producing an answer."

Show `data/gpu_cloud_pricing.PROVENANCE.md` — every row attributed, PII screened.

## 2:50–3:00 — Close

> "Clean Data is enforced, not claimed: attribution is injected by the harness so
> it can't be missed, and PII is a validation failure. You.com for live
> observation, Daytona as the reward oracle, CrewAI for triage and the publish
> gate, One for credentials, action discovery and the write-back."

(One's `mem` lesson store is integrated but never started on this machine, so
lessons are in local JSONL. Say "credentials, action discovery and the
write-back" — not "memory".)

---

## Optional (if you have spare seconds)

```
cleanroom ui
```

The Review tab: click 👍/👎 on an episode and show the posterior move. That is the
*Preferences & Feedback* track in one gesture.

---

## Pre-flight

```
cleanroom doctor            # all green
cleanroom costs --clear     # so the spend table shows only the demo
```

Don't run more than ~8 episodes on camera — Groq's free tier is
requests-per-minute limited and you'll sit through a 30s backoff. The posteriors
and curve come from the full 24-episode run already on disk.
