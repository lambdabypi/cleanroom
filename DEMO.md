# Demo script — 3 minutes

Rubric this is written against: **completed the loop**, technical implementation,
innovation, impact, presentation.

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
cleanroom run -n 6 --pause 4
```

Narrate while it scrolls:

- "You.com Search returns the page **with its markdown inline** — one call."
- "It classifies the page shape, recalls what failed on similar pages, and a
  **contextual bandit** picks one of five extraction strategies."
- "Groq writes a throwaway `extract()` function for that strategy."
- "**That code runs in a Daytona sandbox** — and the fraction of rows that pass
  schema validation *is* the reward. Automatic, verifiable, hundreds of times a run."
- Point at a low score: "that one scored badly, so it repairs and retries."

## 1:10–1:50 — It actually improved

```
cleanroom report
```

- Point at the per-bucket posterior table: "it learned `table_parse` wins on
  table-heavy pages, `regex_fields` on prose."
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
- "A **second bandit** learns how much to spend per page shape. It worked out
  table pages parse fine from a 6k excerpt — 7x fewer tokens, same quality —
  while prose genuinely needs the full context."
- "Total for this run: half a cent. We measured that You.com's Agents API is
  $15/call, so code generation deliberately runs on a free provider while
  retrieval stays on You.com at $0.005."

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
> gate, One for credentials, memory and the write-back."

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
