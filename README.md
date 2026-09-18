# Cleanroom

**A self-improving web-to-clean-dataset ETL agent.**

Cleanroom builds a structured, fully attributed dataset out of the live web. It
does not know in advance how to read any given page, so it learns: it picks an
extraction strategy, writes the code, runs that code in a sandbox, scores the
rows that come out, and updates its beliefs about which strategy works on which
kind of page. Over a run, the fraction of rows that survive validation goes up.

The interesting part is not that an LLM can write a parser. It is that the agent
gets a **verifiable reward** for every attempt and uses it, so improvement is
measured rather than asserted.

```
You.com page ──► classify shape ──► recall past lessons ──► pick strategy (bandit)
                                                                    │
                                              an LLM writes extract()
                                                                    │
                                                     Daytona runs it in a sandbox
                                                                    │
                                            validator scores rows ──► REWARD
                                                                    │
                              ┌─────────────────────────────────────┤
                    update posterior                        write a lesson
                    (which strategy)                        (what went wrong)
                                                                    │
                                            valid rows ──► dataset ──► One ──► GitHub
```

~8,400 lines of Python, 143 tests that run in a few seconds with no credentials
and no network.

---

## Why a bandit and not deep RL

The theme is self-improvement, and the honest engineering answer at this scale is
a **contextual bandit**, not policy-gradient fine-tuning.

| | Verdict |
|---|---|
| PPO / GRPO on a policy network | No reward dataset, no GPU budget, and — worst — a half-trained policy is invisible in a three-minute demo. |
| **Thompson sampling over a discrete strategy set** | Genuinely reinforcement learning. ~130 lines, CPU-only, and its state is a readable JSON file. Converges in 15–30 episodes *in simulation*; on live pages outcome noise is about twice the strategy signal, so a run that size cannot show it — see [What it actually learned](#what-it-actually-learned-and-what-it-did-not). |
| **Experiential memory (Reflexion-style)** | Carries the specific, textual lessons a numeric posterior cannot represent. |

Cleanroom runs the last two together, because they fail differently. The bandit
generalises *numerically* across pages of the same shape but cannot encode "this
site hides the price in a data attribute". Memory encodes exactly that but cannot
rank strategies. Running both is what makes the agent visibly improve *and* able
to quote its own past mistake.

**Reward is continuous**, not a coin flip, so the Beta posterior is updated with
fractional pseudo-counts (`alpha += r`, `beta += 1 - r`) — the standard treatment
for bounded rewards in [0, 1]. Priors are `Beta(1, 1)`, i.e. uniform. A
`discount` of 0.98 pulls old evidence toward the prior each update, so the agent
can change its mind when a site changes shape.

### The action space and the context

Five **strategies** — the arms. Each is a different theory of where records live
on a page, injected into the code-writing prompt as a hint; the model still
writes the parser.

| Strategy | Theory of the page |
|---|---|
| `table_parse` | Parse markdown/HTML tables structurally, one row per table row |
| `heading_sections` | Split on headings, treat each section as one record |
| `regex_fields` | One tuned regex per schema field, scanned over the whole page |
| `label_value_pairs` | Detect `Label: value` and definition-list shapes |
| `list_items` | One record per bullet or numbered list item |

Four **buckets** — the context the posterior is keyed on. Assigned by cheap
deterministic counting, no model call, in this order (structural signals beat
textual ones, because a page with a table and some prose is still best handled as
a table):

| Bucket | Condition |
|---|---|
| `table_heavy` | ≥ 3 table rows |
| `list_heavy` | ≥ 5 list items, and more list items than headings |
| `sectioned` | ≥ 3 headings |
| `prose` | everything else |

So the strategy bandit holds 4 × 5 posteriors and the profile bandit 4 × 3. The
bucket is page *shape*, not page *identity* — which is a real limitation, and a
measured one; see [Honest limitations](#honest-limitations).

Before Thompson sampling takes over, an **exploration floor** (`min_pulls`,
default 1) takes any arm not yet tried that many times in the bucket. With five
arms and identical uniform priors the first draws are near-uniform, so without
the floor a bucket can miss an arm entirely over a short run. One pull is also
too few for a pool with wildly varying page difficulty — again, see the
limitations.

### Where the reward comes from

This is the design decision everything else follows from. Human thumbs-up cannot
produce enough episodes in one day to move a posterior. So the **Daytona sandbox
is the reward function**: run the generated extractor, validate the rows it
returns, and the pass rate is ground truth. Channels are blended:

| Channel | Weight | What it measures |
|---|---:|---|
| `validity` | 1.0 | rows satisfying the schema / rows returned |
| `coverage` | 0.7 | valid rows against the page's expected record count |
| `completeness` | 0.6 | non-empty cells / expected cells |
| `provenance` | 0.15 | a *check*, not an incentive — see below |
| `human` | 2.5 | reviewer verdict, sparse — dominates when present |

A crash scores exactly `0.0`. The multi-channel shape matters: validity alone
would let the agent win by emitting one perfect row and dropping the rest of the
page.

`provenance` used to be weighted 0.4, on the theory that the agent should be
*rewarded* for citing its sources. That was the wrong mechanism, and it cost a
whole run to learn why: the host already knows the source URL, so making the
generated code responsible for it just created a way to fail. Nine perfectly good
rows scored `0.00` because the model left the field `None`. The URL is now
injected by the validator, which makes attribution a structural guarantee rather
than something the agent might learn. The channel survives at low weight as an
alarm — if it ever drops below 1.0, injection is broken.

### What it actually learned, and what it did not

The mechanism runs end to end and reliably: retrieval, sandboxed execution,
scoring, credit assignment, lesson writing, and a real commit. Across the
committed snapshots every published row carries an http source URL (179/179 and
193/193), an independent PII scan of the output is clean, and a run costs
`$0.0006`–`$0.0023` of in-episode spend.

**No live run in this repo demonstrates a learned strategy preference.** Three
runs produced three different winners — `table_parse`, `list_items`,
`heading_sections` — each from 2–9 pulls. Pooled over the 48 episodes of the
control run, `table_parse`, which an earlier draft of this README presented as
the best arm, is the *worst* of the five (0.374 against 0.69–0.72).

A control ran on 2026-09-17: 24 episodes per arm, Thompson sampling against
uniform-random strategy selection, interleaved so both arms see the same page at
the same index, with the execution profile pinned so strategy choice is the only
difference. **No significant difference** — mean paired difference +0.021,
sign-flip p = 0.82, 10 of 24 pairs favouring the bandit.

That null is weak evidence rather than strong. Reward variance here is mostly
code generation, not policy: across the (page, strategy) cells that repeat, the
standard deviation *inside* a cell is 0.258 — the same page with the same
strategy returns `[0.918, 0.0]` — while the spread between strategy means is
0.133. At n=24 the design could only have detected a difference of about 0.3.
Detecting 0.10 would need ~214 episodes per arm. **The honest reading is "not
measured", not "no effect".**

What this repo does support about the learning itself: the strategy bandit
provably converges on a better arm **in simulation**
(`tests/test_observability.py`), with the exploration floor and the on-policy
cost gate measured separately over 20 seeds.

Check any of it without trusting this file:

```bash
python scripts/verify_run.py    runs/2026-09-16-paced-24
python scripts/compare_arms.py  runs/2026-09-17-control-haiku
```

Both are dependency-free and import nothing from `cleanroom`; they recompute
each number from the artifacts and print a `CAUTION` wherever the data is too
thin for the claim. [`runs/README.md`](runs/README.md) lists, per snapshot,
exactly which claims the artifacts support and which must not be cited.

---

## Observability, and learning to spend less

Every outbound call is wrapped in a ledger that records latency, outcome, tokens
and cost to `state/calls.jsonl`. `cleanroom costs` turns that into a per-component
report:

```
┌───────────┬───────┬───────┬───────┬───────┬─────────┬─────────────────────┐
│ component │ calls │ fail% │ p50 s │ p95 s │ total s │ est. $              │
├───────────┼───────┼───────┼───────┼───────┼─────────┼─────────────────────┤
│ writer    │ 40    │ 0%    │ 1.85  │ 3.10  │ 78      │ $0.1425 (free tier) │
│ you       │ 4     │ 75%   │ 0.62  │ 1.90  │ 3       │ $0.0050             │
│ daytona   │ 40    │ 0%    │ 4.20  │ 7.80  │ 171     │ $0.0048             │
└───────────┴───────┴───────┴───────┴───────┴─────────┴─────────────────────┘
estimated total $0.1522   likely billed $0.0098   (difference is free-tier usage)
```

Estimated and billed are reported separately so a free tier does not hide what a
run would cost on a paid plan.

### The efficiency loop

A second, independent contextual bandit chooses an **execution profile** per page
shape, where a profile is a spend budget:

| Profile | Document budget | Repairs |
|---|---|---|
| `lean` | 6k chars | 0 |
| `standard` | 18k chars | 1 |
| `thorough` | 45k chars | 2 |

Its reward is cost-penalised, so the strategy bandit learns *what works* while
the profile bandit learns *what is worth paying for*:

```
utility          = clamp(value - LAMBDA * normalised_cost, 0, 1)
normalised_cost  = min(1, input_tokens / REF_TOKENS
                          + max(0, llm_calls - 1) * REPAIR_COST_UNITS)
```

with `LAMBDA = 0.25`, `REF_TOKENS = 11,250` (the `thorough` budget in tokens, so
the denominator needs no knowledge of the provider) and `REPAIR_COST_UNITS = 0.5`
— a repair resends the document *and* the previous code, so it is not free.

The intent is that the two bandits land in different places for different page
shapes. **No committed run demonstrates that**, and an earlier version of this
section claimed otherwise. The profile posteriors reverse between runs — `lean`
best in the hackathon snapshot (0.645, n=2), `lean` *worst* in `clean-30` (0.462)
and in `paced-24` (0.333, n=1) — and the `prose` bucket has **zero** profile
pulls in every snapshot, so there is no prose column to report at all.

Two things to keep in mind when reading a profile table:

- **A cheap profile topping it is not evidence of equal quality.** The number is
  a utility, already penalised for spend, so `lean` can win on cost alone. That
  inference needs raw reward per profile, which is not what this posterior holds.
- **The on-policy gate below leaves very little data.** Measured across the
  snapshots, only 4 of 14, 10 of 30 and 10 of 24 episodes ever reached the
  profile bandit — 1–5 pulls per arm. `verify_run.py` prints a `CAUTION` on every
  snapshot for exactly this reason.

Cost is normalised in **tokens, not dollars**, because tokens are what the profile
controls and they are provider-independent: a posterior learned on Groq stays
valid after switching to Claude. Dollars are still recorded, they are just not the
learning signal. `CLEANROOM_COST_LAMBDA` (default `0.25`) is the exchange rate
between quality and spend, and it is a product decision rather than a fact.

**One subtlety that took a measurement to find.** Two bandits learning
simultaneously confound each other: while the strategy dimension is still
exploring, every profile scores badly, so the cheapest wins on cost alone and the
posterior can commit to `lean` for a page shape that needs `thorough`. The fix is
on-policy credit assignment for spend — the profile bandit only learns from
episodes that used the currently-best-known strategy. Measured over 10 seeds:

| Episodes | Ungated | Gated |
|---|---|---|
| 20 | 6/10 correct | 8/10 |
| 30 | 7/10 | **10/10** |
| 40 | 9/10 | 10/10 |
| 100 | 10/10 | 10/10 |

So the gate buys convergence *speed*, not asymptotic correctness — and a demo run
lives in exactly that 20–40 episode window.

Two things the ledger caught about itself:

- **Spend is cumulative per state directory**, so a demo inherits every earlier
  probe. `cleanroom costs` splits the total into *attributed to episodes* vs
  *probes and source gathering* — on one run that read `$0.02 / $60.09`, which
  said immediately that the learning loop was never the expensive part.
  `--clear` resets it.
- **`doctor --live` used to cost $15 a run.** While the You.com Agents backend
  was configured, the code-writer probe was a real Agents call. A health check
  must never be the expensive thing, so the paid writer is skipped unless you
  pass `--allow-paid`.

### Adaptive execution when a tool fails

The same ledger feeds `ComponentHealth`, which tracks an EWMA success rate per
component and opens a circuit breaker after 3 consecutive failures, so the agent
stops retrying into a wall and falls back instead. EWMA rather than a lifetime
mean so a component that recovers is trusted again quickly — a tool should not be
punished for an outage that ended ten episodes ago.

Three concrete adaptations, all triggered by real provider behaviour:

| Failure | Response |
|---|---|
| `429` per-minute limit | Honour `Retry-After` and back off, up to 5 attempts capped at 240s total. A 429 is a wait, not a failure — treating it as one threw away 6 of 8 episodes in testing. A token pacer also spaces calls against the provider's discovered per-minute ceiling, so the limit is avoided rather than bounced off. |
| `429` per-**day** limit | Abort, with the provider's own figures. A 429 does not say which limit it means and the `x-ratelimit-*` headers describe only the minute bucket — during a daily refusal they read `remaining-tokens: 8000, reset: 1ms`. Only the response body distinguishes them. Retrying a daily cap spends the whole retry budget and then logs the episode as a code-writer failure, which reads as the agent's fault. |
| `413` payload too large | **Halve the document budget and retry**, then write a lesson so the profile bandit learns that tier's ceiling. |
| `402` credits exhausted | Abort the run immediately. Grinding on would log 20 zero-reward episodes that look like a broken policy rather than a dead API key. |

### The dashboard

`cleanroom ui` serves a Streamlit app with four tabs: **Learning** (the same
three-panel figure the demo uses, plus posteriors and self-written lessons),
**Spend**, **Review**, and **Dataset**. It can also run episodes directly, with
progress streaming as they land.

**Review** is the half that earns its keep. Sandbox reward is automatic and
plentiful; a human verdict is scarce and weighted 2.5x, and otherwise the only
way to give one is to type an episode number. Clicking a thumb routes through
`learning/feedback.py` — the same code path as `cleanroom feedback`, so the two
interfaces cannot drift — and flagging an individual bad row stores a lesson
naming the offending values, which the next synthesis retrieves.

Row rejection also **proposes a schema constraint** that would have caught it:

```
Reviewer rejected a row from jarvislabs.ai: provider='10 hours',
gpu_model='26'. Reason: not a provider. Do not emit rows of this shape.

Suggested schema hardening:
  add to provider: "deny_pattern": "\b\d+\s*(?:hour|hr|day|week|month)s?\b"
```

That distinction matters: a lesson influences the next prompt, whereas a
constraint rejects the whole class of bad rows permanently.

---

## Clean Data, enforced rather than claimed

[Clean Data](http://cleandata.world/) means data that is accurate, attributable,
consented, and free of personal information. Cleanroom treats that as executable,
not as a paragraph in a README:

- **Attribution is structural.** The validator injects the source URL into every
  row from the page the harness actually fetched, so a row physically cannot
  reach the dataset unattributed.
- **PII is a validation failure.** `pipeline/validate.py` rejects any row
  containing an email address, phone number, or government-ID-shaped string, so
  it never reaches the dataset *and* costs the agent reward.
- **Provenance is recorded per source.** Every contributing page is logged with
  retrieval time and method, and `data/<name>.PROVENANCE.md` is published
  alongside the CSV.
- **Sourcing is public-web only**, via You.com's index — no scraping of
  authenticated or personal data.

---

## The stack

| Partner | Role | Where |
|---|---|---|
| **You.com** | Live observation. `POST /search` with `extraction_mode: full_page` gets ranking *and* page markdown in one call; `POST /contents` backfills pages that came back thin. | `partners/you_client.py` |
| **Daytona** | The environment and the reward oracle. Model-written code runs here and nowhere else. | `partners/daytona_env.py` |
| **One** | Credential layer, action discovery, and the GitHub write-back that closes the loop. The `mem` lesson store is integrated too, but its embedded Postgres (`pgserve`) never started on this machine, so every lesson fell back to local JSONL with `synced: false`. Do not read the lessons as being stored in One. | `partners/one_client.py` |
| **CrewAI** | Source triage before the loop, and the publish gate after it. | `crew/crew.py` |
| **The code writer** | Writes and repairs `extract()`. Pluggable: any OpenAI-compatible endpoint, Anthropic, or You.com Agents. | `pipeline/synthesize.py` |

### What integrating One actually took

One's four-tool loop (`list → search → knowledge → execute`) is the right shape,
and reading the action knowledge before executing is what surfaced most of the
following. Recorded here because none of it is guessable and all of it fails
confusingly:

| Symptom | Cause |
|---|---|
| `403` on a GitHub write | `owner`/`repo`/`path` are **path variables**. One's own action knowledge is blunt: *"Do NOT pass path variables in the -d body flag."* |
| `422 "sha" wasn't supplied` | Overwriting a file needs its current blob sha. A demo re-run hits this on the second publish. |
| `404` on a nested path | One matches passthrough routes **by path segment**, so a literal `/` inside a variable adds a segment and the route stops matching. Percent-encoding fixes `PUT`, but *not* `GET` — so sha lookups read the **git tree** instead, which takes a single-segment ref. |
| `The command line is too long` | npm's `one` shim is a `.cmd`, so it routes through cmd.exe and its 8191-byte limit. A base64 CSV blows past it. Writes therefore go over the **HTTP passthrough**, which has no such ceiling. |
| `UnicodeDecodeError` on Windows | `subprocess(text=True)` decodes with the locale codepage; One emits UTF-8 (typographic apostrophes in action titles). Encoding must be explicit. |
| Memory silently empty | `one mem add` takes `<type> <json>`, not raw text, with `--tags` as a CSV and an integer 1–10 weight. Getting it wrong returns non-zero and falls back to local storage with no visible error. |

Two more, documented by One and both silent:

1. **Daytona sandboxes block egress to `*.withone.ai`** (SNI inspection). The
   architecture keeps every One and You.com call on the host; the sandbox only
   ever receives a document and returns a report. It makes no network calls.
2. **CrewAI + One MCP needs input hardening.** CrewAI serialises unset optional
   params as explicit `null`, which One's schema validation rejects, and
   LLM-supplied arguments arrive as JSON strings often enough to matter.
   `crew/mcp_helpers.py` strips nulls, parses stringified objects, and drops any
   agent attempt to set `x-one-*` headers — those carry the credentials, so model
   output is treated as untrusted input.

---

## Setup

Requires Python 3.10+ and Node (for One's CLI and MCP server).

```bash
git clone https://github.com/lambdabypi/cleanroom
cd cleanroom                # the pyproject.toml lives here, not in the parent

python -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[crew,ui,sandbox,viz,dev]"

cp .env.example .env        # then fill it in
npm i -g @withone/cli && one init
one add github              # then `one list` to get your connection key

cleanroom doctor --live     # do this before anything else
```

Use a venv rather than a user install — CrewAI pulls a large dependency tree
(chromadb, lancedb, onnxruntime, pyarrow) and you do not want that in your system
Python. On Windows, if activation is blocked run
`Set-ExecutionPolicy -Scope Process RemoteSigned`, or just call the interpreter
directly as `.\.venv\Scripts\python.exe -m pytest`.

Verified against **crewai 1.15.21, crewai-tools 1.15.21, daytona 0.198.0,
streamlit 1.63.0, anthropic 1.5.0, mcp 1.28.1, One CLI 1.56.1** on Python 3.13.
The dependency floors are the versions actually tested, not the oldest that might
work — crewai 1.x reorganised enough that an optimistic pin is a trap.

`.env` keys:

| Key | Where to get it |
|---|---|
| `YOU_API_KEY` | [you.com/platform](https://you.com/platform) → API Keys |
| `GROQ_API_KEY` (or any provider key) | The code writer — see below. Provider-native names are auto-detected. |
| `DAYTONA_API_KEY` | [app.daytona.io](https://app.daytona.io) → Billing |
| `ONE_SECRET`, `ONE_CONNECTION_KEYS` | `one init`, then `one list` for the connection key |
| `ONE_PUBLISH_TARGET` | `owner/repo` the dataset is committed to |

`cleanroom doctor --live` makes one real call to each partner — including asking
the code writer for an actual extractor and checking that it compiles. Auth
problems found at hour one are cheap; at hour six they are fatal.

### Read this before picking a code writer

Measured on the You.com billing dashboard:

| Product | Price | Cost of one 20-episode run |
|---|---|---|
| Web Search API | $5 / 1000 calls | ~$0.01 |
| **Agent API — Advanced** | **$15 / call** | **$300–450** |

So: **retrieval on You.com is effectively free, and code generation on You.com is
not.** The Agents API works well technically — it produced correct, compiling
extractors at `verbosity: "high"`, while `"medium"` truncated them mid-function —
but eight calls cost $120. `auto` will therefore *never* select it,
`build_synthesizer` refuses rather than silently falling back to it, and the
backend itself caps at 3 calls per process (`YOU_AGENT_MAX_CALLS`).

The code writer is pluggable. Any OpenAI-compatible `/chat/completions` endpoint
works, which is where the usable free tiers are:

| Provider | `CLEANROOM_LLM_BASE_URL` | Model |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| Cerebras | `https://api.cerebras.ai/v1` | `llama-3.3-70b` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` |
| OpenRouter | `https://openrouter.ai/api/v1` | any `...:free` model |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5-coder:7b` |

Setting `GROQ_API_KEY` (or `CEREBRAS_`/`GEMINI_`/`OPENROUTER_`/`TOGETHER_`/
`DEEPSEEK_`) alone is enough — the base URL and a working model get filled in.
Hosted catalogues churn (`llama-3.3-70b-versatile` was decommissioned and
returned a bare 404), so **`cleanroom models`** lists what your key can actually
reach, and a 404 from the writer includes that list in the error.

All backends receive byte-identical prompts, so switching provider mid-project
does not silently change the task and invalidate a run. Output format differs by
backend for a measured reason: Claude gets a JSON schema because structured
output is native, while the others are asked for a fenced ```python block —
JSON-escaping a multi-line program roughly triples its token count and was what
triggered truncation.

---

## Running it

```bash
cleanroom run -n 20                    # the learning loop
cleanroom run -n 20 --publish          # ...and commit the result through One
cleanroom ui                           # dashboard: live run, charts, feedback buttons
cleanroom report                       # posteriors, lessons, dataset state
cleanroom costs                        # per-component latency, failures, spend
cleanroom costs --prices --ops         # ...with the price book and per-operation rows
cleanroom costs --clear                # reset the ledger before a demo run
cleanroom models                       # what your code-writer key can actually reach
cleanroom curve                        # learning-curve PNG for the demo
cleanroom feedback 7 good              # attach human judgement to episode 7
cleanroom feedback 7 bad               # ...verdicts are words: good | ok | bad
cleanroom strategies                   # show the action space
```

Verdicts are spelled out because a bare `-1` is impossible as a positional
argument — any CLI parser reads the leading dash as an option name.

Useful flags:

| Flag | Why |
|---|---|
| `--greedy` | Exploit only. Use *after* learning, and for demos — exploration deliberately samples a wrong strategy now and then, which looks like a bug on camera. |
| `--pause N` | Seconds between episodes. Free tiers are per-minute rate limited. |
| `--seed 42` | Repeatable arm sampling. |
| `--no-crew` | Skip CrewAI triage. Faster while iterating on prompts. |
| `--repairs 2` | More self-repair turns per episode. |
| `CLEANROOM_LOCAL_VALIDATE=1` | Validate in-process, skipping Daytona. Fast, but it runs model-written code with no isolation — development only. |

Posteriors, the profile bandit and the dataset are all written **after every
episode**, not at the end of the run. Saving only at the end meant an interrupted
run threw away everything it had learned and published an empty CSV, because the
rows were still in memory.

Run only one `cleanroom run` at a time — two concurrent runs compete for the same
free-tier rate limit and make each other wait.

### Targeting different data

A run is defined entirely by a schema in `schemas/`. Copy
`gpu_cloud_pricing.json`, change the fields and the `search.topic`, and the whole
pipeline retargets — no code changes. Schemas are checked at load time, including
two rules that exist because violating them silently flatlines the reward: at
least one field must be `required`, and one must be a URL for attribution.

**Make the constraints strict.** This is the single highest-leverage thing in a
new schema, and it is easy to get wrong. A field typed `string` with no
`pattern`/`deny_pattern` accepts anything, so an extractor that reads a page's
*cost-example* table instead of its *pricing* table scores just as well as one
that reads the right table. Observed on a real run before tightening:

```
provider          gpu_model   usd_per_hour   ← scored 0.92
"10 hours"        "26"        215.20
"1 hour"          "2"          21.52
```

Adding `deny_pattern` on `provider`, a required `pattern` on `gpu_model`, and a
sane `max` on `usd_per_hour` rejected 19% of that dataset — precisely the junk —
while keeping every genuine row. Available per-field constraints: `required`,
`min`, `max`, `max_length`, `enum`, `format: url`, `pattern`, `deny_pattern`.

---

## Demo

See **[DEMO.md](DEMO.md)** for a 3-minute script with exact commands, timings and
narration, written against the judging criteria.

---

## Layout

```
src/cleanroom/
  config.py               env-backed settings, provider auto-detection, preflight
  cli.py                  doctor / run / report / costs / curve / ui / models / feedback
  learning/
    strategies.py         action space (5 arms) + page-shape context buckets
    bandit.py             contextual Thompson sampling, fractional + discounted
    budget.py             second bandit: execution profiles, cost-penalised reward
    reward.py             multi-channel reward aggregation
    memory.py             One `mem` lesson store, local JSONL fallback
    feedback.py           human verdicts + row rejection, shared by CLI and UI
    store.py              atomic per-episode persistence + episode log
  observability/
    ledger.py             per-call latency/cost ledger + health circuit breaker
    pricing.py            price book (You.com figures from the dashboard)
  partners/
    you_client.py         Search + Contents, with base-URL probing
    daytona_env.py        sandbox lifecycle; reused across episodes
    one_client.py         four-tool loop, HTTP passthrough, publish + sha lookup
  pipeline/
    schema.py             schema loading and validation
    synthesize.py         shared prompts + backend factory
    openai_compat.py      any OpenAI-compatible endpoint (the default)
    you_agent.py          You.com Agents backend, with a hard spend cap
    validate.py           row scoring, URL injection, PII rejection  (stdlib only)
    sandbox_runner.py     in-sandbox entry point                     (stdlib only)
    dataset.py            accumulating CSV + provenance manifest
    episode.py            the loop
  crew/
    crew.py               Source Scout, Data Steward
    mcp_helpers.py        null-dropping + input hardening for One's MCP tools
  demo/plot_curve.py      the three-panel figure
  ui/app.py               Streamlit dashboard
schemas/                  target schemas
tests/
  test_learning.py        bandit, buckets, validation, reward, URL injection
  test_integration_pieces.py   MCP hardening, schema checks, execution fallback
  test_observability.py   pricing, ledger, circuit breaker, efficiency bandit
  test_feedback.py        verdict parsing, posterior updates, constraint suggestion
  test_viability.py       source screen, exploration floor, token pacer,
                          per-minute vs per-day 429s, per-model request shapes
scripts/
  verify_run.py           recompute one snapshot's claims   (no dependencies)
  compare_arms.py         recompute the bandit-vs-random control  (no dependencies)
runs/                     committed state snapshots + what each one supports
state/                    posteriors, episode log, dataset, memory, ledger  (gitignored)
```

`validate.py` and `sandbox_runner.py` are stdlib-only with no project-relative
imports, because they are uploaded into the sandbox and executed there. That
constraint is what lets local and sandboxed runs score *identically* instead of
quietly disagreeing about what "valid" means.

---

## Tests

```bash
pytest -q      # 143 tests, no credentials, no network, ~3s
```

They cover the property the whole demo rests on — given a genuinely better arm,
the bandit finds it — plus reward monotonicity, PII rejection, source-URL
injection, posterior round-tripping, the credential-stripping in the MCP
hardening layer, the two-bandit confounding gate, and the $15/call Agents price
(asserted so a careless edit breaks a test rather than a budget).

Several exist because a real run failed and the unit tests were green anyway:
the token pacer is driven through `_post_with_backoff` rather than in isolation
(testing the collaborator alone missed a `NameError` on the first live call),
the per-day/per-minute 429 split uses response bodies captured verbatim from the
provider, and the per-model request shape is asserted against a fake client
because that failure mode is a *paid* call that fails.

---

## Submission description (200 words)

Building a structured dataset from the web means writing a bespoke parser per
source, and those parsers break silently when pages change. Teams either
maintain dozens of brittle scrapers or give up on freshness.

Cleanroom is an agent that *learns* to extract rather than being told how. Each
episode it classifies a page's shape, recalls what failed on similar pages, picks
a strategy by contextual Thompson sampling, has an LLM write the extractor, and
runs that code in a Daytona sandbox. The validator's row pass-rate is the reward —
automatic, verifiable ground truth, generated hundreds of times per run — which
updates the per-shape posterior and writes a textual lesson. Across a run,
exploration visibly collapses onto the learned policy.

Stack: the **You.com** Search API (`extraction_mode: full_page`) and Contents API
supply live pages; **Daytona** is the execution environment and reward oracle;
**One** provides managed credentials, the `mem` lesson store, and the GitHub
write-back through its four-tool loop; **CrewAI** agents triage sources and gate
publishing.

Clean Data is enforced rather than claimed: attribution is injected so a row
cannot be unsourced, PII is a validation failure, and a provenance manifest ships
with every dataset. The loop closes with a real commit.

---

## Tunables

Every number the behaviour depends on, in one place, with where it lives. The
defaults are what produced the committed snapshots.

| Constant | Default | Where | What it does |
|---|---:|---|---|
| `WEIGHTS` | 1.0 / 0.7 / 0.6 / 0.15 / 2.5 | `learning/reward.py` | channel weights: validity, coverage, completeness, provenance, human |
| `HUMAN_SCALE` | `-1→0.0, 0→0.5, 1→1.0` | `learning/reward.py` | reviewer verdict mapped into [0, 1] |
| `PRIOR_ALPHA` / `PRIOR_BETA` | 1.0 / 1.0 | `learning/bandit.py` | uniform `Beta(1,1)` prior on every arm |
| `discount` | 0.98 | `learning/store.py` | shrinks old evidence toward the prior each update (the `bandit.py` class default is 1.0; both bandits are constructed with 0.98) |
| `min_pulls` | 1 | `learning/store.py` | cold-start floor: try each arm this often per bucket before sampling |
| `PROFILES` | 6k/0, 18k/1, 45k/2 | `learning/budget.py` | `lean` / `standard` / `thorough` — document budget and repair allowance |
| `LAMBDA` | 0.25 | `learning/budget.py` | quality-per-token exchange rate. **`CLEANROOM_COST_LAMBDA`** |
| `REF_TOKENS` | 11,250 | `learning/budget.py` | cost denominator = the `thorough` budget in tokens |
| `REPAIR_COST_UNITS` | 0.5 | `learning/budget.py` | cost charged per LLM call beyond the first |
| `REPAIR_THRESHOLD` | 0.55 | `pipeline/episode.py` | reward below this triggers a repair turn, if the profile allows one |
| `MIN_USEFUL_CHARS` | 400 | `pipeline/viability.py` | shorter fetches are screened out before an episode is spent |
| `DOC_SHARE_OF_TPM` | 0.35 | `pipeline/openai_compat.py` | share of the provider's per-minute token budget the document may use |
| `MAX_RATE_LIMIT_RETRIES` | 5 | `pipeline/openai_compat.py` | attempts on a per-minute 429 |
| `DEFAULT_BACKOFF_S` | 20.0 | `pipeline/openai_compat.py` | backoff when the provider advises nothing usable |
| `MAX_TOTAL_BACKOFF_S` | 240.0 | `pipeline/openai_compat.py` | total wait for one completion before failing the episode |
| `REQUEST_TIMEOUT` | 180 | `pipeline/openai_compat.py` | per-request timeout, seconds |
| `daytona_auto_stop_minutes` | 15 | `config.py` | idle sandbox teardown |
| `expected_rows_per_page` | per schema | `schemas/*.json` | denominator for the coverage channel; a hand-set guess |

Environment overrides worth knowing: **`CLEANROOM_COST_LAMBDA`**,
**`CLEANROOM_SYNTH_BACKEND`** (`auto` / `compat` / `anthropic` / `you`),
**`CLEANROOM_MODEL`**, **`CLEANROOM_ANTHROPIC_WORKSPACE_ID`** (required if the
Anthropic key is an all-workspaces key — it decides which workspace is billed),
**`CLEANROOM_STATE_DIR`**, and **`CLEANROOM_LOCAL_VALIDATE`** (skip Daytona and
run extractors locally — unsafe, for offline development only).

---

## Honest limitations

- **No live run is large enough to show the bandit works.** The 2026-09-17
  control found no significant difference from uniform-random selection, and was
  far too underpowered to settle it either way (~214 episodes/arm needed to
  detect a 0.10 reward difference; every run here is 14–30 episodes). Treat the
  learning claims as demonstrated in simulation only.
- **The bandit's context is page *shape*, not page *identity*.** Two pricing
  tables with different DOM conventions land in the same bucket. Per-host
  learning is left to the memory channel. Measured consequence: in one control
  arm the bandit scored 0.921 on a page at episode 1 and 0.000 on that same page
  at episode 9 — it cannot represent "this strategy for this host".
- **The cold-start floor is one pull per arm, which is too few for this pool.**
  Per-page difficulty spans 0.306–0.923, so a single trial says more about which
  page came up than which strategy was chosen. In one control arm the best
  available strategy was tried once, on the hardest page, scored 0.000, and was
  effectively discarded; uniform random drew it five times and scored ~0.92.
- **The cost/profile bandit is the least supported thing here.** Its on-policy
  gate only credits episodes that used the currently-best-known strategy, which
  discards most of a run: 4 of 14, 10 of 30, 10 of 24 episodes reached it across
  the three snapshots. That leaves 1–5 pulls per arm, the winner reverses between
  runs (`lean` → `standard` → `thorough`), and because the utilities are
  cost-penalised a cheap profile winning is not evidence of equal quality.
  `verify_run.py` says so on every snapshot.
- **Five arms is a small action space.** It converges fast, which is the point,
  but it cannot invent a strategy that isn't in the list.
- **Revisiting pages inflates apparent learning.** With more episodes than
  sources the loop cycles, and a page seen twice is easier the second time.
- **Raw reward is not comparable across page shapes** — a dense table can reach
  0.95 while prose tops out near 0.70 — so an aggregate reward curve can read
  flat even when the agent learned the right arm in every bucket. That is exactly
  why the figure has a second, bucket-agnostic convergence panel; read both.
- **`expected_rows_per_page` is a hand-set constant** per schema, so the coverage
  channel is only as good as that guess.
- **The CrewAI Data Steward publish gate has been exercised less** than the
  direct publish path. `--publish` falls back to a direct One call if the crew
  path fails.

## License

MIT
