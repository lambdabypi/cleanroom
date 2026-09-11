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
                                                      Claude writes extract()
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

---

## Why a bandit and not deep RL

The theme is self-improvement, and the honest engineering answer at this scale is
a **contextual bandit**, not policy-gradient fine-tuning.

| | Verdict |
|---|---|
| PPO / GRPO on a policy network | No reward dataset, no GPU budget, and — worst — a half-trained policy is invisible in a three-minute demo. |
| **Thompson sampling over a discrete strategy set** | Genuinely reinforcement learning. ~150 lines, CPU-only, converges in 15–30 episodes, and its state is a readable JSON file. |
| **Experiential memory (Reflexion-style)** | Carries the specific, textual lessons a numeric posterior cannot represent. |

Cleanroom runs the last two together, because they fail differently. The bandit
generalises *numerically* across pages of the same shape but cannot encode "this
site hides the price in a data attribute". Memory encodes exactly that but cannot
rank strategies. Running both is what makes the agent visibly improve *and* able
to quote its own past mistake.

**Reward is continuous**, not a coin flip, so the Beta posterior is updated with
fractional pseudo-counts (`alpha += r`, `beta += 1 - r`) — the standard treatment
for bounded rewards in [0, 1]. A `discount` of 0.98 pulls old evidence toward the
prior each update, so the agent can change its mind when a site changes shape.

### Where the reward comes from

This is the design decision everything else follows from. Human thumbs-up cannot
produce enough episodes in one day to move a posterior. So the **Daytona sandbox
is the reward function**: run the generated extractor, validate the rows it
returns, and the pass rate is ground truth. Five channels are blended:

| Channel | Weight | What it measures |
|---|---:|---|
| `validity` | 1.0 | rows satisfying the schema / rows returned |
| `coverage` | 0.7 | valid rows against the page's expected record count |
| `completeness` | 0.6 | non-empty cells / expected cells |
| `provenance` | 0.4 | rows carrying a source URL |
| `human` | 2.5 | reviewer verdict, sparse — dominates when present |

A crash scores exactly `0.0`. The multi-channel shape matters: validity alone
would let the agent win by emitting one perfect row and dropping the rest of the
page.

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
run would cost on a paid plan. The `$120` Agents bill would have been obvious
after two episodes with this in place — which is exactly why it exists.

### The efficiency loop

A second, independent contextual bandit chooses an **execution profile** per page
shape, where a profile is a spend budget:

| Profile | Document budget | Repairs |
|---|---|---|
| `lean` | 6k chars | 0 |
| `standard` | 18k chars | 1 |
| `thorough` | 45k chars | 2 |

Its reward is cost-penalised: `utility = value - LAMBDA * normalised_cost`. So
the strategy bandit learns *what works* while the profile bandit learns *what is
worth paying for*. On a simulated run the two land in different places for
different page shapes, which is the whole point:

```
bucket: table_heavy      bucket: prose
  lean      0.765          thorough  0.531
  standard  0.584          standard  0.386
  thorough  0.483          lean      0.322
```

The agent worked out that clean tables parse fine from a 6k excerpt — a 7x cut in
input tokens at no loss of quality — while prose pages genuinely need the context.

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

Two things the ledger caught about itself, both worth knowing:

- **Spend is cumulative per state directory**, so a demo inherits every earlier
  probe. `cleanroom costs` splits the total into *attributed to episodes* vs
  *probes and source gathering* — on one of our runs that read
  `$0.02 / $60.09`, which said immediately that the learning loop was never the
  expensive part. `--clear` resets it.
- **`doctor --live` used to cost $15 a run.** While the You.com Agents backend
  was configured, the code-writer probe was a real Agents call. A health check
  must never be the expensive thing, so it is now skipped unless you pass
  `--allow-paid`.

### The dashboard

`cleanroom ui` (needs `pip install -e ".[ui]"`) serves a Streamlit app with four
tabs: **Learning** (the same three-panel figure the video uses, plus posteriors
and self-written lessons), **Spend**, **Review**, and **Dataset**. It can also
run episodes directly, with progress streaming as they land.

**Review** is the half that earns its keep. Sandbox reward is automatic and
plentiful; a human verdict is scarce and weighted 2.5x, and until now the only
way to give one was to type an episode number. Clicking a thumb routes through
`learning.feedback` — the same code path as `cleanroom feedback`, so the two
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

### Adaptive execution when a tool fails

The same ledger feeds `ComponentHealth`, which tracks an EWMA success rate per
component and opens a circuit breaker after 3 consecutive failures, so the agent
stops retrying into a wall and falls back instead. EWMA rather than a lifetime
mean so a component that recovers is trusted again quickly — a tool should not be
punished for an outage that ended ten episodes ago.

```
┌───────────┬─────────┬───────────────────┬─────────┐
│ component │ success │ consecutive fails │ circuit │
├───────────┼─────────┼───────────────────┼─────────┤
│ you       │ 0.28    │ 3                 │ open    │
└───────────┴─────────┴───────────────────┴─────────┘
```

---

## Clean Data, enforced rather than claimed

[Clean Data](http://cleandata.world/) means data that is accurate, attributable,
consented, and free of personal information. Cleanroom treats that as executable,
not as a paragraph in a README:

- **Attribution is a reward channel.** Every row must carry the URL it came from;
  rows that don't reduce the score the agent is optimising.
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
| **One** | Credential layer, the `mem` lesson store, and the write-back that closes the loop. | `partners/one_client.py` |
| **CrewAI** | Source triage before the loop, and the publish gate after it. | `crew/crew.py` |
| **Claude Opus 5** | Writes and repairs `extract()`. Structured output; the schema and contract are cached across episodes. | `pipeline/synthesize.py` |

### Two gotchas handled up front

Both are documented by One and both fail *silently*, which is the worst kind:

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
git clone <your-repo-url>
cd cleanroom                # the pyproject.toml lives here, not in the parent

python -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[crew,ui,sandbox,viz,dev]"

cp .env.example .env        # then fill it in
npm i -g @withone/cli && one init
one add you && one add daytona && one add github

cleanroom doctor --live     # do this before anything else
```

Use a venv rather than a user install — CrewAI pulls a large dependency tree
(chromadb, lancedb, onnxruntime, pyarrow) and you do not want that in your system
Python. On Windows, if activation is blocked run
`Set-ExecutionPolicy -Scope Process RemoteSigned`, or just call the interpreter
directly as `.\.venv\Scripts\python.exe -m pytest`.

Verified against **crewai 1.15.21, daytona 0.198.0, streamlit 1.63.0,
anthropic 1.5.0, mcp 1.28.1** on Python 3.13.

`.env` keys:

| Key | Where to get it |
|---|---|
| `YOU_API_KEY` | [you.com/platform](https://you.com/platform) → API Keys ($100 free credit) |
| `CLEANROOM_LLM_*` | The code writer — any free OpenAI-compatible endpoint (see below) |
| `DAYTONA_API_KEY` | [app.daytona.io](https://app.daytona.io) → Billing → redeem `DAYTONA_HACKATHON_NYC_PF5W2GVE` |
| `ONE_SECRET`, `ONE_CONNECTION_KEYS` | [app.withone.ai](https://app.withone.ai/settings/api-keys) — coupon `YOU-NYC-PRO` |
| `ONE_PUBLISH_TARGET` | `owner/repo` the dataset is committed to |

`cleanroom doctor --live` makes one real call to each partner. Auth problems found
at hour one are cheap; at hour six they are fatal.

### Read this before picking a code writer

Measured on the You.com billing dashboard:

| Product | Price | Cost of one 20-episode run |
|---|---|---|
| Web Search API | $5 / 1000 calls | ~$0.01 |
| **Agent API — Advanced** | **$15 / call** | **$300–450** |

So: **retrieval on You.com is effectively free, and code generation on You.com is
not.** The Agents API works well technically — it produced correct, compiling
extractors at `verbosity: "high"` — but eight calls cost $120. `auto` will
therefore *never* select it, `build_synthesizer` refuses rather than silently
falling back to it, and the backend itself caps at 3 calls per process
(`YOU_AGENT_MAX_CALLS`).

The code writer is pluggable. Any OpenAI-compatible `/chat/completions` endpoint
works, which is where the usable free tiers are:

| Provider | `CLEANROOM_LLM_BASE_URL` | Model |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| Cerebras | `https://api.cerebras.ai/v1` | `llama-3.3-70b` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash` |
| OpenRouter | `https://openrouter.ai/api/v1` | any `...:free` model |
| Ollama (local) | `http://localhost:11434/v1` | `qwen2.5-coder:7b` |

Provider-native key names are auto-detected, so `GROQ_API_KEY=...` on its own is
enough — the base URL and a working model get filled in. Hosted catalogues churn
(`llama-3.3-70b-versatile` was decommissioned and returned a bare 404), so
**`cleanroom models`** lists what your key can actually reach and a 404 from the
writer includes that list in the error.

All backends receive byte-identical prompts, so switching provider mid-project
does not silently change the task and invalidate a run.

Free tiers are requests-per-minute limited. A 429 is a wait, not a failure: the
client honours `Retry-After` and backs off up to 3 times, and `--pause N` spaces
episodes out. A 413 (request over the tier's token cap) is handled by *halving
the document budget and retrying*, and writes a lesson so the profile bandit
learns that tier's ceiling.

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
| `--seed 42` | Repeatable arm sampling — pin it for the demo recording. |
| `--no-crew` | Skip CrewAI. Faster while iterating on prompts. |
| `--greedy` | Exploit only. Use *after* learning, to show the learned policy. |
| `--repairs 2` | More self-repair turns per episode. |
| `CLEANROOM_LOCAL_VALIDATE=1` | Validate in-process, skipping Daytona. Fast, but it runs model-written code with no isolation — development only. |

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

## Demo script (3 minutes)

1. `cleanroom doctor --live` — four partners green. **(15s)**
2. `cleanroom run -n 20 --seed 42` — narrate the live table. Early episodes pick
   strategies at random and score badly; watch repairs fire and scores climb. **(70s)**
3. `cleanroom report` then `cleanroom costs` — the posterior table, a stored
   lesson quoted verbatim, and the per-component spend table showing the agent
   learned to use a cheap profile where a cheap profile suffices. **(35s)**
4. `cleanroom curve` — the figure. Read **both** left panels: reward, and
   exploration collapsing onto the learned policy. **(20s)**
5. `cleanroom feedback 12 bad` then `cleanroom report` — one human verdict
   visibly moves the posterior and changes the best arm. **(20s)**
6. `cleanroom run -n 3 --publish --greedy` — the GitHub commit appears, and the
   call shows up in One's log at `app.withone.ai/logs`. **(25s)**

Step 6 is the one that matters most for judging: the agent changed a real system.

---

## Layout

```
src/cleanroom/
  config.py               env-backed settings + preflight
  cli.py                  doctor / run / report / curve / feedback
  learning/
    strategies.py         action space (5 arms) + page-shape context buckets
    bandit.py             contextual Thompson sampling, fractional + discounted
    budget.py             second bandit: execution profiles, cost-penalised reward
    reward.py             five-channel reward aggregation
    memory.py             One mem store, local JSONL fallback
    store.py              atomic posterior persistence + episode log
  observability/
    ledger.py             per-call latency/cost ledger + health circuit breaker
    pricing.py            price book (You.com figures from the dashboard)
  partners/
    you_client.py         Search + Contents, with base-URL probing
    daytona_env.py        sandbox lifecycle; reused across episodes
    one_client.py         four-tool loop + passthrough + publish
  pipeline/
    schema.py             schema loading and validation
    synthesize.py         Claude writes and repairs extract()
    validate.py           row scoring + PII rejection  (stdlib only)
    sandbox_runner.py     in-sandbox entry point       (stdlib only)
    dataset.py            accumulating CSV + provenance manifest
    episode.py            the loop
  crew/
    crew.py               Source Scout, Data Steward
    mcp_helpers.py        null-dropping + input hardening for One's MCP tools
  demo/plot_curve.py      the two-panel figure
schemas/                  target schemas
tests/
  test_learning.py        bandit, buckets, validation, reward
  test_integration_pieces.py
                          MCP hardening, schema checks, execution fallback
  test_observability.py   pricing, ledger, circuit breaker, efficiency bandit
state/                    posterior, episode log, dataset, memory  (gitignored)
```

`validate.py` and `sandbox_runner.py` are stdlib-only with no project-relative
imports, because they are uploaded into the sandbox and executed there. That
constraint is what lets local and sandboxed runs score *identically* instead of
quietly disagreeing about what "valid" means.

---

## Tests

```bash
pytest -q      # 75 tests, no credentials, no network, ~0.5s
```

They cover the property the whole demo rests on — given a genuinely better arm,
the bandit finds it — plus reward monotonicity, PII rejection, posterior
round-tripping, the credential-stripping in the MCP hardening layer, and the
$15/call Agents price (asserted so a careless edit breaks a test rather than a
budget).

---

## Submission description (200 words)

Building a structured dataset from the web means writing a bespoke parser per
source, and those parsers break silently when pages change. Teams either
maintain dozens of brittle scrapers or give up on freshness.

Cleanroom is an agent that *learns* to extract rather than being told how. Each
episode it classifies a page's shape, recalls what failed on similar pages, picks
a strategy by contextual Thompson sampling, has Claude Opus 5 write the
extractor, and runs that code in a Daytona sandbox. The validator's row pass-rate
is the reward — automatic, verifiable ground truth, generated hundreds of times
per run — which updates the per-shape posterior and writes a textual lesson.
Across a run, exploration visibly collapses onto the learned policy.

Stack: the **You.com** Search API (`extraction_mode: full_page`) and Contents API
supply live pages; **Daytona** is the execution environment and reward oracle;
**One** provides managed credentials, the `mem` lesson store, and the GitHub
write-back; **CrewAI** agents triage sources and gate publishing through One's
four-tool loop.

Clean Data is enforced rather than claimed: attribution is a reward channel, PII
is a validation failure, and a provenance manifest ships with every dataset. The
loop closes with a real commit.

---

## Honest limitations

- **The bandit's context is page *shape*, not page *identity*.** Two pricing
  tables with different DOM conventions land in the same bucket. Per-host
  learning is left to the memory channel.
- **Five arms is a small action space.** It converges fast, which is the point,
  but it cannot invent a strategy that isn't in the list.
- **Revisiting pages inflates apparent learning.** With more episodes than
  sources the loop cycles, and a page seen twice is easier the second time.
  Compare the curve against `--seed` runs with more sources before believing a
  number.
- **`expected_rows_per_page` is a hand-set constant** per schema, so the coverage
  channel is only as good as that guess.

## License

MIT
