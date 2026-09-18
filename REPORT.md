# Cleanroom — architecture and status report

**As of 2026-09-18.** Covers what the system is, how it is built, what the
evidence actually supports, and everything that changed on 2026-09-17/18.

Every number here is reproducible from committed artifacts. Two dependency-free
scripts recompute them and import nothing from the project:

```bash
python scripts/verify_run.py    runs/2026-09-16-paced-24
python scripts/compare_arms.py  runs/2026-09-17-control-haiku
```

---

## 1. Status in one page

**What Cleanroom is:** an ETL agent that turns live web pages into a structured,
attributed, PII-screened dataset. It does not ship a parser per source. It picks
an extraction strategy, has an LLM write `extract()` for it, runs that code in a
sandbox, and scores the rows that come back against a schema. That pass rate is
the reward.

**What is demonstrated:**

| Claim | Evidence |
|---|---|
| The loop runs end to end, reliably | 24/24 and 30/30 scored episodes; zero provider failures in the latest runs |
| Every published row is attributed | 179/179 and 193/193 rows carry an http source URL, verified from `dataset.csv` |
| PII never reaches the dataset | Independent scan (regexes restated inside the verifier) comes back clean |
| Cost is measured, not estimated | `$0.0006` and `$0.0023` of in-episode spend; `$0.4373` for the 48-episode control |
| Lessons are written and retrieved | 24/24, 29/30, and 13/14 scored episodes retrieved a prior lesson |
| The bandit converges **in simulation** | 20/20 seeds at 15 episodes with floor + gate (`tests/test_observability.py`) |

**What is not demonstrated — and this is the headline finding:**

A controlled run on 2026-09-17 (24 episodes/arm, interleaved, profile pinned)
found **no significant advantage** for Thompson sampling over uniform-random
strategy selection: mean paired difference **+0.021**, sign-flip **p = 0.82**,
10 of 24 pairs favouring the bandit.

The null is *uninformative rather than negative*, and that distinction is the
most useful result of the whole exercise:

- Within a repeated `(page, strategy)` cell the reward standard deviation is
  **0.258** — the same page with the same strategy returns `[0.918, 0.0]`.
- Between strategy means it is only **0.133**.
- Code-generation noise is roughly **twice** the policy signal, so at n=24 the
  design could only have detected a difference of about 0.3.
- 80% power at α=0.05 needs **214 episodes/arm** for a 0.10 difference, 853 for
  0.05. Every live run in this repo is 14–30 episodes.

So the correct statement is **"not measured"**, never "no effect".

---

## 2. Architecture

### 2.1 Episode flow

One episode is one attempt at one page:

```
You.com Search (extraction_mode: full_page)
        │  page markdown, inline with the ranking
        ▼
viability screen ─────────── drop pages with no evidence of required fields
        │                    (and write a lesson about that host)
        ▼
featurise → bucket          cheap deterministic counting, no model call
        │
        ├─► memory.search(host + bucket)          recall prior lessons
        ├─► strategy = bandit.select(bucket)      5 arms, Thompson + floor
        └─► profile  = profiles.select(bucket)    3 budgets, cost-penalised
        │
        ▼
LLM writes extract()        prompt = contract + schema + strategy hint + lessons
        │                   clipped to the profile's document budget
        ▼
Daytona sandbox             runs the code. No network. Document in, report out.
        │
        ▼
validator                   per-row schema check, source-URL injection,
        │                   PII rejection → reward channels
        ▼
reward ─┬─► bandit.update(bucket, strategy, reward)
        ├─► profiles.update(...)   only if strategy == best-known (on-policy gate)
        ├─► memory.add(lesson)     textual, describing what failed and why
        └─► dataset + provenance, persisted after EVERY episode
                    │
                    ▼
              One → GitHub commit (CSV + PROVENANCE.md)
```

Two things about this shape are load-bearing:

1. **The sandbox is the reward function.** Not an LLM judge, not a human. The
   agent gets ground truth on every attempt because the code either produces
   schema-valid rows or it does not. This is what makes any claim here checkable.
2. **The sandbox makes no network calls.** It receives a document and returns a
   report. Every credentialled call (You.com, One) stays on the host. This was
   forced by a real constraint — Daytona blocks egress to `*.withone.ai` via SNI
   inspection — and it is the right architecture anyway.

### 2.2 Module layout

`src/cleanroom/`, 6,755 lines. Tests and scripts bring the repo to ~8,400.

| Module | Lines | Responsibility |
|---|---:|---|
| `cli.py` | 702 | `doctor` / `run` / `report` / `costs` / `curve` / `ui` / `models` / `feedback` / `strategies` |
| `config.py` | 202 | env-backed settings, provider auto-detection, preflight checks |
| **learning/** | | |
| `strategies.py` | 103 | the action space (5 arms) and the page-shape context (4 buckets) |
| `bandit.py` | 155 | contextual Thompson sampling; fractional updates, discount, cold-start floor |
| `budget.py` | 104 | second bandit over execution profiles; cost-penalised utility |
| `reward.py` | 119 | multi-channel reward aggregation |
| `memory.py` | 291 | lesson store: One `mem` with local JSONL fallback, process-wide breaker |
| `feedback.py` | 160 | human verdicts and row rejection, shared by CLI and UI |
| `store.py` | 132 | atomic per-episode persistence, episode log |
| **pipeline/** | | |
| `episode.py` | 586 | the loop; credit assignment; the on-policy cost gate |
| `synthesize.py` | 450 | shared prompts, the Anthropic backend, backend factory |
| `openai_compat.py` | 444 | any OpenAI-compatible endpoint; token pacer; 429 classification |
| `you_agent.py` | 222 | You.com Agents backend, with a hard spend cap |
| `validate.py` | 207 | row scoring, source-URL injection, PII rejection — **stdlib only** |
| `viability.py` | 101 | source screening against schema constraints |
| `dataset.py` | 156 | accumulating CSV and provenance manifest |
| `schema.py` | 91 | schema loading and validation |
| `sandbox_runner.py` | 95 | in-sandbox entry point — **stdlib only** |
| **partners/** | | |
| `one_client.py` | 462 | four-tool loop, HTTP passthrough, publish, sha lookup |
| `daytona_env.py` | 203 | sandbox lifecycle, reused across episodes |
| `you_client.py` | 198 | Search + Contents, with base-URL probing |
| **observability/** | | |
| `ledger.py` | 229 | per-call latency/outcome/cost; `ComponentHealth` circuit breaker |
| `pricing.py` | 112 | price book, free-tier detection |
| **crew/** | | |
| `crew.py` | 237 | Source Scout (triage) and Data Steward (publish gate) |
| `mcp_helpers.py` | 103 | null-dropping and input hardening for One's MCP tools |
| **demo/**, **ui/** | 414, 363 | three-panel figure; Streamlit dashboard |

`validate.py` and `sandbox_runner.py` are stdlib-only with no project-relative
imports, because they are uploaded into the sandbox and executed there. That
constraint is what lets local and sandboxed runs score *identically* rather than
quietly disagreeing about what "valid" means.

### 2.3 The action space and the context

Five strategies — each a different theory of where records live on a page,
injected into the prompt as a hint. The model still writes the parser.

| Strategy | Theory |
|---|---|
| `table_parse` | markdown/HTML tables, one row per table row |
| `heading_sections` | split on headings, each section is a record |
| `regex_fields` | one tuned regex per schema field over the whole page |
| `label_value_pairs` | `Label: value` and definition-list shapes |
| `list_items` | one record per bullet or numbered item |

Four buckets, assigned by deterministic counting in this order (structural
signals beat textual ones — a page with a table and some prose is still best
handled as a table):

| Bucket | Condition |
|---|---|
| `table_heavy` | ≥ 3 table rows |
| `list_heavy` | ≥ 5 list items, and more list items than headings |
| `sectioned` | ≥ 3 headings |
| `prose` | everything else |

That gives 4 × 5 strategy posteriors and 4 × 3 profile posteriors.

### 2.4 Reward

Sandbox execution produces the reward. Channels are blended:

| Channel | Weight | Measures |
|---|---:|---|
| `validity` | 1.0 | rows satisfying the schema / rows returned |
| `coverage` | 0.7 | valid rows against `expected_rows_per_page` |
| `completeness` | 0.6 | non-empty cells / expected cells |
| `provenance` | 0.15 | a *check*, not an incentive |
| `human` | 2.5 | reviewer verdict, sparse, dominates when present |

A crash scores exactly `0.0`. Multi-channel matters: validity alone would let the
agent win by emitting one perfect row and dropping the rest of the page.

`provenance` is weighted low deliberately. It was 0.4 on the theory that the
agent should be *rewarded* for citing sources; that cost a whole run to unlearn.
Nine good rows scored `0.00` because the model left the field `None`. The URL is
now injected by the validator, making attribution structural rather than
something the agent might learn. The channel survives as an alarm — if it drops
below 1.0, injection is broken.

### 2.5 The two bandits

**Strategy bandit.** Beta posteriors per `(bucket, arm)`, updated with fractional
pseudo-counts (`alpha += r`, `beta += 1 - r`) because reward is continuous, not a
coin flip. `Beta(1,1)` priors. A discount of 0.98 shrinks old evidence toward the
prior each update so the agent can change its mind when a site changes shape. A
cold-start floor (`min_pulls = 1`) takes any untried arm before sampling begins.

**Profile bandit.** Chooses a spend budget per page shape:

| Profile | Document budget | Repairs |
|---|---|---|
| `lean` | 6k chars | 0 |
| `standard` | 18k chars | 1 |
| `thorough` | 45k chars | 2 |

Scored on cost-penalised utility:

```
utility         = clamp(value - LAMBDA * normalised_cost, 0, 1)
normalised_cost = min(1, input_tokens / REF_TOKENS
                        + max(0, llm_calls - 1) * REPAIR_COST_UNITS)
```

`LAMBDA = 0.25`, `REF_TOKENS = 11,250` (the `thorough` budget in tokens, so the
denominator needs no knowledge of the provider), `REPAIR_COST_UNITS = 0.5`.
Cost is normalised in **tokens, not dollars**, so a posterior learned on one
provider stays valid after switching.

**The on-policy gate.** Two bandits learning at once confound each other: while
the strategy dimension explores, every profile scores badly, so the cheapest wins
on cost alone and the posterior commits early. The profile bandit therefore only
learns from episodes that used the currently-best-known strategy. Measured over
20 seeds in simulation (correct answer `thorough`):

| Episodes | no floor, ungated | no floor, gated | floor, ungated | floor, gated |
|---:|---:|---:|---:|---:|
| 15 | 11/20 | 15/20 | 17/20 | **20/20** |
| 30 | 15/20 | 19/20 | 19/20 | **20/20** |
| 100 | 19/20 | 20/20 | 20/20 | **20/20** |

The floor is the larger single lever at short horizons; the gate adds on top.
Both buy convergence *speed*, not asymptotic correctness. **This is a simulation
with stationary arms, which live pages are not.**

The gate has a cost that only shows up live: it discards most of a run. Measured
across the snapshots, only **4 of 14, 10 of 30, and 10 of 24** episodes ever
reached the profile bandit, leaving 1–5 pulls per arm.

### 2.6 Partner integration

| Partner | Role |
|---|---|
| **You.com** | Live observation. `POST /search` with `extraction_mode: full_page` returns ranking *and* page markdown in one call; `/contents` backfills thin pages |
| **Daytona** | The execution environment and the reward oracle. Model-written code runs here and nowhere else |
| **One** | Managed credentials, action discovery, and the GitHub write-back, via its `list → search → knowledge → execute` loop |
| **CrewAI** | Source Scout triages sources before the loop; Data Steward gates publishing |
| **Code writer** | Pluggable: any OpenAI-compatible endpoint, Anthropic, or You.com Agents |

**One's `mem` lesson store is integrated but did not run here.** Its embedded
Postgres (`pgserve`) would not start, so every write failed after ~30s and fell
back to local JSONL with `synced: false`. A process-wide breaker now stops
retrying after the first failure. Do not read the lessons as being stored in One.

Integration details worth keeping, because none are guessable:

- `owner`/`repo`/`path` are **path variables**; passing them in the body gives a 403.
- Overwriting a file needs its current blob sha — a second publish hits this.
- One matches passthrough routes **by path segment**, so a literal `/` inside a
  variable adds a segment and the route stops matching. Percent-encoding fixes
  `PUT` but not `GET`, so sha lookups read the git tree instead.
- npm's `one` shim is a `.cmd`, so it routes through cmd.exe and its 8191-byte
  limit. A base64 CSV blows past it; writes go over the HTTP passthrough.
- `subprocess(text=True)` decodes with the locale codepage; One emits UTF-8.
- CrewAI serialises unset optional params as explicit `null`, which One rejects,
  and LLM-supplied arguments arrive as JSON strings often enough to matter.
  `mcp_helpers.py` strips nulls, parses stringified objects, and drops any agent
  attempt to set `x-one-*` headers — those carry credentials, so model output is
  treated as untrusted input.

### 2.7 Clean Data, enforced

- **Attribution is structural.** The validator injects the source URL from the
  page the harness fetched, so a row physically cannot reach the dataset
  unattributed.
- **PII is a validation failure.** Email, US phone, and SSN-shaped strings are
  rejected, so the row never lands *and* the agent loses reward.
- **Provenance per source.** Every contributing page is logged with retrieval
  time and method; `data/<name>.PROVENANCE.md` is published beside the CSV.
- **Public-web only**, via You.com's index.

### 2.8 Failure handling

| Failure | Response |
|---|---|
| `429` per-minute | Honour `Retry-After`, up to 5 attempts capped at 240s total. A token pacer also spaces calls against the discovered per-minute ceiling |
| `429` per-**day** | Abort with the provider's own figures. The `x-ratelimit-*` headers describe only the minute bucket — during a daily refusal they read `remaining-tokens: 8000, reset: 1ms`. Only the body distinguishes them |
| `413` payload too large | Halve the document budget and retry; write a lesson |
| `402` credits exhausted | Abort. Grinding on would log zero-reward episodes that look like a broken policy rather than a dead key |
| Sandbox stopped | Recreate and retry; `DaytonaError` and `SandboxError` are both caught |
| Component flapping | `ComponentHealth` tracks an EWMA success rate (α=0.35) and trips after 3 consecutive failures |

State is persisted **after every episode**, not at the end. Saving only at the end
meant an interrupted run discarded everything it had learned and published an
empty CSV, because the rows were still in memory.

### 2.9 Tunables

| Constant | Default | Where |
|---|---:|---|
| `WEIGHTS` | 1.0 / 0.7 / 0.6 / 0.15 / 2.5 | `learning/reward.py` |
| `PRIOR_ALPHA` / `PRIOR_BETA` | 1.0 / 1.0 | `learning/bandit.py` |
| `discount` | 0.98 | `learning/store.py` |
| `min_pulls` | 1 | `learning/store.py` |
| `LAMBDA` | 0.25 | `learning/budget.py` (`CLEANROOM_COST_LAMBDA`) |
| `REF_TOKENS` | 11,250 | `learning/budget.py` |
| `REPAIR_COST_UNITS` | 0.5 | `learning/budget.py` |
| `REPAIR_THRESHOLD` | 0.55 | `pipeline/episode.py` |
| `MIN_USEFUL_CHARS` | 400 | `pipeline/viability.py` |
| `DOC_SHARE_OF_TPM` | 0.35 | `pipeline/openai_compat.py` |
| `MAX_RATE_LIMIT_RETRIES` | 5 | `pipeline/openai_compat.py` |
| `MAX_TOTAL_BACKOFF_S` | 240.0 | `pipeline/openai_compat.py` |
| `REQUEST_TIMEOUT` | 180 | `pipeline/openai_compat.py` |
| `daytona_auto_stop_minutes` | 15 | `config.py` |

---

## 3. Evidence status per snapshot

| Snapshot | Episodes | What it is good for |
|---|---|---|
| `2026-09-17-control-haiku` | 24/arm × 2 | **The only causal test. Null.** |
| `2026-09-16-paced-24` | 24/24 | Most complete single-policy run; claims stress-tested |
| `2026-09-15-clean-30` | 30/30 | Internally consistent; cite for *mechanism* |
| `2026-09-11-hackathon` | 17 logged, 14 scored | The submission run. Read the caveats |

**Do not cite, with reasons:**

- *"It learned that strategy X beats Y on real pages."* Three runs produced three
  different winners (`table_parse`, `list_items`, `heading_sections`), each from
  2–9 pulls.
- *`table_parse` is a good arm.* Pooled over the 48 control episodes it is the
  **worst** — 0.374 against 0.69–0.72 for the other four.
- *The `+0.97` pull/reward correlation.* n=5 arms, exact p=0.0333, and one
  adjacent rank swap moves it to p=0.067–0.133.
- *The bandit caused the improvement.* The control says no significant
  difference, and was underpowered to settle it.
- *A learned cost/profile preference.* 1–5 pulls per arm; the winner reverses
  between runs (`lean` → `standard` → `thorough`); and because the utilities are
  cost-penalised, a cheap profile winning says nothing about quality.
- *Lessons are stored in One.* Local JSONL.
- *`$60.09` from the hackathon ledger.* That is development probes, including
  four You.com Agents calls at $15 each. The run itself cost `$0.02`.

---

## 4. What changed on 2026-09-17/18

Eight commits, `301b06a` → `94e5bfc`. 14 source and doc files, +1,213/−142 lines,
plus the control-run artifacts. Tests 137 → **143**.

### 4.1 Ran the control the repo kept asking for — `73c1283`

It took four attempts, and the first three failed for **one** reason worth
recording: the arms ran **sequentially**, so the second competed for a token
budget the first had already spent. Every "uniform random throttles itself" story
previously in this repo was an artefact of running second.

The fix is to **interleave** — bandit ep1, random ep1, bandit ep2, … — so both
arms see the same page at the same index under the same provider conditions, and
a mid-run stop truncates both equally. Profile pinned to `lean` in both, repairs
disabled, pages fetched once and cached so both arms read byte-identical input.

Result: 24 episodes/arm, 24/24 scored in both, zero provider failures, `$0.4373`.
Mean 0.655 vs 0.634; paired difference **+0.021, p=0.82**. Committed as
`runs/2026-09-17-control-haiku` with `scripts/compare_arms.py` to recompute it.

Ran on `claude-haiku-4-5` because Groq's free tier could not fit two arms (below).

### 4.2 Found a provider limit the agent could not see — `7d627f0`

Groq enforces **tokens per day: 200,000 per model**, and the `x-ratelimit-*`
headers describe only the per-minute bucket — during a daily refusal they read
`remaining-tokens: 8000, reset: 1ms`. A run can be blocked for twelve hours while
every header says it is free to proceed. Two 24-episode arms need ~211,000
tokens, so that control **cannot run on the free tier at all**.

A daily refusal now raises `DailyQuotaExhausted` (inheriting `CreditsExhausted`
for the existing abort path), carries the provider's figures, and is exempted
from the synthesis retry **by type** rather than by sniffing the message for
`"402"`. Status branches moved into `_raise_for_status` so each can be driven by
a captured response body.

**A retraction shipped with it.** `runs/README.md` had explained the first
control failure as "bad strategies trigger repairs → double the calls → random
throttles itself". That was inferred from a retry count and never measured — the
code discarded the 429 body. The simpler cause is arm order. The prescription was
wrong too: `--repairs 0` is insufficient, because synthesis retries once
independently of the repair budget and a live profile bandit swings per-call cost
about 2×.

### 4.3 Made the Anthropic backend work on more than one model — `b5966fc`

The backend hardcoded `thinking: {"type": "adaptive"}` and
`output_config.effort`. Haiku 4.5 rejects both, so it 400'd on the first call.
`supports_adaptive_thinking()` now decides by substring, and models outside that
family get neither parameter while keeping structured output.

Also: an **all-workspaces** API key refuses every request that does not name a
workspace, and the header decides which workspace is *billed*.
`CLEANROOM_ANTHROPIC_WORKSPACE_ID` pins it, and the 400 is caught by name and
answered with the variable to set — the provider's own message says a header is
missing without saying what sets it.

Measured: `claude-haiku-4-5` costs **$0.01115/call** (2,815 in + 1,666 out), so a
48-call control run is ~$0.53.

### 4.4 Corrected the committed record — `301b06a`, `341ba7f`, `0aef8af`, `6711757`

The documentation asserted things the artifacts contradict. Four passes, because
each one found more:

- **`runs/README.md`** gained a "What this repository actually supports" section
  splitting supported claims from ones that must not be cited, plus the fragility
  analysis for the `+0.97` correlation (exact permutation p, and the worst p
  reachable by one adjacent rank swap).
- **`verify_run.py`** now reports an exact permutation p-value beside rho and
  prints a `CAUTION` when a single rank swap would cost significance. Its verdict
  keys off **both** rho and p — judging by rho alone had labelled a +0.74 at
  p=0.20 as "effort follows reward".
- **Root `README.md`** was a week stale and led with `table_parse` as the best
  arm, `prose → regex_fields`, and `table_heavy → lean` — all contradicted. The
  section is now "What it actually learned, and what it did not".
- **A fourth stale claim** survived that pass: a posterior table showing `lean`
  best at 0.765 with *"a 7× cut in input tokens at no loss of quality"* — the one
  inference a cost-penalised utility cannot license.
- **`DEMO.md`**, untouched since submission day, narrated the retracted claims
  out loud. Corrected.
- **The bandit's action space and context space were never documented anywhere.**
  Both are now enumerated, along with `normalised_cost` (the formula was given
  without defining its terms) and a Tunables table.
- **A stale measurement**: the gate table said "10 seeds" with figures that no
  longer matched the test producing them. The test measures 20 seeds across four
  conditions, and says the **floor** is the larger lever — the README had
  credited the gate.

### 4.5 Fixed a bug class in my own analysis

The first version of `compare_arms.py` printed **"bandit beats random"** for a
−0.301 difference in which zero pairs favoured the bandit. It checked
significance without checking sign — the same error `verify_run.py` had been
fixed for hours earlier. Both now key off sign and p, and `compare_arms.py`
reports the power figures beside the p-value so a null cannot be quoted without
them.

---

## 5. Known limitations

- **No live run is large enough to show the bandit works.** ~214 episodes/arm
  needed; every run here is 14–30.
- **The context is page *shape*, not page *identity*.** Measured consequence: the
  bandit scored 0.921 on a page at episode 1 and 0.000 on that same page at
  episode 9. It cannot represent "this strategy for this host".
- **`min_pulls = 1` is too few for this pool.** Per-page difficulty spans
  0.306–0.923. In one control arm the best available strategy was tried once, on
  the hardest page, scored 0.000, and was discarded; uniform random drew it five
  times and scored ~0.92.
- **The cost bandit is the least supported thing here** (§2.5).
- **Lesson stores diverge between control arms**, so the comparison is not purely
  strategy selection. Interleaving cannot fix this; only disabling lesson
  retrieval in both arms would.
- **Per-arm cost in the control snapshot is not attributable** — the call ledger
  is a process-wide singleton and both arms ran in one process. The total is
  correct; rewards are unaffected, coming from validation rather than the ledger.
- **Five arms cannot invent a strategy that is not in the list.**
- **`expected_rows_per_page` is a hand-set constant**, so the coverage channel is
  only as good as that guess.
- **Revisiting pages inflates apparent learning** — with more episodes than
  sources the loop cycles.

---

## 6. What would earn "self-improving"

Achievable and cheap; the obstacle is not the algorithm.

1. **Stop regenerating the extractor every episode.** Memoize per
   `(host, strategy)`. A pair's value becomes near-deterministic after one
   observation, it costs *less* to run, and the open question becomes transfer to
   unseen hosts — which is what a page-shape feature is for. A day of work,
   negative running cost. **This is the whole game.**
2. **Make the context predict the target.** Host in the bucket key, or bypass the
   bandit once a host has a known-good extractor. Raise `min_pulls` to 2–3.
3. **Then measure**, with lesson retrieval off in both arms. Either 214
   episodes/arm (~$4.70, ~4.5h) or — better after step 1 — first-pick accuracy
   over 40–60 unseen hosts (~$3), a binomial over hosts rather than a mean over
   noisy episodes. The binding constraint there is **hosts, not tokens**: the
   current pool is 8 pages.

**The ceiling is real.** Once a host has a working extractor there is nothing
further to learn about that host. The durable capability is choosing well on
pages never seen before — a genuine but bounded thing.

**Not worth doing:** more arms (five is not the constraint, the noise is);
policy-gradient anything (same signal-to-noise, far more machinery, far less
interpretability); chasing the aggregate reward curve (confounded by page mix and
drowned in codegen variance).

---

## 7. Verifying this yourself

```bash
pytest -q                                                   # 143 tests, no network
python scripts/verify_run.py    runs/2026-09-16-paced-24    # one snapshot's claims
python scripts/verify_run.py    runs/2026-09-17-control-haiku
python scripts/compare_arms.py  runs/2026-09-17-control-haiku
```

Both scripts are dependency-free, import nothing from `cleanroom`, and read the
primary artifacts rather than any derived summary. Read the `CAUTION` lines —
they are the point.

Tests by file: `test_learning.py` 32, `test_observability.py` 33,
`test_viability.py` 33, `test_feedback.py` 27, `test_integration_pieces.py` 18.

Several exist because a real run failed while the unit tests stayed green: the
token pacer is driven through `_post_with_backoff` rather than in isolation (the
collaborator alone missed a `NameError` on the first live call), the
per-day/per-minute 429 split uses response bodies captured verbatim from the
provider, and the per-model request shape is asserted against a fake client
because that failure mode is a *paid* call that fails.
