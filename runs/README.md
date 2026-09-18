# Run snapshots

Committed copies of `state/` from real runs, so the numbers in the README and in
any writing about this project can be checked rather than taken on trust.
`state/` itself is gitignored (it churns every episode); these are frozen copies.

```bash
python scripts/verify_run.py                      # newest snapshot
python scripts/verify_run.py runs/2026-09-11-hackathon
python scripts/compare_arms.py runs/2026-09-17-control-haiku   # the control run
```

Both scripts import nothing from `cleanroom` and have no dependencies, and both
read the primary artifacts rather than any derived summary. They recompute each
headline number and print a `CAUTION` line wherever the data is too thin to
support a claim. Read those lines — they are the point of the scripts.
`compare_arms.py` additionally reports how much of the reward variance is
code-generation noise and what sample size a real answer would need, because a
null result without that number is not interpretable.

## Snapshot index

| Snapshot | What it is |
|---|---|
| `2026-09-17-control-haiku` | the bandit-vs-random control, 24 episodes/arm, interleaved — **the only causal test here, and it is null** |
| `2026-09-16-paced-24` | the most complete single-policy run; its claims are stress-tested below |
| `2026-09-15-clean-30` | 30 uninterrupted episodes before the fixes; the run to cite for mechanism |
| `2026-09-11-hackathon` | the hackathon run. Read the caveats |

## What this repository actually supports

Stress-tested on 2026-09-17 against the artifacts below. Stated up front because
it is easy to read a snapshot and over-claim from it.

**Supported:**

- The loop runs end to end and reliably — 24/24 and 30/30 scored episodes across
  two runs, zero provider failures in the latest.
- Every published row is attributed (179/179 and 193/193 carry an http source
  URL) and an independent PII scan of the output is clean.
- Lesson retrieval fires on essentially every episode (24/24, 29/30).
- The measured cost of a run: `$0.0006`–`$0.0023` of in-episode spend.
- The reward rise within `paced-24` is **not** a page-composition artefact
  (identical page mix in both thirds — see that snapshot's notes).
- The strategy bandit converges on a better arm **in simulation**
  (`tests/test_observability.py`), with the exploration floor and the on-policy
  cost gate measured separately over 20 seeds.

**Not supported — do not cite:**

- Any claim of the form "it learned that strategy X beats Y on real pages."
  Three runs produced three different winners (`table_parse`, `list_items`,
  `heading_sections`), each from 2–9 pulls.
- The `+0.97` pull/reward correlation as a headline. n=5 arms, p=0.0333, and one
  adjacent rank swap moves it to p=0.067–0.133.
- Any statement that the bandit *caused* the improvement. A valid control ran on
  2026-09-17 and found **no significant difference** from uniform-random
  selection (24 episodes/arm, paired difference +0.021, p=0.82) — and it was far
  too underpowered to settle it either way, needing ~214 episodes/arm to detect
  a 0.10 difference. Neither "it works" nor "it doesn't" is supported.
- That `table_parse` is a good arm. Pooled over the 48 control episodes it is
  the worst (0.374 vs 0.69–0.72 for the other four).
- A learned cost/profile preference. Too few on-policy episodes in every run.
- That lessons are stored in One. They are in local JSONL; One's `mem` store
  could not start on this machine.

## What is in a snapshot

| File | Contents |
|---|---|
| `episodes.jsonl` | one record per episode: bucket, strategy, profile, reward, per-channel breakdown, validator errors, tokens, cost |
| `bandit.json` | strategy posteriors (α, β, pulls, reward sum) per page shape |
| `budget.json` | execution-profile posteriors, same shape |
| `calls.jsonl` | every external call: component, latency, outcome, estimated cost |
| `memory.jsonl` | lessons the agent wrote, with a `synced` flag for One's store |
| `provenance.jsonl` | one record per page attempted: URL, retrieval time, rows contributed |
| `dataset.csv` | the extracted rows |
| `learning_curve.png/.tsv` | the figure and its underlying table |

## Snapshots

### `2026-09-16-paced-24` — the most complete run, with its claims stress-tested

24 episodes with the viability screen, the cold-start exploration floor and the
token pacer all active.

| | `clean-30` (before) | `paced-24` (after) |
|---|---|---|
| episodes scored | 30/30 | 24/24 |
| lost to provider failures | 3 earlier attempts | **0** |
| rate-limit (429) errors | many | **0** |
| reward, first third → last third | 0.691 → 0.627 | **0.644 → 0.891** |
| pull↔reward rank correlation | +0.21 | +0.97 (**see caveat**) |
| best arm sample size | n=3 | n=9 |

Solid regardless of interpretation: 179 rows from 8 sources, 179/179 attributed,
independent PII scan clean, 24/24 episodes retrieved a prior lesson, `$0.0006`
of in-episode spend, and no provider failures.

#### What a verification pass (2026-09-17) established

**The climb is not a page-composition artefact — this one is clean.** The loop
cycles sources in order, so with 8 pages over 24 episodes each third contains all
eight exactly once. Mean per-URL difficulty is **0.765 in both the first and the
last third**, while per-URL difficulty across the pool ranges from 0.306
(`spheron.network`) to 0.923 (`cloudzero.com/blog/h100-gpu-cost`). The
0.644 → 0.891 rise happens on the same pages in the same proportions.

**The +0.97 correlation is one rank swap from non-significance. Do not headline
it.** Computed within `table_heavy`: n=5 arms, exact permutation p=0.0333
(4 of 120 permutations). Three of the four single adjacent rank swaps push p to
0.067–0.133, and two arms are already tied at 6 pulls with rewards 0.870 and
0.668. Pooled equals within only because this run collapsed to a single bucket;
Simpson's risk does apply to the two-bucket `clean-30` run. It is a suggestive
single-run observation that needs replication across 3–4 runs before it is a
result.

**The control now exists, and it finds no significant difference.** See
"The control run (2026-09-17)" below for the result and, more importantly, for
how little this experiment was ever able to detect. The history of the four
attempts it took is kept here because three of them failed for one reason that
is easy to repeat.

An attempt on 2026-09-17 to run 24 episodes with uniform-random strategy
selection over the same pinned 8-URL pool **failed**: only 8 of 24 episodes
scored, the other 16 dying at `synthesis_failed` with "rate limit (429) after 5
attempts".

A second attempt the same day, with `--repairs 0` and the execution profile
pinned, failed too — and established why. **The earlier explanation offered here
was wrong, and is retracted.** It said bad strategies trigger repair turns, which
double the calls, so random selection throttles itself while the bandit does
not. That mechanism was inferred from the retry count, never measured: the code
discarded the 429 body, so nothing recorded *which* limit had been hit.

Reading that body directly gives a simpler and more damaging explanation. The
provider enforces a **per-day** token budget — `tokens per day (TPD): Limit
200000, Used 199033` — which the `x-ratelimit-*` headers do not expose at all;
they describe the per-minute bucket, and during a daily refusal they read
`remaining-tokens: 8000, reset: 1ms`. So a run can be blocked for twelve hours
while every header says it is free to proceed. One 24-episode arm costs about
105,600 tokens, so **whichever arm runs second is starved regardless of which
strategies it picked.** That alone can produce the v1 result, with no
self-throttling story required.

Two consequences:

- **`--repairs 0` is not sufficient**, so the earlier prescription here was also
  wrong. Synthesis retries once on any failure independently of the repair
  budget (`openai_compat.py::_run`), and while the profile bandit is live the
  per-call token cost swings about 2x between `lean` and a clipped `thorough`.
  Equalising effort needs the profile pinned as well.
- **The control as specified cannot run on this tier.** Two 24-episode arms cost
  roughly 211,000 tokens against a 200,000/day ceiling — more than a full day's
  budget before any retries. It needs fewer episodes per arm, a model with its
  own daily bucket, or a paid tier.

The agent now distinguishes the two cases: a per-day refusal raises
`DailyQuotaExhausted` and aborts the run with the provider's own figures,
instead of spending five retries and ~200s of backoff and then logging the
episode as a synthesis failure, which reads as the agent's fault.

### The control run (2026-09-17)

Ran on `claude-haiku-4-5`, because the free tier's per-day cap could not fit two
arms. **24 episodes per arm, both arms 24/24 scored, zero provider failures,
$0.4373 of measured spend.** The two arms are *interleaved* — bandit ep1, random
ep1, bandit ep2, … — so both see the same page at the same index under the same
provider conditions, and a mid-run stop truncates both equally. The execution
profile is pinned to `lean` in both, leaving strategy selection as the only
thing that differs. Pages are fetched once and cached, so both arms read
byte-identical input.

| | bandit | uniform random |
|---|---|---|
| scored | 24/24 | 24/24 |
| mean reward | 0.655 | 0.634 |
| first third → last third | 0.727 → 0.673 | 0.672 → 0.499 |

Paired on page (episode *i* is the same page in both arms):

- **Mean paired difference +0.021, sign-flip p = 0.82**, 10 of 24 pairs favour
  the bandit (5 tied, 19 informative). **No significant difference.**
- Difference-in-differences on the climb: +0.119, randomization p = 0.67. The
  within-run climb is **not** attributable to strategy selection either.

**Why the null is weak evidence, not strong evidence.** Reward here is mostly
noise from code generation, not from the policy. Across the 13 (page, strategy)
cells that repeat, the standard deviation *within* a cell is **0.258** — the same
page with the same strategy returns `[0.918, 0.0]` and `[0.0, 0.918, 0.918]`.
The spread *between* strategy means is only **0.133**. The noise is about twice
the signal, so at n=24 this design could only have detected a difference of
roughly 0.3. For 80% power at α=0.05 it needs **214 episodes per arm to detect a
0.10 reward difference** (~$4.77 on this model) and 853 per arm for 0.05. Every
live run in this repo is 14–30 episodes. **Nothing here can settle the question;
the honest reading is "not measured", not "no effect".**

One strategy-level signal does survive pooling all 48 episodes: `table_parse`
averages **0.374** (n=9) while the other four arms cluster at 0.69–0.72. That is
the *third* run to contradict the hackathon snapshot's "`table_parse` wins on
table-heavy pages", and the first to suggest it is the worst arm rather than a
middling one.

**A second replication on a different model disagrees, which is itself the
result.** The same interleaved design on Groq's `openai/gpt-oss-20b` reached only
12 episodes per arm before the daily cap stopped it, and there uniform-random
*beat* the bandit (mean paired difference −0.301, exact sign-flip p = 0.031, 0 of
10 pairs favouring the bandit). It should not be cited on its own: the random arm
lost 2 episodes to empty completions while the bandit lost none, and the bandit's
zeros were mostly `stage=extract` crashes (`IndexError`, `NameError`,
`extract() exceeded 25s`) rather than a badly chosen strategy. Two runs, two
opposite directions, neither significant in the same place — which is what
"underpowered" looks like from the inside.

Diagnostics worth keeping from that run, because they point at design limits
rather than bugs:

- **`min_pulls=1` is too low for this pool.** The bandit pulled
  `label_value_pairs` exactly once, at the episode that lands on
  `spheron.network` — a page where *both* arms scored 0.000 — and effectively
  discarded the arm. Uniform random drew it 5 times on easier pages and scored
  0.918/0.922/0.921. With per-page difficulty spanning 0.306–0.923, one pull says
  more about which page it landed on than which strategy was chosen.
- **The page-shape bucket is too coarse.** All 8 pages collapse into
  `table_heavy`, so one posterior is averaged over pages that want different
  strategies. The bandit scored 0.921 on `thundercompute/h100` at ep1 with
  `table_parse`, then 0.000 on that same page at ep9 with `list_items`. It cannot
  represent "this strategy for this host".

**Remaining confound, unfixed.** Each arm keeps its own lesson store, so the two
diverge as soon as the arms do. In the Groq run the bandit's lessons became
uniformly failure reports while random's included actionable positives
(`label_value_pairs works well; reward 0.92`). Interleaving removes the
provider-order confound; it cannot remove this one, because any second learning
channel drifts with the arm it is attached to. A clean isolation would disable
lesson retrieval in both arms.

So the honest position on `paced-24` is now: the climb is not composition, and
the control does not attribute it to strategy selection — but the control is far
too underpowered to attribute it to anything else either.

**On the causal story for the improvement — weaker than it first appeared.** The
two `getdeploying.com` subpages the screen excluded were returning 1,455
characters of boilerplate at the time, creating a phantom `list_heavy` bucket in
which every strategy scored 0.000. Re-fetched on 2026-09-17 the same URLs return
61,838 and 53,526 characters and the screen keeps them. The barrenness was a
property of that *crawl*, not those pages, so part of this run's advantage came
from a transient condition rather than from the fix alone.

Still not supported here: the cost/profile bandit. With 10 on-policy episodes
across three arms it remains too thin to claim a learned spend preference, and
the verifier still says so.

### `2026-09-15-clean-30` — 30 uninterrupted episodes, before the fixes.

The run to cite for anything about the *mechanism*, because it is internally
consistent: 30 episodes logged, 30 scored, no code-writer failures, and a dataset
whose sources match its provenance exactly.

What it supports:

- **The loop runs end to end, reliably.** 30/30 episodes produced a scored
  result. 29/30 retrieved at least one prior lesson before acting.
- **193 rows from 8 distinct sources, 193/193 carrying an http source URL**, and
  an independent PII scan (regexes restated inside `verify_run.py`, not imported
  from the project) comes back clean.
- **Cost: `$0.0023` across 77 in-episode calls.** The ledger is clean this time —
  a single `you.search` call at `$0.0050` sits outside the episodes.

What it **contradicts** — and this is the more useful result:

- **`table_parse` did not win on table-heavy pages.** It scored 0.370 over 2
  pulls here, the *worst* arm, having been the best arm (0.899 over 4 pulls) in
  the hackathon snapshot. The two runs disagree.
- **The `lean` profile did not win either.** `standard` led table-heavy at 0.665
  (n=4) with `lean` last at 0.462 (n=2), the reverse of the hackathon run.
- **Aggregate reward fell**, 0.691 → 0.627 across the run.

The honest reading: **neither per-arm preference replicates.** Both were drawn
from 2–4 pulls, which is noise. The strategy bandit provably converges on a
better arm in simulation (`tests/test_observability.py`), but **no live run in
this repo has enough pulls per arm to demonstrate a learned preference over real
pages.** Anything of the form "it learned that X beats Y on real pages" is not
supported by this repository. Claims about the mechanism, the cost, the
attribution and the PII screening are.

One clear finding does emerge: **every `list_heavy` episode scored 0.000** — six
episodes, four different strategies, no valid rows. That is a genuine failure
mode for this schema, not a learning result.

### `2026-09-11-hackathon` — the hackathon run. Read the caveats.

What it supports: `table_heavy → table_parse`, posterior 0.763 / observed 0.899
over 4 pulls, out of 14 scored episodes. Lesson retrieval fired on 13 of 14
episodes. An independent PII scan of the published rows is clean, and all 15 rows
carry an http source URL.

What it does **not** support, and should not be cited for:

- **The `prose` bucket has one pull per arm.** `regex_fields` appearing best there
  is noise, not a finding.
- **The cost/profile bandit has two pulls per arm** in one bucket, and only 4 of
  14 episodes were on-policy for cost learning. This snapshot does **not**
  demonstrate a learned cost preference, and the utilities shown are
  cost-penalised, so a cheap profile winning is not evidence of equal quality.
- **Lessons were stored locally, not in One.** Every record reads
  `synced: false`. One's `mem` commands bootstrap an embedded Postgres
  (`pgserve`) which would not start on this machine, so each write failed after
  ~30s and fell back to local JSONL. One *is* genuinely used for credentials,
  action discovery and the GitHub write-back; it was not used for memory.
- **`$60.09` of the ledger is outside any episode** — development probes,
  including four calls to You.com's Agents API at $15 each. The run itself cost
  `$0.02`. The verifier separates these; do not quote the total.
- **The dataset spans more than one run.** Provenance records 7 contributing
  sources that are absent from `dataset.csv`, because an earlier run was
  interrupted before its rows were flushed. Per-episode persistence was added
  afterwards to prevent exactly this.

The honest summary is that this run demonstrates the *mechanism* end to end —
retrieval, sandboxed execution, scoring, credit assignment, lesson writing and a
real commit — on too few episodes per arm to support most quantitative claims.
