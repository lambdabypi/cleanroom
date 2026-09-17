# Run snapshots

Committed copies of `state/` from real runs, so the numbers in the README and in
any writing about this project can be checked rather than taken on trust.
`state/` itself is gitignored (it churns every episode); these are frozen copies.

```bash
python scripts/verify_run.py                      # newest snapshot
python scripts/verify_run.py runs/2026-09-11-hackathon
```

`verify_run.py` imports nothing from `cleanroom` and has no dependencies. It
recomputes each headline number from the artifacts and prints a `CAUTION` line
wherever the data is too thin to support a claim. Read those lines — they are the
point of the script.

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
- Any statement that the bandit *caused* the improvement. The control run that
  would show this has not successfully run.
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

**No control run exists, so the bandit's contribution is still unproven.** An
attempt on 2026-09-17 to run 24 episodes with uniform-random strategy selection
over the same pinned 8-URL pool **failed**: only 8 of 24 episodes scored, the
other 16 dying at `synthesis_failed` with "rate limit (429) after 5 attempts".
The cause is itself interesting — bad strategies return zero rows, which triggers
a repair turn, which doubles LLM calls per episode, which blows the 8,000
tokens/minute ceiling. Random selection throttles itself and the bandit does not.
That is a real effect, but it contaminates any reward comparison between the two
arms. **A valid control needs `--repairs 0` on both arms** so calls per episode
are equal, or a provider without the ceiling.

Until that control runs, the honest position is: the climb is not composition,
and it is not yet attributed to strategy selection rather than to the profile
bandit or lesson retrieval.

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
