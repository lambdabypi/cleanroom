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

### `2026-09-15-clean-30` — 30 uninterrupted episodes. Read this one first.

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
