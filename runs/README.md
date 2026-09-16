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
