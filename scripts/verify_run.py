#!/usr/bin/env python3
"""Recompute every headline claim from a committed run snapshot.

    python scripts/verify_run.py                 # newest snapshot under runs/
    python scripts/verify_run.py runs/2026-09-15

Deliberately dependency-free and self-contained: it reads only the JSON/JSONL/CSV
in the snapshot directory and imports nothing from `cleanroom`. A reader should be
able to check the numbers without installing the project, and without trusting
any of its code.

Every line it prints is derived from the artifacts. Where an artifact cannot
support a claim, it says so rather than staying quiet.
"""

from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
from collections import Counter

# Same three patterns the validator rejects rows on, restated here so the PII
# check is independent of the code under test.
PII = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
)


def jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def posterior(stat: dict) -> float:
    return stat["alpha"] / (stat["alpha"] + stat["beta"])


def _ranks(values: list[float]) -> list[float]:
    """Ascending ranks, averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, stdlib only. Returns 0.0 when undefined."""
    if len(xs) < 3:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None:
        candidates = sorted(pathlib.Path("runs").glob("*/"), reverse=True)
        if not candidates:
            print("no snapshots under runs/")
            return 1
        root = candidates[0]
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1

    print(f"Verifying snapshot: {root}")

    episodes = jsonl(root / "episodes.jsonl")
    bandit = load(root / "bandit.json")
    budget = load(root / "budget.json")
    calls = jsonl(root / "calls.jsonl")
    prov = jsonl(root / "provenance.jsonl")
    mem = jsonl(root / "memory.jsonl")

    scored = [
        e for e in episodes
        if (e.get("reward_detail") or {}).get("counts_toward_learning") is not False
    ]
    unscored = len(episodes) - len(scored)

    # -- episodes ---------------------------------------------------------
    section("Episodes")
    print(f"  logged                 {len(episodes)}")
    print(f"  scored (policy)        {len(scored)}")
    print(f"  excluded (writer fail) {unscored}")
    if not scored:
        print("  nothing scored; no claim about learning is supported.")
        return 1

    rewards = [float(e.get("reward") or 0.0) for e in scored]
    print(f"  reward min/mean/max    {min(rewards):.3f} / "
          f"{sum(rewards)/len(rewards):.3f} / {max(rewards):.3f}")
    cut = max(2, len(rewards) // 3)
    early, late = sum(rewards[:cut]) / cut, sum(rewards[-cut:]) / cut
    print(f"  first {cut} vs last {cut}     {early:.3f} -> {late:.3f} ({late-early:+.3f})")
    print("  NOTE: reward is not comparable across page shapes, so this aggregate")
    print("        is confounded by which shapes came up. Read per-bucket below.")
    print(f"  buckets seen           {dict(Counter(e['bucket'] for e in scored))}")
    hosts = {e["url"].split("/")[2] for e in scored if e.get("url", "").count("/") > 2}
    print(f"  distinct source hosts  {len(hosts)}")

    # -- strategy bandit --------------------------------------------------
    section("Strategy bandit (per page shape)")
    for bucket, arms in (bandit.get("stats") or {}).items():
        pulled = {a: s for a, s in arms.items() if s["pulls"]}
        if not pulled:
            continue
        best = max(pulled, key=lambda a: posterior(pulled[a]))
        total = sum(s["pulls"] for s in pulled.values())
        print(f"  {bucket}  (n={total})")
        for arm, s in sorted(pulled.items(), key=lambda kv: -posterior(kv[1])):
            obs = s["reward_sum"] / s["pulls"]
            mark = " <- best" if arm == best else ""
            print(f"    {arm:<20} posterior {posterior(s):.3f}  "
                  f"observed {obs:.3f}  n={s['pulls']}{mark}")
        if pulled[best]["pulls"] < 3:
            print(f"    CAUTION: best arm has only {pulled[best]['pulls']} pull(s); "
                  "not a supported claim.")

        # Which arm "wins" varies run to run when several are close. The
        # behaviour that should hold regardless is that effort follows reward:
        # more pulls for arms that score better. That is checkable.
        if len(pulled) >= 3:
            rho = _spearman(
                [s["pulls"] for s in pulled.values()],
                [s["reward_sum"] / s["pulls"] for s in pulled.values()],
            )
            verdict = ("effort follows reward" if rho >= 0.6
                       else "WEAK -- effort does not track reward")
            print(f"    pull/reward rank correlation: {rho:+.2f}  ({verdict})")
    print(f"  discount in use        {bandit.get('discount')}")

    # -- profile bandit ---------------------------------------------------
    section("Cost/profile bandit")
    any_profile = False
    for bucket, arms in (budget.get("stats") or {}).items():
        pulled = {a: s for a, s in arms.items() if s["pulls"]}
        if not pulled:
            continue
        any_profile = True
        best = max(pulled, key=lambda a: posterior(pulled[a]))
        print(f"  {bucket}")
        for arm, s in sorted(pulled.items(), key=lambda kv: -posterior(kv[1])):
            mark = " <- best" if arm == best else ""
            print(f"    {arm:<10} utility {posterior(s):.3f}  n={s['pulls']}{mark}")
        print("    NOTE: these are cost-penalised utilities, not quality. A cheap")
        print("          profile winning does NOT by itself show equal quality.")
        if sum(s["pulls"] for s in pulled.values()) < 12:
            print("    CAUTION: too few pulls to claim a learned cost preference.")
    if not any_profile:
        print("  no profile pulls recorded.")
    gated = sum(1 for e in scored if (e.get("reward_detail") or {}).get("cost_learned"))
    print(f"  on-policy (cost-learning) episodes: {gated}/{len(scored)}")

    # -- spend ------------------------------------------------------------
    section("Spend (from the call ledger)")
    in_ep = [c for c in calls if c.get("episode") is not None]
    out_ep = [c for c in calls if c.get("episode") is None]
    by_op = Counter(c["operation"] for c in calls)
    cost = Counter()
    for c in calls:
        cost[c["operation"]] += float(c.get("cost_usd") or 0.0)
    for op, n in by_op.most_common():
        print(f"  {op:<18} n={n:<4} ${cost[op]:.4f}")
    print(f"  attributed to episodes  {len(in_ep)} calls  "
          f"${sum(float(c.get('cost_usd') or 0) for c in in_ep):.4f}")
    print(f"  outside episodes        {len(out_ep)} calls  "
          f"${sum(float(c.get('cost_usd') or 0) for c in out_ep):.4f}")
    print("  NOTE: 'outside episodes' includes development probes and health")
    print("        checks. Only the first figure is the cost of the run.")

    # -- lessons ----------------------------------------------------------
    section("Lessons")
    synced = sum(1 for m in mem if m.get("synced"))
    print(f"  written            {len(mem)}")
    print(f"  synced to One      {synced}")
    if mem and synced == 0:
        print("  => stored in LOCAL JSONL, not One's mem store.")
    used = sum(1 for e in scored if e.get("memory_hits"))
    print(f"  episodes that retrieved at least one lesson: {used}/{len(scored)}")
    for m in mem[:3]:
        print(f"    - {str(m.get('text'))[:110]}")

    # -- provenance & clean data -----------------------------------------
    section("Provenance and Clean Data")
    print(f"  provenance records          {len(prov)}")
    print(f"  every record has a URL      {all(p.get('url') for p in prov)}")
    print(f"  every record has retrieved_at {all(p.get('retrieved_at') for p in prov)}")
    print("  NOTE: retrieved_at is written when the episode ends, not at the")
    print("        moment of the HTTP fetch.")

    csv_path = root / "dataset.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        url_field = "source_url" if rows and "source_url" in rows[0] else None
        attributed = sum(1 for r in rows if url_field and (r.get(url_field) or "").startswith("http"))
        hits = [
            (r, f) for r in rows for f, v in r.items()
            if isinstance(v, str) and any(p.search(v) for p in PII)
        ]
        print(f"  dataset rows                {len(rows)}")
        print(f"  rows with an http source    {attributed}/{len(rows)}")
        print(f"  distinct sources            "
              f"{len({r.get(url_field) for r in rows}) if url_field else 'n/a'}")
        print(f"  independent PII scan        "
              f"{'CLEAN' if not hits else f'{len(hits)} HIT(S)'}")
        if len(prov) and url_field:
            contributing = {p.get("url") for p in prov
                            if int(p.get("rows_contributed") or 0) > 0}
            in_csv = {r.get(url_field) for r in rows}
            if contributing - in_csv:
                print(f"  CAUTION: {len(contributing - in_csv)} source(s) recorded as")
                print("           contributing are absent from dataset.csv --")
                print("           the snapshot spans more than one run.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
