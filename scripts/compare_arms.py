#!/usr/bin/env python3
"""Recompute the bandit-vs-random control result from a committed snapshot.

Imports nothing from `cleanroom` and has no dependencies, for the same reason
`verify_run.py` does not: a number that can only be reproduced by the code that
produced it is not evidence. Reads the two arms' `episodes.jsonl` directly --
the primary artifact -- rather than the derived `summary.json`.

    python scripts/compare_arms.py runs/2026-09-17-control-haiku

Two questions, kept apart:

  Q1  Does bandit selection beat uniform-random selection on mean reward?
      Paired sign-flip test. Paired, because episode *i* is the same page in
      both arms and between-page difficulty (0.306-0.923 in this pool) is far
      larger than any policy effect -- an unpaired test spends its power there.

  Q2  Is the within-run climb attributable to strategy selection?
      Difference-in-differences on the paired differences, randomized over
      episode position.

It also reports how much of the reward variance is code-generation noise rather
than policy, and the sample size that would be needed to see a real effect. A
null result means nothing without that number.
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import random
import statistics as st
import sys
from collections import defaultdict

ARMS = ("bandit", "random")


def load(snapshot: pathlib.Path, arm: str) -> list[dict]:
    path = snapshot / arm / "episodes.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        detail = rec.get("reward_detail") or {}
        # A provider failure scores 0.0 but says nothing about the policy.
        # Averaging those in makes a flaky API look like a bad agent.
        rec["_scored"] = detail.get("counts_toward_learning") is not False and \
            detail.get("stage") != "synthesis_failed"
        out.append(rec)
    return out


def signflip_p(d: list[float], iters: int = 200_000, seed: int = 7) -> tuple[float, str]:
    """Two-sided sign-flip test on paired differences; exact when it can be."""
    n = len(d)
    obs = abs(st.mean(d))
    if n <= 18:
        hits = sum(
            1 for signs in itertools.product((1, -1), repeat=n)
            if abs(st.mean([s * x for s, x in zip(signs, d)])) >= obs - 1e-12
        )
        return hits / 2 ** n, "exact"
    rng = random.Random(seed)
    hits = sum(
        1 for _ in range(iters)
        if abs(st.mean([x if rng.random() < 0.5 else -x for x in d])) >= obs - 1e-12
    )
    return (hits + 1) / (iters + 1), f"MC {iters:,}"


def trend_p(d: list[float], iters: int = 200_000, seed: int = 11) -> tuple[float, float]:
    third = max(1, len(d) // 3)
    obs = abs(st.mean(d[-third:]) - st.mean(d[:third]))
    rng = random.Random(seed)
    perm = list(d)
    hits = 0
    for _ in range(iters):
        rng.shuffle(perm)
        if abs(st.mean(perm[-third:]) - st.mean(perm[:third])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1), obs


def main() -> None:
    snapshot = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                            else "runs/2026-09-17-control-haiku")
    eps = {arm: load(snapshot, arm) for arm in ARMS}

    print(f"--- {snapshot}")
    losses = {}
    for arm in ARMS:
        scored = [e for e in eps[arm] if e["_scored"]]
        rewards = [float(e["reward"]) for e in scored]
        losses[arm] = len(eps[arm]) - len(scored)
        third = max(1, len(rewards) // 3)
        print(f"  {arm:<7} {len(scored)}/{len(eps[arm])} scored   "
              f"mean {st.mean(rewards):.3f}   "
              f"first third {st.mean(rewards[:third]):.3f} -> "
              f"last third {st.mean(rewards[-third:]):.3f}")

    # Equal effort is the precondition. Three earlier attempts at this control
    # were invalid because one arm was starved of provider budget.
    if max(losses.values()) > 2 and abs(losses["bandit"] - losses["random"]) >= 2:
        print("  CAUTION: episode losses are lopsided "
              f"({losses}); the arm that lost more was starved. Do not report "
              "the comparison.")

    # --- paired on page --------------------------------------------------
    by_ep = {arm: {e["episode"]: e for e in eps[arm] if e["_scored"]} for arm in ARMS}
    shared = sorted(set(by_ep["bandit"]) & set(by_ep["random"]))
    pairs = [(e, by_ep["bandit"][e], by_ep["random"][e]) for e in shared
             if by_ep["bandit"][e].get("url") == by_ep["random"][e].get("url")]
    if len(pairs) < 4:
        print("  too few paired episodes for a test")
        return

    d = [float(b["reward"]) - float(c["reward"]) for _, b, c in pairs]
    ties = sum(1 for x in d if x == 0)
    p1, kind = signflip_p(d)
    favour = sum(1 for x in d if x > 0)
    print(f"\n  Q1 paired difference (bandit - random): {st.mean(d):+.3f}  "
          f"(n={len(d)}, {ties} tied, sign-flip p={p1:.4f} {kind})")
    print(f"     {favour} of {len(d)} pairs favour the bandit")
    # The verdict keys off the sign as well as the p-value: checking only
    # significance once printed "bandit beats random" for a -0.301 difference
    # in which no pair favoured the bandit.
    if p1 > 0.05:
        print("     -> NO SIGNIFICANT DIFFERENCE at this sample size")
    elif st.mean(d) > 0:
        print("     -> bandit beats uniform random")
    else:
        print("     -> UNIFORM RANDOM BEATS THE BANDIT")

    p2, obs = trend_p(d)
    print(f"\n  Q2 difference-in-differences on the climb: {obs:+.3f} "
          f"(randomization p={p2:.4f})")
    print("     -> " + ("climb attributable to strategy selection" if p2 <= 0.05
                        else "climb NOT attributable to strategy selection"))

    # --- is the null even informative? -----------------------------------
    pooled = [e for arm in ARMS for e in eps[arm] if e["_scored"]]
    cells = defaultdict(list)
    for e in pooled:
        cells[(e.get("url"), e.get("strategy"))].append(float(e["reward"]))
    repeats = [v for v in cells.values() if len(v) > 1]
    by_strategy = defaultdict(list)
    for e in pooled:
        by_strategy[e.get("strategy")].append(float(e["reward"]))

    print("\n  Where the variance lives:")
    if repeats:
        within = st.mean([st.pstdev(v) for v in repeats])
        print(f"    same page + same strategy, repeated ({len(repeats)} cells): "
              f"sd {within:.3f}   <- code-generation noise, not policy")
    means = {k: st.mean(v) for k, v in by_strategy.items()}
    print(f"    between strategy means: sd {st.pstdev(list(means.values())):.3f}")
    for k, v in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<18} {v:.3f}  (n={len(by_strategy[k])})")

    sd_d = st.pstdev(d)
    print(f"\n  Episodes per arm for 80% power at alpha=0.05 (sd of paired "
          f"differences = {sd_d:.3f}):")
    for effect in (0.05, 0.10, 0.20):
        n = math.ceil(((1.96 + 0.84) * sd_d / effect) ** 2)
        print(f"    to detect a {effect:.2f} reward difference: n = {n:,} per arm")
    print("  CAUTION: read those before quoting the p-value above. A null result "
          "at\n  this sample size is 'not measured', not 'no effect'.")


if __name__ == "__main__":
    main()
