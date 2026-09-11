"""Human feedback -- the sparse, high-weight reward channel.

Shared by the CLI (`cleanroom feedback`) and the dashboard's thumbs buttons, so
a verdict means exactly the same thing whichever way it arrives. When these
diverge you get the worst kind of bug: two interfaces that both look like they
work and disagree about what the agent learned.

Two grains of feedback, because they teach different things:

* **Episode verdict** (`good` / `ok` / `bad`) updates the strategy posterior. It
  answers "was this approach right for this page?"
* **Row rejection** writes a lesson naming the offending values. A posterior
  cannot represent "rows whose provider is a duration are wrong"; a retrieved
  lesson can, and the next synthesis sees it.

The episode verdict is applied as repeated pseudo-observations rather than by
recomputing history, so a reviewer's opinion moves the posterior without
invalidating the automatic signal already banked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cleanroom.config import Settings, settings
from cleanroom.learning.memory import MemoryStore
from cleanroom.learning.reward import HUMAN_SCALE, WEIGHTS
from cleanroom.learning.store import load_bandit, load_episodes, save_bandit

#: Word forms accepted from every interface.
VERDICTS: dict[str, int] = {
    "good": 1, "up": 1, "+1": 1, "1": 1, "yes": 1,
    "ok": 0, "neutral": 0, "0": 0,
    "bad": -1, "down": -1, "-1": -1, "no": -1,
}


class FeedbackError(ValueError):
    pass


@dataclass
class FeedbackResult:
    episode: int
    bucket: str
    strategy: str
    value: int
    posterior_mean: float
    pulls: int
    best_arm: str
    changed_best: bool


def parse_verdict(raw: str) -> int:
    value = VERDICTS.get(str(raw).strip().lower())
    if value is None:
        raise FeedbackError(f"verdict must be one of good | ok | bad (got {raw!r})")
    if value not in HUMAN_SCALE:
        raise FeedbackError(f"verdict {raw!r} maps outside the reward scale")
    return value


def find_episode(episode: int, cfg: Settings | None = None) -> dict[str, Any]:
    cfg = cfg or settings
    records = {int(r.get("episode", -1)): r for r in load_episodes(cfg)}
    record = records.get(int(episode))
    if not record:
        known = sorted(records)[-8:]
        raise FeedbackError(f"episode {episode} not found (recent: {known})")
    return record


def apply_episode_feedback(
    episode: int,
    verdict: str | int,
    cfg: Settings | None = None,
    *,
    memory: MemoryStore | None = None,
) -> FeedbackResult:
    """Fold a reviewer's verdict on one episode into the strategy posterior."""
    cfg = cfg or settings
    value = verdict if isinstance(verdict, int) else parse_verdict(verdict)
    if value not in HUMAN_SCALE:
        raise FeedbackError(f"verdict {verdict!r} maps outside the reward scale")

    record = find_episode(episode, cfg)
    bucket, strategy = record["bucket"], record["strategy"]

    bandit = load_bandit(cfg)
    best_before = bandit.best_arm(bucket)

    # Weighted as repeated observations so one review outranks one auto-episode,
    # which is the whole point of a sparse high-trust channel.
    repeats = max(1, int(round(WEIGHTS["human"])))
    stats = None
    for _ in range(repeats):
        stats = bandit.update(bucket, strategy, HUMAN_SCALE[value])
    save_bandit(bandit, cfg)

    (memory or MemoryStore(cfg)).add(
        f"Reviewer scored {strategy} on {bucket} as {value:+d} "
        f"(episode {episode}, {str(record.get('url', ''))[:80]}).",
        tags=(bucket, strategy, "human-feedback"),
        weight=2.0,
    )

    best_after = bandit.best_arm(bucket)
    return FeedbackResult(
        episode=int(episode),
        bucket=bucket,
        strategy=strategy,
        value=value,
        posterior_mean=stats.posterior_mean if stats else 0.0,
        pulls=stats.pulls if stats else 0,
        best_arm=best_after,
        changed_best=best_after != best_before,
    )


def _describe(row: dict[str, Any], limit: int = 3) -> str:
    """A short, quotable description of a row, for the lesson text."""
    parts = []
    for key, value in row.items():
        if key == "source_url" or value in (None, ""):
            continue
        parts.append(f"{key}={value!r}")
        if len(parts) >= limit:
            break
    return ", ".join(parts)


def reject_row(
    row: dict[str, Any],
    *,
    reason: str = "",
    episode: int | None = None,
    cfg: Settings | None = None,
    memory: MemoryStore | None = None,
) -> str:
    """Record that a specific extracted row is wrong, and why.

    This is the grain of feedback a posterior cannot hold. The lesson names the
    actual values, so the next synthesis on a similar page sees a concrete
    counterexample rather than a number.
    """
    cfg = cfg or settings
    url = str(row.get("source_url") or "")
    host = re.sub(r"^www\.", "", (re.split(r"/+", url)[1] if "//" in url else url).lower())

    detail = reason.strip() or "reviewer marked it wrong"
    text = (
        f"Reviewer rejected a row from {host or 'an unknown source'}: "
        f"{_describe(row)}. Reason: {detail}. Do not emit rows of this shape."
    )
    tags = tuple(t for t in (host, "row-rejected", "human-feedback") if t)
    (memory or MemoryStore(cfg)).add(text, tags=tags, weight=2.2)
    return text


def suggest_constraint(row: dict[str, Any], schema: dict) -> str | None:
    """Propose a schema constraint that would have caught this row.

    Feedback is more valuable when it hardens the validator than when it only
    nudges a posterior: a `deny_pattern` rejects the whole *class* of bad rows
    forever, whereas a lesson only influences the next prompt.
    """
    numeric_types = {"number", "integer"}
    specs = {f.get("name"): f for f in (schema.get("fields") or []) if f.get("name")}

    for name, value in row.items():
        spec = specs.get(name)
        if not spec or value in (None, ""):
            continue

        if spec.get("type") in numeric_types:
            try:
                number = float(str(value).replace(",", ""))
            except (TypeError, ValueError):
                continue
            cap = spec.get("max")
            if cap is not None and number > float(cap) * 0.5:
                return f'tighten {name}: "max" is {cap}, but {number} is implausible'
            continue

        text = str(value)
        if re.search(r"\b\d+\s*(?:hour|hr|day|week|month)s?\b", text, re.I):
            return (
                f'add to {name}: "deny_pattern": '
                r'"\\b\\d+\\s*(?:hour|hr|day|week|month)s?\\b"'
            )
        if len(text) > 45:
            return f'add to {name}: "max_length": 40  (got {len(text)} chars)'
    return None
