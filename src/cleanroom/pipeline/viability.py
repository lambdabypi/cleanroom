"""Can this page possibly satisfy the schema, before we spend an episode on it?

A page that contains none of the data the schema asks for will return zero rows
no matter which strategy is chosen. Spending an episode on it costs an LLM call,
a sandbox run and a bandit pull -- and, worse, writes a `0.0` reward against a
strategy that did nothing wrong. Six such episodes on one barren source flattened
an entire bucket's posteriors in a measured run: every arm looked broken when the
truth was that the page was empty.

Observed case: `getdeploying.com/gpus/nvidia-h100` returns 1,455 characters of
"How this list works" boilerplate with zero prices and zero GPU names. The real
table is rendered client-side and never reaches the markdown. The correct
response is to skip the page, not to blame `list_items` for it.

The check is deliberately cheap and schema-driven: it looks for *evidence* that
each required field could be populated, using constraints the schema already
declares. No model call, no network, microseconds per page. It is a screen for
pages that are impossible, not a judgement of pages that are merely hard --
anything genuinely ambiguous is allowed through, because a false skip loses data
while a false accept only costs one episode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A *standalone* number, with optional thousands separators and decimals.
#: The boundaries are load-bearing. Without them the `100` inside `H100` counts
#: as a number in range, so any page naming a GPU appeared to have evidence of
#: an hourly price and the numeric check never rejected anything. The lookbehind
#: excludes digits embedded in identifiers (H100, A100, RTX4090) and the
#: lookahead excludes unit-suffixed specs (80GB, 100TB).
_NUMBER = re.compile(r"(?<![A-Za-z0-9.,])-?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")

#: A page shorter than this cannot hold a useful number of records; the
#: retrieval layer already drops pages under 200 chars as "no content".
MIN_USEFUL_CHARS = 400


@dataclass(frozen=True)
class Viability:
    ok: bool
    reason: str = ""
    #: Required fields for which no supporting evidence was found.
    missing: tuple[str, ...] = ()

    def __bool__(self) -> bool:  # lets callers write `if viability(...)`
        return self.ok


def _numbers_in_range(text: str, low: float | None, high: float | None) -> bool:
    """Is there at least one number that could satisfy this field's bounds?"""
    for match in _NUMBER.finditer(text):
        try:
            value = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        if low is not None and value < low:
            continue
        if high is not None and value > high:
            continue
        return True
    return False


def assess(document: str, schema: dict) -> Viability:
    """Decide whether `document` could plausibly yield a valid row.

    Only *required* fields are considered, and only where the schema declares a
    constraint concrete enough to test. A required string field with no `pattern`
    (a company name, say) is untestable and is therefore not held against the
    page.
    """
    text = document or ""
    if len(text.strip()) < MIN_USEFUL_CHARS:
        return Viability(False, f"only {len(text.strip())} chars of content")

    missing: list[str] = []
    for spec in schema.get("fields") or []:
        name = spec.get("name")
        if not name or not spec.get("required"):
            continue

        # The URL field is injected by the harness, so its absence from the page
        # says nothing.
        if spec.get("format") == "url" or name == "source_url":
            continue

        pattern = spec.get("pattern")
        if pattern:
            try:
                if not re.search(pattern, text, re.IGNORECASE):
                    missing.append(name)
            except re.error:
                pass  # unusable pattern is the schema's problem, not the page's
            continue

        if spec.get("type") in ("number", "integer"):
            if not _numbers_in_range(text, spec.get("min"), spec.get("max")):
                missing.append(name)

    if missing:
        return Viability(
            False,
            "no evidence for required field(s): " + ", ".join(missing),
            tuple(missing),
        )
    return Viability(True)


def filter_sources(sources, schema: dict):
    """Split sources into (viable, skipped) preserving order.

    `skipped` carries the reason so the caller can report *why* a page was
    dropped rather than silently shrinking the candidate list.
    """
    viable, skipped = [], []
    for source in sources:
        verdict = assess(getattr(source, "markdown", "") or "", schema)
        (viable if verdict.ok else skipped).append(
            source if verdict.ok else (source, verdict)
        )
    return viable, skipped
