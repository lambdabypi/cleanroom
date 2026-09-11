"""Experiential memory -- the second, textual learning channel.

The bandit learns *which* strategy to reach for. It cannot learn "this site puts
the price in a data-attribute, not the cell text". That kind of lesson is
specific, textual, and worth carrying into the next attempt, so it goes here.

Primary backend is One's memory store (`one mem add` / `one mem search`), which is
what gives the lessons tags and weights and makes them visible in One's dashboard
during a demo. When the One CLI is absent we fall back to a local JSONL file with
token-overlap scoring -- not as good, but it keeps the loop running offline, which
matters when hackathon wifi fails.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

from cleanroom.config import Settings, settings
from cleanroom.learning.store import append_jsonl, read_jsonl, utcnow

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or of to in for on with is was be from that this it as at by".split()
)

#: Record type used in One's memory store. One namespaces by type, so a stable
#: name keeps Cleanroom's lessons separable from anything else in the workspace.
ONE_RECORD_TYPE = "cleanroom_lesson"


def _run(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run the One CLI with explicit UTF-8 decoding.

    `text=True` alone decodes with the locale codepage; on Windows that is
    cp1252, and One emits UTF-8 (typographic apostrophes in action titles),
    which raises UnicodeDecodeError inside subprocess's reader thread.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _one_weight(weight: float) -> int:
    """Map our 1.0-2.5 importance scale onto One's integer 1-10.

    One rejects out-of-range weights, and our floats are not its units.
    """
    return max(1, min(10, int(round(weight * 3))))


@dataclass(frozen=True)
class Lesson:
    text: str
    tags: tuple[str, ...]
    weight: float
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tags": list(self.tags),
            "weight": self.weight,
            "timestamp": self.timestamp or utcnow(),
        }


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOPWORDS}


class MemoryStore:
    """One-backed lesson store with a local fallback."""

    def __init__(self, cfg: Settings | None = None, *, prefer_one: bool = True) -> None:
        self.cfg = cfg or settings
        self._one = shutil.which("one") if prefer_one else None
        self.backend = "one" if (self._one and self.cfg.one_secret) else "local"

    # -- writes ------------------------------------------------------------

    def add(self, text: str, *, tags: Sequence[str] = (), weight: float = 1.0) -> Lesson:
        lesson = Lesson(text=text.strip(), tags=tuple(tags), weight=weight, timestamp=utcnow())
        if self.backend == "one" and self._push_to_one(lesson):
            # Mirror locally too: One is the shared source of truth, the mirror
            # keeps `cleanroom report` working without a network round-trip.
            append_jsonl(self.cfg.memory_path, {**lesson.as_dict(), "synced": True})
            return lesson
        append_jsonl(self.cfg.memory_path, {**lesson.as_dict(), "synced": False})
        return lesson

    def _push_to_one(self, lesson: Lesson) -> bool:
        """`one mem add <type> <data-json> --tags <csv> --weight <1-10>`.

        Signature verified against One CLI 1.56.1. The first positional is a
        record *type*, not the text, and the second must be JSON -- passing the
        lesson text as the type silently fails and falls back to local storage,
        which is the kind of bug that only shows up as "why is the One dashboard
        empty".
        """
        payload = json.dumps(
            {
                "content": lesson.text,
                "tags": list(lesson.tags),
                "source": "cleanroom",
            }
        )
        cmd = [self._one, "--agent", "mem", "add", ONE_RECORD_TYPE, payload]
        if lesson.tags:
            cmd += ["--tags", ",".join(lesson.tags)]
        cmd += ["--weight", str(_one_weight(lesson.weight))]
        try:
            done = _run(cmd)
            return done.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    # -- reads -------------------------------------------------------------

    def search(self, query: str, *, limit: int = 3) -> list[Lesson]:
        """Retrieve lessons relevant to the page we are about to attempt."""
        if self.backend == "one":
            hits = self._search_one(query, limit=limit)
            if hits:
                return hits
        return self._search_local(query, limit=limit)

    def _search_one(self, query: str, *, limit: int) -> list[Lesson]:
        """`one mem search <query> --type <type> --limit <n>`."""
        cmd = [
            self._one, "--agent", "mem", "search", query,
            "--type", ONE_RECORD_TYPE,
            "--limit", str(limit),
        ]
        try:
            done = _run(cmd)
            if done.returncode != 0 or not done.stdout.strip():
                return []
            payload = json.loads(done.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return []

        rows = payload if isinstance(payload, list) else (
            payload.get("results") or payload.get("records") or payload.get("data") or []
        )
        lessons: list[Lesson] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            # The text lives inside the record's `data` payload we wrote, but be
            # liberal -- the envelope shape is not something to bet a run on.
            data = row.get("data") if isinstance(row.get("data"), dict) else {}
            text = (
                data.get("content")
                or row.get("content")
                or row.get("text")
                or row.get("memory")
                or ""
            )
            if not text:
                continue
            tags = data.get("tags") or row.get("tags") or ()
            lessons.append(
                Lesson(
                    text=str(text),
                    tags=tuple(tags) if isinstance(tags, (list, tuple)) else (),
                    weight=float(row.get("weight") or 5.0) / 3.0,
                    timestamp=str(row.get("created_at") or row.get("timestamp") or ""),
                )
            )
        return lessons

    def recent(self, limit: int = 10) -> list[Lesson]:
        """Most recently written lessons, newest first.

        Distinct from `search()` on purpose. A *display* surface -- the dashboard
        or `cleanroom report` -- wants "what has this agent learned", which is a
        recency question. Searching for a fixed term instead is how the lessons
        panel ended up silently empty: no lesson text happens to contain the word
        "extraction", so token overlap was zero and the panel just did not render.
        Retrieval during an episode still uses `search()`, where relevance to the
        page at hand is exactly what matters.
        """
        rows = list(read_jsonl(self.cfg.memory_path))
        lessons = [
            Lesson(
                text=str(raw.get("text") or ""),
                tags=tuple(raw.get("tags") or ()),
                weight=float(raw.get("weight") or 1.0),
                timestamp=str(raw.get("timestamp") or ""),
            )
            for raw in rows
            if raw.get("text")
        ]
        # The local mirror is append-only, so file order is chronological.
        return list(reversed(lessons))[:limit]

    def _search_local(self, query: str, *, limit: int) -> list[Lesson]:
        q = _tokens(query)
        if not q:
            return []
        scored: list[tuple[float, Lesson]] = []
        for raw in read_jsonl(self.cfg.memory_path):
            text = raw.get("text") or ""
            tags = tuple(raw.get("tags") or ())
            weight = float(raw.get("weight") or 1.0)
            overlap = _tokens(f"{text} {' '.join(tags)}") & q
            if not overlap:
                continue
            # Normalise by query length so long lessons do not automatically win.
            scored.append((len(overlap) / len(q) * weight, Lesson(text, tags, weight,
                                                                  str(raw.get("timestamp") or ""))))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [lesson for _, lesson in scored[:limit]]

    def count(self) -> int:
        return sum(1 for _ in read_jsonl(self.cfg.memory_path))


def lesson_from_failure(
    *,
    url: str,
    bucket: str,
    strategy: str,
    reward: float,
    report: dict | None,
    stderr: str = "",
) -> Lesson | None:
    """Write a lesson only when there is something specific to say.

    Logging "attempt scored 0.4" teaches nothing and pollutes retrieval. We keep
    the cases with a concrete, reusable cause.
    """
    host = re.sub(r"^www\.", "", (re.split(r"/+", url)[1] if "//" in url else url).lower())
    tags = (host, bucket, strategy)

    if stderr.strip():
        first = stderr.strip().splitlines()[-1][:280]
        return Lesson(
            text=f"{strategy} on {host} ({bucket}) crashed: {first}",
            tags=tags,
            weight=1.5,
        )

    if report and int(report.get("rows_total") or 0) == 0:
        return Lesson(
            text=(
                f"{strategy} on {host} ({bucket}) parsed cleanly but found zero rows -- "
                "the record boundary assumption is wrong for this layout."
            ),
            tags=tags,
            weight=1.3,
        )

    if report:
        errors = report.get("sample_errors") or []
        if errors:
            joined = "; ".join(str(e)[:120] for e in errors[:3])
            return Lesson(
                text=f"{strategy} on {host} ({bucket}) field errors: {joined}",
                tags=tags,
                weight=1.2,
            )

    if reward >= 0.85:
        return Lesson(
            text=f"{strategy} works well on {host} ({bucket}); reward {reward:.2f}.",
            tags=tags,
            weight=1.0,
        )
    return None
