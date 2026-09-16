"""Durable state: the bandit posterior and the episode log.

Learning that does not survive a process restart is not learning, it is a long
function call. Writes go through a temp-file rename so a crash mid-episode cannot
leave a truncated posterior on disk -- which matters more than it sounds, because
a corrupt bandit.json silently resets the agent to naive and the demo curve
flattens with no error.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from cleanroom.config import Settings, settings
from cleanroom.learning.bandit import ThompsonBandit
from cleanroom.learning.strategies import BUCKETS, STRATEGY_IDS


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_bandit(cfg: Settings | None = None, *, seed: int | None = None) -> ThompsonBandit:
    """Load the posterior, or start a fresh one on first run / corrupt state."""
    cfg = cfg or settings
    path = cfg.bandit_path
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            bandit = ThompsonBandit.restore(blob, seed=seed)
            if bandit.arms:
                return bandit
        except (json.JSONDecodeError, TypeError, ValueError):
            # Corrupt state is recoverable; keep the bad file for inspection so a
            # flat learning curve during the demo has an explanation on disk.
            path.replace(path.with_suffix(".corrupt.json"))
    return ThompsonBandit(arms=STRATEGY_IDS, buckets=BUCKETS, discount=0.98,
                           seed=seed, min_pulls=1)


def save_bandit(bandit: ThompsonBandit, cfg: Settings | None = None) -> Path:
    cfg = cfg or settings
    _write_atomic(cfg.bandit_path, json.dumps(bandit.snapshot(), indent=2, sort_keys=True))
    return cfg.bandit_path


def load_profile_bandit(cfg: Settings | None = None, *, seed: int | None = None):
    """Load the execution-profile posterior, or start fresh.

    Kept in its own file from the strategy bandit: the two have different action
    spaces and different reward definitions, and conflating them in one blob
    would make either one hard to reset independently while tuning.
    """
    from cleanroom.learning.budget import PROFILE_IDS, ProfileBandit

    cfg = cfg or settings
    path = cfg.budget_path
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            bandit = ProfileBandit.restore(blob, seed=seed)
            if bandit.bandit.arms:
                return bandit
        except (json.JSONDecodeError, TypeError, ValueError):
            path.replace(path.with_suffix(".corrupt.json"))
    return ProfileBandit(ThompsonBandit(arms=PROFILE_IDS, buckets=BUCKETS,
                                        discount=0.98, seed=seed))


def save_profile_bandit(bandit, cfg: Settings | None = None) -> Path:
    cfg = cfg or settings
    _write_atomic(cfg.budget_path, json.dumps(bandit.snapshot(), indent=2, sort_keys=True))
    return cfg.budget_path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())

    def _gen() -> Iterator[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    return _gen()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EpisodeRecord:
    """One pull of one arm. The episode log is the demo's learning curve."""

    episode: int
    timestamp: str
    url: str
    bucket: str
    strategy: str
    reward: float
    rows_total: int
    rows_valid: int
    crashed: bool
    reward_detail: dict[str, Any]
    memory_hits: list[str]
    duration_s: float
    published: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "timestamp": self.timestamp,
            "url": self.url,
            "bucket": self.bucket,
            "strategy": self.strategy,
            "reward": round(self.reward, 4),
            "rows_total": self.rows_total,
            "rows_valid": self.rows_valid,
            "crashed": self.crashed,
            "reward_detail": self.reward_detail,
            "memory_hits": self.memory_hits,
            "duration_s": round(self.duration_s, 2),
            "published": self.published,
        }


def log_episode(record: EpisodeRecord, cfg: Settings | None = None) -> None:
    cfg = cfg or settings
    append_jsonl(cfg.episodes_path, record.as_dict())


def episode_count(cfg: Settings | None = None) -> int:
    cfg = cfg or settings
    return sum(1 for _ in read_jsonl(cfg.episodes_path))


def load_episodes(cfg: Settings | None = None) -> list[dict[str, Any]]:
    cfg = cfg or settings
    return list(read_jsonl(cfg.episodes_path))

