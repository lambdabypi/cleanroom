"""Tests for the shared human-feedback path.

Shared by the CLI and the dashboard, so these guard against the two interfaces
drifting -- the failure mode where both look like they work and disagree about
what the agent learned.

Each test gets its own state dir via a Settings override, so nothing here
touches a real run's posterior.
"""

from __future__ import annotations

import json

import pytest

from cleanroom.config import Settings
from cleanroom.learning.feedback import (
    FeedbackError,
    apply_episode_feedback,
    parse_verdict,
    reject_row,
    suggest_constraint,
)
from cleanroom.learning.memory import MemoryStore
from cleanroom.learning.store import EpisodeRecord, load_bandit, log_episode, save_bandit, utcnow

SCHEMA = {
    "name": "t",
    "fields": [
        {"name": "provider", "type": "string", "required": True, "max_length": 40},
        {"name": "usd_per_hour", "type": "number", "required": True, "min": 0.01, "max": 100},
        {"name": "source_url", "type": "string", "required": True, "format": "url"},
    ],
}


@pytest.fixture()
def cfg(tmp_path) -> Settings:
    return Settings(state_dir=tmp_path, you_api_key="x", anthropic_api_key="x")


def _seed_episode(cfg: Settings, *, episode: int = 1, bucket: str = "table_heavy",
                  strategy: str = "table_parse") -> None:
    log_episode(
        EpisodeRecord(
            episode=episode, timestamp=utcnow(), url="https://example.com/p",
            bucket=bucket, strategy=strategy, reward=0.5, rows_total=4, rows_valid=2,
            crashed=False,
            reward_detail={"counts_toward_learning": True, "repairs": 0},
            memory_hits=[], duration_s=1.0,
        ),
        cfg,
    )


# -- verdict parsing ---------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("good", 1), ("GOOD", 1), ("+1", 1), ("1", 1), ("yes", 1),
    ("ok", 0), ("neutral", 0), ("0", 0),
    ("bad", -1), ("down", -1), ("-1", -1), ("no", -1),
])
def test_verdict_words_and_numbers_both_parse(raw, expected):
    assert parse_verdict(raw) == expected


def test_unknown_verdict_is_rejected():
    with pytest.raises(FeedbackError):
        parse_verdict("sideways")


# -- episode feedback --------------------------------------------------------


def test_good_verdict_raises_the_posterior(cfg):
    _seed_episode(cfg)
    before = load_bandit(cfg).stats_for("table_heavy")["table_parse"].posterior_mean
    result = apply_episode_feedback(1, "good", cfg)
    assert result.posterior_mean > before
    assert result.value == 1


def test_bad_verdict_lowers_the_posterior(cfg):
    _seed_episode(cfg)
    before = load_bandit(cfg).stats_for("table_heavy")["table_parse"].posterior_mean
    result = apply_episode_feedback(1, "bad", cfg)
    assert result.posterior_mean < before


def test_feedback_persists_across_reloads(cfg):
    """Feedback that does not survive a restart is not feedback."""
    _seed_episode(cfg)
    result = apply_episode_feedback(1, "good", cfg)
    reloaded = load_bandit(cfg).stats_for("table_heavy")["table_parse"]
    assert reloaded.posterior_mean == pytest.approx(result.posterior_mean)
    assert reloaded.pulls == result.pulls


def test_human_verdict_outweighs_a_single_auto_episode(cfg):
    """The channel is weighted 2.5x; one review must beat one machine episode."""
    _seed_episode(cfg)
    bandit = load_bandit(cfg)
    bandit.update("table_heavy", "table_parse", 0.0)   # one bad auto-episode
    save_bandit(bandit, cfg)
    low = load_bandit(cfg).stats_for("table_heavy")["table_parse"].posterior_mean

    result = apply_episode_feedback(1, "good", cfg)
    assert result.posterior_mean > low + 0.15


def test_feedback_can_change_the_best_arm(cfg):
    _seed_episode(cfg, strategy="list_items")
    bandit = load_bandit(cfg)
    for _ in range(6):
        bandit.update("table_heavy", "table_parse", 0.9)
    save_bandit(bandit, cfg)
    assert load_bandit(cfg).best_arm("table_heavy") == "table_parse"

    # Hammer the incumbent down, then promote the challenger.
    for _ in range(3):
        apply_episode_feedback(1, "good", cfg)
    result = apply_episode_feedback(1, "good", cfg)
    assert result.best_arm == "list_items"
    assert result.changed_best or load_bandit(cfg).best_arm("table_heavy") == "list_items"


def test_unknown_episode_is_reported_clearly(cfg):
    _seed_episode(cfg)
    with pytest.raises(FeedbackError, match="not found"):
        apply_episode_feedback(999, "good", cfg)


def test_feedback_writes_a_retrievable_lesson(cfg):
    _seed_episode(cfg)
    apply_episode_feedback(1, "bad", cfg)
    lessons = MemoryStore(cfg, prefer_one=False).search("table_parse table_heavy", limit=5)
    assert any("Reviewer scored" in l.text for l in lessons)


# -- row rejection -----------------------------------------------------------


def test_reject_row_names_the_offending_values(cfg):
    row = {"provider": "10 hours", "usd_per_hour": "215.20",
           "source_url": "https://example.com/p"}
    text = reject_row(row, reason="not a provider", cfg=cfg,
                      memory=MemoryStore(cfg, prefer_one=False))
    assert "10 hours" in text
    assert "not a provider" in text
    assert "example.com" in text


def test_rejected_row_lesson_is_retrievable(cfg):
    memory = MemoryStore(cfg, prefer_one=False)
    reject_row(
        {"provider": "10 hours", "source_url": "https://example.com/p"},
        reason="duration", cfg=cfg, memory=memory,
    )
    assert any("rejected" in l.text.lower() for l in memory.search("example.com", limit=5))


def test_reject_row_works_without_a_reason(cfg):
    text = reject_row({"provider": "x", "source_url": "https://e.com/p"}, cfg=cfg,
                      memory=MemoryStore(cfg, prefer_one=False))
    assert "marked it wrong" in text


# -- constraint suggestion ---------------------------------------------------


def test_suggests_a_deny_pattern_for_a_duration():
    hint = suggest_constraint({"provider": "10 hours", "usd_per_hour": 2.0}, SCHEMA)
    assert hint and "deny_pattern" in hint and "provider" in hint


def test_suggests_tightening_max_for_an_implausible_number():
    hint = suggest_constraint({"provider": "Lambda", "usd_per_hour": 515.0}, SCHEMA)
    assert hint and "usd_per_hour" in hint


def test_suggests_max_length_for_prose():
    long_name = "One GPU serving for 3 hours a day over 30 days of production"
    hint = suggest_constraint({"provider": long_name}, SCHEMA)
    assert hint and ("max_length" in hint or "deny_pattern" in hint)


def test_no_suggestion_for_a_clean_row():
    assert suggest_constraint(
        {"provider": "Lambda", "usd_per_hour": 2.49,
         "source_url": "https://e.com/p"},
        SCHEMA,
    ) is None
