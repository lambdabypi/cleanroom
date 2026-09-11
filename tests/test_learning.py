"""Tests for the learning core.

These run with no credentials and no network, in about a second. That matters at
a hackathon: it means you can refactor the bandit at 2am and know immediately
whether you broke the thing the whole demo rests on.
"""

from __future__ import annotations

import json

import pytest

from cleanroom.learning.bandit import PRIOR_ALPHA, PRIOR_BETA, ThompsonBandit
from cleanroom.learning.reward import compute_reward
from cleanroom.learning.strategies import BUCKETS, STRATEGY_IDS, bucket_for, featurise
from cleanroom.pipeline.validate import validate_row, validate_rows

SCHEMA = {
    "name": "test",
    "fields": [
        {"name": "provider", "type": "string", "required": True},
        {"name": "usd_per_hour", "type": "number", "required": True, "min": 0, "max": 100},
        {"name": "gpu_count", "type": "integer", "required": False, "min": 1},
        {"name": "source_url", "type": "string", "required": True, "format": "url"},
    ],
}


def _row(**over):
    base = {
        "provider": "Lambda",
        "usd_per_hour": 2.49,
        "gpu_count": 8,
        "source_url": "https://example.com/pricing",
    }
    base.update(over)
    return base


# -- bandit ------------------------------------------------------------------


def test_bandit_converges_on_the_better_arm():
    """The whole demo rests on this: given a genuinely better arm, find it."""
    bandit = ThompsonBandit(arms=["good", "bad"], buckets=["b"], seed=7)
    for _ in range(120):
        arm = bandit.select("b")
        bandit.update("b", arm, 0.9 if arm == "good" else 0.1)

    assert bandit.best_arm("b") == "good"
    # And it should have stopped wasting pulls on the loser.
    stats = bandit.stats_for("b")
    assert stats["good"].pulls > stats["bad"].pulls * 2


def test_fractional_reward_updates_posterior_proportionally():
    bandit = ThompsonBandit(arms=["a"], buckets=["b"], seed=1)
    bandit.update("b", "a", 0.25)
    stats = bandit.stats_for("b")["a"]
    assert stats.alpha == pytest.approx(PRIOR_ALPHA + 0.25)
    assert stats.beta == pytest.approx(PRIOR_BETA + 0.75)
    assert stats.pulls == 1


def test_reward_is_clamped_to_unit_interval():
    bandit = ThompsonBandit(arms=["a"], buckets=["b"], seed=1)
    bandit.update("b", "a", 5.0)
    bandit.update("b", "a", -3.0)
    stats = bandit.stats_for("b")["a"]
    # Two pulls, one clamped to 1.0 and one to 0.0.
    assert stats.reward_sum == pytest.approx(1.0)
    assert 0.0 <= stats.posterior_mean <= 1.0


def test_discount_lets_the_agent_change_its_mind():
    """Non-stationarity: an arm that stops working must lose its crown."""
    bandit = ThompsonBandit(arms=["a", "b"], buckets=["x"], discount=0.8, seed=3)
    for _ in range(30):
        bandit.update("x", "a", 1.0)
    assert bandit.best_arm("x") == "a"
    for _ in range(30):
        bandit.update("x", "a", 0.0)
        bandit.update("x", "b", 1.0)
    assert bandit.best_arm("x") == "b"


def test_greedy_selection_is_deterministic():
    bandit = ThompsonBandit(arms=["a", "b"], buckets=["x"], seed=5)
    bandit.update("x", "a", 1.0)
    assert {bandit.select("x", greedy=True) for _ in range(20)} == {"a"}


def test_snapshot_round_trips_through_json():
    bandit = ThompsonBandit(arms=list(STRATEGY_IDS), buckets=list(BUCKETS), discount=0.98, seed=2)
    for _ in range(15):
        arm = bandit.select("prose")
        bandit.update("prose", arm, 0.6)

    restored = ThompsonBandit.restore(json.loads(json.dumps(bandit.snapshot())))
    assert restored.total_pulls == bandit.total_pulls
    assert restored.discount == bandit.discount
    for arm, stats in bandit.stats_for("prose").items():
        assert restored.stats_for("prose")[arm].alpha == pytest.approx(stats.alpha)


def test_unseen_bucket_is_registered_not_fatal():
    bandit = ThompsonBandit(arms=["a"], buckets=["known"], seed=1)
    assert bandit.select("brand_new") == "a"
    assert "brand_new" in bandit.buckets


def test_unknown_arm_is_rejected():
    bandit = ThompsonBandit(arms=["a"], buckets=["b"], seed=1)
    with pytest.raises(KeyError):
        bandit.update("b", "nope", 1.0)


# -- context bucketing -------------------------------------------------------


def test_buckets_discriminate_page_shapes():
    table = "| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
    listing = "\n".join(f"- item {i}" for i in range(9))
    sectioned = "\n\n".join(f"## Section {i}\n\nSome prose here." for i in range(5))
    prose = "Just a paragraph of ordinary text with nothing structural about it."

    assert bucket_for(table) == "table_heavy"
    assert bucket_for(listing) == "list_heavy"
    assert bucket_for(sectioned) == "sectioned"
    assert bucket_for(prose) == "prose"
    assert bucket_for("") == "prose"


def test_featurise_counts_structure():
    features = featurise("| a |\n| b |\n- one\n- two\n# Head")
    assert features.table_rows >= 2
    assert features.list_items == 2
    assert features.headings == 1


# -- validation --------------------------------------------------------------


def test_valid_row_passes_and_coerces():
    clean, errors = validate_row(_row(usd_per_hour="$2.49", gpu_count="8"), SCHEMA)
    assert errors == []
    assert clean["usd_per_hour"] == pytest.approx(2.49)
    assert clean["gpu_count"] == 8


def test_missing_required_field_fails():
    clean, errors = validate_row(_row(provider=None), SCHEMA)
    assert clean is None
    assert any("provider" in e for e in errors)


def test_out_of_range_and_bad_url_fail():
    _, errors = validate_row(_row(usd_per_hour=5000), SCHEMA)
    assert any("above max" in e for e in errors)
    _, errors = validate_row(_row(source_url="not-a-url"), SCHEMA)
    assert any("http(s) URL" in e for e in errors)


def test_pii_is_rejected():
    """Clean Data, enforced where the reward can see it."""
    _, errors = validate_row(_row(provider="contact sales@vendor.com"), SCHEMA)
    assert any("PII" in e for e in errors)
    _, errors = validate_row(_row(provider="call 415-555-0199"), SCHEMA)
    assert any("PII" in e for e in errors)


def test_non_integer_rejected_for_integer_field():
    _, errors = validate_row(_row(gpu_count=2.5), SCHEMA)
    assert any("not an integer" in e for e in errors)


def test_blank_sentinels_count_as_missing():
    _, errors = validate_row(_row(provider="n/a"), SCHEMA)
    assert any("provider" in e for e in errors)


def test_validate_rows_dedupes_and_scores():
    rows = [_row(), _row(), _row(provider="RunPod"), {"provider": "broken"}]
    report = validate_rows(rows, SCHEMA)
    assert report["rows_total"] == 4
    assert report["rows_valid"] == 2      # one duplicate dropped, one invalid
    assert report["rows_duplicate"] == 1
    assert report["rows_with_source"] == 2
    assert report["field_error_counts"]


TIGHT_SCHEMA = {
    "name": "tight",
    "fields": [
        {
            "name": "provider",
            "type": "string",
            "required": True,
            "max_length": 40,
            "deny_pattern": r"\b\d+\s*(?:hour|hr|day)s?\b|lasting|experiment",
        },
        {
            "name": "gpu_model",
            "type": "string",
            "required": True,
            "pattern": r"H100|A100|B200|RTX",
        },
        {"name": "usd_per_hour", "type": "number", "required": True, "min": 0.01, "max": 100},
        {"name": "source_url", "type": "string", "required": True, "format": "url"},
    ],
}


def _tight(**over):
    base = {
        "provider": "Lambda",
        "gpu_model": "H100 SXM",
        "usd_per_hour": 2.49,
        "source_url": "https://example.com/p",
    }
    base.update(over)
    return base


def test_deny_pattern_rejects_durations_masquerading_as_providers():
    """The real failure: an extractor reading a cost-example table, scoring 0.92."""
    for junk in ("10 hours", "1 hour", "One-GPU experiment lasting 4 hours"):
        clean, errors = validate_row(_tight(provider=junk), TIGHT_SCHEMA)
        assert clean is None, f"{junk!r} should be rejected"
        assert any("provider" in e for e in errors)


def test_required_pattern_rejects_a_gpu_model_that_names_no_gpu():
    clean, errors = validate_row(_tight(gpu_model="1"), TIGHT_SCHEMA)
    assert clean is None
    assert any("gpu_model" in e for e in errors)


def test_total_cost_figures_are_rejected_as_hourly_rates():
    """$215/hour is a 10-hour total, not a price. The max bound catches it."""
    for total in (215.2, 516.48, 242.1):
        clean, _ = validate_row(_tight(usd_per_hour=total), TIGHT_SCHEMA)
        assert clean is None


def test_a_genuine_row_still_passes_the_tighter_schema():
    clean, errors = validate_row(_tight(), TIGHT_SCHEMA)
    assert errors == []
    assert clean["provider"] == "Lambda"


def test_a_broken_schema_pattern_does_not_block_rows():
    """A typo in a schema regex must not silently reject everything."""
    broken = {
        "name": "broken",
        "fields": [{"name": "provider", "type": "string", "required": True,
                    "pattern": "((("}],
    }
    clean, errors = validate_row({"provider": "Lambda"}, broken)
    assert errors == []
    assert clean == {"provider": "Lambda"}


def test_source_url_is_injected_when_the_model_omits_it():
    """The bug this prevents: 9 perfectly good rows scoring 0.0.

    The model was asked to hardcode the source URL into generated code and left
    it None, so every row failed `source_url: missing (required)`. The host knows
    the URL, so it injects it -- attribution is structural, not learned.
    """
    rows = [_row(source_url=None), _row(provider="RunPod", source_url=None)]
    report = validate_rows(rows, SCHEMA, "https://example.com/pricing")
    assert report["rows_valid"] == 2
    assert report["rows_with_source"] == 2
    assert all(r["source_url"] == "https://example.com/pricing" for r in report["rows"])


def test_injection_does_not_overwrite_a_real_url():
    rows = [_row(source_url="https://original.example/page")]
    report = validate_rows(rows, SCHEMA, "https://injected.example/other")
    assert report["rows"][0]["source_url"] == "https://original.example/page"


def test_without_injection_missing_urls_still_fail():
    """Injection is opt-in per call, so the validator stays strict on its own."""
    report = validate_rows([_row(source_url=None)], SCHEMA)
    assert report["rows_valid"] == 0
    assert any("source_url" in e for e in report["sample_errors"])


def test_validate_rows_handles_non_list():
    report = validate_rows({"not": "a list"}, SCHEMA)
    assert report["rows_valid"] == 0
    assert report["sample_errors"]


# -- reward ------------------------------------------------------------------


def test_crash_scores_zero():
    assert compute_reward(None, crashed=True).total == 0.0
    assert compute_reward({"rows_total": 0}).total == 0.0


def test_perfect_extraction_scores_near_one():
    report = {
        "rows_total": 10,
        "rows_valid": 10,
        "cells_expected": 40,
        "cells_filled": 40,
        "rows_with_source": 10,
    }
    assert compute_reward(report, expected_rows=10).total == pytest.approx(1.0)


def test_partial_extraction_lands_between():
    report = {
        "rows_total": 10,
        "rows_valid": 5,
        "cells_expected": 20,
        "cells_filled": 12,
        "rows_with_source": 5,
    }
    total = compute_reward(report).total
    assert 0.2 < total < 0.85


def test_unattributed_rows_are_penalised():
    """Two runs, identical except attribution. The cited one must win."""
    base = {"rows_total": 10, "rows_valid": 10, "cells_expected": 40, "cells_filled": 40}
    cited = compute_reward({**base, "rows_with_source": 10}).total
    uncited = compute_reward({**base, "rows_with_source": 0}).total
    assert cited > uncited


def test_human_channel_dominates_when_present():
    report = {
        "rows_total": 10,
        "rows_valid": 10,
        "cells_expected": 40,
        "cells_filled": 40,
        "rows_with_source": 10,
    }
    assert compute_reward(report, human=-1).total < 0.45
    assert compute_reward(report, human=1).total == pytest.approx(1.0)


def test_out_of_range_human_signal_is_noted_not_fatal():
    result = compute_reward(
        {"rows_total": 2, "rows_valid": 2, "cells_expected": 8,
         "cells_filled": 8, "rows_with_source": 2},
        human=9,
    )
    assert result.notes
    assert result.total > 0
