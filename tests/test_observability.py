"""Tests for the ledger, pricing, health, and the efficiency bandit.

The cost tests matter more than they look: a wrong price silently understates
spend, which is exactly the failure that produced a $120 surprise on the You.com
Agents API. The $15/call figure is asserted here so a careless edit breaks a test
instead of a budget.
"""

from __future__ import annotations

import pytest

from cleanroom.learning.budget import (
    LAMBDA,
    PROFILE_IDS,
    PROFILES,
    REF_TOKENS,
    ProfileBandit,
    normalised_cost,
    utility,
)
from cleanroom.observability.ledger import CallLedger, ComponentHealth
from cleanroom.observability.pricing import (
    call_cost,
    estimate_tokens,
    is_free_tier,
    llm_cost,
    sandbox_cost,
    token_price,
)


@pytest.fixture()
def ledger() -> CallLedger:
    # path=None keeps it entirely in memory -- no state dir side effects.
    return CallLedger(path=None)


# -- pricing -----------------------------------------------------------------


def test_you_agents_is_priced_at_fifteen_dollars_per_call():
    """Measured on the billing dashboard. Eight calls came to $120."""
    assert call_cost("you.agents").usd == pytest.approx(15.00)
    assert call_cost("you.agents").usd * 8 == pytest.approx(120.00)


def test_search_is_effectively_free_next_to_the_agents_api():
    search = call_cost("you.search").usd
    assert search == pytest.approx(0.005)
    assert call_cost("you.agents").usd / search == pytest.approx(3000)


def test_unpriced_operation_is_zero_not_an_error():
    assert call_cost("something.unknown").usd == 0.0


def test_llm_cost_uses_token_prices():
    cost = llm_cost(model="claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    assert cost.usd == pytest.approx(5.00)
    cost = llm_cost(model="claude-opus-5", input_tokens=0, output_tokens=1_000_000)
    assert cost.usd == pytest.approx(25.00)


def test_unknown_model_is_flagged_rather_than_guessed():
    cost = llm_cost(model="some-new-model", input_tokens=1000, output_tokens=100)
    assert cost.usd == 0.0
    assert "unknown model" in cost.basis


def test_free_tier_hosts_are_recognised():
    assert is_free_tier("https://api.groq.com/openai/v1")
    assert is_free_tier("http://localhost:11434/v1")
    assert not is_free_tier("https://api.anthropic.com")


def test_free_tier_reports_estimate_but_zero_billed():
    cost = llm_cost(
        model="llama-3.3-70b",
        input_tokens=1_000_000,
        output_tokens=0,
        base_url="https://api.groq.com/openai/v1",
    )
    assert cost.usd > 0        # what it would have cost
    assert cost.billed_usd == 0.0   # what you were probably charged


def test_token_price_matches_by_substring():
    assert token_price("anthropic/claude-opus-5") is not None
    assert token_price("groq/llama-3.3-70b-versatile") is not None
    assert token_price("totally-made-up") is None


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 400) == 100


def test_sandbox_cost_scales_with_seconds():
    assert sandbox_cost(0).usd == 0.0
    assert sandbox_cost(100).usd > sandbox_cost(10).usd


# -- ledger ------------------------------------------------------------------


def test_track_records_latency_and_success(ledger):
    with ledger.track("you.search") as span:
        span.charge(call_cost("you.search"))
    assert len(ledger.records) == 1
    record = ledger.records[0]
    assert record.component == "you"
    assert record.ok
    assert record.latency_s >= 0
    assert record.cost_usd == pytest.approx(0.005)


def test_exception_is_recorded_then_reraised(ledger):
    with pytest.raises(ValueError):
        with ledger.track("daytona.run"):
            raise ValueError("sandbox exploded")

    assert len(ledger.records) == 1
    assert not ledger.records[0].ok
    assert "sandbox exploded" in ledger.records[0].error


def test_soft_failure_is_recorded_without_raising(ledger):
    with ledger.track("you.contents") as span:
        span.fail("returned no markdown")
    assert not ledger.records[0].ok
    assert "no markdown" in ledger.records[0].error


def test_cost_is_charged_even_when_the_call_then_fails(ledger):
    """At $15/call an unusable Agents response still costs $15."""
    with pytest.raises(RuntimeError):
        with ledger.track("you.agents") as span:
            span.charge(call_cost("you.agents"))
            raise RuntimeError("truncated output")
    assert ledger.total_usd == pytest.approx(15.00)


def test_stats_aggregate_per_component(ledger):
    for _ in range(3):
        with ledger.track("you.search") as span:
            span.charge(call_cost("you.search"))
    with pytest.raises(ValueError):
        with ledger.track("you.search"):
            raise ValueError("boom")

    stats = ledger.stats()["you"]
    assert stats.calls == 4
    assert stats.failures == 1
    assert stats.failure_rate == pytest.approx(0.25)
    assert stats.cost_usd == pytest.approx(0.015)


def test_percentiles_do_not_crash_on_a_single_call(ledger):
    with ledger.track("you.search"):
        pass
    stats = ledger.stats()["you"]
    assert stats.p50 >= 0
    assert stats.p95 >= 0


def test_episode_attribution(ledger):
    ledger.episode = 7
    with ledger.track("you.search") as span:
        span.charge(call_cost("you.search"))
    assert ledger.episode_cost(7) == pytest.approx(0.005)
    assert ledger.episode_cost(8) == 0.0


# -- health / circuit breaker ------------------------------------------------


def test_circuit_opens_after_consecutive_failures():
    health = ComponentHealth(trip_after=3)
    for _ in range(2):
        health.record("you", False, "500")
    assert not health.is_open("you")
    health.record("you", False, "500")
    assert health.is_open("you")


def test_one_success_closes_the_circuit():
    health = ComponentHealth(trip_after=2)
    health.record("you", False)
    health.record("you", False)
    assert health.is_open("you")
    health.record("you", True)
    assert not health.is_open("you")


def test_ewma_recovers_faster_than_a_lifetime_mean():
    """A tool that comes back must be trusted again, not punished forever."""
    health = ComponentHealth(alpha=0.35)
    for _ in range(10):
        health.record("you", False)
    low = health.success_rate("you")
    for _ in range(5):
        health.record("you", True)
    assert health.success_rate("you") > low + 0.5


def test_prefer_orders_healthy_components_first():
    health = ComponentHealth(trip_after=2)
    for _ in range(3):
        health.record("contents", False)
    health.record("full_page", True)
    assert health.prefer(["contents", "full_page"])[0] == "full_page"


# -- efficiency bandit -------------------------------------------------------


def test_normalised_cost_is_bounded():
    assert normalised_cost(input_tokens=0, llm_calls=1) == 0.0
    assert normalised_cost(input_tokens=10**9, llm_calls=9) == 1.0


def test_repairs_add_cost():
    one = normalised_cost(input_tokens=1000, llm_calls=1)
    two = normalised_cost(input_tokens=1000, llm_calls=2)
    assert two > one


def test_utility_penalises_an_expensive_tie():
    """Equal quality, unequal spend -> the cheap profile must win."""
    cheap = utility(value=0.9, input_tokens=1500, llm_calls=1)
    pricey = utility(value=0.9, input_tokens=REF_TOKENS, llm_calls=1)
    assert cheap > pricey


def test_utility_still_prefers_quality_when_the_gap_is_large():
    cheap_bad = utility(value=0.2, input_tokens=1500, llm_calls=1)
    pricey_good = utility(value=0.95, input_tokens=REF_TOKENS, llm_calls=1)
    assert pricey_good > cheap_bad


def test_utility_is_clamped_to_unit_interval():
    assert 0.0 <= utility(value=0.0, input_tokens=10**9, llm_calls=5) <= 1.0
    assert 0.0 <= utility(value=1.0, input_tokens=0, llm_calls=1) <= 1.0


def test_lambda_controls_thrift():
    value, tokens = 0.9, REF_TOKENS
    assert utility(value=value, input_tokens=tokens, llm_calls=1) == pytest.approx(
        value - LAMBDA, abs=1e-6
    )


def test_profile_bandit_learns_the_cheap_profile_when_quality_matches():
    """The headline claim of the efficiency loop, on a synthetic environment
    where the lean profile is genuinely good enough."""
    bandit = ProfileBandit(seed=11)
    for _ in range(120):
        profile = bandit.select("table_heavy")
        # A clean table parses fine from a small excerpt: same value either way.
        bandit.update(
            "table_heavy",
            profile.id,
            value=0.92,
            input_tokens=profile.doc_chars // 4,
            llm_calls=1 + profile.max_repairs,
        )
    assert bandit.best("table_heavy").id == "lean"


def test_profile_bandit_pays_up_when_the_cheap_profile_fails():
    bandit = ProfileBandit(seed=13)
    quality = {"lean": 0.1, "standard": 0.45, "thorough": 0.95}
    for _ in range(150):
        profile = bandit.select("prose")
        bandit.update(
            "prose",
            profile.id,
            value=quality[profile.id],
            input_tokens=profile.doc_chars // 4,
            llm_calls=1,
        )
    assert bandit.best("prose").id == "thorough"


def _run_coupled_learners(
    *, gated: bool, episodes: int = 160, seed: int = 21, floor: int = 1
) -> str:
    """Two bandits learning at once on a page shape that genuinely needs context.

    Returns the profile the cost bandit settles on. `gated` applies the
    on-policy rule: only learn about spend from episodes that used the
    currently-best-known strategy. `floor` is the cold-start minimum-pulls
    setting, exposed so the two interventions can be measured apart.
    """
    from cleanroom.learning.bandit import ThompsonBandit
    from cleanroom.learning.strategies import BUCKETS, STRATEGY_IDS

    strategies = ThompsonBandit(arms=STRATEGY_IDS, buckets=BUCKETS,
                                discount=0.98, seed=seed, min_pulls=floor)
    profiles = ProfileBandit(
        ThompsonBandit(arms=PROFILE_IDS, buckets=BUCKETS, discount=0.98,
                       seed=seed, min_pulls=floor)
    )

    right_strategy = "regex_fields"
    # Prose: the cheap profile is genuinely bad, the expensive one genuinely good.
    quality = {"lean": 0.12, "standard": 0.45, "thorough": 0.90}

    for _ in range(episodes):
        strategy = strategies.select("prose")
        profile = profiles.select("prose")
        best_known = strategies.best_arm("prose")

        value = quality[profile.id] * (1.0 if strategy == right_strategy else 0.3)
        strategies.update("prose", strategy, value)

        if not gated or strategy == best_known:
            profiles.update(
                "prose", profile.id, value=value,
                input_tokens=profile.doc_chars // 4, llm_calls=1,
            )

    return profiles.best("prose").id


def _correct_fraction(*, gated: bool, episodes: int, seeds: int = 20,
                      floor: int = 1) -> int:
    return sum(
        1
        for seed in range(seeds)
        if _run_coupled_learners(gated=gated, episodes=episodes, seed=seed,
                                 floor=floor) == "thorough"
    )


def test_cold_start_floor_and_on_policy_gate_compose():
    """Two independent fixes for the same failure, measured apart.

    Two bandits learning at once confound each other: while the strategy
    dimension explores, every profile scores badly, so the cheapest wins on cost
    alone and the posterior can commit to the wrong profile. Separately, with
    identical Beta(1,1) priors and five arms, Thompson sampling can simply never
    try an arm in a short run.

    Measured over 20 seeds (correct answer = `thorough`):

        episodes | no floor, ungated | no floor, gated | floor, ungated | floor, gated
              15 |             11/20 |           15/20 |          17/20 |        20/20
              20 |             15/20 |           17/20 |          18/20 |        20/20
              30 |             15/20 |           19/20 |          19/20 |        20/20
              50 |             18/20 |           20/20 |          20/20 |        20/20
             100 |             19/20 |           20/20 |          20/20 |        20/20

    The floor is the larger single lever at short horizons; the gate still adds
    on top of it. Asserted at 15 episodes, where the spread is widest.
    """
    both = _correct_fraction(gated=True, episodes=15, floor=1)
    floor_only = _correct_fraction(gated=False, episodes=15, floor=1)
    gate_only = _correct_fraction(gated=True, episodes=15, floor=0)
    neither = _correct_fraction(gated=False, episodes=15, floor=0)

    assert both >= 19, f"floor+gate should be near-perfect at 15 episodes, got {both}/20"
    assert both >= floor_only, f"gate should not hurt: {both} vs {floor_only}"
    assert floor_only > neither, f"floor should help: {floor_only} vs {neither}"
    assert gate_only > neither, f"gate should help: {gate_only} vs {neither}"


def test_everything_converges_given_enough_episodes():
    """Both fixes buy speed, not asymptotic correctness -- state that honestly."""
    assert _correct_fraction(gated=False, episodes=160, floor=0) >= 19
    assert _correct_fraction(gated=True, episodes=160, floor=1) == 20


def test_profiles_are_ordered_by_increasing_budget():
    budgets = [PROFILES[p].doc_chars for p in PROFILE_IDS]
    assert budgets == sorted(budgets)


def test_profile_bandit_round_trips():
    bandit = ProfileBandit(seed=3)
    for _ in range(10):
        profile = bandit.select("prose")
        bandit.update("prose", profile.id, value=0.6,
                      input_tokens=2000, llm_calls=1)
    restored = ProfileBandit.restore(bandit.snapshot())
    assert restored.total_pulls == bandit.total_pulls
