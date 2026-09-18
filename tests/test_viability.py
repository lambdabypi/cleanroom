"""Tests for the source viability screen and the cold-start exploration floor.

Both exist because of one measured failure: a 30-episode run spent six episodes
on `getdeploying.com/gpus/nvidia-h100`, which returns 1,455 characters of
"How this list works" boilerplate with no prices and no GPU names because the
real table is client-rendered. Every episode correctly returned zero rows, and
every `0.0` was recorded against whichever strategy happened to be sampled --
flattening the whole `list_heavy` bucket. Thompson sampling also never sampled
`list_items` there across those six episodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from cleanroom.learning.bandit import ThompsonBandit
from cleanroom.pipeline.viability import MIN_USEFUL_CHARS, assess, filter_sources

SCHEMA = {
    "name": "gpu_pricing",
    "fields": [
        {"name": "provider", "type": "string", "required": True, "max_length": 40},
        {
            "name": "gpu_model",
            "type": "string",
            "required": True,
            "pattern": r"H100|A100|B200|L40|RTX",
        },
        {"name": "usd_per_hour", "type": "number", "required": True,
         "min": 0.01, "max": 100},
        {"name": "region", "type": "string", "required": False},
        {"name": "source_url", "type": "string", "required": True, "format": "url"},
    ],
}

# Condensed from the real page that caused the failure.
BARREN = (
    "### How this list works\n\n#### The two views\n\n"
    "By provider shows one card per company, placed where its best offer ranked. "
    "By configuration lists every offer.\n\n#### Order\n\n"
    "What you can rent comes first: in stock, then waitlist, then not reported, "
    "then out of stock. Priced offers rank ahead of quote-only.\n\n"
    "Within a group, five factors set the order:\n\n"
    "- Location: datacenter proximity, blended with provider HQ.\n"
    "- Price: hourly price, per GPU and in total.\n"
    "- Billing type: reserved ahead of spot.\n"
    "- Specs: more VRAM, vCPUs and RAM.\n"
    "- Provider diversity: repeat rows rank slightly lower.\n"
)

GOOD = (
    "# Cloud GPU pricing\n\nCompare prices across providers.\n\n"
    "| Provider | GPU | $/hr |\n| --- | --- | --- |\n"
    "| Lambda | H100 SXM | 2.49 |\n| RunPod | A100 40GB | 1.19 |\n"
    "| CoreWeave | H100 PCIe | 4.76 |\n" + "filler text to clear the length floor. " * 20
)


@dataclass
class FakeSource:
    url: str
    markdown: str


# -- the screen ---------------------------------------------------------------


def test_barren_page_is_rejected():
    """The exact failure case: boilerplate with no prices and no GPU names."""
    verdict = assess(BARREN, SCHEMA)
    assert not verdict.ok
    assert "gpu_model" in verdict.missing
    assert "usd_per_hour" in verdict.missing


def test_page_with_real_data_is_accepted():
    assert assess(GOOD, SCHEMA).ok


def test_short_page_is_rejected_with_a_length_reason():
    verdict = assess("H100 $2.49", SCHEMA)
    assert not verdict.ok
    assert "chars" in verdict.reason


def test_length_floor_boundary():
    padded = ("H100 at 2.49 per hour. " * 40)[: MIN_USEFUL_CHARS + 50]
    assert len(padded.strip()) >= MIN_USEFUL_CHARS
    assert assess(padded, SCHEMA).ok


def test_out_of_range_numbers_do_not_count_as_evidence():
    """A page quoting only $50,000 annual figures cannot fill an hourly rate."""
    text = ("The H100 costs 250000 per year and 30000 per month. " * 12)
    verdict = assess(text, SCHEMA)
    assert not verdict.ok
    assert "usd_per_hour" in verdict.missing
    assert "gpu_model" not in verdict.missing  # H100 is present


def test_untestable_required_field_is_not_held_against_the_page():
    """`provider` has no pattern, so its absence is unknowable and must not fail."""
    verdict = assess(GOOD, SCHEMA)
    assert "provider" not in verdict.missing


def test_url_field_is_never_required_of_the_page():
    """source_url is injected by the harness; the page need not contain it."""
    assert "source_url" not in (assess(BARREN, SCHEMA).missing)


def test_broken_pattern_does_not_reject_every_page():
    schema = {
        "name": "x",
        "fields": [{"name": "a", "type": "string", "required": True, "pattern": "((("}],
    }
    assert assess(GOOD, schema).ok


def test_verdict_is_truthy_and_falsy():
    assert assess(GOOD, SCHEMA)
    assert not assess(BARREN, SCHEMA)


def test_filter_sources_splits_and_keeps_reasons():
    sources = [
        FakeSource("https://good.example/p", GOOD),
        FakeSource("https://barren.example/p", BARREN),
    ]
    viable, skipped = filter_sources(sources, SCHEMA)
    assert [s.url for s in viable] == ["https://good.example/p"]
    assert len(skipped) == 1
    source, verdict = skipped[0]
    assert source.url == "https://barren.example/p"
    assert verdict.missing


# -- the exploration floor ----------------------------------------------------


def test_floor_tries_every_arm_before_exploiting():
    """The measured failure: 6 pulls over 5 arms never sampled `list_items`."""
    arms = ["a", "b", "c", "d", "e"]
    bandit = ThompsonBandit(arms=arms, buckets=["x"], seed=3, min_pulls=1)
    seen = []
    for _ in range(len(arms)):
        arm = bandit.select("x")
        seen.append(arm)
        bandit.update("x", arm, 0.0)
    assert sorted(seen) == sorted(arms), f"floor missed an arm: {seen}"


def test_without_the_floor_an_arm_can_be_missed():
    """Contrast case, so the floor's value is visible rather than asserted."""
    arms = ["a", "b", "c", "d", "e"]
    missed_somewhere = False
    for seed in range(12):
        bandit = ThompsonBandit(arms=arms, buckets=["x"], seed=seed, min_pulls=0)
        seen = set()
        for _ in range(len(arms)):
            arm = bandit.select("x")
            seen.add(arm)
            bandit.update("x", arm, 0.0)
        if len(seen) < len(arms):
            missed_somewhere = True
            break
    assert missed_somewhere, "expected at least one seed to skip an arm in 5 pulls"


def test_floor_does_not_prevent_later_exploitation():
    bandit = ThompsonBandit(arms=["good", "bad"], buckets=["x"], seed=5, min_pulls=1)
    for _ in range(80):
        arm = bandit.select("x")
        bandit.update("x", arm, 0.95 if arm == "good" else 0.05)
    assert bandit.best_arm("x") == "good"
    stats = bandit.stats_for("x")
    assert stats["good"].pulls > stats["bad"].pulls


def test_greedy_ignores_the_floor():
    bandit = ThompsonBandit(arms=["a", "b"], buckets=["x"], seed=1, min_pulls=3)
    bandit.update("x", "a", 1.0)
    assert bandit.select("x", greedy=True) == "a"


def test_min_pulls_round_trips_through_a_snapshot():
    bandit = ThompsonBandit(arms=["a", "b"], buckets=["x"], seed=1, min_pulls=2)
    restored = ThompsonBandit.restore(bandit.snapshot())
    assert restored.min_pulls == 2


def test_negative_min_pulls_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        ThompsonBandit(arms=["a"], buckets=["x"], min_pulls=-1)


# -- provider token ceiling ---------------------------------------------------


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


def _synth():
    """A compat synthesizer without touching the network."""
    from cleanroom.config import Settings
    from cleanroom.pipeline.openai_compat import OpenAICompatSynthesizer

    cfg = Settings(
        llm_base_url="https://example.test/v1",
        llm_api_key="k",
        llm_model="test-model",
    )
    return OpenAICompatSynthesizer(cfg)


def test_doc_cap_is_unknown_until_a_response_is_seen():
    """The limit is discovered, never assumed."""
    assert _synth().max_doc_chars is None


def test_doc_cap_is_learned_from_response_headers():
    """Measured on Groq's free tier: 8000 tokens per minute."""
    synth = _synth()
    synth._note_rate_limits(_FakeResponse({"x-ratelimit-limit-tokens": "8000"}))
    assert synth.tokens_per_minute == 8000
    assert synth.max_doc_chars == int(8000 * synth.DOC_SHARE_OF_TPM * 4)


def test_learned_cap_clips_the_oversized_profiles():
    """`thorough` asks for more tokens than a whole minute's allowance."""
    from cleanroom.learning.budget import PROFILES

    synth = _synth()
    synth._note_rate_limits(_FakeResponse({"x-ratelimit-limit-tokens": "8000"}))
    cap = synth.max_doc_chars
    assert PROFILES["lean"].doc_chars <= cap, "lean should fit without clipping"
    assert PROFILES["standard"].doc_chars > cap
    assert PROFILES["thorough"].doc_chars > cap


def test_a_generous_limit_clips_nothing():
    from cleanroom.learning.budget import PROFILES

    synth = _synth()
    synth._note_rate_limits(_FakeResponse({"x-ratelimit-limit-tokens": "500000"}))
    assert synth.max_doc_chars > PROFILES["thorough"].doc_chars


def test_backoff_path_executes_end_to_end():
    """Guards the integration, not just the pacer in isolation.

    `TokenPacer` was unit-tested and green while `_post_with_backoff` crashed on
    the first real call with `NameError: estimate_tokens` -- the helper was
    imported inside a different method. Testing the collaborator alone could not
    catch that; this drives the actual method.
    """
    synth = _synth()
    synth.tokens_per_minute = 8000
    calls = []

    def fake_post(system, user):
        calls.append((system, user))
        return "```python\ndef extract(d):\n    return []\n```", {
            "prompt_tokens": 100, "completion_tokens": 50,
        }

    synth._post = fake_post
    text, usage = synth._post_with_backoff("sys", "user")
    assert calls, "the underlying post was never reached"
    assert "def extract" in text
    assert usage["prompt_tokens"] == 100


def test_pacer_allows_calls_that_fit_the_window():
    from cleanroom.pipeline.openai_compat import TokenPacer

    pacer = TokenPacer()
    assert pacer.delay_for(1000, 8000) == 0.0
    pacer.note(1000)
    assert pacer.delay_for(1000, 8000) == 0.0


def test_pacer_waits_once_the_window_is_full():
    """5,260-token calls against an 8,000/min ceiling: the second must wait."""
    from cleanroom.pipeline.openai_compat import TokenPacer

    pacer = TokenPacer()
    pacer.note(5260)
    delay = pacer.delay_for(5260, 8000)
    assert delay > 0, "a second full-size call should not be allowed immediately"
    assert delay <= 61


def test_pacer_is_inert_without_a_known_limit():
    """The ceiling is discovered; before that, never stall the run."""
    from cleanroom.pipeline.openai_compat import TokenPacer

    assert TokenPacer().delay_for(999_999, None) == 0.0
    assert TokenPacer().delay_for(999_999, 0) == 0.0


def test_pacer_forgets_spend_older_than_a_minute():
    from cleanroom.pipeline.openai_compat import TokenPacer

    pacer = TokenPacer()
    # Backdate an event beyond the window.
    pacer._events.append((time.monotonic() - 61.0, 8000))
    assert pacer.delay_for(5000, 8000) == 0.0


def test_pacer_safety_margin_reserves_headroom():
    """The next call's cost is an estimate, so do not plan to the last token."""
    from cleanroom.pipeline.openai_compat import TokenPacer

    pacer = TokenPacer(safety=0.85)
    pacer.note(7000)
    assert pacer.delay_for(500, 8000) > 0  # 7500 > 6800 budget


def test_missing_or_junk_headers_leave_the_cap_unset():
    synth = _synth()
    synth._note_rate_limits(_FakeResponse({}))
    assert synth.tokens_per_minute is None
    synth._note_rate_limits(_FakeResponse({"x-ratelimit-limit-tokens": "not-a-number"}))
    assert synth.tokens_per_minute is None
    synth._note_rate_limits(_FakeResponse({"x-ratelimit-limit-tokens": "0"}))
    assert synth.tokens_per_minute is None


# -- per-minute vs per-day refusals -------------------------------------------
#
# These use response bodies captured verbatim from Groq on 2026-09-17, during a
# run that lost episodes to a daily cap while every header advertised a full
# per-minute bucket. Three wrong diagnoses were made before the body was read.


class _FakeHTTPResponse:
    """Enough of requests.Response for the status-code branches."""

    def __init__(self, status_code, headers=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.ok = 200 <= status_code < 300


_TPD_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in '
    "organization `org_01kx` service tier `on_demand` on tokens per day (TPD): "
    "Limit 200000, Used 199033, Requested 2721. Please try again in 12m37.728s. "
    'Need more tokens? Upgrade to Dev Tier today","type":"tokens",'
    '"code":"rate_limit_exceeded"}}'
)

_TPM_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` in '
    "organization `org_01kx` service tier `on_demand` on tokens per minute (TPM): "
    "Limit 8000, Used 7200, Requested 2721. Please try again in 5.52s. "
    'Need more tokens?","type":"tokens","code":"rate_limit_exceeded"}}'
)

#: The headers a daily refusal actually carried: the per-minute bucket reads as
#: completely free, which is why trusting headers alone is not enough.
_MISLEADING_HEADERS = {
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "8000",
    "x-ratelimit-reset-tokens": "1ms",
    "retry-after": "758",
}


def test_a_daily_cap_is_not_treated_as_a_retryable_rate_limit():
    """The whole point: waiting cannot clear a per-day budget inside a run."""
    import pytest

    from cleanroom.pipeline.synthesize import DailyQuotaExhausted, RateLimited

    synth = _synth()
    with pytest.raises(DailyQuotaExhausted) as caught:
        synth._raise_for_status(
            _FakeHTTPResponse(429, _MISLEADING_HEADERS, _TPD_BODY)
        )
    # Must NOT be routed down the retry path, which would spend five attempts
    # and ~200s of backoff and then blame the strategy for the failure.
    assert not isinstance(caught.value, RateLimited)
    message = str(caught.value)
    assert "DAILY" in message
    assert "199,033" in message and "200,000" in message, "restate the real figures"
    assert "12m37.728s" in message, "say when it resets"


def test_a_per_minute_cap_stays_retryable():
    import pytest

    from cleanroom.pipeline.synthesize import DailyQuotaExhausted, RateLimited

    synth = _synth()
    with pytest.raises(RateLimited) as caught:
        synth._raise_for_status(
            _FakeHTTPResponse(429, {"retry-after": "5.52"}, _TPM_BODY)
        )
    assert not isinstance(caught.value, DailyQuotaExhausted)
    assert caught.value.retry_after == pytest.approx(5.52)
    assert "7,200 of 8,000" in str(caught.value)


def test_daily_cap_aborts_rather_than_retrying_synthesis_twice():
    """`_run` retries once on failure; a spent daily budget must skip that.

    Otherwise one clear "the tier is exhausted" message is rewritten as
    "failed twice", which sends the reader hunting for a bug in the model's
    output instead of checking the quota.
    """
    import pytest

    from cleanroom.pipeline.synthesize import DailyQuotaExhausted

    synth = _synth()
    calls = []

    def fake_call(system, user):
        calls.append(user)
        raise DailyQuotaExhausted("daily budget gone")

    synth._call = fake_call
    with pytest.raises(DailyQuotaExhausted):
        synth._run("sys", "user", "table_parse", None)
    assert len(calls) == 1, f"should not retry a daily cap, made {len(calls)} calls"


def test_quota_detail_survives_an_unparseable_body():
    """A provider that phrases it differently must not crash the error path."""
    from cleanroom.pipeline.openai_compat import _quota_detail

    assert "no figures" in _quota_detail("429 Too Many Requests")
    assert "no figures" in _quota_detail("")


# -- per-model request shape (Anthropic backend) ------------------------------


def test_thinking_and_effort_only_go_to_models_that_accept_them():
    """Haiku 4.5 rejects `effort` and does not take adaptive thinking.

    Sending either fails the first call of a run, which is a costly way to find
    out the configured model and the request shape disagree.
    """
    from cleanroom.pipeline.synthesize import supports_adaptive_thinking

    for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
                  "claude-fable-5-1", "claude-sonnet-4-6"):
        assert supports_adaptive_thinking(model), model
    for model in ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-3-5-haiku", ""):
        assert not supports_adaptive_thinking(model), model


def test_haiku_request_omits_effort_and_thinking():
    """Drive the real request builder, not just the predicate."""
    from cleanroom.config import Settings
    from cleanroom.pipeline.synthesize import AnthropicSynthesizer

    sent = {}

    class _FakeMessages:
        def create(self, **kwargs):
            sent.update(kwargs)
            raise RuntimeError("stop here -- we only care about the request shape")

    class _FakeClient:
        messages = _FakeMessages()

    for model, expect_tuning in (("claude-haiku-4-5", False), ("claude-opus-5", True)):
        sent.clear()
        synth = AnthropicSynthesizer(
            Settings(anthropic_api_key="k", model=model), client=_FakeClient()
        )
        try:
            synth.synthesize(
                schema=SCHEMA, document="| gpu | price |\n| H100 | 2.50 |",
                source_url="https://example.test/p", strategy_id="table_parse",
                bucket="table_heavy",
            )
        except Exception:
            pass  # the fake always raises; the assertions are on what it received

        assert sent["model"] == model
        assert sent["output_config"]["format"]["type"] == "json_schema", \
            "structured output must be requested on every model"
        if expect_tuning:
            assert sent.get("thinking") == {"type": "adaptive"}
            assert sent["output_config"].get("effort") == "medium"
        else:
            assert "thinking" not in sent, "Haiku 4.5 does not take adaptive thinking"
            assert "effort" not in sent["output_config"], "Haiku 4.5 rejects effort"

