"""The second learning dimension: how *expensively* to solve each page.

The strategy bandit learns which extraction approach works. It has no opinion on
cost, so left alone it will happily spend 45,000 characters of context and two
repair turns on a page that a 6,000-character excerpt would have cracked.

This module adds a second contextual bandit over **execution profiles**. Same
Thompson sampling machinery, different action space and a different reward:

    utility = value - LAMBDA * normalised_cost

`value` is the extraction reward already computed. `normalised_cost` is what the
agent actually controls -- input tokens shipped to the code writer, plus a penalty
per extra repair call. So the profile bandit learns statements like "table-heavy
pages don't need the big context" while the strategy bandit separately learns
"table-heavy pages want table_parse". Neither could represent the other.

Cost is normalised in **tokens, not dollars**, on purpose. Tokens are what the
profile controls and they are provider-independent, so a posterior learned on
Groq stays meaningful after switching to Claude. Dollars are still recorded in
the ledger for reporting -- they just are not the learning signal.

LAMBDA is the exchange rate between quality and spend, and it is a product
decision rather than a fact. The default deliberately favours thrift: a profile
must earn its extra context, because the cheap profile is usually close.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cleanroom.learning.bandit import ThompsonBandit
from cleanroom.learning.strategies import BUCKETS


@dataclass(frozen=True)
class Profile:
    id: str
    doc_chars: int
    max_repairs: int
    summary: str


PROFILES: dict[str, Profile] = {
    "lean": Profile(
        id="lean",
        doc_chars=6_000,
        max_repairs=0,
        summary="6k chars of page, no repair turn. Cheapest; enough for clean tables.",
    ),
    "standard": Profile(
        id="standard",
        doc_chars=18_000,
        max_repairs=1,
        summary="18k chars, one repair turn. The general-purpose middle.",
    ),
    "thorough": Profile(
        id="thorough",
        doc_chars=45_000,
        max_repairs=2,
        summary="45k chars, two repairs. For sprawling or prose-heavy pages.",
    ),
}

PROFILE_IDS: tuple[str, ...] = tuple(PROFILES)

#: Reference cost = the most expensive profile's document budget, in tokens.
#: Using the max profile as the denominator keeps `normalised_cost` in [0, 1]
#: without needing to know anything about the provider.
REF_TOKENS = PROFILES["thorough"].doc_chars // 4

#: Cost units charged per repair call beyond the first synthesis call. A repair
#: resends the document *and* the previous code, so it is not free.
REPAIR_COST_UNITS = 0.5

#: Quality-per-token exchange rate. Higher means thriftier.
LAMBDA = float(os.getenv("CLEANROOM_COST_LAMBDA", "0.25"))


def normalised_cost(*, input_tokens: int, llm_calls: int) -> float:
    """Cost of one episode in [0, 1], from the levers the profile controls."""
    token_units = max(0, input_tokens) / REF_TOKENS
    repair_units = max(0, llm_calls - 1) * REPAIR_COST_UNITS
    return min(1.0, token_units + repair_units)


def utility(*, value: float, input_tokens: int, llm_calls: int) -> float:
    """Cost-penalised reward for the profile bandit, clamped to [0, 1]."""
    cost = normalised_cost(input_tokens=input_tokens, llm_calls=llm_calls)
    return max(0.0, min(1.0, value - LAMBDA * cost))


class ProfileBandit:
    """Contextual bandit over execution profiles, keyed by page shape."""

    def __init__(self, bandit: ThompsonBandit | None = None, *, seed: int | None = None) -> None:
        self.bandit = bandit or ThompsonBandit(
            arms=PROFILE_IDS, buckets=BUCKETS, discount=0.98, seed=seed, min_pulls=1
        )

    def select(self, bucket: str, *, greedy: bool = False) -> Profile:
        return PROFILES[self.bandit.select(bucket, greedy=greedy)]

    def update(
        self,
        bucket: str,
        profile_id: str,
        *,
        value: float,
        input_tokens: int,
        llm_calls: int,
    ) -> float:
        score = utility(value=value, input_tokens=input_tokens, llm_calls=llm_calls)
        self.bandit.update(bucket, profile_id, score)
        return score

    def best(self, bucket: str) -> Profile:
        return PROFILES[self.bandit.best_arm(bucket)]

    def ranking(self, bucket: str):
        return self.bandit.ranking(bucket)

    def seen_buckets(self) -> list[str]:
        return self.bandit.seen_buckets()

    @property
    def total_pulls(self) -> int:
        return self.bandit.total_pulls

    def snapshot(self) -> dict:
        return self.bandit.snapshot()

    @classmethod
    def restore(cls, blob: dict, *, seed: int | None = None) -> "ProfileBandit":
        return cls(ThompsonBandit.restore(blob, seed=seed))

