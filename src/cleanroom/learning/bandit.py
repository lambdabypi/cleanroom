"""A contextual Thompson-sampling bandit over extraction strategies.

Why a bandit and not policy-gradient RL: we need something that demonstrably
converges inside a hackathon demo. With ~5 arms per bucket a Beta-Bernoulli
posterior separates good arms from bad ones in 15-30 episodes, needs no GPU, and
its state is a readable JSON file rather than an opaque checkpoint.

Reward is continuous in [0, 1] (fraction of extracted rows that validate), not a
coin flip. The standard treatment for bounded rewards applies: update the Beta
posterior with fractional pseudo-counts, `alpha += r` and `beta += 1 - r`. The
posterior mean stays an unbiased estimate of the arm's expected reward, and
Thompson sampling's regret guarantees carry over for rewards in [0, 1].

`discount` handles non-stationarity. Sites change shape, and an arm that was
right yesterday can stop working; pulling old evidence toward the prior each
update keeps the agent able to change its mind.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0


@dataclass
class ArmStats:
    """Beta posterior plus bookkeeping for one (bucket, strategy) pair."""

    alpha: float = PRIOR_ALPHA
    beta: float = PRIOR_BETA
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def observed_mean(self) -> float:
        """Raw average reward. Differs from the posterior mean at low pull counts,
        where the prior still dominates -- useful for spotting under-explored arms."""
        return self.reward_sum / self.pulls if self.pulls else 0.0

    @property
    def posterior_sd(self) -> float:
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))


class ThompsonBandit:
    """Per-bucket Thompson sampling over a fixed set of arms."""

    def __init__(
        self,
        arms: Sequence[str],
        buckets: Sequence[str],
        *,
        discount: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not arms:
            raise ValueError("bandit needs at least one arm")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        self.arms = list(arms)
        self.buckets = list(buckets)
        self.discount = discount
        self._rng = random.Random(seed)
        self._stats: dict[str, dict[str, ArmStats]] = {
            bucket: {arm: ArmStats() for arm in self.arms} for bucket in self.buckets
        }

    # -- core loop ---------------------------------------------------------

    def select(self, bucket: str, *, greedy: bool = False) -> str:
        """Pick an arm. Sampling from each posterior and taking the argmax *is* the
        exploration mechanism -- there is no epsilon to tune."""
        stats = self._bucket(bucket)
        if greedy:
            return max(self.arms, key=lambda arm: stats[arm].posterior_mean)
        draws = {arm: self._rng.betavariate(s.alpha, s.beta) for arm, s in stats.items()}
        return max(draws, key=draws.__getitem__)

    def update(self, bucket: str, arm: str, reward: float) -> ArmStats:
        """Fold one observed reward into the posterior. `reward` is clamped to [0, 1]."""
        if arm not in self.arms:
            raise KeyError(f"unknown arm {arm!r}")
        r = min(1.0, max(0.0, float(reward)))
        stats = self._bucket(bucket)[arm]

        if self.discount < 1.0:
            # Shrink accumulated evidence toward the prior before adding the new
            # observation, so old data fades instead of dominating forever.
            stats.alpha = PRIOR_ALPHA + self.discount * (stats.alpha - PRIOR_ALPHA)
            stats.beta = PRIOR_BETA + self.discount * (stats.beta - PRIOR_BETA)

        stats.alpha += r
        stats.beta += 1.0 - r
        stats.pulls += 1
        stats.reward_sum += r
        return stats

    # -- introspection (drives the demo table) -----------------------------

    def _bucket(self, bucket: str) -> dict[str, ArmStats]:
        if bucket not in self._stats:
            # Unseen bucket: register it rather than fail. Keeps the loop running
            # if page shapes drift beyond the predefined set.
            self._stats[bucket] = {arm: ArmStats() for arm in self.arms}
            self.buckets.append(bucket)
        return self._stats[bucket]

    def stats_for(self, bucket: str) -> dict[str, ArmStats]:
        return dict(self._bucket(bucket))

    def ranking(self, bucket: str) -> list[tuple[str, ArmStats]]:
        return sorted(
            self._bucket(bucket).items(),
            key=lambda kv: kv[1].posterior_mean,
            reverse=True,
        )

    def best_arm(self, bucket: str) -> str:
        return self.ranking(bucket)[0][0]

    @property
    def total_pulls(self) -> int:
        return sum(s.pulls for arms in self._stats.values() for s in arms.values())

    def seen_buckets(self) -> list[str]:
        return [b for b, arms in self._stats.items() if any(s.pulls for s in arms.values())]

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "arms": self.arms,
            "buckets": self.buckets,
            "discount": self.discount,
            "stats": {
                bucket: {arm: asdict(s) for arm, s in arms.items()}
                for bucket, arms in self._stats.items()
            },
        }

    @classmethod
    def restore(cls, blob: dict, *, seed: int | None = None) -> "ThompsonBandit":
        bandit = cls(
            arms=blob.get("arms") or [],
            buckets=blob.get("buckets") or [],
            discount=float(blob.get("discount", 1.0)),
            seed=seed,
        )
        for bucket, arms in (blob.get("stats") or {}).items():
            target = bandit._bucket(bucket)
            for arm, raw in arms.items():
                if arm in target:
                    target[arm] = ArmStats(**raw)
        return bandit
