"""The learning half of the agent: action space, bandit, reward, memory, state."""

from cleanroom.learning.bandit import ArmStats, ThompsonBandit
from cleanroom.learning.reward import RewardReport, compute_reward
from cleanroom.learning.strategies import (
    BUCKETS,
    STRATEGIES,
    STRATEGY_IDS,
    bucket_for,
    featurise,
)

__all__ = [
    "ArmStats",
    "ThompsonBandit",
    "RewardReport",
    "compute_reward",
    "BUCKETS",
    "STRATEGIES",
    "STRATEGY_IDS",
    "bucket_for",
    "featurise",
]
