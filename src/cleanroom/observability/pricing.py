"""What each external call costs.

These are **list prices recorded on 2026-09-11**, and the You.com figures were
read off the billing dashboard rather than inferred:

    Web Search API        $5 per 1000 calls   -> $0.005 / call
    Agent API - Advanced  $15 per call        -> eight calls cost $120

They will drift. `cleanroom costs` prints the price book it used so a stale
number is visible in the output instead of silently wrong, and every entry can be
overridden from the environment (see `_env_overrides`).

Token counts come from the provider's own `usage` block whenever one is returned;
the chars/4 heuristic is only a fallback so an un-instrumented provider still
produces a usable estimate rather than a zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Per-call prices for fixed-price endpoints, in USD.
CALL_PRICES: dict[str, float] = {
    "you.search": 0.005,        # $5 / 1k calls
    "you.contents": 0.005,      # metered with search on the dashboard
    "you.agents": 15.00,        # Agent API - Advanced. Yes, per call.
}

#: (input, output) USD per million tokens, matched by substring against the
#: configured model name so that provider prefixes and tags do not matter.
TOKEN_PRICES: dict[str, tuple[float, float]] = {
    # Free tiers -- priced at their paid rate so a run still reports what it
    # *would* cost, with `free_tier` noting that you were probably not charged.
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "qwen": (0.0, 0.0),          # local via Ollama
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Providers whose free tier normally absorbs a hackathon-sized run.
FREE_TIER_HOSTS = ("groq.com", "cerebras.ai", "localhost", "127.0.0.1", "openrouter.ai")

#: Daytona bills for sandbox uptime. The rate depends on the resource tier, and
#: it is small next to the code writer, so this is a rough placeholder -- the
#: ledger records real wall-clock seconds either way, which is the number that
#: actually matters when deciding whether to reuse a sandbox.
DAYTONA_USD_PER_SANDBOX_SECOND = float(
    os.getenv("DAYTONA_USD_PER_SANDBOX_SECOND", "0.00003")
)


def _env_overrides() -> None:
    """Allow `CLEANROOM_PRICE_you.agents=0` style overrides."""
    for key, value in os.environ.items():
        if not key.startswith("CLEANROOM_PRICE_"):
            continue
        name = key[len("CLEANROOM_PRICE_") :]
        try:
            CALL_PRICES[name] = float(value)
        except ValueError:
            continue


_env_overrides()


@dataclass(frozen=True)
class CostEstimate:
    usd: float
    basis: str
    #: True when the configured provider's free tier very likely covered this.
    free_tier: bool = False

    @property
    def billed_usd(self) -> float:
        return 0.0 if self.free_tier else self.usd


def estimate_tokens(text: str) -> int:
    """Rough token count. Only used when the provider returns no usage block."""
    return max(1, len(text or "") // 4)


def token_price(model: str) -> tuple[float, float] | None:
    needle = (model or "").lower()
    for key, price in TOKEN_PRICES.items():
        if key in needle:
            return price
    return None


def is_free_tier(base_url: str) -> bool:
    host = (base_url or "").lower()
    return any(marker in host for marker in FREE_TIER_HOSTS)


def llm_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    base_url: str = "",
) -> CostEstimate:
    price = token_price(model)
    free = is_free_tier(base_url)
    if price is None:
        return CostEstimate(
            0.0,
            f"unknown model {model!r} -- add it to TOKEN_PRICES for a real figure",
            free_tier=free,
        )
    usd = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
    basis = f"{input_tokens} in + {output_tokens} out @ ${price[0]}/${price[1]} per Mtok"
    return CostEstimate(usd, basis, free_tier=free)


def call_cost(operation: str) -> CostEstimate:
    if operation in CALL_PRICES:
        return CostEstimate(CALL_PRICES[operation], f"${CALL_PRICES[operation]:.4f} per call")
    return CostEstimate(0.0, "not priced")


def sandbox_cost(seconds: float) -> CostEstimate:
    usd = max(0.0, seconds) * DAYTONA_USD_PER_SANDBOX_SECOND
    return CostEstimate(usd, f"{seconds:.1f}s @ ${DAYTONA_USD_PER_SANDBOX_SECOND}/s")


def price_book() -> dict[str, object]:
    """Everything the cost report was computed from, for printing alongside it."""
    return {
        "recorded": "2026-09-11 (You.com figures from the billing dashboard)",
        "per_call_usd": dict(CALL_PRICES),
        "per_mtok_usd": {k: {"in": v[0], "out": v[1]} for k, v in TOKEN_PRICES.items()},
        "daytona_usd_per_sandbox_second": DAYTONA_USD_PER_SANDBOX_SECOND,
        "free_tier_hosts": list(FREE_TIER_HOSTS),
    }
