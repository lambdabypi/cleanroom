"""Per-component observability: latency, failure rates, and spend.

Two consumers, which is why this is its own package rather than logging:

* `cleanroom costs` reads it for the human-facing report.
* `ComponentHealth` feeds it back into the agent, so a failing tool changes
  behaviour instead of just appearing in a log file.
"""

from cleanroom.observability.ledger import (
    CallLedger,
    CallRecord,
    ComponentHealth,
    ComponentStats,
    get_ledger,
    reset_ledger,
    set_ledger,
)
from cleanroom.observability.pricing import (
    CostEstimate,
    call_cost,
    llm_cost,
    price_book,
    sandbox_cost,
)

__all__ = [
    "CallLedger",
    "CallRecord",
    "ComponentHealth",
    "ComponentStats",
    "get_ledger",
    "reset_ledger",
    "set_ledger",
    "CostEstimate",
    "call_cost",
    "llm_cost",
    "price_book",
    "sandbox_cost",
]
