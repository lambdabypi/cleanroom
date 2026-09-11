"""Per-component call ledger: latency, failures, and money.

Every outbound call is wrapped in `ledger.track(...)`, which records how long it
took, whether it worked, and what it cost. Two things fall out of that:

1. **Observability.** `cleanroom costs` can show, per component, the call count,
   failure rate, p50/p95 latency and total spend. The $120 Agents bill would have
   been obvious after two episodes with this in place.
2. **Adaptive execution.** `ComponentHealth` turns the same stream into a signal
   the agent acts on: a component that keeps failing gets its circuit opened and
   the agent falls back instead of retrying into a wall. That is the difference
   between "remembers what worked" and "notices the tool is down".

The ledger is process-wide by default because threading it through every call
site would add a parameter to functions that otherwise do not care. It is still
injectable for tests.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

from cleanroom.config import Settings, settings
from cleanroom.learning.store import append_jsonl, read_jsonl, utcnow
from cleanroom.observability.pricing import CostEstimate


@dataclass
class CallRecord:
    component: str          # "you" | "daytona" | "writer" | "one"
    operation: str          # "you.search", "daytona.exec", "writer.synthesize", ...
    latency_s: float
    ok: bool
    cost_usd: float = 0.0
    free_tier: bool = False
    cost_basis: str = ""
    error: str = ""
    episode: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        blob = asdict(self)
        blob["timestamp"] = self.timestamp or utcnow()
        blob["latency_s"] = round(self.latency_s, 3)
        blob["cost_usd"] = round(self.cost_usd, 6)
        return blob


class Span:
    """Mutable handle yielded by `track`, so a call can report what it learned."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self.cost_usd = 0.0
        self.free_tier = False
        self.cost_basis = ""
        self.meta: dict[str, Any] = {}
        self.ok: bool | None = None
        self.error = ""

    def charge(self, estimate: CostEstimate) -> None:
        self.cost_usd = estimate.usd
        self.free_tier = estimate.free_tier
        self.cost_basis = estimate.basis

    def note(self, **fields: Any) -> None:
        self.meta.update(fields)

    def fail(self, error: str) -> None:
        """Mark a soft failure -- a call that returned but did not work."""
        self.ok = False
        self.error = error[:300]


@dataclass
class ComponentStats:
    component: str
    calls: int = 0
    failures: int = 0
    cost_usd: float = 0.0
    billed_usd: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]

    @property
    def total_latency(self) -> float:
        return sum(self.latencies)


class ComponentHealth:
    """Rolling success rate with a circuit breaker.

    EWMA rather than a plain mean so that a component which recovers is trusted
    again quickly -- a lifetime average would keep punishing a tool for an outage
    that ended ten episodes ago.
    """

    def __init__(self, alpha: float = 0.35, trip_after: int = 3) -> None:
        self.alpha = alpha
        self.trip_after = trip_after
        self._success: dict[str, float] = {}
        self._consecutive: dict[str, int] = {}
        self._last_error: dict[str, str] = {}

    def record(self, component: str, ok: bool, error: str = "") -> None:
        prior = self._success.get(component, 1.0)
        self._success[component] = (1 - self.alpha) * prior + self.alpha * (1.0 if ok else 0.0)
        if ok:
            self._consecutive[component] = 0
        else:
            self._consecutive[component] = self._consecutive.get(component, 0) + 1
            self._last_error[component] = error[:300]

    def success_rate(self, component: str) -> float:
        return self._success.get(component, 1.0)

    def consecutive_failures(self, component: str) -> int:
        return self._consecutive.get(component, 0)

    def last_error(self, component: str) -> str:
        return self._last_error.get(component, "")

    def is_open(self, component: str) -> bool:
        """True when the component should be skipped rather than retried."""
        return self.consecutive_failures(component) >= self.trip_after

    def prefer(self, candidates: list[str]) -> list[str]:
        """Order fallback candidates by observed reliability, healthiest first."""
        return sorted(
            candidates,
            key=lambda c: (self.is_open(c), -self.success_rate(c)),
        )

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            component: {
                "success_rate": round(rate, 3),
                "consecutive_failures": self.consecutive_failures(component),
                "circuit_open": self.is_open(component),
                "last_error": self.last_error(component),
            }
            for component, rate in sorted(self._success.items())
        }


class CallLedger:
    def __init__(self, cfg: Settings | None = None, path: Path | None = None) -> None:
        self.cfg = cfg or settings
        self.path = path if path is not None else (self.cfg.state_dir / "calls.jsonl")
        self.records: list[CallRecord] = []
        self.health = ComponentHealth()
        self.episode: int | None = None

    # -- recording ---------------------------------------------------------

    @contextmanager
    def track(self, operation: str, *, component: str | None = None) -> Iterator[Span]:
        span = Span(operation)
        started = time.monotonic()
        comp = component or operation.split(".", 1)[0]
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            self._commit(comp, span, time.monotonic() - started, False,
                         f"{type(exc).__name__}: {exc}")
            raise
        else:
            ok = True if span.ok is None else span.ok
            self._commit(comp, span, time.monotonic() - started, ok, span.error)

    def _commit(self, component: str, span: Span, latency: float, ok: bool, error: str) -> None:
        record = CallRecord(
            component=component,
            operation=span.operation,
            latency_s=latency,
            ok=ok,
            cost_usd=span.cost_usd,
            free_tier=span.free_tier,
            cost_basis=span.cost_basis,
            error=error,
            episode=self.episode,
            meta=dict(span.meta),
            timestamp=utcnow(),
        )
        self.records.append(record)
        self.health.record(component, ok, error)
        if self.path is not None:
            append_jsonl(self.path, record.as_dict())

    # -- aggregation -------------------------------------------------------

    def stats(self, records: list[CallRecord] | None = None) -> dict[str, ComponentStats]:
        out: dict[str, ComponentStats] = {}
        for record in records if records is not None else self.records:
            stats = out.setdefault(record.component, ComponentStats(record.component))
            stats.calls += 1
            stats.failures += 0 if record.ok else 1
            stats.cost_usd += record.cost_usd
            stats.billed_usd += 0.0 if record.free_tier else record.cost_usd
            stats.latencies.append(record.latency_s)
        return out

    @property
    def total_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def billed_usd(self) -> float:
        return sum(0.0 if r.free_tier else r.cost_usd for r in self.records)

    def episode_cost(self, episode: int) -> float:
        return sum(r.cost_usd for r in self.records if r.episode == episode)

    def load_history(self) -> list[CallRecord]:
        """Every call ever recorded for this state dir, not just this process."""
        history: list[CallRecord] = []
        if self.path is None:
            return history
        for raw in read_jsonl(self.path):
            try:
                history.append(
                    CallRecord(
                        component=str(raw.get("component") or "?"),
                        operation=str(raw.get("operation") or "?"),
                        latency_s=float(raw.get("latency_s") or 0.0),
                        ok=bool(raw.get("ok")),
                        cost_usd=float(raw.get("cost_usd") or 0.0),
                        free_tier=bool(raw.get("free_tier")),
                        cost_basis=str(raw.get("cost_basis") or ""),
                        error=str(raw.get("error") or ""),
                        episode=raw.get("episode"),
                        meta=raw.get("meta") or {},
                        timestamp=str(raw.get("timestamp") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
        return history


# -- process-wide default ----------------------------------------------------

_LEDGER: CallLedger | None = None


def get_ledger(cfg: Settings | None = None) -> CallLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = CallLedger(cfg)
    return _LEDGER


def set_ledger(ledger: CallLedger | None) -> None:
    """Swap the default ledger. Tests use this to get an isolated, file-less one."""
    global _LEDGER
    _LEDGER = ledger


def reset_ledger() -> None:
    set_ledger(None)
