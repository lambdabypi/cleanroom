"""Turning a sandbox run into a scalar reward.

The reward is deliberately multi-channel. A single "did it parse" bit would let
the agent win by emitting one perfect row and dropping the rest of the page, so
validity is balanced against coverage and completeness.

Attribution used to be a reward channel here, on the theory that the agent should
be *rewarded* for citing its sources. That was the wrong mechanism: the host knows
the source URL authoritatively, so making the generated code responsible for it
just created a way to fail. It now gets injected by the validator, which makes
attribution a structural guarantee rather than something the agent might learn.
The channel is retained at low weight as a *check* -- if it ever drops below 1.0,
injection is broken and that should be visible in the reward, not silent.

Human feedback is a separate channel with a high weight but sparse arrival. When
present it dominates; when absent the automatic channels still produce a usable
gradient, which is what makes hundreds of episodes possible in one day.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Channel:
    name: str
    value: float
    weight: float
    detail: str = ""


@dataclass(frozen=True)
class RewardReport:
    total: float
    channels: tuple[Channel, ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "total": round(self.total, 4),
            "channels": {
                c.name: {"value": round(c.value, 4), "weight": c.weight, "detail": c.detail}
                for c in self.channels
            },
            "notes": list(self.notes),
        }

    def explain(self) -> str:
        parts = [f"{c.name}={c.value:.2f}(w{c.weight:g})" for c in self.channels]
        return f"reward {self.total:.3f} <- " + " ".join(parts)


#: Weights. Human feedback outranks everything; provenance is weighted low enough
#: that it shapes behaviour without letting a page of well-cited garbage score well.
WEIGHTS = {
    "validity": 1.0,
    "coverage": 0.7,
    "completeness": 0.6,
    # Low: attribution is enforced by injection, so this reads 1.0 in healthy
    # runs and only moves if that enforcement breaks.
    "provenance": 0.15,
    "human": 2.5,
}

HUMAN_SCALE = {-1: 0.0, 0: 0.5, 1: 1.0}


def compute_reward(
    report: dict | None,
    *,
    human: int | None = None,
    crashed: bool = False,
    expected_rows: int | None = None,
) -> RewardReport:
    """Collapse a validation report into a reward in [0, 1].

    `report` is the JSON emitted by the in-sandbox runner: it carries
    `rows_total`, `rows_valid`, `cells_expected`, `cells_filled`, and
    `rows_with_source`. A crash or an unparseable report scores exactly 0 -- the
    agent should feel broken code as strongly as it feels a wrong answer.
    """
    notes: list[str] = []

    if crashed or not report:
        notes.append("extractor crashed or produced no report")
        return RewardReport(total=0.0, channels=(), notes=tuple(notes))

    rows_total = max(0, int(report.get("rows_total") or 0))
    rows_valid = max(0, int(report.get("rows_valid") or 0))
    cells_expected = max(0, int(report.get("cells_expected") or 0))
    cells_filled = max(0, int(report.get("cells_filled") or 0))
    rows_sourced = max(0, int(report.get("rows_with_source") or 0))

    if rows_total == 0:
        notes.append("extractor ran but returned zero rows")
        return RewardReport(total=0.0, channels=(), notes=tuple(notes))

    channels: list[Channel] = [
        Channel(
            "validity",
            rows_valid / rows_total,
            WEIGHTS["validity"],
            f"{rows_valid}/{rows_total} rows satisfy the schema",
        ),
        Channel(
            "completeness",
            (cells_filled / cells_expected) if cells_expected else 0.0,
            WEIGHTS["completeness"],
            f"{cells_filled}/{cells_expected} cells non-empty",
        ),
        Channel(
            "provenance",
            rows_sourced / rows_total,
            WEIGHTS["provenance"],
            f"{rows_sourced}/{rows_total} rows carry a source URL",
        ),
    ]

    # Coverage only means something when we have a prior expectation of how many
    # records the page holds; the page featuriser supplies it where it can.
    if expected_rows and expected_rows > 0:
        channels.append(
            Channel(
                "coverage",
                min(1.0, rows_valid / expected_rows),
                WEIGHTS["coverage"],
                f"{rows_valid} valid rows against ~{expected_rows} expected",
            )
        )

    if human is not None:
        scaled = HUMAN_SCALE.get(int(human))
        if scaled is None:
            notes.append(f"ignored out-of-range human signal {human!r}")
        else:
            channels.append(
                Channel("human", scaled, WEIGHTS["human"], f"reviewer said {human:+d}")
            )

    weight_total = sum(c.weight for c in channels)
    total = sum(c.value * c.weight for c in channels) / weight_total if weight_total else 0.0
    return RewardReport(total=total, channels=tuple(channels), notes=tuple(notes))
