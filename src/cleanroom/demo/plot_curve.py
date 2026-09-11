"""The demo figure: does the agent actually get better?

Three panels, because "did it learn" needs two different answers and one piece of
evidence.

* **Top left -- reward per episode.** Dots plus a rolling mean.
* **Bottom left -- convergence.** The share of recent episodes where the sampled
  arm was that bucket's current best-known arm, against the 1/k random baseline.
* **Right -- final posteriors.** Which strategy won the most-seen bucket, with
  pull counts.

Why the second panel exists, and it is not decoration. Raw reward is **not
comparable across buckets**: a dense pricing table can reach ~0.95 while a prose
page tops out near 0.70, so an aggregate reward mean drifts with whichever bucket
happened to come up rather than with what the agent learned. A run where the
bandit correctly identified the best arm in all four buckets can still show a
*flat or negative* aggregate reward trend. Convergence is bucket-agnostic --
it asks "is the agent concentrating its pulls?" -- so the two panels together
distinguish "learned nothing" from "learned, but the pages got harder".

This is the canonical pairing from the bandit literature (Sutton & Barto Fig 2.2
plots average reward and % optimal action side by side for exactly this reason).
One honest caveat, stated on the chart: convergence measures concentration
against the agent's *own* current estimate, not against ground truth, so it can
rise while the agent is confidently wrong. Read it next to panel one, never alone.

Design notes, since these were deliberate:

* Each measure gets its own axis. Never a dual-axis chart -- reward and
  convergence are different measures, so they are stacked, not overlaid.
* One series per panel, so no legend box; lines are labelled in place.
* Strategy identity is *not* encoded as scatter color. Scatter compares every
  pair of colors at once and the palette only guarantees three all-pairs
  distinguishable hues; five strategies would fail that gate. Identity moves to
  the bar panel as length, which needs no hue at all.
* Dark mode is a selected set of steps for the dark surface, not an inverted
  light palette. Both modes were run through the palette validator.
* A sibling `.tsv` is written next to the PNG so the figure has a table view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Tokens:
    """Design tokens for one mode. Series color validated against the surface."""

    surface: str
    primary: str
    secondary: str
    muted: str
    grid: str
    baseline: str
    series: str
    good: str


LIGHT = Tokens(
    surface="#fcfcfb",
    primary="#0b0b0b",
    secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    series="#2a78d6",
    good="#006300",
)

DARK = Tokens(
    surface="#1a1a19",
    primary="#ffffff",
    secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    series="#3987e5",
    good="#0ca30c",
)

#: matplotlib resolves real family names, not CSS generics, so "system-ui" is
#: omitted -- it only produces findfont warnings.
FONT_STACK = ["Segoe UI", "Arial", "DejaVu Sans", "sans-serif"]


def _trailing_mean(values: Sequence[float], window: int) -> list[float]:
    """Rolling mean that expands over the first `window-1` points.

    A trailing window that only starts at episode `window` leaves the most
    interesting part of a short demo run unplotted, so the early points use an
    expanding mean instead of nothing.
    """
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _convergence_series(
    episodes: Sequence[dict[str, Any]], window: int
) -> tuple[list[float], float]:
    """Rolling share of episodes that pulled the bucket's then-best-known arm.

    Computed by **replaying** the episode log through a fresh bandit, so the
    judgement at episode `t` uses only what the agent knew at episode `t` -- not
    the final posterior. Scoring against the end state would leak hindsight and
    make any run look like it converged.

    Returns the series plus the 1/k random-choice baseline.
    """
    from cleanroom.learning.bandit import ThompsonBandit
    from cleanroom.learning.strategies import BUCKETS, STRATEGY_IDS

    replay = ThompsonBandit(arms=list(STRATEGY_IDS), buckets=list(BUCKETS), discount=0.98, seed=0)
    hits: list[float] = []

    for record in episodes:
        bucket = str(record.get("bucket") or "prose")
        arm = str(record.get("strategy") or "")
        stats = replay.stats_for(bucket)
        pulled_any = any(s.pulls for s in stats.values())
        # Before any evidence exists every arm ties, so "best" is meaningless;
        # count those episodes as exploration rather than a free hit.
        best = replay.best_arm(bucket) if pulled_any else None
        hits.append(1.0 if (best is not None and arm == best) else 0.0)
        replay.update(bucket, arm, float(record.get("reward") or 0.0))

    return _trailing_mean(hits, window), 1.0 / max(1, len(STRATEGY_IDS))


def _rounded_bars(ax, ys, widths, *, color, height=0.58, radius=0.012):
    """Horizontal bars with a rounded data-end and a square baseline end.

    The rounded box is started slightly left of zero and the axes x-limit is
    pinned at zero, so the left rounding is clipped away by the frame. That gets
    a square baseline and a rounded tip without compositing two patches.
    """
    from matplotlib.patches import BoxStyle, FancyBboxPatch

    for y, width in zip(ys, widths):
        if width <= 0:
            continue
        ax.add_patch(
            FancyBboxPatch(
                (-radius, y - height / 2),
                width + radius,
                height,
                boxstyle=BoxStyle("Round", pad=0, rounding_size=radius),
                linewidth=0,
                facecolor=color,
                mutation_aspect=0.4,
                clip_on=True,
                zorder=2,
            )
        )


def _style_axes(ax, tokens: Tokens, *, xgrid: bool = False) -> None:
    ax.set_facecolor(tokens.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(tokens.baseline)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=tokens.muted, labelsize=9, length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(
        axis="x" if xgrid else "y",
        color=tokens.grid,
        linewidth=0.8,
        alpha=1.0,
    )
    ax.grid(axis="y" if xgrid else "x", visible=False)


def render_curve(
    episodes: Sequence[dict[str, Any]],
    output: Path,
    *,
    window: int = 5,
    dark: bool = False,
    bandit: Any | None = None,
) -> Path:
    """Render the three-panel demo figure to a PNG. Returns the path."""
    import matplotlib.pyplot as plt

    fig, ordered = build_figure(episodes, window=window, dark=dark, bandit=bandit)
    tokens = DARK if dark else LIGHT

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=tokens.surface)
    plt.close(fig)

    _write_table(ordered, output.with_suffix(".tsv"))
    return output


def build_figure(
    episodes: Sequence[dict[str, Any]],
    *,
    window: int = 5,
    dark: bool = False,
    bandit: Any | None = None,
):
    """Build the figure without writing it.

    Split out so the dashboard can embed the *same* figure the demo video shows,
    rather than maintaining a second set of charts that drift apart.
    Returns `(figure, scored_episodes)`.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('matplotlib missing; pip install -e ".[viz]"') from exc

    from cleanroom.learning.store import load_bandit
    from cleanroom.learning.strategies import STRATEGY_IDS

    tokens = DARK if dark else LIGHT
    plt.rcParams["font.family"] = FONT_STACK

    # Drop episodes where the code writer failed. They score 0.0 but measure the
    # provider, not the policy, so plotting them would turn a dead API key into
    # what looks like a collapsing agent. The count is stated on the figure.
    everything = sorted(episodes, key=lambda r: int(r.get("episode") or 0))
    ordered = [
        r
        for r in everything
        if (r.get("reward_detail") or {}).get("counts_toward_learning") is not False
    ]
    excluded = len(everything) - len(ordered)
    if not ordered:
        raise RuntimeError(
            f"all {len(everything)} logged episodes failed at the code-writing step; "
            "nothing to plot. Check `cleanroom doctor`."
        )
    xs = [int(r.get("episode") or i + 1) for i, r in enumerate(ordered)]
    ys = [float(r.get("reward") or 0.0) for r in ordered]
    smooth = _trailing_mean(ys, window)

    converge, baseline = _convergence_series(ordered, window)

    fig = plt.figure(figsize=(13.0, 6.4))
    fig.patch.set_facecolor(tokens.surface)
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.75, 1.0], height_ratios=[1.35, 1.0], hspace=0.32, wspace=0.28
    )
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_conv = fig.add_subplot(grid[1, 0], sharex=ax_curve)
    ax_bars = fig.add_subplot(grid[:, 1])

    # ---- panel 1: the learning curve -----------------------------------
    _style_axes(ax_curve, tokens)

    ax_curve.scatter(
        xs,
        ys,
        s=46,
        facecolor=tokens.muted,
        edgecolor=tokens.surface,
        linewidth=1.6,  # 2px surface ring so overlapping points stay countable
        alpha=0.75,
        zorder=2,
        label="_nolegend_",
    )
    ax_curve.plot(xs, smooth, color=tokens.series, linewidth=2.0, zorder=3, solid_capstyle="round")

    # Direct label instead of a legend box: one series, named in place.
    ax_curve.annotate(
        f"rolling mean (window {window})",
        xy=(xs[-1], smooth[-1]),
        xytext=(-6, 12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=tokens.series,
        fontweight="bold",
    )
    ax_curve.annotate(
        "each dot = one episode",
        xy=(xs[0], ys[0]),
        xytext=(4, -16),
        textcoords="offset points",
        fontsize=8.5,
        color=tokens.muted,
    )

    # Early-vs-late callout: the headline number, stated rather than implied.
    if len(ys) >= 6:
        cut = max(2, len(ys) // 3)
        early, late = sum(ys[:cut]) / cut, sum(ys[-cut:]) / cut
        delta = late - early
        ax_curve.annotate(
            f"first {cut}: {early:.2f}  ->  last {cut}: {late:.2f}  ({delta:+.2f})",
            xy=(0.5, 1.02),
            xycoords="axes fraction",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color=tokens.good if delta > 0 else tokens.secondary,
        )

    ax_curve.set_ylim(-0.04, 1.06)
    ax_curve.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_curve.set_ylabel("reward", fontsize=9.5, color=tokens.secondary, labelpad=8)
    ax_curve.tick_params(labelbottom=False)  # x axis is shared with the panel below
    ax_curve.set_title(
        "Extraction reward per episode",
        fontsize=12.5,
        fontweight="bold",
        color=tokens.primary,
        loc="left",
        pad=26,
    )

    # ---- panel 2: convergence ------------------------------------------
    # Reward is not comparable across buckets, so this panel carries the
    # bucket-agnostic half of the "did it learn" question.
    _style_axes(ax_conv, tokens)

    ax_conv.axhline(baseline, color=tokens.muted, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax_conv.annotate(
        f"random choice ({baseline:.2f})",
        xy=(xs[0], baseline),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=tokens.muted,
    )
    ax_conv.plot(xs, converge, color=tokens.series, linewidth=2.0, zorder=3,
                 solid_capstyle="round")
    ax_conv.annotate(
        "share of episodes pulling the best-known arm",
        xy=(xs[-1], converge[-1]),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=tokens.series,
        fontweight="bold",
    )

    ax_conv.set_ylim(-0.04, 1.06)
    ax_conv.set_yticks([0.0, 0.5, 1.0])
    ax_conv.set_ylabel("convergence", fontsize=9.5, color=tokens.secondary, labelpad=8)
    ax_conv.set_xlabel("episode", fontsize=9.5, color=tokens.secondary, labelpad=8)
    ax_conv.set_title(
        "Exploration collapsing onto the learned policy",
        fontsize=10.5,
        fontweight="bold",
        color=tokens.primary,
        loc="left",
        pad=10,
    )

    # ---- panel 3: where the posterior landed ---------------------------
    bandit = bandit or load_bandit()
    seen = bandit.seen_buckets()
    if seen:
        bucket = max(
            seen,
            key=lambda b: sum(s.pulls for s in bandit.stats_for(b).values()),
        )
    else:
        bucket = "prose"

    ranking = [(arm, st) for arm, st in bandit.ranking(bucket) if arm in STRATEGY_IDS]
    # Best at the top: barh draws bottom-up, so feed it reversed.
    ranking = list(reversed(ranking))
    positions = list(range(len(ranking)))

    _style_axes(ax_bars, tokens, xgrid=True)

    # An arm with zero pulls sits at its untouched prior (0.50), which would read
    # as a real measurement next to arms that earned their value. Render those
    # recessive and say so, so the panel cannot be misread as evidence.
    for y, (arm, stats) in zip(positions, ranking):
        pulled = stats.pulls > 0
        _rounded_bars(
            ax_bars,
            [y],
            [stats.posterior_mean],
            color=tokens.series if pulled else tokens.grid,
        )
        ax_bars.annotate(
            f"{stats.posterior_mean:.2f}" if pulled else "prior only",
            xy=(stats.posterior_mean, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=tokens.secondary if pulled else tokens.muted,
            fontweight="bold" if pulled else "normal",
        )

    labels = [
        f"{arm}  (n={st.pulls})" if st.pulls else f"{arm}  (untried)"
        for arm, st in ranking
    ]

    ax_bars.set_yticks(positions)
    ax_bars.set_yticklabels(labels, fontsize=9, color=tokens.secondary)
    ax_bars.set_xlim(0, 1.0)
    ax_bars.set_ylim(-0.7, len(ranking) - 0.3)
    ax_bars.set_xticks([0.0, 0.5, 1.0])
    ax_bars.set_xlabel("posterior mean reward", fontsize=9.5,
                       color=tokens.secondary, labelpad=8)
    ax_bars.set_title(
        f"Strategy value -- bucket: {bucket}",
        fontsize=12.5,
        fontweight="bold",
        color=tokens.primary,
        loc="left",
        pad=26,
    )

    # Placed with fig.text rather than suptitle: suptitle sits in the same band
    # as the first panel's title and the two collide.
    fig.text(
        0.012,
        0.962,
        f"Cleanroom -- {len(ordered)} scored episodes, {bandit.total_pulls} arm pulls"
        + (f"  ({excluded} excluded: code writer failed)" if excluded else ""),
        fontsize=10,
        color=tokens.muted,
        ha="left",
    )
    # State the limitation on the chart, not only in the docs: a reader who sees
    # convergence rise should know it is measured against the agent's own belief.
    fig.text(
        0.012,
        0.018,
        "Reward is not comparable across buckets (different achievable ceilings), so read both left panels together. "
        "Convergence is measured against the agent's own running estimate, not ground truth.",
        fontsize=8,
        color=tokens.muted,
        ha="left",
    )
    # subplots_adjust rather than tight_layout: the gridspec already sets the
    # spacing, and tight_layout would discard it.
    fig.subplots_adjust(left=0.075, right=0.975, top=0.865, bottom=0.10)
    return fig, ordered


def _write_table(episodes: Sequence[dict[str, Any]], path: Path) -> Path:
    """Table view of the figure -- identity is never carried by the picture alone."""
    header = ["episode", "bucket", "strategy", "reward", "rows_valid", "rows_total", "repairs", "url"]
    lines = ["\t".join(header)]
    for record in episodes:
        detail = record.get("reward_detail") or {}
        lines.append(
            "\t".join(
                str(value)
                for value in (
                    record.get("episode"),
                    record.get("bucket"),
                    record.get("strategy"),
                    record.get("reward"),
                    record.get("rows_valid"),
                    record.get("rows_total"),
                    detail.get("repairs", 0),
                    record.get("url"),
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

