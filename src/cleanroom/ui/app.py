"""Streamlit dashboard: watch the agent learn, and teach it.

Launch with `cleanroom ui` (which wraps `streamlit run` on this file).

The read-only half mirrors what `cleanroom report` and `cleanroom costs` print --
deliberately the *same* figure the demo video shows, built by
`demo.plot_curve.build_figure`, so there is one set of charts rather than two
that drift.

The half that earns its keep is **Review**. Reward from the sandbox is automatic
and plentiful; a human verdict is scarce and high-trust, and until now the only
way to give one was to type an episode number. Clicking a thumb, or flagging a
single bad row, routes through `learning.feedback` -- the same code path as the
CLI -- so the posterior moves and a lesson gets stored that the next synthesis
will retrieve.

Row rejection also proposes a *schema constraint* that would have caught the row.
That matters: a `deny_pattern` rejects the whole class of bad rows permanently,
while a lesson only influences the next prompt.
"""

from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st

from cleanroom.config import settings
from cleanroom.learning.budget import LAMBDA, PROFILES
from cleanroom.learning.feedback import (
    FeedbackError,
    apply_episode_feedback,
    reject_row,
    suggest_constraint,
)
from cleanroom.learning.memory import MemoryStore
from cleanroom.learning.store import (
    load_bandit,
    load_episodes,
    load_profile_bandit,
)
from cleanroom.learning.strategies import STRATEGIES
from cleanroom.observability.ledger import CallLedger
from cleanroom.pipeline.schema import SchemaError, available_schemas, load_schema

st.set_page_config(page_title="Cleanroom", page_icon="*", layout="wide")

REFRESH_KEYS = ("episodes", "bandit", "profiles", "ledger")


def _clear_cache() -> None:
    for key in REFRESH_KEYS:
        st.session_state.pop(key, None)


# -- data loading ------------------------------------------------------------


def load_state():
    episodes = load_episodes(settings)
    bandit = load_bandit(settings)
    profiles = load_profile_bandit(settings)
    ledger = CallLedger(settings)
    history = ledger.load_history()
    for record in history:
        ledger.health.record(record.component, record.ok, record.error)
    return episodes, bandit, profiles, ledger, history


def scored(episodes: list[dict]) -> list[dict]:
    return [
        e
        for e in episodes
        if (e.get("reward_detail") or {}).get("counts_toward_learning") is not False
    ]


# -- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Cleanroom")

    names = [p.stem for p in available_schemas()] or ["gpu_cloud_pricing"]
    schema_name = st.selectbox("Schema", names)
    try:
        schema = load_schema(schema_name)
    except SchemaError as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"code writer: **{settings.resolved_backend}**")
    if settings.resolved_backend == "compat":
        st.caption(f"`{settings.llm_model}`")

    missing = settings.missing_for(need_one=False)
    if missing:
        st.warning("missing: " + ", ".join(missing))

    st.divider()
    st.subheader("Run")
    n_episodes = st.number_input("Episodes", 1, 100, 12)
    pause = st.number_input("Pause between episodes (s)", 0.0, 60.0, 5.0, step=1.0)
    use_crew = st.checkbox("CrewAI source triage", value=False)
    go = st.button("Run episodes", type="primary", use_container_width=True)

    st.divider()
    if st.button("Refresh", use_container_width=True):
        _clear_cache()
        st.rerun()
    st.caption(f"state: `{settings.state_dir}`")


# -- run ---------------------------------------------------------------------

if go:
    from cleanroom.pipeline.episode import Learner
    from cleanroom.pipeline.synthesize import CreditsExhausted

    blocking = settings.missing_for(need_one=False)
    if blocking:
        st.error("Cannot run; missing: " + ", ".join(blocking))
        st.stop()

    progress = st.progress(0.0, text="gathering sources from You.com...")
    log = st.container()
    learner = Learner(schema, settings)
    try:
        sources = learner.gather_sources()
        if use_crew:
            try:
                from cleanroom.crew.crew import triage_sources

                sources = triage_sources(sources, schema, settings).ordered
            except Exception as exc:  # noqa: BLE001
                log.warning(f"triage unavailable ({exc}); using search order")

        if not sources:
            st.error("You.com returned no usable pages. Try a different schema topic.")
            st.stop()

        import time as _time

        for index in range(int(n_episodes)):
            if pause and index:
                _time.sleep(pause)
            source = sources[index % len(sources)]
            outcome = learner.run_episode(
                source,
                episode=learner._episode_offset + index + 1,  # noqa: SLF001
                max_repairs=1,
            )
            progress.progress(
                (index + 1) / float(n_episodes),
                text=f"episode {index + 1}/{int(n_episodes)}",
            )
            if not outcome.counts_toward_learning:
                log.error(f"ep{outcome.episode}: code writer failed -- {outcome.synthesis_error}")
            else:
                report = outcome.report or {}
                log.write(
                    f"**ep{outcome.episode}** `{outcome.bucket}` / `{outcome.strategy}` "
                    f"/ `{outcome.profile}` -> **{outcome.score:.2f}** "
                    f"({report.get('rows_valid', 0)}/{report.get('rows_total', 0)} rows, "
                    f"${outcome.cost_usd:.4f})"
                )
        learner.dataset.save()
    except CreditsExhausted as exc:
        st.error(f"Run aborted -- out of credits: {exc}")
    finally:
        learner.close()

    _clear_cache()
    st.success("Run complete.")


episodes, bandit, profiles, ledger, history = load_state()

if not episodes:
    st.info("No episodes yet. Set the episode count in the sidebar and hit **Run episodes**.")
    st.stop()

ok = scored(episodes)

# -- headline ----------------------------------------------------------------

cols = st.columns(5)
cols[0].metric("Episodes", len(ok), delta=f"{len(episodes) - len(ok)} unscored" or None)
cols[1].metric("Mean reward", f"{sum(e['reward'] for e in ok) / max(1, len(ok)):.2f}")
if len(ok) >= 6:
    cut = max(2, len(ok) // 3)
    early = sum(e["reward"] for e in ok[:cut]) / cut
    late = sum(e["reward"] for e in ok[-cut:]) / cut
    cols[2].metric(f"Last {cut} vs first {cut}", f"{late:.2f}", delta=f"{late - early:+.2f}")
else:
    cols[2].metric("Trend", "need 6+ eps")
cols[3].metric("Arm pulls", bandit.total_pulls)
# Labelled "lifetime" deliberately. The ledger is append-only across runs, and an
# unqualified "Spend" of $45 next to an 8-episode run that cost half a cent is
# actively misleading -- $45 of that was three $15 health-check probes.
cols[4].metric("Spend (lifetime)", f"${sum(r.cost_usd for r in history):.4f}")

tab_learn, tab_spend, tab_review, tab_data = st.tabs(
    ["Learning", "Spend", "Review", "Dataset"]
)

# -- learning ----------------------------------------------------------------

with tab_learn:
    if len(ok) >= 3:
        from cleanroom.demo.plot_curve import build_figure

        fig, _ = build_figure(episodes, window=5, bandit=bandit)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Three scored episodes needed before the curve is meaningful.")

    st.subheader("Strategy posteriors")
    st.caption("Which extraction approach works, per page shape.")
    for bucket in bandit.seen_buckets():
        rows = [
            {
                "strategy": arm,
                "posterior": round(stat.posterior_mean, 3),
                "+/-": round(stat.posterior_sd, 3),
                "pulls": stat.pulls,
                "what it does": STRATEGIES[arm].summary if arm in STRATEGIES else "",
            }
            for arm, stat in bandit.ranking(bucket)
            if stat.pulls
        ]
        if rows:
            st.markdown(f"**{bucket}**")
            st.dataframe(rows, hide_index=True, use_container_width=True)

    st.subheader("Lessons the agent wrote itself")
    memory = MemoryStore(settings)
    lessons = memory.recent(limit=10)
    if lessons:
        st.caption(
            f"{memory.count()} stored in the `{memory.backend}` backend; "
            "retrieved before the next attempt on a similar page."
        )
        for lesson in lessons:
            st.markdown(f"- {lesson.text}")
    else:
        st.info("No lessons yet — they get written as episodes succeed and fail.")

# -- spend -------------------------------------------------------------------

with tab_spend:
    stats = ledger.stats(history)
    st.dataframe(
        [
            {
                "component": name,
                "calls": s.calls,
                "fail %": round(s.failure_rate * 100, 1),
                "p50 s": round(s.p50, 2),
                "p95 s": round(s.p95, 2),
                "est. $": round(s.cost_usd, 4),
                "billed $": round(s.billed_usd, 4),
            }
            for name, s in sorted(stats.items(), key=lambda kv: -kv[1].cost_usd)
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Cumulative across every run in this state directory, not just the last "
        "one. Estimated and billed differ where a free tier covered the call. "
        "You.com Search is $0.005/call; the Agents API is $15/call, which is why "
        "code generation does not run there."
    )

    by_episode = [r for r in history if r.episode is not None]
    if by_episode:
        episode_spend = sum(r.cost_usd for r in by_episode)
        overhead = sum(r.cost_usd for r in history) - episode_spend
        c1, c2 = st.columns(2)
        c1.metric("Attributed to episodes", f"${episode_spend:.4f}")
        c2.metric(
            "Outside episodes",
            f"${overhead:.4f}",
            help="Health checks, probes and source gathering -- calls with no "
                 "episode number attached.",
        )

    st.subheader("Learned execution profiles")
    st.caption(f"utility = reward - {LAMBDA} x normalised cost. A cheap profile "
               "winning means the agent found it does not need the big context there.")
    for bucket in profiles.seen_buckets():
        rows = [
            {
                "profile": arm,
                "utility": round(stat.posterior_mean, 3),
                "pulls": stat.pulls,
                "budget": f"{PROFILES[arm].doc_chars // 1000}k chars, "
                          f"{PROFILES[arm].max_repairs} repair(s)" if arm in PROFILES else "",
            }
            for arm, stat in profiles.ranking(bucket)
        ]
        st.markdown(f"**{bucket}**")
        st.dataframe(rows, hide_index=True, use_container_width=True)

    health = ledger.health.snapshot()
    if health:
        st.subheader("Component health")
        st.dataframe(
            [
                {
                    "component": name,
                    "success rate": info["success_rate"],
                    "consecutive fails": info["consecutive_failures"],
                    "circuit": "OPEN" if info["circuit_open"] else "closed",
                    "last error": info["last_error"][:80],
                }
                for name, info in health.items()
            ],
            hide_index=True,
            use_container_width=True,
        )

    failures = [r for r in history if not r.ok]
    if failures:
        with st.expander(f"{len(failures)} failed call(s)"):
            for record in failures[-20:]:
                st.code(f"{record.operation}: {record.error[:200]}")

# -- review (the feedback channel) -------------------------------------------

with tab_review:
    st.caption(
        "Sandbox reward is automatic and plentiful; your judgement is scarce and "
        "weighted 2.5x. A verdict updates the strategy posterior. Flagging a row "
        "stores a lesson naming the bad values, which the next synthesis retrieves."
    )

    for record in reversed(ok[-12:]):
        detail = record.get("reward_detail") or {}
        header = (
            f"ep{record['episode']} - {record['bucket']} / {record['strategy']} "
            f"- reward {record['reward']:.2f} "
            f"({record.get('rows_valid', 0)}/{record.get('rows_total', 0)} rows)"
        )
        with st.expander(header):
            st.caption(record.get("url", ""))
            if detail.get("sample_errors"):
                st.markdown("**Why rows were rejected**")
                for err in detail["sample_errors"][:5]:
                    st.code(err)

            left, mid, right = st.columns([1, 1, 6])
            if left.button("Good", key=f"g{record['episode']}"):
                try:
                    result = apply_episode_feedback(record["episode"], "good", settings)
                    st.success(
                        f"{result.strategy} on {result.bucket} -> "
                        f"{result.posterior_mean:.3f}"
                        + ("  (best arm changed!)" if result.changed_best else "")
                    )
                    _clear_cache()
                except FeedbackError as exc:
                    st.error(str(exc))
            if mid.button("Bad", key=f"b{record['episode']}"):
                try:
                    result = apply_episode_feedback(record["episode"], "bad", settings)
                    st.warning(
                        f"{result.strategy} on {result.bucket} -> "
                        f"{result.posterior_mean:.3f}. Best arm is now "
                        f"{result.best_arm}."
                    )
                    _clear_cache()
                except FeedbackError as exc:
                    st.error(str(exc))

    st.divider()
    st.subheader("Flag a bad row")
    st.caption("Row-level rejection is the grain a posterior cannot represent.")

    dataset_path = Path(settings.dataset_path)
    if not dataset_path.exists():
        st.info("No dataset yet.")
    else:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            st.info("Dataset is empty.")
        else:
            labels = [
                f"{i}: " + " | ".join(f"{v}" for k, v in r.items() if k != "source_url" and v)
                for i, r in enumerate(rows[:200])
            ]
            picked = st.selectbox("Row", range(len(labels)), format_func=lambda i: labels[i])
            reason = st.text_input("What is wrong with it?", placeholder="not a provider")
            if st.button("Reject this row"):
                text = reject_row(rows[picked], reason=reason, cfg=settings)
                st.success("Lesson stored.")
                st.code(text)
                hint = suggest_constraint(rows[picked], schema)
                if hint:
                    st.markdown(
                        "**Suggested schema hardening** — a constraint rejects this "
                        "whole class of row permanently, not just on the next prompt:"
                    )
                    st.code(hint)
                _clear_cache()

# -- dataset -----------------------------------------------------------------

with tab_data:
    path = Path(settings.dataset_path)
    if not path.exists():
        st.info("No dataset yet.")
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        st.metric("Rows", len(rows))
        st.dataframe(rows, hide_index=True, use_container_width=True, height=520)
        st.download_button(
            "Download CSV",
            path.read_text(encoding="utf-8"),
            file_name=path.name,
            mime="text/csv",
        )
