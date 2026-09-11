"""Cleanroom command line.

    cleanroom doctor          # verify every credential before you need it
    cleanroom run -n 20       # run the learning loop
    cleanroom report          # what the agent has learned so far
    cleanroom curve           # learning curve PNG for the demo video
    cleanroom feedback 7 +1   # attach human judgement to an episode
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cleanroom.config import settings
from cleanroom.learning.memory import MemoryStore
from cleanroom.learning.reward import HUMAN_SCALE, WEIGHTS
from cleanroom.learning.store import load_bandit, load_episodes, save_bandit
from cleanroom.learning.strategies import BUCKETS, STRATEGIES
from cleanroom.pipeline.schema import SchemaError, available_schemas, load_schema
from cleanroom.pipeline.synthesize import CreditsExhausted

app = typer.Typer(add_completion=False, help="A self-improving web-to-clean-dataset ETL agent.")
console = Console()


def _fail(message: str) -> None:
    console.print(f"[bold red]x[/] {message}")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------- doctor ----


@app.command()
def doctor(
    live: bool = typer.Option(False, "--live", help="Also make one real call to each partner."),
    allow_paid: bool = typer.Option(
        False,
        "--allow-paid",
        help="Permit live checks that cost more than a cent (i.e. the You.com Agents writer).",
    ),
) -> None:
    """Check credentials and connectivity. Run this first."""
    table = Table("Component", "Status", "Detail", title="Cleanroom preflight")

    def row(name: str, ok: bool | None, detail: str) -> None:
        mark = {True: "[green]ok[/]", False: "[red]missing[/]", None: "[yellow]optional[/]"}[ok]
        table.add_row(name, mark, detail)

    backend = settings.resolved_backend
    row("YOU_API_KEY", bool(settings.you_api_key), f"{settings.you_api_base}  (search ~$0.005/call)")

    writer_detail = {
        "compat": f"{settings.llm_model} @ {settings.llm_base_url}",
        "anthropic": f"Anthropic {settings.model}",
        "you": "[red]You.com Agents -- $15 PER CALL[/]",
        "unconfigured": "none configured -- set CLEANROOM_LLM_BASE_URL/_MODEL",
    }[backend]
    row("code writer", backend != "unconfigured", writer_detail)
    row(
        "DAYTONA_API_KEY",
        bool(settings.daytona_api_key) or settings.local_validate,
        "local validation on -- sandbox bypassed"
        if settings.local_validate
        else f"auto-stop {settings.daytona_auto_stop_minutes}m",
    )
    row("ONE_SECRET", bool(settings.one_secret), "credential layer + memory + publish")
    # The template ships a placeholder, and a green tick against `your-org/your-repo`
    # is worse than a red one -- it hides the fact that nothing can be published.
    target = settings.one_publish_target
    placeholder = target in ("", "your-org/your-repo") or target.startswith("your-org/")
    row(
        "ONE_PUBLISH_TARGET",
        None if placeholder else True,
        f"[yellow]{target or 'unset'} -- still the placeholder, nothing will publish[/]"
        if placeholder
        else target,
    )

    memory = MemoryStore(settings)
    row("memory backend", True, f"{memory.backend} ({memory.count()} lessons stored)")

    try:
        from cleanroom.partners.one_client import OneClient

        status = OneClient(settings).status()
        row("one CLI", status["cli_installed"] or None,
            "installed" if status["cli_installed"] else "npm i -g @withone/cli")
    except Exception as exc:  # noqa: BLE001
        row("one CLI", None, str(exc)[:60])

    # Bracketed extras must be escaped or rich swallows them as console markup.
    try:
        import crewai  # noqa: F401

        row("crewai", True, "installed")
    except ImportError:
        row("crewai", None, r'pip install -e ".\[crew]"')

    try:
        import daytona  # noqa: F401

        row("daytona SDK", True, "installed")
    except ImportError:
        row("daytona SDK", None, r'pip install -e ".\[sandbox]"')

    schemas = available_schemas()
    row("schemas", bool(schemas), ", ".join(p.stem for p in schemas) or "none in schemas/")
    console.print(table)

    if live:
        console.print("\n[bold]Live checks[/]")
        _live_checks(allow_paid=allow_paid)

    missing = settings.missing_for()
    if missing:
        console.print(
            Panel(
                "Set these in .env before running:\n  " + "\n  ".join(missing),
                title="[red]incomplete[/]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)
    console.print("[green]Ready.[/] Try: cleanroom run -n 12")


def _live_checks(*, allow_paid: bool = False) -> None:
    """One real call per partner. Auth problems found now are cheap.

    "Cheap" is the operative word, and it was not always true: while the You.com
    Agents backend was configured, each `--live` run quietly spent $15 on the
    code-writer probe. A health check must never be the expensive thing, so the
    paid writer is skipped unless explicitly permitted.
    """
    try:
        from cleanroom.partners.you_client import YouClient

        client = YouClient(settings)
        payload = client.search("gpu cloud pricing", count=2, full_page=False)
        hits = len((payload.get("results") or {}).get("web") or [])
        console.print(f"  [green]ok[/] You.com: {hits} results via {client.resolved_base}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]x[/] You.com: {exc}")

    # Exercise whichever code writer is actually configured, with a real (tiny)
    # extraction task -- "can it reach the API" is not the question; "can it
    # return a module that compiles" is.
    if settings.resolved_backend == "you" and not allow_paid:
        console.print(
            "  [yellow]-[/] code writer skipped: the You.com Agents backend bills "
            "$15/call. Re-run with [bold]--allow-paid[/] to probe it anyway."
        )
        _live_one_check()
        return

    try:
        from cleanroom.pipeline.synthesize import build_synthesizer

        synth = build_synthesizer(settings)
        probe_schema = {
            "name": "probe",
            "fields": [
                {"name": "provider", "type": "string", "required": True},
                {"name": "usd_per_hour", "type": "number", "required": True, "min": 0},
                {"name": "source_url", "type": "string", "required": True, "format": "url"},
            ],
        }
        extractor = synth.synthesize(
            schema=probe_schema,
            document="| Provider | $/hr |\n| --- | --- |\n| Lambda | 2.49 |\n| RunPod | 1.19 |\n",
            source_url="https://example.com/pricing",
            strategy_id="table_parse",
            bucket="table_heavy",
        )
        console.print(
            f"  [green]ok[/] code writer ({synth.backend}): "
            f"{len(extractor.code)} chars, compiles"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]x[/] code writer: {exc}")

    if settings.local_validate:
        console.print("  [yellow]-[/] Daytona skipped (CLEANROOM_LOCAL_VALIDATE=1)")
    else:
        try:
            from cleanroom.partners.daytona_env import DaytonaEnvironment

            env = DaytonaEnvironment(settings)
            try:
                result = env.run(
                    "def extract(document):\n    return [{'source_url': 'https://example.com'}]\n",
                    "probe",
                    {"fields": [{"name": "source_url", "type": "string",
                                 "required": True, "format": "url"}]},
                )
                console.print(
                    f"  [green]ok[/] Daytona sandbox {env.sandbox_id}: "
                    f"stage={result.stage} rows_valid={(result.report or {}).get('rows_valid')}"
                )
            finally:
                env.close()
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]x[/] Daytona: {exc}")

    _live_one_check()


def _live_one_check() -> None:
    try:
        from cleanroom.partners.one_client import OneClient

        one = OneClient(settings)
        if one.has_cli:
            console.print(f"  [green]ok[/] One integrations: "
                          f"{json.dumps(one.list_integrations())[:160]}")
        else:
            console.print("  [yellow]-[/] One CLI not installed; passthrough only")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]x[/] One: {exc}")


# ------------------------------------------------------------------- run ----


@app.command()
def run(
    episodes: int = typer.Option(12, "--episodes", "-n", help="How many episodes to run."),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Override the search topic."),
    schema_path: Optional[str] = typer.Option(None, "--schema", "-s", help="Schema name or path."),
    publish: bool = typer.Option(False, "--publish", help="Write the dataset out through One."),
    crew: bool = typer.Option(True, "--crew/--no-crew", help="Use CrewAI for triage and publish."),
    greedy: bool = typer.Option(False, "--greedy", help="Exploit only; no exploration."),
    repairs: int = typer.Option(1, "--repairs", help="Repair turns allowed per episode."),
    pause: float = typer.Option(
        0.0, "--pause", help="Seconds to wait between episodes (free-tier RPM limits)."
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed the sampler for a repeatable demo."),
) -> None:
    """Run the learning loop."""
    from cleanroom.pipeline.episode import Learner

    missing = settings.missing_for(need_one=publish)
    if missing:
        _fail(f"missing credentials: {', '.join(missing)} (run `cleanroom doctor`)")

    try:
        schema = load_schema(schema_path)
    except SchemaError as exc:
        _fail(str(exc))

    console.print(
        Panel(
            f"schema   [bold]{schema['name']}[/]\n"
            f"episodes [bold]{episodes}[/]   repairs/episode [bold]{repairs}[/]\n"
            f"sandbox  [bold]{'local (unsafe)' if settings.local_validate else 'daytona'}[/]\n"
            f"publish  [bold]{'yes -> ' + settings.one_publish_target if publish else 'no'}[/]",
            title="cleanroom run",
        )
    )

    learner = Learner(schema, settings, seed=seed)
    outcomes: list = []  # defined before the try so the abort handler can read it
    try:
        sources = learner.gather_sources(topic)
        console.print(f"You.com returned [bold]{len(sources)}[/] pages with usable content.")
        if not sources:
            _fail("no usable pages; try a broader --topic")

        if crew:
            from cleanroom.crew.crew import triage_sources

            triaged = triage_sources(sources, schema, settings)
            sources = triaged.ordered
            if triaged.rationale:
                console.print(f"[dim]Scout: {triaged.rationale}[/]")

        table = Table("#", "bucket", "strategy", "reward", "valid/total", "rep", "source")
        for index in range(episodes):
            if pause and index:
                time.sleep(pause)
            source = sources[index % len(sources)]
            outcome = learner.run_episode(
                source,
                episode=learner._episode_offset + index + 1,  # noqa: SLF001
                max_repairs=repairs,
                greedy=greedy,
            )
            outcomes.append(outcome)
            report = outcome.report or {}

            if not outcome.counts_toward_learning:
                # Distinguish a provider failure from a bad strategy, loudly.
                # These look identical on the reward axis and are not the same
                # thing at all.
                table.add_row(
                    str(outcome.episode), outcome.bucket, outcome.strategy,
                    "[red]n/a[/]", "-", "-", "code writer failed",
                )
                console.print(
                    f"  ep{outcome.episode:>3} [red]code writer failed[/] "
                    f"(not counted): {str(outcome.synthesis_error)[:110]}"
                )
                continue

            colour = "green" if outcome.score >= 0.7 else "yellow" if outcome.score >= 0.4 else "red"
            table.add_row(
                str(outcome.episode),
                outcome.bucket,
                outcome.strategy,
                f"[{colour}]{outcome.score:.2f}[/]",
                f"{report.get('rows_valid', 0)}/{report.get('rows_total', 0)}",
                str(outcome.repairs),
                outcome.source.url[:44],
            )
            note = ", repaired" if outcome.repairs else ""
            if outcome.repair_error:
                note = ", [yellow]repair unavailable[/]"
            console.print(
                f"  ep{outcome.episode:>3} {outcome.bucket:<12} {outcome.strategy:<18} "
                f"[{colour}]{outcome.score:.2f}[/] "
                f"[dim]{outcome.profile:<9}[/] "
                f"({report.get('rows_valid', 0)}/{report.get('rows_total', 0)} rows, "
                f"{outcome.input_tokens // 1000}k tok, ${outcome.cost_usd:.4f}{note})"
            )

        save_bandit(learner.bandit, settings)
        learner.dataset.save()
        console.print(table)

        failures = [o for o in outcomes if not o.counts_toward_learning]
        if failures:
            console.print(
                f"\n[yellow]![/] {len(failures)}/{len(outcomes)} episodes could not be scored "
                "because the code writer failed. They are logged but excluded from "
                "the learning stats (they measure the provider, not the policy)."
            )

        scores = [o.score for o in outcomes if o.counts_toward_learning]
        if scores:
            cut = max(2, len(scores) // 3)
            early = sum(scores[:cut]) / cut
            late = sum(scores[-cut:]) / cut
            arrow = "[green]up[/]" if late > early else "[yellow]flat/down[/]"
            console.print(
                f"\nmean reward first {cut}: [bold]{early:.2f}[/] -> "
                f"last {cut}: [bold]{late:.2f}[/]  {arrow}"
            )

        console.print(
            f"dataset: [bold]{len(learner.dataset.rows)}[/] rows from "
            f"[bold]{learner.dataset.source_count}[/] sources -> {learner.dataset.path}"
        )

        spent = learner.ledger.total_usd
        billed = learner.ledger.billed_usd
        console.print(
            f"spend this run: est. [bold]${spent:.4f}[/] "
            f"(likely billed [bold]${billed:.4f}[/]) across "
            f"{len(learner.ledger.records)} external calls -> [bold]cleanroom costs[/]"
        )

        if publish:
            _do_publish(learner, schema, crew)
        else:
            console.print("[dim]Nothing published. Add --publish to close the loop.[/]")

    except CreditsExhausted as exc:
        # Save what was learned before the money ran out, then stop. Grinding on
        # would log a page of 0.00 episodes that look like a broken policy rather
        # than a dead credit balance.
        save_bandit(learner.bandit, settings)
        learner.dataset.save()
        console.print(
            Panel(
                str(exc),
                title="[red]run aborted -- code writer out of credits[/]",
                border_style="red",
            )
        )
        scored = [o for o in outcomes if o.counts_toward_learning]
        console.print(
            f"Kept {len(scored)} scored episode(s) and "
            f"{len(learner.dataset.rows)} dataset rows from before the abort."
        )
        raise typer.Exit(code=1)
    finally:
        learner.close()

    console.print("\nNext: [bold]cleanroom report[/] and [bold]cleanroom curve[/]")


def _do_publish(learner, schema: dict, use_crew: bool) -> None:
    from cleanroom.pipeline.dataset import clean_data_manifest, manifest_markdown

    manifest = clean_data_manifest(schema, settings)
    if not settings.one_publish_target:
        console.print("[yellow]![/] ONE_PUBLISH_TARGET unset; skipping publish.")
        return

    console.print(f"\nPublishing to [bold]{settings.one_publish_target}[/] via One...")
    if use_crew:
        try:
            from cleanroom.crew.crew import review_and_publish

            report = review_and_publish(
                schema=schema,
                summary_blob={
                    "dataset_rows": len(learner.dataset.rows),
                    "distinct_sources": learner.dataset.source_count,
                    "bandit": learner.bandit.snapshot()["stats"],
                    "recent_episodes": load_episodes(settings)[-12:],
                },
                dataset_csv=learner.dataset.to_csv(),
                manifest_md=manifest_markdown(manifest),
                cfg=settings,
            )
            console.print(Panel(report[:2000], title="Data Steward"))
            return
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]![/] crew publish failed ({exc}); falling back to direct call")

    result = learner.publish(dry_run=False)
    style = "green" if result.ok else "red"
    console.print(f"[{style}]{result.detail}[/]")


# ---------------------------------------------------------------- report ----


@app.command()
def report(
    schema_path: Optional[str] = typer.Option(None, "--schema", "-s"),
    lessons: int = typer.Option(8, "--lessons", help="How many stored lessons to show."),
) -> None:
    """Show what the agent has learned: posteriors, lessons, dataset state."""
    bandit = load_bandit(settings)
    episodes = load_episodes(settings)

    if not bandit.total_pulls:
        console.print("[yellow]No learning yet.[/] Run `cleanroom run` first.")
        raise typer.Exit()

    console.print(
        Panel(
            f"episodes logged [bold]{len(episodes)}[/]   bandit pulls [bold]{bandit.total_pulls}[/]",
            title="learned state",
        )
    )

    for bucket in bandit.seen_buckets() or list(BUCKETS):
        table = Table(
            "strategy", "posterior", "+/-", "pulls", "observed", "summary",
            title=f"bucket: {bucket}",
        )
        for arm, stats in bandit.ranking(bucket):
            if not stats.pulls and bandit.total_pulls:
                continue
            table.add_row(
                arm,
                f"{stats.posterior_mean:.3f}",
                f"{stats.posterior_sd:.3f}",
                str(stats.pulls),
                f"{stats.observed_mean:.3f}" if stats.pulls else "-",
                STRATEGIES[arm].summary[:52] if arm in STRATEGIES else "",
            )
        console.print(table)

    memory = MemoryStore(settings)
    recent = memory.search("extraction", limit=lessons)
    if recent:
        panel = "\n".join(f"- {l.text[:150]}" for l in recent)
        console.print(Panel(panel, title=f"lessons ({memory.backend} backend)"))

    try:
        schema = load_schema(schema_path)
        from cleanroom.pipeline.dataset import Dataset, clean_data_manifest

        dataset = Dataset(schema, settings)
        manifest = clean_data_manifest(schema, settings)
        console.print(
            Panel(
                f"rows            [bold]{len(dataset.rows)}[/]\n"
                f"distinct sources [bold]{dataset.source_count}[/]\n"
                f"pages attempted  [bold]{manifest['pages_attempted']}[/]\n"
                f"csv              {dataset.path}",
                title=f"dataset: {schema['name']}",
            )
        )
    except SchemaError as exc:
        console.print(f"[yellow]![/] {exc}")


# -------------------------------------------------------------------- ui ----


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", help="Port to serve on."),
    headless: bool = typer.Option(False, "--headless", help="Do not open a browser."),
) -> None:
    """Launch the dashboard: live run view, posteriors, spend, and feedback buttons."""
    import subprocess
    import sys

    from cleanroom.ui import APP_PATH

    try:
        import streamlit  # noqa: F401
    except ImportError:
        _fail('streamlit is not installed. Run: pip install -e ".[ui]"')

    # `streamlit run` rather than importing: Streamlit needs to own the process
    # and its own script-rerun loop.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    console.print(f"Starting dashboard on [bold]http://localhost:{port}[/] (ctrl-c to stop)")
    raise typer.Exit(subprocess.call(cmd))


# ---------------------------------------------------------------- models ----


@app.command()
def models() -> None:
    """List the models your configured code-writer key can actually reach.

    Hosted catalogues churn -- a model that worked last month returns an opaque
    404 today. This turns that into a list you can copy from.
    """
    if settings.resolved_backend != "compat":
        console.print(
            f"Backend is [bold]{settings.resolved_backend}[/]; this command lists "
            "models for the OpenAI-compatible backend only."
        )
        raise typer.Exit()

    from cleanroom.pipeline.openai_compat import OpenAICompatSynthesizer

    synth = OpenAICompatSynthesizer(settings)
    available = synth.available_models()
    if not available:
        _fail(f"could not list models from {settings.llm_base_url}/models")

    table = Table("model", "note", title=f"{settings.llm_base_url}")
    # Surface the ones worth using for code generation rather than the whole list.
    code_hints = ("gpt-oss", "qwen", "coder", "llama", "deepseek", "gemini")
    audio = ("whisper", "orpheus", "tts", "guard", "safeguard")
    for model in available:
        lower = model.lower()
        if any(marker in lower for marker in audio):
            note = "[dim]not a code model[/]"
        elif any(marker in lower for marker in code_hints):
            note = "[green]good for codegen[/]"
        else:
            note = ""
        marker = " [bold](configured)[/]" if model == settings.llm_model else ""
        table.add_row(model + marker, note)
    console.print(table)
    console.print(
        f"\ncurrent: [bold]{settings.llm_model}[/]  "
        f"-- change with CLEANROOM_LLM_MODEL in .env"
    )


# ----------------------------------------------------------------- costs ----


@app.command()
def costs(
    prices: bool = typer.Option(False, "--prices", help="Also print the price book used."),
    operations: bool = typer.Option(False, "--ops", help="Break down by operation, not component."),
    clear: bool = typer.Option(
        False, "--clear", help="Delete the call ledger and start fresh (useful before a demo)."
    ),
) -> None:
    """Per-component observability: calls, failures, latency, and spend."""
    if clear:
        # The ledger is append-only across runs, so a demo inherits every probe
        # and experiment that came before it.
        if settings.calls_path.exists():
            settings.calls_path.unlink()
            console.print(f"[green]Cleared[/] {settings.calls_path}")
        else:
            console.print("Nothing to clear.")
        raise typer.Exit()

    from cleanroom.learning.budget import LAMBDA, PROFILES
    from cleanroom.learning.store import load_profile_bandit
    from cleanroom.observability.ledger import CallLedger
    from cleanroom.observability.pricing import price_book

    ledger = CallLedger(settings)
    history = ledger.load_history()
    if not history:
        console.print(
            "[yellow]No calls recorded yet.[/] Run `cleanroom run` first "
            f"(ledger lives at {settings.calls_path})."
        )
        raise typer.Exit()

    # -- per-component table
    key = (lambda r: r.operation) if operations else (lambda r: r.component)
    grouped: dict[str, list] = {}
    for record in history:
        grouped.setdefault(key(record), []).append(record)

    table = Table(
        "component" if not operations else "operation",
        "calls", "fail%", "p50 s", "p95 s", "total s", "est. $",
        title="external calls",
    )
    total_cost = total_billed = 0.0
    for name in sorted(grouped, key=lambda n: -sum(r.cost_usd for r in grouped[n])):
        records = grouped[name]
        stats = list(ledger.stats(records).values())[0]
        billed = sum(0.0 if r.free_tier else r.cost_usd for r in records)
        total_cost += stats.cost_usd
        total_billed += billed
        fail_colour = "red" if stats.failure_rate > 0.2 else "white"
        free_note = " [dim](free tier)[/]" if billed == 0 and stats.cost_usd > 0 else ""
        table.add_row(
            name,
            str(stats.calls),
            f"[{fail_colour}]{stats.failure_rate * 100:.0f}%[/]",
            f"{stats.p50:.2f}",
            f"{stats.p95:.2f}",
            f"{stats.total_latency:.0f}",
            f"${stats.cost_usd:.4f}{free_note}",
        )
    console.print(table)
    episode_spend = sum(r.cost_usd for r in history if r.episode is not None)
    console.print(
        f"estimated total [bold]${total_cost:.4f}[/]   "
        f"likely billed [bold]${total_billed:.4f}[/]   "
        f"(difference is free-tier usage)"
    )
    console.print(
        f"[dim]cumulative across all runs in {settings.state_dir.name}/  --  "
        f"${episode_spend:.4f} attributed to episodes, "
        f"${total_cost - episode_spend:.4f} to probes and source gathering[/]"
    )

    # -- failures worth seeing
    failures = [r for r in history if not r.ok]
    if failures:
        ftable = Table("operation", "error", title=f"{len(failures)} failed call(s)")
        seen: dict[str, int] = {}
        for record in failures:
            seen[f"{record.operation}|{record.error[:90]}"] = (
                seen.get(f"{record.operation}|{record.error[:90]}", 0) + 1
            )
        for combined, count in sorted(seen.items(), key=lambda kv: -kv[1])[:10]:
            op, _, err = combined.partition("|")
            ftable.add_row(f"{op} (x{count})", err or "-")
        console.print(ftable)

    # -- what the agent learned about spending
    profiles = load_profile_bandit(settings)
    if profiles.total_pulls:
        console.print(
            f"\n[bold]Learned execution profiles[/] "
            f"(utility = reward - {LAMBDA} x normalised cost)"
        )
        for bucket in profiles.seen_buckets():
            ptable = Table("profile", "utility", "pulls", "budget", title=f"bucket: {bucket}")
            for arm, stats in profiles.ranking(bucket):
                spec = PROFILES.get(arm)
                ptable.add_row(
                    arm,
                    f"{stats.posterior_mean:.3f}",
                    str(stats.pulls),
                    f"{spec.doc_chars // 1000}k chars, {spec.max_repairs} repair(s)"
                    if spec else "",
                )
            console.print(ptable)
        console.print(
            "[dim]A cheap profile winning a bucket means the agent found it does "
            "not need the big context there.[/]"
        )

    # Health is normally built up live during a run; replay the log so the table
    # reflects history rather than this (empty) process.
    for record in history:
        ledger.health.record(record.component, record.ok, record.error)

    # -- component health / circuit breakers
    snapshot = ledger.health.snapshot()
    if snapshot:
        htable = Table("component", "success", "consecutive fails", "circuit",
                       title="health (this process)")
        for component, info in snapshot.items():
            htable.add_row(
                component,
                f"{info['success_rate']:.2f}",
                str(info["consecutive_failures"]),
                "[red]open[/]" if info["circuit_open"] else "closed",
            )
        console.print(htable)

    if prices:
        console.print(Panel(json.dumps(price_book(), indent=2), title="price book"))


# ----------------------------------------------------------------- curve ----


@app.command()
def curve(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="PNG path."),
    window: int = typer.Option(5, "--window", help="Rolling-mean window."),
    dark: bool = typer.Option(False, "--dark", help="Dark-surface variant for dark slides."),
) -> None:
    """Render the learning curve. This is the shot for the demo video."""
    from cleanroom.demo.plot_curve import render_curve

    episodes = load_episodes(settings)
    if len(episodes) < 3:
        _fail(f"only {len(episodes)} episodes logged; run more before plotting")

    target = output or (settings.state_dir / "learning_curve.png")
    path = render_curve(episodes, target, window=window, dark=dark)
    console.print(f"[green]Wrote[/] {path}  (table view: {path.with_suffix('.tsv').name})")


# -------------------------------------------------------------- feedback ----


@app.command()
def feedback(
    episode: int = typer.Argument(..., help="Episode number from `cleanroom run`."),
    verdict: str = typer.Argument(..., help="good | ok | bad  (or +1 / 0 / -1)."),
) -> None:
    """Attach human judgement to a past episode and fold it into the posterior.

    Shares its implementation with the dashboard's thumbs buttons, so a verdict
    means the same thing whichever interface it arrives through.
    """
    from cleanroom.learning.feedback import FeedbackError, apply_episode_feedback

    try:
        result = apply_episode_feedback(episode, verdict, settings)
    except FeedbackError as exc:
        _fail(str(exc))

    console.print(
        f"[green]Recorded[/] {result.value:+d} for [bold]{result.strategy}[/] on "
        f"[bold]{result.bucket}[/]. Posterior now {result.posterior_mean:.3f} "
        f"over {result.pulls} pulls."
    )
    verb = "[bold]changed[/] to" if result.changed_best else "still"
    console.print(f"Best arm for {result.bucket} is {verb} [bold]{result.best_arm}[/].")


# ------------------------------------------------------------ strategies ----


@app.command()
def strategies() -> None:
    """List the bandit's action space."""
    table = Table("strategy", "what it does", title="action space")
    for strategy in STRATEGIES.values():
        table.add_row(strategy.id, strategy.summary)
    console.print(table)
    console.print(f"\ncontexts: {', '.join(BUCKETS)}")
    console.print(f"arms total: {len(STRATEGIES)} x {len(BUCKETS)} contexts")


if __name__ == "__main__":
    app()
