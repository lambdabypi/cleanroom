"""CrewAI layer: source triage and the publish decision.

An honest note on the division of labour, because it is a design choice and not
an accident. The learning loop in `pipeline/episode.py` is deterministic Python.
Reinforcement learning needs the mapping from arm to reward to be unambiguous,
and an agent that can improvise its way around the strategy it was told to use
would corrupt the credit assignment. So the crew sits on either side of the loop
rather than inside it:

* **Source Scout** triages what You.com returned -- which pages plausibly contain
  the records, which are marketing fluff. Judgement, no ground truth, good fit
  for an LLM.
* **Data Steward** reads the finished run (rewards, per-bucket posteriors,
  provenance) and decides whether the dataset is fit to publish. A gate with
  taste in it, which is exactly what a deterministic threshold cannot express.

The Steward holds One's four MCP tools and performs the write itself, so the
loop-closing action is genuinely agent-taken rather than a hardcoded call.

Requires the extras: `pip install -e ".[crew]"`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

from cleanroom.config import Settings, settings
from cleanroom.crew.mcp_helpers import harden_tools, select_tools
from cleanroom.partners.you_client import Source

#: The four-tool loop, and nothing else.
ONE_TOOLS = (
    "list_one_integrations",
    "search_one_platform_actions",
    "get_one_action_knowledge",
    "execute_one_action",
)


class CrewUnavailable(RuntimeError):
    pass


def _quiet_crewai_first_run() -> None:
    """Suppress CrewAI's interactive first-run prompts.

    CrewAI 1.15 asks "Would you like to view your execution traces? [y/N]" on
    first use, with a 20-second timeout. In a scripted run that is a 20-second
    stall; in a fresh clone -- which is what a judge reproducing the repo has --
    it is a blocking prompt in the middle of the demo. Both env vars must be set
    before crewai is imported, which is why this lives here rather than in a
    shell profile.

    Set CREWAI_TRACING_ENABLED=true in .env if you actually want traces.
    """
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def _require_crewai():
    _quiet_crewai_first_run()
    try:
        from crewai import Agent, Crew, LLM, Process, Task  # noqa: PLC0415
        from crewai_tools import MCPServerAdapter  # noqa: PLC0415
        from mcp import StdioServerParameters  # noqa: PLC0415
    except ImportError as exc:
        raise CrewUnavailable(
            'CrewAI extras missing. Install with: pip install -e ".[crew]"'
        ) from exc
    return Agent, Crew, LLM, Process, Task, MCPServerAdapter, StdioServerParameters


def one_server_params(cfg: Settings | None = None):
    """Stdio params for One's local MCP server.

    The local server reads scoping from the environment rather than browser
    OAuth, which is what makes it usable from a framework process.
    """
    cfg = cfg or settings
    _, _, _, _, _, _, StdioServerParameters = _require_crewai()
    return StdioServerParameters(
        command="npx",
        args=["-y", "@withone/mcp"],
        env={
            **os.environ,
            "ONE_SECRET": cfg.one_secret,
            "ONE_CONNECTION_KEYS": cfg.one_connection_keys,
        },
    )


def build_llm(cfg: Settings | None = None):
    """Point CrewAI at whichever backend is actually configured.

    CrewAI routes through LiteLLM, so the model string needs a provider prefix.
    This used to hardcode `anthropic/...`, which fails outright when the project
    is running on a free OpenAI-compatible provider and no Anthropic key exists
    -- the common case. LiteLLM treats an `openai/` prefix plus an explicit
    `base_url` as a generic OpenAI-compatible endpoint, which covers Groq,
    Cerebras, OpenRouter, Together and Ollama in one branch.
    """
    cfg = cfg or settings
    _, _, LLM, _, _, _, _ = _require_crewai()

    backend = cfg.resolved_backend
    if backend == "anthropic":
        return LLM(model=f"anthropic/{cfg.model}", api_key=cfg.anthropic_api_key)
    if backend == "compat":
        return LLM(
            model=f"openai/{cfg.llm_model}",
            base_url=cfg.llm_base_url,
            # Ollama and other local servers need no key, but LiteLLM wants a
            # non-empty string.
            api_key=cfg.llm_api_key or "not-needed",
        )

    raise CrewUnavailable(
        f"no LiteLLM-compatible model for backend {backend!r}. CrewAI needs either "
        "ANTHROPIC_API_KEY or an OpenAI-compatible endpoint "
        "(CLEANROOM_LLM_BASE_URL + CLEANROOM_LLM_MODEL)."
    )


# -- source triage -----------------------------------------------------------


@dataclass
class TriageResult:
    ordered: list[Source]
    rationale: str = ""


def triage_sources(
    sources: Sequence[Source],
    schema: dict,
    cfg: Settings | None = None,
) -> TriageResult:
    """Rank pages by how likely they are to hold extractable records.

    Falls back to the original ordering on any failure. Triage is an
    optimisation, not a dependency -- the loop must still run when the crew
    cannot start.
    """
    cfg = cfg or settings
    if not sources:
        return TriageResult(ordered=[], rationale="no sources")

    try:
        Agent, Crew, _, Process, Task, _, _ = _require_crewai()
    except CrewUnavailable:
        return TriageResult(ordered=list(sources), rationale="crewai unavailable; original order")

    catalogue = [
        {
            "index": i,
            "url": s.url,
            "title": s.title[:120],
            "description": s.description[:200],
            "chars": len(s.markdown),
        }
        for i, s in enumerate(sources)
    ]
    fields = ", ".join(f.get("name", "") for f in schema.get("fields") or [])

    scout = Agent(
        role="Source Scout",
        goal=(
            "Identify which web pages actually contain structured records for the "
            "target schema, and reject pages that only talk about the topic."
        ),
        backstory=(
            "You have spent years building datasets from the public web. You can tell "
            "a real pricing table from a landing page that merely mentions prices, and "
            "you know that one dense table beats ten blog posts."
        ),
        llm=build_llm(cfg),
        verbose=False,
    )

    task = Task(
        description=(
            f"Target schema '{schema.get('name')}' with fields: {fields}.\n\n"
            f"Candidate pages:\n{json.dumps(catalogue, indent=2)}\n\n"
            "Rank these by how likely each is to yield many valid rows. Drop any page "
            "that is clearly editorial with no tabular or list data. Respond with JSON "
            'only: {"order": [<indices, best first>], "rationale": "<one sentence>"}'
        ),
        expected_output='JSON object with keys "order" and "rationale".',
        agent=scout,
    )

    try:
        raw = str(Crew(agents=[scout], tasks=[task], process=Process.sequential).kickoff())
        start, end = raw.find("{"), raw.rfind("}")
        parsed = json.loads(raw[start : end + 1]) if start >= 0 < end else {}
        order = [int(i) for i in (parsed.get("order") or []) if 0 <= int(i) < len(sources)]
        if not order:
            raise ValueError("empty order")
        seen: set[int] = set()
        ranked = [sources[i] for i in order if not (i in seen or seen.add(i))]
        # Keep unranked pages at the back rather than discarding them.
        ranked += [s for i, s in enumerate(sources) if i not in seen]
        return TriageResult(ordered=ranked, rationale=str(parsed.get("rationale") or ""))
    except Exception as exc:  # noqa: BLE001
        return TriageResult(ordered=list(sources), rationale=f"triage failed ({exc}); original order")


# -- the publish gate --------------------------------------------------------


def review_and_publish(
    *,
    schema: dict,
    summary_blob: dict,
    dataset_csv: str,
    manifest_md: str,
    cfg: Settings | None = None,
) -> str:
    """Let the Data Steward judge the run and, if it passes, write it out via One.

    Returns the agent's report. This is the call that changes a real system, so it
    is only ever reached from `cleanroom run --publish`.
    """
    cfg = cfg or settings
    Agent, Crew, _, Process, Task, MCPServerAdapter, _ = _require_crewai()

    target = f"{cfg.one_publish_platform}:{cfg.one_publish_target}"
    preview = dataset_csv[:4000]

    with MCPServerAdapter(one_server_params(cfg)) as raw_tools:
        tools = harden_tools(select_tools(raw_tools, ONE_TOOLS))
        if not tools:
            return (
                "One MCP exposed none of the expected four tools; nothing published. "
                "Check `one list` and ONE_SECRET."
            )

        steward = Agent(
            role="Data Steward",
            goal=(
                "Decide whether this dataset is fit to publish, and if so publish it "
                f"to {target} together with its provenance manifest."
            ),
            backstory=(
                "You are accountable for what gets committed. You care that rows are "
                "attributed, that no personal data slipped through, and that the run "
                "actually learned something rather than getting lucky once. You would "
                "rather block a publish than ship a dirty dataset."
            ),
            tools=tools,
            llm=build_llm(cfg),
            verbose=False,
        )

        task = Task(
            description=(
                f"Run summary:\n{json.dumps(summary_blob, indent=2, default=str)}\n\n"
                f"Provenance manifest:\n{manifest_md[:2500]}\n\n"
                f"Dataset preview (CSV):\n{preview}\n\n"
                "Assess: do most episodes score well, is every row attributed, and did "
                "the per-bucket posteriors move? If the dataset is not fit, say so and "
                "publish nothing.\n\n"
                "If it is fit, use the One tools in order: list_one_integrations, then "
                "search_one_platform_actions to find the action that creates or updates "
                "a file in a repository, then get_one_action_knowledge to read its exact "
                "schema (do this before executing -- the schema is authoritative), then "
                f"execute_one_action to write 'data/{schema.get('name')}.csv' to {target}. "
                "Use the exact actionId from the search results.\n\n"
                "Report what you decided, which actionId you used, and the result."
            ),
            expected_output=(
                "A short report: the publish decision, the reasoning, the actionId used, "
                "and the outcome of the write."
            ),
            agent=steward,
        )

        crew = Crew(agents=[steward], tasks=[task], process=Process.sequential)
        return str(crew.kickoff())
