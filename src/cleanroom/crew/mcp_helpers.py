"""Hardening wrappers for One's MCP tools inside CrewAI.

One's docs flag two specific failure modes when wiring its MCP server into
CrewAI, and both are silent -- you get empty results or a rejected schema rather
than an exception, which is a bad way to lose an hour:

* **`dropping_nulls`** -- CrewAI serialises every declared optional parameter,
  sending `null` for the ones the agent left unset. One validates against the
  action's schema, where an explicit `null` is not the same as an absent key, so
  the call fails or silently returns nothing. Strip them before dispatch.

* **`harden_execute`** -- LLM-generated arguments arrive as JSON *strings* rather
  than objects often enough to matter, so nested params need parsing. The same
  wrapper drops any agent-supplied `x-one-*` header override: header fields carry
  the credentials, and a model that can set them can redirect the call to another
  connection. Model output is untrusted input here.

Wrapped tools keep CrewAI's `BaseTool` interface, so they drop straight into
`Agent(tools=...)`.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

#: Header-shaped keys an agent must never be able to set.
_FORBIDDEN_KEYS = frozenset(
    {
        "x-one-secret",
        "x-one-connection-key",
        "x-one-action-id",
        "authorization",
        "headers",
        "_headers",
    }
)


def drop_nulls(payload: Any) -> Any:
    """Recursively remove keys whose value is None."""
    if isinstance(payload, dict):
        return {k: drop_nulls(v) for k, v in payload.items() if v is not None}
    if isinstance(payload, list):
        return [drop_nulls(v) for v in payload]
    return payload


def parse_json_strings(payload: Any, *, depth: int = 0) -> Any:
    """Turn JSON-encoded strings into real objects.

    Bounded depth: a pathological input should not recurse forever, and two
    levels covers every real case (a stringified object containing stringified
    fields).
    """
    if depth > 2:
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if text[:1] in ("{", "[") and text[-1:] in ("}", "]"):
            try:
                return parse_json_strings(json.loads(text), depth=depth + 1)
            except json.JSONDecodeError:
                return payload
        return payload
    if isinstance(payload, dict):
        return {k: parse_json_strings(v, depth=depth + 1) for k, v in payload.items()}
    if isinstance(payload, list):
        return [parse_json_strings(v, depth=depth + 1) for v in payload]
    return payload


def strip_credential_overrides(payload: Any) -> Any:
    """Remove agent attempts to set credential or header fields."""
    if isinstance(payload, dict):
        return {
            k: strip_credential_overrides(v)
            for k, v in payload.items()
            if k.lower() not in _FORBIDDEN_KEYS
        }
    if isinstance(payload, list):
        return [strip_credential_overrides(v) for v in payload]
    return payload


def sanitize(payload: Any) -> Any:
    """The full inbound treatment, in the order that matters.

    Parse first (so nested objects become inspectable), then strip credentials
    (so a stringified header override cannot smuggle itself past), then drop
    nulls last (so keys emptied by stripping do not linger).
    """
    return drop_nulls(strip_credential_overrides(parse_json_strings(payload)))


def harden_tool(tool: Any) -> Any:
    """Wrap one CrewAI MCP tool so every call is sanitised.

    Patches the instance's `_run` rather than subclassing: MCP tools are built
    dynamically by `MCPServerAdapter`, so there is no static class to extend.
    """
    original = getattr(tool, "_run", None)
    if original is None or getattr(tool, "_cleanroom_hardened", False):
        return tool

    def _run(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **sanitize(kwargs))

    try:
        tool._run = _run  # noqa: SLF001 - intentional instance patch
        tool._cleanroom_hardened = True
    except (AttributeError, TypeError):
        # Frozen/pydantic-locked tool: better to run unwrapped than to crash.
        return tool
    return tool


def harden_tools(tools: Iterable[Any]) -> list[Any]:
    return [harden_tool(t) for t in tools]


def select_tools(tools: Iterable[Any], wanted: Sequence[str]) -> list[Any]:
    """Keep only the named tools.

    One's whole design point is that agents *search* for actions instead of
    receiving a 780-app tool list, so handing a CrewAI agent every tool undoes
    that and blows up the prompt. Narrow to the four-tool loop.
    """
    wanted_lower = {w.lower() for w in wanted}
    return [t for t in tools if str(getattr(t, "name", "")).lower() in wanted_lower]
