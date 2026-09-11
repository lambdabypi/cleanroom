"""Tests for the glue: MCP input hardening, schema checks, execution fallback.

No network and no credentials. The sanitizer tests matter most -- they encode a
security property (model output cannot set credential headers), and a regression
there would be silent.
"""

from __future__ import annotations

import pytest

from cleanroom.crew.mcp_helpers import (
    drop_nulls,
    harden_tool,
    parse_json_strings,
    sanitize,
    select_tools,
    strip_credential_overrides,
)
from cleanroom.partners.daytona_env import LocalEnvironment, _parse_report
from cleanroom.pipeline.schema import check_schema, load_schema
from cleanroom.pipeline.sandbox_runner import BEGIN, END

SCHEMA = {
    "name": "test",
    "fields": [
        {"name": "provider", "type": "string", "required": True},
        {"name": "usd_per_hour", "type": "number", "required": True, "min": 0},
        {"name": "source_url", "type": "string", "required": True, "format": "url"},
    ],
}


# -- MCP input hardening -----------------------------------------------------


def test_nulls_are_dropped_recursively():
    assert drop_nulls({"a": 1, "b": None, "c": {"d": None, "e": 2}}) == {"a": 1, "c": {"e": 2}}


def test_json_strings_are_parsed():
    assert parse_json_strings('{"a": 1}') == {"a": 1}
    assert parse_json_strings('[1, 2]') == [1, 2]
    # Plain strings must survive untouched.
    assert parse_json_strings("just text") == "just text"
    assert parse_json_strings("{not json") == "{not json"


def test_credential_headers_are_stripped():
    """A model that can set x-one-* could redirect the call to another connection."""
    payload = {
        "owner": "acme",
        "x-one-secret": "leak",
        "X-One-Connection-Key": "leak",
        "authorization": "Bearer leak",
        "headers": {"x-one-action-id": "evil"},
    }
    clean = strip_credential_overrides(payload)
    assert clean == {"owner": "acme"}


def test_sanitize_order_catches_stringified_credential_override():
    """Parse must run before stripping, or a JSON-encoded header slips through."""
    payload = {"body": '{"x-one-secret": "leak", "path": "a.csv", "branch": null}'}
    clean = sanitize(payload)
    assert clean == {"body": {"path": "a.csv"}}


def test_sanitize_is_idempotent():
    payload = {"a": 1, "b": None, "x-one-secret": "leak"}
    assert sanitize(sanitize(payload)) == sanitize(payload)


class _FakeTool:
    name = "execute_one_action"

    def __init__(self):
        self.seen: dict | None = None

    def _run(self, **kwargs):
        self.seen = kwargs
        return "done"


def test_harden_tool_sanitizes_calls():
    tool = harden_tool(_FakeTool())
    tool._run(owner="acme", repo=None, **{"x-one-secret": "leak"})
    assert tool.seen == {"owner": "acme"}


def test_harden_tool_is_applied_once():
    tool = _FakeTool()
    assert harden_tool(harden_tool(tool)) is tool
    tool._run(a=1, b=None)
    assert tool.seen == {"a": 1}


def test_select_tools_narrows_to_the_four_tool_loop():
    class T:
        def __init__(self, name):
            self.name = name

    tools = [T("execute_one_action"), T("send_email"), T("list_one_integrations")]
    kept = select_tools(tools, ["execute_one_action", "list_one_integrations"])
    assert {t.name for t in kept} == {"execute_one_action", "list_one_integrations"}


# -- schema checks -----------------------------------------------------------


def test_bundled_schema_is_valid():
    assert check_schema(load_schema()) == []


def test_schema_without_required_field_is_rejected():
    problems = check_schema({"name": "x", "fields": [{"name": "a", "type": "string"}]})
    assert any("required" in p for p in problems)
    assert any("url" in p for p in problems)


def test_schema_with_bad_type_is_rejected():
    problems = check_schema(
        {"name": "x", "fields": [{"name": "a", "type": "strng", "required": True,
                                  "format": "url"}]}
    )
    assert any("strng" in p for p in problems)


def test_duplicate_field_names_are_rejected():
    problems = check_schema(
        {
            "name": "x",
            "fields": [
                {"name": "a", "type": "string", "required": True},
                {"name": "a", "type": "string", "format": "url"},
            ],
        }
    )
    assert any("duplicate" in p for p in problems)


# -- sandbox report parsing --------------------------------------------------


def test_report_is_recovered_from_noisy_stdout():
    """Generated extractors print their own debug output around the report."""
    noisy = f'debug line\n{BEGIN}\n{{"ok": true, "rows_valid": 3}}\n{END}\ntrailing junk\n'
    assert _parse_report(noisy) == {"ok": True, "rows_valid": 3}


def test_missing_sentinels_yield_no_report():
    assert _parse_report("total garbage") is None
    assert _parse_report(f"{BEGIN}\nnot json\n{END}") is None


# -- local execution fallback ------------------------------------------------


def test_local_environment_runs_and_scores():
    env = LocalEnvironment()
    code = (
        "def extract(document):\n"
        "    return [{'provider': 'Lambda', 'usd_per_hour': 2.49,\n"
        "             'source_url': 'https://example.com/p'}]\n"
    )
    result = env.run(code, "anything", SCHEMA)
    assert not result.crashed
    assert result.report["rows_valid"] == 1


def test_local_environment_reports_syntax_errors_as_crashes():
    result = LocalEnvironment().run("def extract(:\n", "x", SCHEMA)
    assert result.crashed
    assert result.report is None


def test_local_environment_reports_runtime_errors_as_crashes():
    result = LocalEnvironment().run(
        "def extract(d):\n    raise ValueError('boom')\n", "x", SCHEMA
    )
    assert result.crashed
    assert "boom" in result.stderr


def test_missing_extract_contract_is_a_crash():
    result = LocalEnvironment().run("def other():\n    pass\n", "x", SCHEMA)
    assert result.crashed
    assert "extract()" in result.stderr
