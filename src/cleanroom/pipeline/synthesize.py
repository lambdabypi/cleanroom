"""Extractor synthesis and repair, with a swappable code-writing backend.

Claude writes a small pure-Python `extract(document) -> list[dict]` function; so
does You.com's Agents API. Both backends receive **byte-identical prompts** built
by the shared functions in this module, so switching backend does not silently
change the task and invalidate a learning run mid-way.

The chosen bandit arm enters through exactly one place -- the strategy's
`prompt_hint` -- which keeps the chain from "arm pulled" to "reward observed"
clean enough that the posterior means something.

Two backends, because they have different costs:

* **`you`** (default when no Anthropic key is present) -- the You.com Agents API,
  billed against the same You.com credits already used for retrieval. Free on the
  hackathon's $100 grant, and it makes You.com three endpoints deep.
* **`anthropic`** -- Claude with structured output and prompt caching. More
  reliable and much faster per call, but it is a paid API.

Output format differs by backend for a measured reason. Claude gets a JSON schema
because structured output is native. The Agents API is asked for a fenced
```python block instead: it is an answer-writing agent, and JSON-escaping a
multi-line program made it truncate mid-function in testing (the escaped `\\n`
form roughly triples the token count of the code). A fenced block needs no
escaping and came back intact.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from cleanroom.config import Settings, settings
from cleanroom.learning.memory import Lesson
from cleanroom.learning.strategies import STRATEGIES

MAX_DOCUMENT_CHARS = 60_000

CONTRACT = """\
You write small, deterministic Python data-extraction functions.

The module you write must:

1. Define exactly one public function: `extract(document: str) -> list[dict]`.
2. Import ONLY from the Python standard library (`re`, `json`, `html`, `csv`,
   `io`, `itertools`, `collections` are all available). No requests, no bs4, no
   pandas, no lxml -- they are not installed and the module will fail to import.
3. Make no network calls and touch no files. The input document is the only
   source of data.
4. Return a list of dicts whose keys are exactly the schema field names.
5. Never raise on malformed input. Skip a record you cannot parse rather than
   throwing -- a crash scores zero, whereas partial output scores partially.
6. Terminate quickly. No unbounded `while` loops, no catastrophic backtracking
   in regexes (avoid nested quantifiers like `(\\w+)*`).

Rules about the data itself:

- IGNORE the source-URL field entirely. Set it to None, or leave the key out.
  The harness fills it in from the page it fetched, which it knows reliably --
  do not try to hardcode a URL into your code.
- Never emit personal data -- no emails, phone numbers, or government IDs. Rows
  containing them are rejected by the validator.
- Emit numbers as numbers, not strings with units. Strip currency symbols,
  thousands separators, and trailing units.
- Do not invent values. A missing field should be `None`, not a plausible guess.

Write the simplest function that satisfies the strategy you are given. Prefer
explicit, readable parsing over cleverness.
"""


class SynthesisError(RuntimeError):
    pass


class RateLimited(SynthesisError):
    """Provider said 429. Recoverable by waiting, so it carries the delay."""

    def __init__(self, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PayloadTooLarge(SynthesisError):
    """Provider said 413. Recoverable by sending less of the document."""


class CreditsExhausted(SynthesisError):
    """The code-writing provider is out of credits.

    Separated from the generic error because it is *not* worth retrying and not
    worth continuing a run through: every remaining episode would score 0.0 and
    the learning curve would look like a failed policy rather than a dead API
    key. The loop aborts on this rather than grinding out a misleading run.
    """


@dataclass
class Extractor:
    code: str
    approach: str
    strategy: str
    backend: str = ""
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    repaired_from: str | None = None


# -- shared prompt construction ---------------------------------------------


def schema_text(schema: dict) -> str:
    """Stable schema rendering.

    `sort_keys=True` is load-bearing for the Anthropic backend -- an unsorted dump
    reorders between runs and silently destroys every prompt-cache hit.
    """
    publishable = {k: v for k, v in schema.items() if not k.startswith("_")}
    return json.dumps(publishable, indent=2, sort_keys=True)


def _document_excerpt(document: str, limit: int | None = None) -> str:
    """Trim the page to `limit` characters.

    `limit` is chosen per-episode by the profile bandit, and is the main lever on
    what an episode costs -- input tokens dominate the bill.
    """
    cap = limit or MAX_DOCUMENT_CHARS
    if len(document) <= cap:
        return document
    # Keep both ends: records often start near the top, and tables frequently
    # continue to the bottom. Truncating only the tail loses the latter.
    head = document[: int(cap * 0.7)]
    tail = document[-int(cap * 0.3) :]
    return f"{head}\n\n... [{len(document) - cap} chars elided] ...\n\n{tail}"


def _lessons_block(lessons: Sequence[Lesson]) -> str:
    if not lessons:
        return ""
    body = "\n".join(f"- {l.text}" for l in lessons)
    return (
        "\nLESSONS FROM EARLIER ATTEMPTS (these are things that actually went "
        f"wrong or right on similar pages -- respect them):\n{body}\n"
    )


def _strategy(strategy_id: str):
    strategy = STRATEGIES.get(strategy_id)
    if strategy is None:
        raise SynthesisError(f"unknown strategy {strategy_id!r}")
    return strategy


def build_synthesis_prompt(
    *,
    document: str,
    source_url: str,
    strategy_id: str,
    bucket: str,
    lessons: Sequence[Lesson] = (),
    doc_chars: int | None = None,
) -> str:
    strategy = _strategy(strategy_id)
    return (
        f"SOURCE_URL = {source_url!r}\n"
        f"PAGE SHAPE (auto-classified): {bucket}\n\n"
        f"STRATEGY TO USE -- {strategy.id}: {strategy.summary}\n"
        f"{strategy.prompt_hint}\n"
        f"{_lessons_block(lessons)}\n"
        "DOCUMENT\n"
        "--------\n"
        f"{_document_excerpt(document, doc_chars)}\n"
    )


def build_repair_prompt(
    *,
    document: str,
    source_url: str,
    strategy_id: str,
    bucket: str,
    previous_code: str,
    failure: str,
    report: dict | None = None,
    lessons: Sequence[Lesson] = (),
    doc_chars: int | None = None,
) -> str:
    strategy = _strategy(strategy_id)
    observed = ""
    if report:
        observed = (
            "\nVALIDATOR REPORT\n"
            f"  rows returned : {report.get('rows_total')}\n"
            f"  rows valid    : {report.get('rows_valid')}\n"
            f"  duplicates    : {report.get('rows_duplicate')}\n"
            f"  field errors  : {json.dumps(report.get('field_error_counts') or {})}\n"
            f"  sample errors : {json.dumps((report.get('sample_errors') or [])[:6], indent=2)}\n"
        )
    return (
        f"SOURCE_URL = {source_url!r}\n"
        f"PAGE SHAPE: {bucket}\n\n"
        f"STRATEGY (do not switch strategies) -- {strategy.id}: {strategy.summary}\n"
        f"{strategy.prompt_hint}\n\n"
        "Your previous attempt failed. Fix it.\n\n"
        "PREVIOUS CODE\n"
        "-------------\n"
        f"{previous_code}\n\n"
        f"WHAT WENT WRONG\n---------------\n{failure.strip() or 'unknown failure'}\n{observed}"
        f"{_lessons_block(lessons)}\n"
        "Diagnose the specific cause and rewrite the module. Stay within the "
        "same strategy; change the parsing details, not the approach.\n\n"
        "DOCUMENT\n"
        "--------\n"
        f"{_document_excerpt(document, doc_chars)}\n"
    )


# -- code post-processing ----------------------------------------------------

_FENCE_TAGGED = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)
_FENCE_BARE = re.compile(r"```[ \t]*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    """Pull the largest fenced code block out of a prose answer."""
    for pattern in (_FENCE_TAGGED, _FENCE_BARE):
        blocks = pattern.findall(text)
        if blocks:
            return max(blocks, key=len).strip()
    return None


def clean_code(code: str) -> str:
    text = (code or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip() + "\n"


def check_code(code: str) -> str | None:
    """Local syntax + contract check. Returns an error string, or None if fine.

    Worth doing before spending a sandbox round trip: truncated output is the
    dominant failure mode on the Agents backend, and `compile()` catches it in
    microseconds instead of ~10 seconds.
    """
    if "def extract" not in code:
        return "module does not define extract()"
    try:
        compile(code, "extractor.py", "exec")
    except SyntaxError as exc:
        return (
            f"generated code is not valid Python ({exc.msg} at line {exc.lineno}) "
            "-- most likely the response was truncated"
        )
    return None


# -- backend protocol --------------------------------------------------------


class Synthesizer(Protocol):
    backend: str

    def synthesize(
        self,
        *,
        schema: dict,
        document: str,
        source_url: str,
        strategy_id: str,
        bucket: str,
        lessons: Sequence[Lesson] = (),
        doc_chars: int | None = None,
    ) -> Extractor: ...

    def repair(
        self,
        *,
        schema: dict,
        document: str,
        source_url: str,
        strategy_id: str,
        bucket: str,
        previous: Extractor,
        failure: str,
        report: dict | None = None,
        lessons: Sequence[Lesson] = (),
        doc_chars: int | None = None,
    ) -> Extractor: ...


# -- Anthropic backend -------------------------------------------------------

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "approach": {
            "type": "string",
            "description": "One or two sentences on how the code locates records.",
        },
        "code": {
            "type": "string",
            "description": "Complete Python module defining extract(document).",
        },
    },
    "required": ["approach", "code"],
    "additionalProperties": False,
}


class AnthropicSynthesizer:
    """Claude backend. Structured output, and the contract + schema are cached."""

    backend = "anthropic"

    def __init__(self, cfg: Settings | None = None, client=None) -> None:
        import anthropic

        self._anthropic = anthropic
        self.cfg = cfg or settings
        if client is not None:
            self.client = client
        else:
            if not self.cfg.anthropic_api_key:
                raise SynthesisError("ANTHROPIC_API_KEY is not set; run `cleanroom doctor`")
            self.client = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)

    def _system_blocks(self, schema: dict) -> list[dict]:
        return [
            {"type": "text", "text": CONTRACT},
            {
                "type": "text",
                "text": f"TARGET SCHEMA\n\n{schema_text(schema)}",
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def _request(self, schema: dict, user_content: str, strategy_id: str,
                 repaired_from: str | None) -> Extractor:
        from cleanroom.observability.ledger import get_ledger
        from cleanroom.observability.pricing import llm_cost

        anthropic = self._anthropic
        ledger = get_ledger(self.cfg)
        with ledger.track("writer.anthropic", component="writer") as span:
            extractor = self._request_inner(schema, user_content, strategy_id,
                                            repaired_from, anthropic)
            span.charge(
                llm_cost(
                    model=self.cfg.model,
                    input_tokens=extractor.input_tokens,
                    output_tokens=extractor.output_tokens,
                )
            )
            span.note(
                model=self.cfg.model,
                input_tokens=extractor.input_tokens,
                output_tokens=extractor.output_tokens,
                cache_read_tokens=extractor.cache_read_tokens,
                prompt_chars=len(user_content),
            )
            return extractor

    def _request_inner(self, schema: dict, user_content: str, strategy_id: str,
                       repaired_from: str | None, anthropic) -> Extractor:
        try:
            response = self.client.messages.create(
                model=self.cfg.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "medium",
                    "format": {"type": "json_schema", "schema": _PLAN_SCHEMA},
                },
                system=self._system_blocks(schema),
                messages=[{"role": "user", "content": user_content}],
            )
        except anthropic.AuthenticationError as exc:
            raise SynthesisError("Anthropic rejected ANTHROPIC_API_KEY") from exc
        except anthropic.RateLimitError as exc:
            raise SynthesisError("Anthropic rate limit hit; slow the loop down") from exc
        except anthropic.APIStatusError as exc:
            raise SynthesisError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise SynthesisError(f"could not reach Anthropic: {exc}") from exc

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "category", None)
            raise SynthesisError(f"model declined the request (category={detail})")

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            raise SynthesisError("model returned no text block")
        try:
            plan = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SynthesisError(f"structured output was not valid JSON: {text[:200]}") from exc

        code = clean_code(plan.get("code") or "")
        problem = check_code(code)
        if problem:
            raise SynthesisError(problem)

        usage = response.usage
        return Extractor(
            code=code,
            approach=(plan.get("approach") or "").strip(),
            strategy=strategy_id,
            backend=self.backend,
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            repaired_from=repaired_from,
        )

    def synthesize(self, *, schema, document, source_url, strategy_id, bucket,
                   lessons=(), doc_chars=None):
        prompt = build_synthesis_prompt(
            document=document, source_url=source_url, strategy_id=strategy_id,
            bucket=bucket, lessons=lessons, doc_chars=doc_chars,
        )
        return self._request(schema, prompt, strategy_id, None)

    def repair(self, *, schema, document, source_url, strategy_id, bucket, previous,
               failure, report=None, lessons=(), doc_chars=None):
        prompt = build_repair_prompt(
            document=document, source_url=source_url, strategy_id=strategy_id, bucket=bucket,
            previous_code=previous.code, failure=failure, report=report, lessons=lessons,
            doc_chars=doc_chars,
        )
        return self._request(schema, prompt, strategy_id, previous.code)


#: Backwards-compatible alias -- the Anthropic backend was the original class.
ExtractorSynthesizer = AnthropicSynthesizer


# -- factory -----------------------------------------------------------------


def build_synthesizer(cfg: Settings | None = None) -> Synthesizer:
    """Pick the code-writing backend from config.

    Imported lazily so that a missing `anthropic` package never breaks the
    You.com path, which is the default when no Anthropic key is present.
    """
    cfg = cfg or settings
    backend = cfg.resolved_backend
    if backend == "anthropic":
        return AnthropicSynthesizer(cfg)
    if backend == "compat":
        from cleanroom.pipeline.openai_compat import OpenAICompatSynthesizer

        return OpenAICompatSynthesizer(cfg)
    if backend == "you":
        from cleanroom.pipeline.you_agent import YouAgentSynthesizer

        return YouAgentSynthesizer(cfg)

    raise SynthesisError(
        "No code writer is configured. Set an OpenAI-compatible endpoint in .env:\n"
        "  CLEANROOM_LLM_BASE_URL=https://api.groq.com/openai/v1\n"
        "  CLEANROOM_LLM_API_KEY=<your key>\n"
        "  CLEANROOM_LLM_MODEL=llama-3.3-70b-versatile\n"
        "...or set ANTHROPIC_API_KEY.\n\n"
        "The You.com Agents API also works but bills $15 PER CALL "
        "($300+ for one 20-episode run), so it is never selected automatically. "
        "Opt in with CLEANROOM_SYNTH_BACKEND=you only if you mean it."
    )
