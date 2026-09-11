"""You.com Agents API as the code-writing backend.

`POST https://api.you.com/v1/agents/runs`, `Authorization: Bearer <YOU_API_KEY>`.
Billed against the same You.com credits as retrieval, which makes the whole agent
runnable on the hackathon's free grant with no second paid provider -- and takes
the submission to three You.com endpoints (search, contents, agents).

Two things were measured rather than assumed, both the hard way:

* **`verbosity` must be `"high"`.** At `"medium"` the response is cut off
  mid-function and the module fails to compile. `"low"` is rejected outright
  (422: must be `medium` or `high`).
* **Ask for a fenced ```python block, never JSON.** This is an answer-writing
  agent, and JSON-escaping a multi-line program roughly triples its token count
  via `\\n` escapes, which is what triggered the truncation. A fenced block needs
  no escaping and survives intact.

The response is prose with the code embedded -- instructions to return bare code
are not reliably followed -- so `extract_code_block` mines the largest fenced
block out of the answer, and `check_code` compiles it locally before anything is
sent to a sandbox. One internal retry covers truncation, since that failure is
both common and recoverable by simply asking again.
"""

from __future__ import annotations

import os

import requests

from cleanroom.config import Settings, settings
from cleanroom.pipeline.synthesize import (
    CONTRACT,
    CreditsExhausted,
    Extractor,
    SynthesisError,
    build_repair_prompt,
    build_synthesis_prompt,
    check_code,
    clean_code,
    extract_code_block,
    schema_text,
)

#: Generous: a cold Agents run took ~20s in testing, and repairs are longer.
REQUEST_TIMEOUT = 240

#: MEASURED PRICE, from the You.com billing dashboard: `Agent API - Advanced`
#: bills **$15 per call**. Eight calls came to $120. This constant exists so the
#: number is visible in code review rather than discovered on an invoice.
USD_PER_CALL = 15.00

#: Hard per-process ceiling. A learning run wants 20-30 synthesis calls, which
#: at $15 each is $300-450, so the backend refuses to be the thing that quietly
#: spends it. Raise with YOU_AGENT_MAX_CALLS if you genuinely intend to.
DEFAULT_MAX_CALLS = 3

OUTPUT_INSTRUCTION = """\
OUTPUT FORMAT -- follow this exactly:
Respond with ONE fenced code block tagged `python`, containing the complete
module. Do not split the code across multiple blocks. Do not abbreviate, and do
not write `# ... rest unchanged` -- the block must be the entire runnable module.
"""


class YouAgentSynthesizer:
    backend = "you"

    def __init__(self, cfg: Settings | None = None, session: requests.Session | None = None) -> None:
        self.cfg = cfg or settings
        if not self.cfg.you_api_key:
            raise SynthesisError("YOU_API_KEY is not set; run `cleanroom doctor`")
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.cfg.you_api_key}",
                "Content-Type": "application/json",
            }
        )
        self.calls = 0
        self.max_calls = _max_calls()

    @property
    def spent_usd(self) -> float:
        return self.calls * USD_PER_CALL

    # -- transport ---------------------------------------------------------

    def _call(self, prompt: str) -> str:
        if self.calls >= self.max_calls:
            raise CreditsExhausted(
                f"refusing to exceed {self.max_calls} You.com Agents calls "
                f"(~${self.spent_usd:.0f} spent at ${USD_PER_CALL:.0f}/call). "
                "Switch the code writer to a free provider:\n"
                "  CLEANROOM_LLM_BASE_URL=https://api.groq.com/openai/v1\n"
                "  CLEANROOM_LLM_MODEL=llama-3.3-70b-versatile\n"
                "or raise YOU_AGENT_MAX_CALLS if you intend the spend."
            )
        from cleanroom.observability.ledger import get_ledger
        from cleanroom.observability.pricing import call_cost

        self.calls += 1
        ledger = get_ledger(self.cfg)
        with ledger.track("you.agents", component="writer") as span:
            # Charged up front: at $15 a call this must be recorded even if the
            # response turns out to be unusable.
            span.charge(call_cost("you.agents"))
            span.note(call_index=self.calls, prompt_chars=len(prompt))
            return self._post(prompt)

    def _post(self, prompt: str) -> str:
        body = {
            "agent": "advanced",
            "input": prompt,
            "stream": False,
            # Anything below "high" truncates the module. Do not lower this.
            "verbosity": "high",
            # The task is pure code generation over a document that is already in
            # the prompt, so there is nothing to research -- one step keeps it fast.
            "workflow_config": {"max_workflow_steps": 1},
        }
        try:
            resp = self.session.post(self.cfg.you_agents_url, json=body, timeout=REQUEST_TIMEOUT)
        except requests.Timeout as exc:
            raise SynthesisError(f"You.com Agents timed out after {REQUEST_TIMEOUT}s") from exc
        except requests.RequestException as exc:
            raise SynthesisError(f"could not reach You.com Agents: {exc}") from exc

        if resp.status_code in (401, 403):
            raise SynthesisError(
                f"You.com rejected the key for the Agents API ({resp.status_code}). "
                "The Agents API is a separate entitlement from Search -- check your plan."
            )
        if resp.status_code == 402:
            # Measured: roughly half a dozen `advanced` runs at verbosity "high"
            # exhausted a fresh grant, while Search and Contents kept working --
            # they meter separately. This must be loud, because the symptom is
            # every episode scoring 0.0, which looks like a learning failure.
            raise CreditsExhausted(
                "You.com Agents credits are exhausted (402). Search and Contents "
                "usually still work, so retrieval is fine -- it is only code "
                "generation that is blocked. Switch the code writer to a free "
                "provider:\n"
                "  CLEANROOM_LLM_BASE_URL=https://api.groq.com/openai/v1\n"
                "  CLEANROOM_LLM_API_KEY=<groq key>\n"
                "  CLEANROOM_LLM_MODEL=llama-3.3-70b-versatile\n"
                "...or top up at https://you.com/platform"
            )
        if resp.status_code == 422:
            raise SynthesisError(f"You.com Agents rejected the request body: {resp.text[:300]}")
        if resp.status_code == 429:
            raise SynthesisError("You.com Agents rate limit (429); slow the episode loop down")
        if not resp.ok:
            raise SynthesisError(f"You.com Agents {resp.status_code}: {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise SynthesisError("You.com Agents returned non-JSON") from exc

        # Shape: {"output": [{"text": "...", "type": "message.answer"}], ...}
        blocks = payload.get("output") or []
        text = "\n".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict)
        ).strip()
        if not text:
            raise SynthesisError(f"You.com Agents returned no answer text: {str(payload)[:300]}")
        return text

    def _synthesize_once(self, prompt: str) -> tuple[str, str]:
        """Returns (code, answer_text). Raises SynthesisError on unusable output."""
        text = self._call(prompt)
        block = extract_code_block(text)
        if not block:
            raise SynthesisError(
                f"no fenced code block in the Agents answer (got {len(text)} chars of prose)"
            )
        code = clean_code(block)
        problem = check_code(code)
        if problem:
            raise SynthesisError(problem)
        return code, text

    def _run(self, prompt: str, strategy_id: str, repaired_from: str | None) -> Extractor:
        attempts: list[str] = []
        for attempt in range(2):
            # Retrying a 402 just wastes 20 seconds per episode.
            nudge = ""
            if attempt:
                # Truncation is the dominant failure here and asking again fixes
                # it more often than not; say what went wrong so the retry is
                # informed rather than identical.
                nudge = (
                    "\nIMPORTANT: your previous response was unusable -- "
                    f"{attempts[-1]}. Keep the module short and complete; "
                    "prefer fewer comments over an unfinished function.\n"
                )
            try:
                code, text = self._synthesize_once(prompt + nudge)
            except CreditsExhausted:
                raise
            except SynthesisError as exc:
                attempts.append(str(exc))
                continue
            return Extractor(
                code=code,
                approach=_first_sentence(text),
                strategy=strategy_id,
                backend=self.backend,
                repaired_from=repaired_from,
            )
        raise SynthesisError(f"Agents backend failed twice: {attempts[-1]}")

    # -- public API --------------------------------------------------------

    def _prefix(self, schema: dict) -> str:
        # The Agents API has no system-prompt field, so the contract and schema
        # are prepended to the input instead.
        return f"{CONTRACT}\n\nTARGET SCHEMA\n\n{schema_text(schema)}\n\n{OUTPUT_INSTRUCTION}\n"

    def synthesize(self, *, schema, document, source_url, strategy_id, bucket,
                   lessons=(), doc_chars=None):
        prompt = self._prefix(schema) + build_synthesis_prompt(
            document=document, source_url=source_url, strategy_id=strategy_id,
            bucket=bucket, lessons=lessons, doc_chars=doc_chars,
        )
        return self._run(prompt, strategy_id, None)

    def repair(self, *, schema, document, source_url, strategy_id, bucket, previous,
               failure, report=None, lessons=(), doc_chars=None):
        prompt = self._prefix(schema) + build_repair_prompt(
            document=document, source_url=source_url, strategy_id=strategy_id, bucket=bucket,
            previous_code=previous.code, failure=failure, report=report, lessons=lessons,
            doc_chars=doc_chars,
        )
        return self._run(prompt, strategy_id, previous.code)


def _max_calls() -> int:
    raw = (os.getenv("YOU_AGENT_MAX_CALLS") or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_CALLS
    except ValueError:
        return DEFAULT_MAX_CALLS


def _first_sentence(text: str, limit: int = 240) -> str:
    """Best-effort 'approach' line from the prose around the code block."""
    for line in text.splitlines():
        stripped = line.strip().lstrip("#>*- ").strip()
        if len(stripped) > 40 and "```" not in stripped:
            return stripped[:limit]
    return ""
