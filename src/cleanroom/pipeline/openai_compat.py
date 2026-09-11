"""Code-writing backend for any OpenAI-compatible `/chat/completions` endpoint.

One code path covers every provider that speaks the OpenAI chat protocol, which
is where the usable free tiers are:

| Provider  | `CLEANROOM_LLM_BASE_URL`                                  | Suggested model                  |
|-----------|-----------------------------------------------------------|----------------------------------|
| Groq      | `https://api.groq.com/openai/v1`                          | `llama-3.3-70b-versatile`        |
| Cerebras  | `https://api.cerebras.ai/v1`                              | `llama-3.3-70b`                  |
| Gemini    | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash`               |
| OpenRouter| `https://openrouter.ai/api/v1`                            | any `...:free` model             |
| Ollama    | `http://localhost:11434/v1`                               | `qwen2.5-coder:7b`               |

Deliberately built on `requests` rather than the `openai` package: the surface
used here is a single POST, and adding an SDK dependency to reach providers that
are not OpenAI buys nothing. The Anthropic backend uses the real Anthropic SDK,
because there it is the first-party client.

`temperature` is low but not zero. Some exploration in the generated code is
useful -- two attempts at the same strategy on the same page should not be
byte-identical, or a repair turn cannot discover anything new.
"""

from __future__ import annotations

import time

import requests

from cleanroom.config import Settings, settings
from cleanroom.pipeline.synthesize import (
    CONTRACT,
    Extractor,
    PayloadTooLarge,
    RateLimited,
    SynthesisError,
    build_repair_prompt,
    build_synthesis_prompt,
    check_code,
    clean_code,
    extract_code_block,
    schema_text,
)

REQUEST_TIMEOUT = 180

#: 429 retries. Free tiers are per-minute limited, so a short wait usually clears.
MAX_RATE_LIMIT_RETRIES = 3
DEFAULT_BACKOFF_S = 20.0


def _retry_after(resp: requests.Response) -> float:
    """Honour the provider's own advice when it gives any."""
    for header in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw = (resp.headers.get(header) or "").strip().rstrip("s")
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if 0 < value <= 120:
            return value
    return DEFAULT_BACKOFF_S

OUTPUT_INSTRUCTION = """\
OUTPUT FORMAT -- follow this exactly:
Respond with ONE fenced code block tagged `python` containing the complete
module. No prose before or after it. Do not abbreviate and do not write
`# ... unchanged` -- the block must be the entire runnable module.
"""


class OpenAICompatSynthesizer:
    backend = "compat"

    def __init__(self, cfg: Settings | None = None, session: requests.Session | None = None) -> None:
        self.cfg = cfg or settings
        if not self.cfg.llm_base_url or not self.cfg.llm_model:
            raise SynthesisError(
                "CLEANROOM_LLM_BASE_URL and CLEANROOM_LLM_MODEL must both be set "
                "for the compat backend; run `cleanroom doctor`"
            )
        self.session = session or requests.Session()
        headers = {"Content-Type": "application/json"}
        if self.cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.llm_api_key}"
        self.session.headers.update(headers)
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    @property
    def endpoint(self) -> str:
        return f"{self.cfg.llm_base_url}/chat/completions"

    def available_models(self) -> list[str]:
        """Model ids this key can reach. Best-effort; used to explain a 404."""
        try:
            resp = self.session.get(f"{self.cfg.llm_base_url}/models", timeout=30)
            if not resp.ok:
                return []
            data = resp.json().get("data") or []
            return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))
        except (requests.RequestException, ValueError):
            return []

    # -- transport ---------------------------------------------------------

    def _call(self, system: str, user: str) -> str:
        """POST one completion. Records latency, tokens and cost in the ledger."""
        from cleanroom.observability.ledger import get_ledger
        from cleanroom.observability.pricing import estimate_tokens, llm_cost

        ledger = get_ledger(self.cfg)
        with ledger.track("writer.compat", component="writer") as span:
            text, usage = self._post_with_backoff(system, user)
            # Prefer the provider's own usage block; the chars/4 heuristic is a
            # fallback so an un-instrumented provider still yields an estimate.
            in_tokens = int(usage.get("prompt_tokens") or 0) or estimate_tokens(system + user)
            out_tokens = int(usage.get("completion_tokens") or 0) or estimate_tokens(text)
            span.charge(
                llm_cost(
                    model=self.cfg.llm_model,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    base_url=self.cfg.llm_base_url,
                )
            )
            span.note(
                model=self.cfg.llm_model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                measured=bool(usage.get("prompt_tokens")),
                prompt_chars=len(system) + len(user),
            )
            self.last_input_tokens = in_tokens
            self.last_output_tokens = out_tokens
            return text

    def _post_with_backoff(self, system: str, user: str) -> tuple[str, dict]:
        """Retry 429s with the provider's advised delay.

        Free tiers are requests-per-minute limited, so a rate limit is a wait,
        not a failure -- treating it as one threw away 6 of 8 episodes in testing.
        """
        last: RateLimited | None = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                return self._post(system, user)
            except RateLimited as exc:
                last = exc
                if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                    break
                delay = exc.retry_after * (attempt + 1)
                print(f"    [rate limited, waiting {delay:.0f}s]", flush=True)
                time.sleep(delay)
        raise SynthesisError(
            f"{last} after {MAX_RATE_LIMIT_RETRIES} attempts -- lower -n, raise "
            "--pause, or use a smaller model"
        )

    def _post(self, system: str, user: str) -> tuple[str, dict]:
        body = {
            "model": self.cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
        }
        try:
            resp = self.session.post(self.endpoint, json=body, timeout=REQUEST_TIMEOUT)
        except requests.Timeout as exc:
            raise SynthesisError(f"{self.cfg.llm_model} timed out after {REQUEST_TIMEOUT}s") from exc
        except requests.RequestException as exc:
            raise SynthesisError(f"could not reach {self.endpoint}: {exc}") from exc

        if resp.status_code in (401, 403):
            raise SynthesisError(
                f"{self.endpoint} rejected the credentials ({resp.status_code}). "
                "Check CLEANROOM_LLM_API_KEY."
            )
        if resp.status_code == 402:
            raise SynthesisError(f"{self.cfg.llm_model}: credits exhausted (402)")
        if resp.status_code == 404:
            # Overwhelmingly this is a decommissioned model rather than a bad
            # URL -- hosted catalogues churn fast. Ask the provider what it
            # actually serves so the error is actionable instead of a puzzle.
            available = self.available_models()
            hint = (
                "models your key can reach: " + ", ".join(available[:12])
                if available
                else "could not list models either -- check CLEANROOM_LLM_BASE_URL "
                "(it should end at /v1)"
            )
            raise SynthesisError(
                f"404 from {self.endpoint} for model {self.cfg.llm_model!r}. {hint}"
            )
        if resp.status_code == 429:
            raise RateLimited(
                f"{self.cfg.llm_model} rate limit (429)",
                retry_after=_retry_after(resp),
            )
        if resp.status_code == 413:
            # Free tiers cap tokens per request. This is recoverable by sending
            # less of the page, so it gets its own type and the caller retries
            # with a smaller budget instead of losing the episode.
            raise PayloadTooLarge(
                f"{self.cfg.llm_model} rejected the request as too large (413); "
                "the document budget exceeds this tier's per-request token cap"
            )
        if not resp.ok:
            raise SynthesisError(f"{self.endpoint} {resp.status_code}: {resp.text[:300]}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise SynthesisError(f"{self.endpoint} returned non-JSON") from exc

        choices = payload.get("choices") or []
        if not choices:
            error = (payload.get("error") or {}).get("message") or str(payload)[:200]
            raise SynthesisError(f"no choices returned: {error}")
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not content:
            raise SynthesisError("model returned empty content")
        return content, (payload.get("usage") or {})

    def _run(self, system: str, user: str, strategy_id: str,
             repaired_from: str | None) -> Extractor:
        attempts: list[str] = []
        for attempt in range(2):
            nudge = ""
            if attempt:
                nudge = (
                    f"\nIMPORTANT: your previous response was unusable -- {attempts[-1]}. "
                    "Return one complete fenced python block and nothing else.\n"
                )
            try:
                text = self._call(system, user + nudge)
                block = extract_code_block(text)
                # Some models ignore the fence instruction and return bare code.
                if block is None and "def extract" in text:
                    block = text
                if block is None:
                    raise SynthesisError(
                        f"no code block in the response ({len(text)} chars of prose)"
                    )
                code = clean_code(block)
                problem = check_code(code)
                if problem:
                    raise SynthesisError(problem)
            except PayloadTooLarge:
                # Retrying the same oversized prompt is pointless; the caller
                # shrinks the document budget and tries again.
                raise
            except SynthesisError as exc:
                attempts.append(str(exc))
                # A credit or auth failure will not fix itself on retry.
                if any(s in str(exc) for s in ("402", "rejected the credentials", "404 from")):
                    raise
                continue
            return Extractor(
                code=code,
                approach=_first_prose_line(text),
                strategy=strategy_id,
                backend=self.backend,
                input_tokens=self.last_input_tokens,
                output_tokens=self.last_output_tokens,
                repaired_from=repaired_from,
            )
        raise SynthesisError(f"{self.cfg.llm_model} failed twice: {attempts[-1]}")

    # -- public API --------------------------------------------------------

    def _system(self, schema: dict) -> str:
        return f"{CONTRACT}\n\nTARGET SCHEMA\n\n{schema_text(schema)}\n\n{OUTPUT_INSTRUCTION}"

    def synthesize(self, *, schema, document, source_url, strategy_id, bucket,
                   lessons=(), doc_chars=None):
        return self._run(
            self._system(schema),
            build_synthesis_prompt(
                document=document, source_url=source_url, strategy_id=strategy_id,
                bucket=bucket, lessons=lessons, doc_chars=doc_chars,
            ),
            strategy_id,
            None,
        )

    def repair(self, *, schema, document, source_url, strategy_id, bucket, previous,
               failure, report=None, lessons=(), doc_chars=None):
        return self._run(
            self._system(schema),
            build_repair_prompt(
                document=document, source_url=source_url, strategy_id=strategy_id,
                bucket=bucket, previous_code=previous.code, failure=failure,
                report=report, lessons=lessons, doc_chars=doc_chars,
            ),
            strategy_id,
            previous.code,
        )


def _first_prose_line(text: str, limit: int = 240) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#>*- ").strip()
        if len(stripped) > 40 and "```" not in stripped:
            return stripped[:limit]
    return ""
