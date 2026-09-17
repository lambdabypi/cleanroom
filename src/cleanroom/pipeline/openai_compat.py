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

import re
import time
from collections import deque

import requests

from cleanroom.config import Settings, settings
from cleanroom.pipeline.synthesize import (
    CONTRACT,
    CreditsExhausted,
    DailyQuotaExhausted,
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

#: 429 retries. Free tiers are limited per *minute*, and the binding constraint
#: is usually tokens rather than requests -- so a 45k-character prompt can consume
#: a whole window on its own and the next few requests fail no matter how fast
#: they are. Three attempts over ~60s was not enough: a measured 30-episode run
#: lost 3 episodes to exhausted retries. Five attempts with a growing delay
#: covers a full 60s window plus slack, and a retry is always cheaper than
#: discarding the episode's LLM work.
MAX_RATE_LIMIT_RETRIES = 5
DEFAULT_BACKOFF_S = 20.0

#: Never wait longer than this in total for one completion. Past it, failing the
#: episode is better than stalling a run indefinitely.
MAX_TOTAL_BACKOFF_S = 240.0


class TokenPacer:
    """Keeps a rolling one-minute token spend under the provider's ceiling.

    Retrying after a 429 wastes the round trip; not provoking one is strictly
    better. The numbers make the case -- measured on Groq's free tier with the
    `lean` profile:

        per-minute allowance      8,000 tokens
        one synthesis call        5,260 tokens  (3,432 prompt + 1,828 completion)
        => sustainable rate       1.5 calls/min, i.e. 39s apart

    A `--pause 6` flag meant attempting roughly four calls a minute, four times
    over budget, so 429s were not bad luck but arithmetic. The pacer replaces
    that guess: it records what each call actually cost, and before the next one
    sleeps just long enough for the window to carry it.

    `safety` leaves headroom because the estimate for the *next* call is exactly
    that -- an estimate -- and the provider's window boundary is not observable.
    """

    def __init__(self, safety: float = 0.85) -> None:
        self._events: deque[tuple[float, int]] = deque()
        self.safety = safety
        self.total_waited = 0.0

    def _spent_in_window(self, now: float) -> int:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()
        return sum(tokens for _, tokens in self._events)

    def note(self, tokens: int) -> None:
        self._events.append((time.monotonic(), max(0, tokens)))

    def delay_for(self, estimated_tokens: int, limit: int | None) -> float:
        """Seconds to wait before spending `estimated_tokens`."""
        if not limit or limit <= 0:
            return 0.0
        budget = limit * self.safety
        now = time.monotonic()
        spent = self._spent_in_window(now)
        if spent + estimated_tokens <= budget:
            return 0.0
        # Wait for the oldest events to age out of the window, one at a time,
        # until the projected spend fits.
        for timestamp, tokens in list(self._events):
            spent -= tokens
            if spent + estimated_tokens <= budget:
                return max(0.0, 60.0 - (now - timestamp)) + 0.5
        return 60.0


#: A 429 body is the *only* place an OpenAI-compatible tier says which limit it
#: enforced. Measured on Groq: a per-day refusal arrives with
#: `x-ratelimit-remaining-tokens: 8000` and `x-ratelimit-reset-tokens: 1ms`,
#: because those headers describe the per-minute bucket. Reading the status code
#: alone, or trusting the headers, makes a 12-hour wall look like a 1-second one.
_DAILY_LIMIT = re.compile(r"tokens per day|\bTPD\b", re.IGNORECASE)
_LIMIT_FIGURES = re.compile(r"Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+)")
_TRY_AGAIN_IN = re.compile(r"try again in ([0-9hms.]+)")


def _quota_detail(body: str) -> str:
    """Restate the provider's own figures, so the log says what ran out."""
    figures = _LIMIT_FIGURES.search(body)
    when = _TRY_AGAIN_IN.search(body)
    parts = []
    if figures:
        limit, used, requested = (int(g) for g in figures.groups())
        parts.append(f"used {used:,} of {limit:,} tokens, this call needed {requested:,}")
    if when:
        parts.append(f"resets in {when.group(1)}")
    return "; ".join(parts) or "the provider gave no figures"


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
        #: Provider's tokens-per-minute ceiling, learned from response headers.
        self.tokens_per_minute: int | None = None
        self.pacer = TokenPacer()
        #: Rolling mean completion size, so the next call's cost can be
        #: estimated rather than guessed. Seeded from a measured run: gpt-oss
        #: emits reasoning alongside the module, so completions run ~1,800
        #: tokens, not the few hundred the output alone would suggest.
        self._completion_estimate = 1800.0

    #: Share of a one-minute token budget one request may use for the document.
    #: The rest covers the system prompt (~900 tokens) and the completion
    #: (~1500), and leaves room for roughly two episodes per minute rather than
    #: one request that consumes the whole window.
    DOC_SHARE_OF_TPM = 0.35

    @property
    def max_doc_chars(self) -> int | None:
        """Largest document budget that fits this provider's per-minute limit.

        Measured on Groq's free tier: `x-ratelimit-limit-tokens: 8000` *per
        minute*. The configured profiles ask for up to 45,000 characters
        (~11,250 tokens), which exceeds the entire minute budget in a single
        request -- so the `thorough` profile could never succeed there, and
        `standard` (~4,500 tokens) capped throughput at one episode per minute.
        Both showed up as unexplained 429s and 413s rather than as a
        configuration problem.

        Returns None until a response has been seen, since the limit is
        discovered rather than assumed.
        """
        if not self.tokens_per_minute:
            return None
        return int(self.tokens_per_minute * self.DOC_SHARE_OF_TPM * 4)

    def _note_rate_limits(self, resp: requests.Response) -> None:
        raw = (resp.headers.get("x-ratelimit-limit-tokens") or "").strip()
        if not raw:
            return
        try:
            limit = int(float(raw))
        except ValueError:
            return
        if limit > 0 and limit != self.tokens_per_minute:
            self.tokens_per_minute = limit

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
            # Feed the real cost back so the next call is paced on measurement
            # rather than the seed estimate.
            self.pacer.note(in_tokens + out_tokens)
            self._completion_estimate = 0.7 * self._completion_estimate + 0.3 * out_tokens
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
        from cleanroom.observability.pricing import estimate_tokens

        # Wait our turn before spending, rather than discovering the ceiling by
        # bouncing off it. Only possible once a response has revealed the limit.
        projected = estimate_tokens(system + user) + int(self._completion_estimate)
        wait = self.pacer.delay_for(projected, self.tokens_per_minute)
        if wait > 0:
            self.pacer.total_waited += wait
            print(
                f"    [pacing {wait:.0f}s -- next call needs ~{projected:,} of "
                f"{self.tokens_per_minute:,} tokens/min]",
                flush=True,
            )
            time.sleep(wait)

        last: RateLimited | None = None
        spent = 0.0
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            try:
                return self._post(system, user)
            except RateLimited as exc:
                last = exc
                if attempt == MAX_RATE_LIMIT_RETRIES - 1:
                    break
                delay = min(exc.retry_after * (attempt + 1),
                            max(0.0, MAX_TOTAL_BACKOFF_S - spent))
                if delay <= 0:
                    break
                spent += delay
                print(
                    f"    [rate limited, waiting {delay:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})]",
                    flush=True,
                )
                time.sleep(delay)
        raise SynthesisError(
            f"{last} after {MAX_RATE_LIMIT_RETRIES} attempts and {spent:.0f}s of "
            "backoff -- raise --pause, lower the profile budget, or use a smaller model"
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

        # Learn the provider's limits from every response, including failures --
        # a 429 carries the headers too, and that is exactly when we need them.
        self._note_rate_limits(resp)
        self._raise_for_status(resp)

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

    def _raise_for_status(self, resp: requests.Response) -> None:
        """Turn an error response into the narrowest exception that fits.

        Split out from `_post` so each branch can be driven by a test with a
        captured response body, rather than only by a live provider having a
        bad day.
        """
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
            # Separate "this minute is full" from "today is gone" before
            # retrying. Retrying the second wastes the whole retry budget and
            # then logs the episode as a synthesis failure, which reads as the
            # agent writing bad code rather than the tier being spent.
            body = resp.text[:600]
            if _DAILY_LIMIT.search(body):
                raise DailyQuotaExhausted(
                    f"{self.cfg.llm_model}: the provider's DAILY token budget is "
                    f"exhausted ({_quota_detail(body)}). Waiting cannot fix this "
                    "inside a run -- switch model, use another key, or resume "
                    "after the reset.",
                    retry_after=_retry_after(resp),
                )
            raise RateLimited(
                f"{self.cfg.llm_model} rate limit (429): {_quota_detail(body)}",
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
            except CreditsExhausted:
                # Credits gone, or the daily token budget spent. A second
                # attempt cannot succeed, and spending it rewrites one clear
                # "the tier is exhausted" message into "failed twice", which
                # sends the reader looking for a bug in the code writer.
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
