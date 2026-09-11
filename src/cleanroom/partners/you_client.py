"""You.com client -- the agent's live-web observation layer.

Two endpoints, both POST with an `X-API-Key` header:

* `POST {base}/search`   -- ranked web/news results. With
  `extraction.extraction_mode = "full_page"` it returns page markdown *inline*,
  so discovery and fetching collapse into a single billed call.
* `POST {base}/contents` -- markdown/HTML/metadata for specific URLs. Used as the
  fallback when a search hit comes back without usable content.

The published docs give the base as `https://ydc-index.io/v1` while the hackathon
PDF's working curl uses `https://api.ydc-index.io/v1`. Rather than bet on one, the
client probes and remembers whichever answers -- a five-line safeguard against
losing an hour to a DNS error on the one API the submission requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import requests

from cleanroom.config import Settings, settings

CANDIDATE_BASES = ("https://ydc-index.io/v1", "https://api.ydc-index.io/v1")
DEFAULT_TIMEOUT = 45


class YouError(RuntimeError):
    pass


@dataclass
class Source:
    """One candidate page: where it came from and what it said."""

    url: str
    title: str = ""
    description: str = ""
    markdown: str = ""
    page_age: str = ""
    origin: str = "web"

    @property
    def has_content(self) -> bool:
        return len(self.markdown.strip()) > 200

    def as_provenance(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "page_age": self.page_age,
            "origin": self.origin,
            "retrieved_via": "you.com",
        }


class YouClient:
    def __init__(self, cfg: Settings | None = None, session: requests.Session | None = None) -> None:
        self.cfg = cfg or settings
        if not self.cfg.you_api_key:
            raise YouError("YOU_API_KEY is not set; run `cleanroom doctor`")
        self.session = session or requests.Session()
        self.session.headers.update(
            {"X-API-Key": self.cfg.you_api_key, "Content-Type": "application/json"}
        )
        self._base: str | None = None

    # -- transport ---------------------------------------------------------

    def _bases(self) -> list[str]:
        if self._base:
            return [self._base]
        configured = self.cfg.you_api_base.rstrip("/")
        ordered = [configured] + [b for b in CANDIDATE_BASES if b != configured]
        return ordered

    def _post(self, path: str, body: dict) -> Any:
        from cleanroom.observability.ledger import get_ledger
        from cleanroom.observability.pricing import call_cost

        operation = f"you.{path.strip('/')}"
        ledger = get_ledger(self.cfg)
        with ledger.track(operation, component="you") as span:
            span.charge(call_cost(operation))
            result = self._post_inner(path, body)
            span.note(base=self._base, query=str(body.get("query") or "")[:80])
            return result

    def _post_inner(self, path: str, body: dict) -> Any:
        last_error: Exception | None = None
        for base in self._bases():
            url = f"{base}/{path.lstrip('/')}"
            try:
                resp = self.session.post(url, json=body, timeout=DEFAULT_TIMEOUT)
            except requests.RequestException as exc:
                last_error = exc
                continue

            # While probing, a wrong base is indistinguishable from a bad key by
            # status alone: api.ydc-index.io answers 403 for a key that works
            # fine on ydc-index.io. So during probing these are "try the next
            # candidate", and only become fatal once every base has been tried.
            if resp.status_code in (401, 403, 404) and self._base is None:
                last_error = YouError(f"{resp.status_code} from {url}")
                continue
            if resp.status_code in (401, 403):
                raise YouError(f"You.com rejected the API key ({resp.status_code}). Check YOU_API_KEY.")
            if resp.status_code == 429:
                raise YouError("You.com rate limit hit (429). Slow the episode loop down.")
            if not resp.ok:
                raise YouError(f"You.com {resp.status_code} from {url}: {resp.text[:300]}")

            self._base = base
            try:
                return resp.json()
            except ValueError as exc:
                raise YouError(f"You.com returned non-JSON from {url}") from exc

        raise YouError(
            f"no You.com base URL accepted the request (tried {self._bases()}). "
            f"Last error: {last_error}. If this is 401/403 everywhere, YOU_API_KEY is wrong."
        )

    @property
    def resolved_base(self) -> str | None:
        return self._base

    # -- endpoints ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        full_page: bool = True,
    ) -> dict:
        body: dict[str, Any] = {"query": query, "count": count}
        if freshness:
            body["freshness"] = freshness
        if include_domains:
            body["include_domains"] = list(include_domains)
        if exclude_domains:
            body["exclude_domains"] = list(exclude_domains)
        if full_page:
            # Pull the page body down with the ranking in one round trip.
            body["extraction"] = {"extraction_mode": "full_page"}
            body["crawl_timeout"] = 20
        return self._post("search", body)

    def contents(
        self,
        urls: Sequence[str],
        *,
        formats: Sequence[str] = ("markdown", "metadata"),
        crawl_timeout: int = 20,
    ) -> list[dict]:
        if not urls:
            return []
        payload = self._post(
            "contents",
            {
                "urls": list(urls),
                "formats": list(formats),
                "crawl_timeout": crawl_timeout,
            },
        )
        if isinstance(payload, list):
            return payload
        return payload.get("results") or payload.get("contents") or []

    # -- the call the pipeline actually makes -------------------------------

    def find_sources(
        self,
        topic: str,
        *,
        count: int = 8,
        include_domains: Sequence[str] | None = None,
        freshness: str | None = None,
        backfill: bool = True,
    ) -> list[Source]:
        """Search for pages about `topic` and return them with content attached."""
        payload = self.search(
            topic,
            count=count,
            include_domains=include_domains,
            freshness=freshness,
            full_page=True,
        )
        results = (payload.get("results") or {}) if isinstance(payload, dict) else {}

        sources: list[Source] = []
        for origin in ("web", "news"):
            for hit in results.get(origin) or []:
                if not isinstance(hit, dict) or not hit.get("url"):
                    continue
                contents = hit.get("contents") or {}
                sources.append(
                    Source(
                        url=hit["url"],
                        title=hit.get("title") or "",
                        description=hit.get("description") or "",
                        markdown=(contents.get("markdown") or "").strip(),
                        page_age=hit.get("page_age") or "",
                        origin=origin,
                    )
                )

        # Some hits come back without a body (paywalls, slow crawls). One extra
        # Contents call fills them in rather than wasting the episode.
        if backfill:
            thin = [s for s in sources if not s.has_content]
            if thin:
                by_url = {s.url: s for s in thin}
                try:
                    fetched = self.contents(list(by_url)[:10])
                except YouError:
                    fetched = []
                for page in fetched:
                    url = page.get("url")
                    target = by_url.get(url)
                    if target and page.get("markdown"):
                        target.markdown = page["markdown"].strip()
                        if not target.title:
                            target.title = page.get("title") or ""

        return [s for s in sources if s.has_content]
