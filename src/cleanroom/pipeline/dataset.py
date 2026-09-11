"""The artifact: an accumulating, deduplicated, fully attributed CSV.

This is the thing that outlives the run and gets published. It accumulates across
episodes rather than being overwritten, so the dataset improves as the agent does.

Provenance is recorded per source rather than per project. "Clean Data" means the
output is accurate, attributable, non-PII, and gathered from the public web with
consent -- claims that are only worth making if there is a record backing them, so
every contributing page is logged with when it was retrieved and how.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from cleanroom.config import Settings, settings
from cleanroom.learning.store import append_jsonl, read_jsonl, utcnow
from cleanroom.pipeline.schema import field_names, url_field


def _row_key(row: dict, schema: dict) -> tuple:
    required = [f["name"] for f in (schema.get("fields") or []) if f.get("required")]
    keys = required or field_names(schema)
    return tuple(str(row.get(k)).strip().lower() for k in keys)


@dataclass
class MergeStats:
    added: int
    duplicates: int
    total: int


class Dataset:
    """CSV-backed row store with schema-stable column order."""

    def __init__(self, schema: dict, cfg: Settings | None = None, path: Path | None = None) -> None:
        self.cfg = cfg or settings
        self.schema = schema
        self.columns = field_names(schema)
        self.path = path or self.cfg.dataset_path
        self.rows: list[dict] = []
        self._keys: set[tuple] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row = {c: (raw.get(c) or None) for c in self.columns}
                key = _row_key(row, self.schema)
                if key in self._keys:
                    continue
                self._keys.add(key)
                self.rows.append(row)

    def merge(self, new_rows: Iterable[dict]) -> MergeStats:
        added = duplicates = 0
        for row in new_rows:
            trimmed = {c: row.get(c) for c in self.columns}
            key = _row_key(trimmed, self.schema)
            if key in self._keys:
                duplicates += 1
                continue
            self._keys.add(key)
            self.rows.append(trimmed)
            added += 1
        return MergeStats(added=added, duplicates=duplicates, total=len(self.rows))

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        return buffer.getvalue()

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.to_csv(), encoding="utf-8")
        return self.path

    @property
    def source_count(self) -> int:
        field = url_field(self.schema)
        if not field:
            return 0
        return len({row.get(field) for row in self.rows if row.get(field)})


def record_provenance(
    *,
    source: dict[str, Any],
    episode: int,
    strategy: str,
    bucket: str,
    reward: float,
    rows_contributed: int,
    cfg: Settings | None = None,
) -> None:
    cfg = cfg or settings
    append_jsonl(
        cfg.provenance_path,
        {
            **source,
            "retrieved_at": utcnow(),
            "episode": episode,
            "strategy": strategy,
            "bucket": bucket,
            "reward": round(reward, 4),
            "rows_contributed": rows_contributed,
            "pii_screened": True,
            "license_basis": "public web page, retrieved via You.com Search API",
        },
    )


def clean_data_manifest(schema: dict, cfg: Settings | None = None) -> dict[str, Any]:
    """A summary of where the dataset came from, suitable for publishing alongside it."""
    cfg = cfg or settings
    entries = list(read_jsonl(cfg.provenance_path))
    contributing = [e for e in entries if int(e.get("rows_contributed") or 0) > 0]
    return {
        "dataset": schema.get("name"),
        "generated_at": utcnow(),
        "pages_attempted": len(entries),
        "pages_contributing": len(contributing),
        "distinct_sources": len({e.get("url") for e in contributing if e.get("url")}),
        "rows_published": sum(int(e.get("rows_contributed") or 0) for e in contributing),
        "retrieval_method": "You.com Search + Contents APIs (public web)",
        "pii_policy": (
            "Rows containing email addresses, phone numbers, or government "
            "identifiers are rejected by the validator before scoring."
        ),
        "attribution_policy": "Every row carries the URL of the page it was read from.",
        "sources": [
            {
                "url": e.get("url"),
                "title": e.get("title"),
                "retrieved_at": e.get("retrieved_at"),
                "rows_contributed": e.get("rows_contributed"),
            }
            for e in contributing
        ],
    }


def manifest_markdown(manifest: dict) -> str:
    """Render the manifest for committing next to the CSV."""
    lines = [
        f"# Provenance -- {manifest.get('dataset')}",
        "",
        f"Generated: {manifest.get('generated_at')}",
        f"Rows published: {manifest.get('rows_published')}",
        f"Distinct sources: {manifest.get('distinct_sources')} "
        f"(from {manifest.get('pages_attempted')} pages attempted)",
        f"Retrieval: {manifest.get('retrieval_method')}",
        "",
        "## Clean Data policy",
        "",
        f"- {manifest.get('pii_policy')}",
        f"- {manifest.get('attribution_policy')}",
        "",
        "## Sources",
        "",
        "| Rows | Retrieved | Source |",
        "| ---: | --------- | ------ |",
    ]
    for source in manifest.get("sources") or []:
        title = (source.get("title") or source.get("url") or "").replace("|", "\\|")[:70]
        lines.append(
            f"| {source.get('rows_contributed')} | {str(source.get('retrieved_at'))[:19]} "
            f"| [{title}]({source.get('url')}) |"
        )
    return "\n".join(lines) + "\n"
