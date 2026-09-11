"""Loading and sanity-checking target schemas.

The schema is the contract three separate things agree on: the prompt Claude is
given, the validator that scores the rows, and the CSV that gets published. A
typo here silently tanks every reward, so it is checked once up front rather than
discovered as a flat learning curve.
"""

from __future__ import annotations

import json
from pathlib import Path

from cleanroom.config import REPO_ROOT
from cleanroom.pipeline.validate import FIELD_TYPES

SCHEMA_DIR = REPO_ROOT / "schemas"
DEFAULT_SCHEMA = SCHEMA_DIR / "gpu_cloud_pricing.json"


class SchemaError(ValueError):
    pass


def available_schemas() -> list[Path]:
    if not SCHEMA_DIR.exists():
        return []
    return sorted(SCHEMA_DIR.glob("*.json"))


def check_schema(schema: dict) -> list[str]:
    """Return a list of problems; empty means the schema is usable."""
    problems: list[str] = []

    if not schema.get("name"):
        problems.append("schema has no 'name'")

    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        return problems + ["schema has no 'fields' list"]

    seen: set[str] = set()
    has_required = False
    has_url = False

    for index, spec in enumerate(fields):
        where = f"fields[{index}]"
        if not isinstance(spec, dict):
            problems.append(f"{where} is not an object")
            continue

        name = spec.get("name")
        if not name:
            problems.append(f"{where} has no 'name'")
            continue
        if name in seen:
            problems.append(f"{where}: duplicate field name {name!r}")
        seen.add(name)

        ftype = spec.get("type", "string")
        if ftype not in FIELD_TYPES:
            problems.append(f"{where} ({name}): type {ftype!r} not in {FIELD_TYPES}")

        if spec.get("required"):
            has_required = True
        if spec.get("format") == "url" or name == "source_url":
            has_url = True

        for bound in ("min", "max"):
            if bound in spec and not isinstance(spec[bound], (int, float)):
                problems.append(f"{where} ({name}): '{bound}' must be numeric")

    if not has_required:
        problems.append(
            "no field is marked required, so every row validates trivially and the "
            "reward signal carries no information"
        )
    if not has_url:
        problems.append(
            "no field has format 'url' (or the name 'source_url'), so the provenance "
            "reward channel cannot be scored -- add one to satisfy Clean Data attribution"
        )

    return problems


def load_schema(path: str | Path | None = None, *, strict: bool = True) -> dict:
    target = Path(path) if path else DEFAULT_SCHEMA
    if not target.is_absolute():
        # Accept both 'schemas/foo.json' and a bare 'foo' / 'foo.json'.
        for candidate in (REPO_ROOT / target, SCHEMA_DIR / target.name, SCHEMA_DIR / f"{target.name}.json"):
            if candidate.exists():
                target = candidate
                break

    if not target.exists():
        known = ", ".join(p.stem for p in available_schemas()) or "none found"
        raise SchemaError(f"schema not found: {target} (available: {known})")

    try:
        schema = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{target.name} is not valid JSON: {exc}") from exc

    problems = check_schema(schema)
    if problems and strict:
        joined = "\n  - ".join(problems)
        raise SchemaError(f"{target.name} is not usable:\n  - {joined}")

    schema["_path"] = str(target)
    return schema


def field_names(schema: dict) -> list[str]:
    return [f["name"] for f in (schema.get("fields") or []) if f.get("name")]


def url_field(schema: dict) -> str | None:
    for spec in schema.get("fields") or []:
        if spec.get("format") == "url" or spec.get("name") == "source_url":
            return spec.get("name")
    return None
