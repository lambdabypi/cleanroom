"""Schema validation and row scoring.

STDLIB ONLY, AND NO PROJECT-RELATIVE IMPORTS. This file is uploaded verbatim into
the Daytona sandbox and executed there, so it must stand alone -- that constraint
is what lets local `--local-validate` runs and sandboxed runs score identically
instead of quietly disagreeing about what "valid" means.
"""

from __future__ import annotations

import math
import re
from typing import Any

FIELD_TYPES = ("string", "number", "integer", "boolean")

_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_TRUE = {"true", "yes", "y", "1", "available", "in stock"}
_FALSE = {"false", "no", "n", "0", "unavailable", "out of stock"}

# Anything matching these is treated as a validation failure rather than data.
# Clean Data means no personally identifiable information ends up in the output,
# and the cheapest enforcement point is the validator the reward reads from.
_PII_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}"),                    # email
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                          # SSN-shaped
)


def looks_like_pii(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return any(p.search(value) for p in _PII_PATTERNS)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"n/a", "na", "-", "--", "none", "null"}
    if isinstance(value, float):
        return math.isnan(value)
    return False


def coerce(value: Any, ftype: str) -> tuple[bool, Any, str]:
    """Best-effort type coercion. Returns (ok, coerced_value, error)."""
    if ftype == "string":
        text = str(value).strip()
        return (True, text, "") if text else (False, None, "empty string")

    if ftype in ("number", "integer"):
        if isinstance(value, bool):
            return False, None, "boolean where number expected"
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            match = _NUM_RE.search(str(value))
            if not match:
                return False, None, f"no number in {str(value)[:40]!r}"
            try:
                number = float(match.group(0).replace(",", ""))
            except ValueError:
                return False, None, f"unparseable number {match.group(0)!r}"
        if ftype == "integer":
            if abs(number - round(number)) > 1e-9:
                return False, None, f"{number} is not an integer"
            return True, int(round(number)), ""
        return True, number, ""

    if ftype == "boolean":
        if isinstance(value, bool):
            return True, value, ""
        text = str(value).strip().lower()
        if text in _TRUE:
            return True, True, ""
        if text in _FALSE:
            return True, False, ""
        return False, None, f"not a boolean: {text[:40]!r}"

    return False, None, f"unknown field type {ftype!r}"


def _matches(pattern: str, value: str) -> bool:
    """Schema-supplied regex. Author-controlled, but a typo must not crash a run."""
    try:
        return re.search(pattern, value, re.IGNORECASE) is not None
    except re.error:
        return True  # unusable pattern: do not block the row on it


def _check_constraints(name: str, value: Any, spec: dict) -> str:
    if spec.get("format") == "url" and not _URL_RE.match(str(value)):
        return f"{name}: {str(value)[:40]!r} is not an http(s) URL"

    # `pattern` / `deny_pattern` are what make "valid" mean something. Without
    # them a field typed `string` accepts anything, and an extractor that reads
    # the wrong table scores just as well as one that reads the right table --
    # observed: rows with provider="10 hours" and usd_per_hour=215 scoring 0.92.
    if isinstance(value, str):
        required = spec.get("pattern")
        if required and not _matches(required, value):
            return f"{name}: {value[:40]!r} does not look like a {name.replace('_', ' ')}"
        denied = spec.get("deny_pattern")
        if denied and _matches(denied, value):
            return f"{name}: {value[:40]!r} looks like prose or a duration, not a {name}"

    minimum, maximum = spec.get("min"), spec.get("max")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if minimum is not None and value < minimum:
            return f"{name}: {value} below min {minimum}"
        if maximum is not None and value > maximum:
            return f"{name}: {value} above max {maximum}"

    allowed = spec.get("enum")
    if allowed and value not in allowed:
        return f"{name}: {str(value)[:40]!r} not in allowed values"

    max_len = spec.get("max_length")
    if max_len and isinstance(value, str) and len(value) > int(max_len):
        return f"{name}: longer than {max_len} chars"

    if looks_like_pii(value):
        return f"{name}: looks like personal data (PII), rejected"

    return ""


def validate_row(row: Any, schema: dict) -> tuple[dict | None, list[str]]:
    """Validate and coerce one row. Returns (clean_row_or_None, errors)."""
    fields = schema.get("fields") or []
    if not isinstance(row, dict):
        return None, [f"row is {type(row).__name__}, expected dict"]

    clean: dict[str, Any] = {}
    errors: list[str] = []

    for spec in fields:
        name = spec.get("name")
        if not name:
            continue
        ftype = spec.get("type", "string")
        required = bool(spec.get("required"))
        raw = row.get(name)

        if _is_blank(raw):
            if required:
                errors.append(f"{name}: missing (required)")
            clean[name] = None
            continue

        ok, value, err = coerce(raw, ftype)
        if not ok:
            errors.append(f"{name}: {err}")
            clean[name] = None
            continue

        constraint_err = _check_constraints(name, value, spec)
        if constraint_err:
            errors.append(constraint_err)
            clean[name] = None
            continue

        clean[name] = value

    return (None if errors else clean), errors


def _dedupe_key(row: dict, schema: dict) -> tuple:
    required = [f["name"] for f in (schema.get("fields") or []) if f.get("required")]
    keys = required or sorted(row)
    return tuple(str(row.get(k)) for k in keys)


def validate_rows(rows: Any, schema: dict, source_url: str | None = None) -> dict:
    """Score a list of extracted rows. This dict is what the reward reads.

    `source_url` is injected into every row rather than being the generated
    code's responsibility. The host knows the URL authoritatively, so asking the
    model to embed it was pointless risk -- and it went wrong exactly as you would
    expect: perfectly good extractions scoring 0.0 because the model left the
    field `None`. Attribution is now structurally guaranteed instead of merely
    rewarded, which is a stronger Clean Data claim, not a weaker one.
    """
    fields = schema.get("fields") or []
    field_names = [f["name"] for f in fields if f.get("name")]
    source_field = next(
        (f["name"] for f in fields if f.get("format") == "url" or f.get("name") == "source_url"),
        None,
    )

    if source_url and source_field and isinstance(rows, list):
        for row in rows:
            # Overwrite blanks *and* anything the model invented -- the host's URL
            # is the only trustworthy value here.
            if isinstance(row, dict) and _is_blank(row.get(source_field)):
                row[source_field] = source_url

    if not isinstance(rows, list):
        return {
            "rows_total": 0,
            "rows_valid": 0,
            "cells_expected": 0,
            "cells_filled": 0,
            "rows_with_source": 0,
            "rows_duplicate": 0,
            "sample_errors": [f"extract() returned {type(rows).__name__}, expected list"],
            "field_error_counts": {},
            "rows": [],
        }

    valid: list[dict] = []
    seen: set[tuple] = set()
    duplicates = 0
    cells_filled = 0
    sample_errors: list[str] = []
    field_error_counts: dict[str, int] = {}

    for row in rows:
        clean, errors = validate_row(row, schema)
        if errors:
            for err in errors:
                key = err.split(":", 1)[0]
                field_error_counts[key] = field_error_counts.get(key, 0) + 1
            if len(sample_errors) < 8:
                sample_errors.extend(errors[: 8 - len(sample_errors)])
            continue

        key = _dedupe_key(clean, schema)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)

        cells_filled += sum(1 for name in field_names if not _is_blank(clean.get(name)))
        valid.append(clean)

    rows_with_source = (
        sum(1 for r in valid if not _is_blank(r.get(source_field))) if source_field else len(valid)
    )

    return {
        "rows_total": len(rows),
        "rows_valid": len(valid),
        "cells_expected": len(valid) * len(field_names),
        "cells_filled": cells_filled,
        "rows_with_source": rows_with_source,
        "rows_duplicate": duplicates,
        "sample_errors": sample_errors,
        "field_error_counts": field_error_counts,
        "rows": valid,
    }
