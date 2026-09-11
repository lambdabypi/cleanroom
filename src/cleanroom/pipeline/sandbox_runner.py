"""Entry point executed *inside* the Daytona sandbox.

STDLIB ONLY. Uploaded next to `validate.py`, `extractor.py`, and `payload.json`,
then run with `python3 sandbox_runner.py`. Everything it learns about the attempt
is printed to stdout as a single JSON object between sentinels, so the host can
recover the report even when the generated extractor prints debug noise of its own.

Untrusted model-written code runs here and nowhere else. That is the whole reason
the sandbox exists: `extract()` can loop forever, exhaust memory, or try to read
the filesystem, and the blast radius stays inside an ephemeral container.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import traceback

BEGIN = "<<<CLEANROOM_REPORT_BEGIN>>>"
END = "<<<CLEANROOM_REPORT_END>>>"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXTRACT_TIMEOUT_S = int(os.environ.get("CLEANROOM_EXTRACT_TIMEOUT", "25"))
MAX_ROWS = 5000


class _Timeout(Exception):
    pass


def _emit(report: dict) -> None:
    print(BEGIN)
    print(json.dumps(report, default=str))
    print(END)


def main() -> int:
    try:
        with open(os.path.join(HERE, "payload.json"), "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        document = payload["document"]
        schema = payload["schema"]
        source_url = payload.get("source_url") or None
    except Exception as exc:  # noqa: BLE001 - report, never raise, to the host
        _emit({"ok": False, "stage": "payload", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    try:
        import validate  # uploaded alongside this file
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "stage": "import_validate", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    # Import the generated extractor. A syntax error here is the single most
    # common failure mode, and it must come back as reward 0 rather than a crash.
    try:
        import extractor
    except Exception as exc:  # noqa: BLE001
        _emit({
            "ok": False,
            "stage": "import_extractor",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        })
        return 1

    if not hasattr(extractor, "extract"):
        _emit({"ok": False, "stage": "contract", "error": "extractor.py defines no extract()"})
        return 1

    # SIGALRM is POSIX-only, which is fine -- this file only ever runs in the
    # Linux sandbox. An extractor that hangs would otherwise pin the sandbox open
    # until the TTL expires and stall the whole learning run.
    if hasattr(signal, "SIGALRM"):
        def _on_alarm(signum, frame):  # noqa: ANN001, ARG001
            raise _Timeout(f"extract() exceeded {EXTRACT_TIMEOUT_S}s")

        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(EXTRACT_TIMEOUT_S)

    try:
        rows = extractor.extract(document)
    except _Timeout as exc:
        _emit({"ok": False, "stage": "extract", "error": str(exc), "timeout": True})
        return 1
    except Exception as exc:  # noqa: BLE001
        _emit({
            "ok": False,
            "stage": "extract",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        })
        return 1
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    # Guard against a runaway generator flooding the report.
    if isinstance(rows, list) and len(rows) > MAX_ROWS:
        rows = rows[:MAX_ROWS]

    try:
        report = validate.validate_rows(rows, schema, source_url)
    except Exception as exc:  # noqa: BLE001
        _emit({"ok": False, "stage": "validate", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    report["ok"] = True
    report["stage"] = "done"
    _emit(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
