"""Daytona sandbox -- the environment, in the reinforcement-learning sense.

This is where the reward comes from. The agent writes extraction code, the code
runs here, and the validation report that comes back is ground truth: automatic,
verifiable, and generatable hundreds of times in an afternoon. Human thumbs alone
could never produce enough episodes to move a posterior.

Two decisions worth knowing about:

* **One sandbox, many episodes.** Creation dominates episode latency, so the
  sandbox is created lazily and reused, with only `extractor.py` and
  `payload.json` re-uploaded per attempt. Learning curves need tens of episodes;
  at ~20s of setup each, per-episode sandboxes would cap the demo at a handful.
* **Nothing calls out from inside.** Daytona blocks egress to `*.withone.ai`
  (SNI inspection), so One and You.com are only ever called from the host. The
  sandbox receives a document and a schema and returns a report -- no network.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cleanroom.config import Settings, settings
from cleanroom.pipeline import validate as validate_mod
from cleanroom.pipeline.sandbox_runner import BEGIN, END

SANDBOX_DIR = "/home/daytona/cleanroom"
_RUNNER_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "sandbox_runner.py"
_VALIDATE_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "validate.py"


class SandboxError(RuntimeError):
    pass


@dataclass
class RunResult:
    """Outcome of one extractor execution."""

    report: dict | None
    exit_code: int
    stdout: str
    stderr: str
    crashed: bool

    @property
    def stage(self) -> str:
        return (self.report or {}).get("stage", "unknown")


def _parse_report(stdout: str) -> dict | None:
    """Pull the JSON report out from between the sentinels.

    Generated extractors print their own debug output, so the report cannot just
    be "the last line of stdout".
    """
    if BEGIN not in stdout or END not in stdout:
        return None
    chunk = stdout.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    try:
        parsed = json.loads(chunk)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class LocalEnvironment:
    """In-process fallback for `CLEANROOM_LOCAL_VALIDATE=1`.

    Fast enough to iterate on prompts against, but it executes model-written code
    in *this* process with no isolation. Use it while developing; run the demo
    against Daytona.
    """

    backend = "local"

    def ensure_ready(self) -> None:  # noqa: D102
        return None

    def run(self, extractor_code: str, document: str, schema: dict,
            source_url: str | None = None) -> RunResult:
        namespace: dict[str, Any] = {}
        try:
            exec(compile(extractor_code, "extractor.py", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            return RunResult(None, 1, "", f"{type(exc).__name__}: {exc}", True)

        fn = namespace.get("extract")
        if not callable(fn):
            return RunResult(None, 1, "", "extractor defines no extract()", True)

        try:
            rows = fn(document)
        except Exception as exc:  # noqa: BLE001
            return RunResult(None, 1, "", f"{type(exc).__name__}: {exc}", True)

        report = validate_mod.validate_rows(rows, schema, source_url)
        report["ok"] = True
        report["stage"] = "done"
        return RunResult(report, 0, "", "", False)

    def close(self) -> None:  # noqa: D102
        return None


class DaytonaEnvironment:
    """Real sandboxed execution."""

    backend = "daytona"

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        if not self.cfg.daytona_api_key:
            raise SandboxError("DAYTONA_API_KEY is not set; run `cleanroom doctor`")
        self._client = None
        self._sandbox = None
        self.sandbox_id: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def ensure_ready(self) -> None:
        if self._sandbox is not None:
            return
        try:
            from daytona import Daytona, DaytonaConfig
        except ImportError as exc:  # pragma: no cover
            raise SandboxError("daytona SDK missing; pip install 'cleanroom[sandbox]'") from exc

        self._client = Daytona(DaytonaConfig(api_key=self.cfg.daytona_api_key))
        self._sandbox = self._create_sandbox()
        self.sandbox_id = getattr(self._sandbox, "id", None)
        self._install_harness()

    def _create_sandbox(self):
        from daytona import CreateSandboxFromImageParams, Image, Resources

        # auto_stop_interval is the guard against orphaned sandboxes eating the
        # coupon. The kwarg name has moved between SDK versions, so fall back
        # rather than fail the run over it.
        params_kwargs: dict[str, Any] = {
            "image": Image.debian_slim("3.12"),
            "resources": Resources(cpu=1, memory=2, disk=4),
        }
        try:
            return self._client.create(
                CreateSandboxFromImageParams(
                    auto_stop_interval=self.cfg.daytona_auto_stop_minutes,
                    **params_kwargs,
                )
            )
        except TypeError:
            return self._client.create(CreateSandboxFromImageParams(**params_kwargs))

    def _install_harness(self) -> None:
        """Upload the runner and validator once per sandbox."""
        self._sandbox.process.exec(f"mkdir -p {SANDBOX_DIR}")
        for path in (_RUNNER_PATH, _VALIDATE_PATH):
            self._upload(path.read_bytes(), f"{SANDBOX_DIR}/{path.name}")

    def _upload(self, content: bytes, remote_path: str) -> None:
        try:
            self._sandbox.fs.upload_file(content, remote_path)
        except TypeError:
            # Older signature ordering.
            self._sandbox.fs.upload_file(remote_path, content)

    # -- the reward call ---------------------------------------------------

    def run(self, extractor_code: str, document: str, schema: dict,
            source_url: str | None = None) -> RunResult:
        from cleanroom.observability.ledger import get_ledger
        from cleanroom.observability.pricing import sandbox_cost

        ledger = get_ledger(self.cfg)
        with ledger.track("daytona.run", component="daytona") as span:
            started = time.monotonic()
            result = self._run_inner(extractor_code, document, schema, source_url)
            elapsed = time.monotonic() - started
            span.charge(sandbox_cost(elapsed))
            span.note(
                stage=result.stage,
                rows_valid=(result.report or {}).get("rows_valid"),
                code_chars=len(extractor_code),
            )
            # A crashed extractor is a legitimate outcome of the loop, not an
            # infrastructure failure -- marking it failed would open the circuit
            # breaker on Daytona for doing its job correctly.
            return result

    def _run_inner(self, extractor_code: str, document: str, schema: dict,
                   source_url: str | None = None) -> RunResult:
        self.ensure_ready()

        payload = json.dumps(
            {"document": document, "schema": schema, "source_url": source_url}
        )
        self._upload(extractor_code.encode("utf-8"), f"{SANDBOX_DIR}/extractor.py")
        self._upload(payload.encode("utf-8"), f"{SANDBOX_DIR}/payload.json")

        # Clear any stale bytecode so a previous episode's extractor cannot be
        # imported instead of this one.
        self._sandbox.process.exec(f"rm -rf {SANDBOX_DIR}/__pycache__")

        response = self._sandbox.process.exec(
            f"cd {SANDBOX_DIR} && python3 sandbox_runner.py"
        )
        stdout = getattr(response, "result", "") or ""
        exit_code = int(getattr(response, "exit_code", 0) or 0)

        report = _parse_report(stdout)
        if report is None:
            return RunResult(None, exit_code, stdout, stdout[-1500:], True)

        crashed = not report.get("ok", False)
        stderr = ""
        if crashed:
            stderr = "\n".join(
                str(report.get(k, "")) for k in ("error", "traceback") if report.get(k)
            )
        return RunResult(report, exit_code, stdout, stderr, crashed)

    def close(self) -> None:
        if self._sandbox is None:
            return
        for attempt in (
            lambda: self._sandbox.delete(),
            lambda: self._client.delete(self._sandbox),
        ):
            try:
                attempt()
                break
            except Exception:  # noqa: BLE001,S110 - best-effort teardown
                continue
        self._sandbox = None
        self.sandbox_id = None

    def __enter__(self) -> "DaytonaEnvironment":
        self.ensure_ready()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_environment(cfg: Settings | None = None):
    """Pick the execution backend from config."""
    cfg = cfg or settings
    if cfg.local_validate:
        return LocalEnvironment()
    return DaytonaEnvironment(cfg)
