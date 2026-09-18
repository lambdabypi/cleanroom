"""Environment-backed settings.

One dataclass, loaded once, so that every module agrees on where state lives and
which credentials exist. `Settings.missing_for()` powers `cleanroom doctor`,
which is the first thing to run at a hackathon -- auth problems found at hour one
are cheap, at hour six they are fatal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


#: Provider-native key names, so a `.env` written the way each provider documents
#: it just works. Without this, pasting `GROQ_API_KEY=...` leaves the code writer
#: silently unconfigured -- the key is present, just under a name nothing reads.
#: (env var, base URL, default model)
#: Model defaults are the ones actually served as of 2026-09-11 -- hosted model
#: catalogues churn, and a decommissioned default fails as an opaque 404. Run
#: `cleanroom models` to see what your key can reach right now.
KNOWN_PROVIDERS: tuple[tuple[str, str, str], ...] = (
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    ("CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "llama-3.3-70b"),
    (
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
    (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
    ),
    (
        "GOOGLE_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
    ),
    ("TOGETHER_API_KEY", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-chat"),
)


def _detect_provider() -> tuple[str, str, str]:
    """Find a configured provider from its native env var name.

    Explicit `CLEANROOM_LLM_*` always wins; this only fills the gaps.
    """
    base = _env("CLEANROOM_LLM_BASE_URL").rstrip("/")
    key = _env("CLEANROOM_LLM_API_KEY")
    model = _env("CLEANROOM_LLM_MODEL")

    if base and model:
        return base, key, model

    for var, default_base, default_model in KNOWN_PROVIDERS:
        found = _env(var)
        if found:
            return (base or default_base, key or found, model or default_model)

    return base, key, model


@dataclass(frozen=True)
class Settings:
    # You.com
    you_api_key: str = ""
    you_api_base: str = "https://ydc-index.io/v1"
    you_agents_url: str = "https://api.you.com/v1/agents/runs"

    # Which backend writes the extractor code.
    #   "auto"      -> compat if configured, else anthropic, else you
    #   "compat"    -> any OpenAI-compatible endpoint (Groq, Cerebras, Gemini's
    #                  compat layer, OpenRouter, Ollama) -- the free-tier route
    #   "anthropic" -> Claude
    #   "you"       -> You.com Agents API (burns You.com credits fast)
    synthesis_backend: str = "auto"

    # Anthropic (optional -- only used by the anthropic backend)
    anthropic_api_key: str = ""
    model: str = "claude-opus-5"
    #: Required when the key is an *all-workspaces* key: such a key refuses
    #: every request that does not name a workspace, and which workspace it
    #: bills depends entirely on this value. A key scoped to one workspace
    #: ignores it. Setting it explicitly is also the only way to be sure a run
    #: charges the workspace you meant rather than a colleague's.
    anthropic_workspace_id: str = ""

    # OpenAI-compatible backend (optional). One code path covers every provider
    # that speaks /chat/completions, so a free tier is a three-line .env change.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Daytona
    daytona_api_key: str = ""
    daytona_auto_stop_minutes: int = 15

    # One
    one_secret: str = ""
    one_connection_keys: str = ""
    one_publish_platform: str = "github"
    one_publish_target: str = ""

    # Local
    state_dir: Path = field(default=REPO_ROOT / "state")
    local_validate: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        detected_base, detected_key, detected_model = _detect_provider()
        state = _env("CLEANROOM_STATE_DIR", "state")
        state_path = Path(state)
        if not state_path.is_absolute():
            state_path = REPO_ROOT / state_path
        return cls(
            you_api_key=_env("YOU_API_KEY"),
            you_api_base=_env("YOU_API_BASE", "https://ydc-index.io/v1").rstrip("/"),
            you_agents_url=_env("YOU_AGENTS_URL", "https://api.you.com/v1/agents/runs"),
            synthesis_backend=_env("CLEANROOM_SYNTH_BACKEND", "auto").lower(),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            model=_env("CLEANROOM_MODEL", "claude-opus-5"),
            anthropic_workspace_id=_env(
                "CLEANROOM_ANTHROPIC_WORKSPACE_ID", _env("ANTHROPIC_WORKSPACE_ID")
            ),
            llm_base_url=detected_base,
            llm_api_key=detected_key,
            llm_model=detected_model,
            daytona_api_key=_env("DAYTONA_API_KEY"),
            daytona_auto_stop_minutes=_env_int("DAYTONA_AUTO_STOP_MINUTES", 15),
            one_secret=_env("ONE_SECRET"),
            one_connection_keys=_env("ONE_CONNECTION_KEYS"),
            one_publish_platform=_env("ONE_PUBLISH_PLATFORM", "github"),
            one_publish_target=_env("ONE_PUBLISH_TARGET"),
            state_dir=state_path,
            local_validate=_env_bool("CLEANROOM_LOCAL_VALIDATE"),
        )

    # -- derived paths -----------------------------------------------------

    @property
    def bandit_path(self) -> Path:
        return self.state_dir / "bandit.json"

    @property
    def episodes_path(self) -> Path:
        return self.state_dir / "episodes.jsonl"

    @property
    def budget_path(self) -> Path:
        return self.state_dir / "budget.json"

    @property
    def calls_path(self) -> Path:
        return self.state_dir / "calls.jsonl"

    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memory.jsonl"

    @property
    def dataset_path(self) -> Path:
        return self.state_dir / "dataset.csv"

    @property
    def provenance_path(self) -> Path:
        return self.state_dir / "provenance.jsonl"

    def ensure_state_dir(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir

    # -- preflight ---------------------------------------------------------

    @property
    def compat_configured(self) -> bool:
        # Ollama and other local servers need no key, so a base URL plus a model
        # is enough to count as configured.
        return bool(self.llm_base_url and self.llm_model)

    @property
    def resolved_backend(self) -> str:
        """Which synthesis backend will actually be used.

        `auto` will NEVER select the You.com Agents API. Measured on the You.com
        dashboard, `Agent API - Advanced` bills **$15 per call** -- eight calls
        consumed $120. A learning run needs 20-30 synthesis calls, so autoselecting
        it would mean $300-450 for one demo. Search, by contrast, is $5 per 1000
        calls, which is why retrieval stays on You.com and codegen does not.

        Using it requires typing `CLEANROOM_SYNTH_BACKEND=you` explicitly.
        """
        if self.synthesis_backend in ("you", "anthropic", "compat"):
            return self.synthesis_backend
        if self.compat_configured:
            return "compat"
        if self.anthropic_api_key:
            return "anthropic"
        return "unconfigured"

    def missing_for(self, *, need_sandbox: bool = True, need_one: bool = True) -> list[str]:
        """Return the names of env vars required for a full run but unset."""
        missing: list[str] = []
        if not self.you_api_key:
            # Needed for retrieval either way, and for synthesis on the "you" backend.
            missing.append("YOU_API_KEY")
        backend = self.resolved_backend
        if backend == "anthropic" and not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if backend == "compat" and not self.compat_configured:
            missing.append("CLEANROOM_LLM_BASE_URL + CLEANROOM_LLM_MODEL")
        if backend == "unconfigured":
            missing.append("CLEANROOM_LLM_BASE_URL + CLEANROOM_LLM_MODEL (a code writer)")
        if need_sandbox and not self.local_validate and not self.daytona_api_key:
            missing.append("DAYTONA_API_KEY")
        if need_one and not self.one_secret:
            missing.append("ONE_SECRET")
        return missing


settings = Settings.from_env()
