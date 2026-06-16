"""Runtime configuration helpers for GeneAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OPENAI_ENV_PATH = PROJECT_ROOT / "OPENAI.env"


def load_env_file(path: Path = OPENAI_ENV_PATH, *, override: bool = True) -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


@dataclass(frozen=True)
class OpenAISettings:
    provider: str
    api_key: str
    endpoint: str | None
    api_version: str | None
    model: str
    detection_model: str
    reasoning_effort: str | None
    store_responses: bool


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_openai_settings() -> OpenAISettings:
    """Return OpenAI/Azure settings, loading OPENAI.env first."""

    load_env_file()

    provider = os.getenv("OPENAI_PROVIDER", "azure").lower()
    model = (
        os.getenv("GENEAGENT_MODEL")
        or os.getenv("AZURE_OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5.1"
    )
    detection_model = os.getenv("GENEAGENT_DETECT_MODEL", model)

    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        endpoint = os.getenv("OPENAI_BASE_URL")
        api_version = None

    return OpenAISettings(
        provider=provider,
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        model=model,
        detection_model=detection_model,
        reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "low") or None,
        store_responses=_env_bool("OPENAI_STORE_RESPONSES", False),
    )


def require_openai_settings() -> OpenAISettings:
    """Validate settings only when an API call is about to be made."""

    settings = get_openai_settings()
    if not settings.api_key or settings.api_key.startswith("replace-"):
        raise RuntimeError(
            "OpenAI credentials are not configured. Add the rotated key to OPENAI.env."
        )
    if settings.provider == "azure" and (
        not settings.endpoint or "your-rotated-resource" in settings.endpoint
    ):
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not configured. Add the rotated endpoint to OPENAI.env."
        )
    if settings.model.startswith("your-"):
        raise RuntimeError(
            "GENEAGENT_MODEL is not configured. Add your GPT-5 deployment name to OPENAI.env."
        )
    return settings
