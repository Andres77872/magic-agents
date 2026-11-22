"""Credential helpers for the opt-in live-provider tests.

Credential-free runs blank the provider variables explicitly (see CI and
AGENTS.md). Presence and truthiness mean different things here: an *empty*
shell variable must win over ``.env.test``, otherwise a developer-local file
could silently re-enable live calls. Only a genuinely absent variable may fall
back to the dotenv file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_TEST_PATH = PROJECT_ROOT / ".env.test"

_PLACEHOLDER_PATTERNS = ("your-", "-here", "placeholder", "xxx", "changeme", "insert-")


def _parse_dotenv(path: str | Path) -> dict[str, str]:
    """Parse the simple ``KEY=VALUE`` syntax used by ``.env.test``."""
    result: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists():
        return result

    with env_path.open(encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key:
                result[key] = value
    return result


def _is_placeholder_value(value: str) -> bool:
    """Return whether a configured credential is clearly a placeholder."""
    if not value or len(value) < 10:
        return True
    return any(pattern in value.lower() for pattern in _PLACEHOLDER_PATTERNS)


def _load_env_test() -> dict[str, str]:
    """Load values from the repository's ``.env.test`` file."""
    return _parse_dotenv(ENV_TEST_PATH)


def _env_or_dotenv(name: str, dotenv_vars: dict[str, str]) -> str:
    return os.environ[name] if name in os.environ else dotenv_vars.get(name, "")


def _resolve_api_keys() -> dict[str, str]:
    """Resolve live API keys with the shell environment taking priority."""
    dotenv_vars = _load_env_test()
    keys: dict[str, str] = {}
    openai_key = _env_or_dotenv("OPENAI_API_KEY", dotenv_vars)
    serper_key = _env_or_dotenv("SERPER_API_KEY", dotenv_vars)
    if openai_key:
        keys["openai_key"] = openai_key
    if serper_key:
        keys["serper_key"] = serper_key
    return keys


def skip_if_no_api_keys(api_keys: dict[str, str] | None = None) -> dict[str, str]:
    """Skip a live-provider test unless a non-placeholder OpenAI key exists."""
    resolved = _resolve_api_keys() if api_keys is None else api_keys
    openai_key = resolved.get("openai_key", "")
    if not openai_key or _is_placeholder_value(openai_key):
        pytest.skip("No real OPENAI_API_KEY configured (env var or .env.test)")
    return resolved
