"""Import-safe helpers shared by both pytest suites.

Pytest treats ``conftest.py`` as configuration, not as a normal import module.
Helpers that tests call directly therefore live here so collection is independent
of which test root pytest discovers first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterable

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent


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
    placeholder_patterns = (
        "your-",
        "-here",
        "placeholder",
        "xxx",
        "changeme",
        "insert-",
    )
    return any(pattern in value.lower() for pattern in placeholder_patterns)


def _load_env_test() -> dict[str, str]:
    """Load values from the repository's ``.env.test`` file."""
    return _parse_dotenv(PROJECT_ROOT / ".env.test")


def _populate_os_environ_from_dotenv() -> None:
    """Populate missing process variables without overriding shell values."""
    for key, value in _load_env_test().items():
        os.environ.setdefault(key, value)


def _resolve_api_keys() -> dict[str, str]:
    """Resolve live API keys with shell environment taking priority."""
    dotenv_vars = _load_env_test()

    # Presence and truthiness have different meanings here.  The credential-free
    # targets deliberately export provider variables as empty strings; falling
    # back to .env.test in that case could silently re-enable live calls.  Only a
    # genuinely absent variable may use the developer-local dotenv fallback.
    openai_key = (
        os.environ["OPENAI_API_KEY"]
        if "OPENAI_API_KEY" in os.environ
        else dotenv_vars.get("OPENAI_API_KEY", "")
    )
    serper_key = (
        os.environ["SERPER_API_KEY"]
        if "SERPER_API_KEY" in os.environ
        else dotenv_vars.get("SERPER_API_KEY", "")
    )

    keys: dict[str, str] = {}
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


async def collect_all_from_generator(async_gen: AsyncIterable[Any]) -> list[Any]:
    """Consume an async generator and return all emitted values."""
    return [item async for item in async_gen]
