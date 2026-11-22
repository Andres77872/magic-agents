"""Import-safe helpers shared by the test suite.

Pytest treats ``conftest.py`` as configuration rather than an importable
module, so helpers that tests import directly live in this package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterable

from tests.support.env import (
    _is_placeholder_value,
    _load_env_test,
    _parse_dotenv,
    _resolve_api_keys,
    skip_if_no_api_keys,
)
from tests.support.hooks import AsyncRecordingHook, RecordingHook

FIXTURE_GRAPHS = Path(__file__).resolve().parents[1] / "fixtures" / "graphs"


def fixture_graph_path(name: str) -> Path:
    """Return the path of a tracked graph fixture such as ``"browsing.json"``."""
    path = FIXTURE_GRAPHS / name
    if not path.is_file():
        raise FileNotFoundError(f"tracked graph fixture missing: {path}")
    return path


def load_fixture_graph(name: str) -> dict[str, Any]:
    """Load a tracked graph fixture as a fresh dict."""
    return json.loads(fixture_graph_path(name).read_text(encoding="utf-8"))


def debug_errors(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the ``Node.yield_debug_error`` payloads among runtime events."""
    return [
        event["content"] for event in events
        if event.get("type") == "debug"
        and isinstance(event.get("content"), dict)
        and event["content"].get("error_type")
    ]


async def collect_all_from_generator(async_gen: AsyncIterable[Any]) -> list[Any]:
    """Consume an async generator and return all emitted values."""
    return [item async for item in async_gen]


__all__ = [
    "AsyncRecordingHook",
    "FIXTURE_GRAPHS",
    "RecordingHook",
    "_is_placeholder_value",
    "_load_env_test",
    "_parse_dotenv",
    "_resolve_api_keys",
    "collect_all_from_generator",
    "debug_errors",
    "fixture_graph_path",
    "load_fixture_graph",
    "skip_if_no_api_keys",
]
