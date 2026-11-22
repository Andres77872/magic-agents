"""Opt-in live-provider tests.

Every module here declares ``pytestmark = LIVE`` (``needs_api``,
``credential_gated``, ``slow``), and the default ``-m`` selection in setup.cfg
deselects them. Run explicitly, with real keys in the shell or ``.env.test``::

    python -m pytest tests/live -m needs_api

Tests skip (never fail) when no real OpenAI key is configured.
"""

from __future__ import annotations

import os

import pytest

from tests.support.env import _load_env_test, skip_if_no_api_keys


@pytest.fixture
def api_keys(monkeypatch):
    """Resolve real keys (shell first, then ``.env.test``) or skip the test.

    Keys that only exist in ``.env.test`` are exported for the duration of the
    test so graphs using ``{{env.OPENAI_API_KEY}}`` placeholders see them.
    """
    keys = skip_if_no_api_keys()
    for name, value in _load_env_test().items():
        if name not in os.environ:
            monkeypatch.setenv(name, value)
    return keys
