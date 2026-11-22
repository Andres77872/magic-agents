"""
Slice 17 — Mocked fetch node integration tests (no API keys).

Tests fetch node with mocked HTTP responses. No real network calls.
Tests the fetch node directly rather than through the full executor
to avoid hanging issues with complex graph execution.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from magic_agents.node_system import NodeFetch
from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


class TestFetchMockedHTTP:
    """Tests for fetch node with mocked HTTP responses."""

    def _make_mock_session(self, response_data, status=200):
        """Create a fully mocked aiohttp session with response."""
        mock_response = MagicMock()
        mock_response.status = status
        mock_response.json = AsyncMock(return_value=response_data)
        mock_response.raise_for_status = MagicMock()

        # Async context manager for the response
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        # session.request must be a MagicMock (not AsyncMock) that returns the cm directly
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_cm)
        # Make the session itself an async context manager
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        return mock_session

    @pytest.mark.asyncio
    async def test_fetch_body_alias_supports_env_placeholders(self, monkeypatch):
        """Fetch body alias maps to data and resolves env placeholders."""
        monkeypatch.setenv("API_TOKEN", "body-secret")
        mock_response_data = {"ok": True}
        mock_session = self._make_mock_session(mock_response_data)

        fetch_node = NodeFetch(
            data=FetchNodeModel(
                url="https://api.example.com/items",
                method="POST",
                body={
                    "token": "{{env.API_TOKEN}}",
                    "query": "{{ handle_fetch_input }}",
                },
            ),
            node_id="fetch_body_alias",
            debug=False,
        )
        fetch_node.inputs["handle_fetch_input"] = "widget"

        chat_log = ModelAgentRunLog()
        with patch("aiohttp.ClientSession", return_value=mock_session):
            async for _ in fetch_node(chat_log):
                pass

        call_kwargs = mock_session.request.call_args
        assert call_kwargs.kwargs["data"] == {
            "token": "body-secret",
            "query": "widget",
        }

