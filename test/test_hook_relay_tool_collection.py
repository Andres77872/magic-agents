"""
Regression test: HookRelay accumulates tool call/result data for API persistence.

When NodeLLM runs with callable tools, the agent loop consumes tool data
internally. HookRelay (via AgentHooks on_tool_start/on_tool_complete) is the
ONLY surviving path for tool data. This test verifies HookRelay correctly
captures and exposes that data for NodeLLM to yield to the API layer.
"""
import json
from unittest.mock import MagicMock

from magic_llm.agent.types import ToolResult
from magic_agents.hooks.hook_relay import HookRelay


class TestHookRelayToolCollection:
    """HookRelay must accumulate tool call/result data during lifecycle."""

    def test_on_tool_start_accumulates_tool_call(self):
        relay = HookRelay(node_id="llm-1", graph_id="graph-1", run_id="run-1")

        relay.on_tool_start(
            tool_name="web_search",
            tool_call_id="call-abc123",
            arguments={"query": "test"},
            state=MagicMock(),
        )

        assert len(relay.collected_tool_calls) == 1
        tc = relay.collected_tool_calls[0]
        assert tc["id"] == "call-abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "test"}

    def test_on_tool_start_multiple_calls_accumulates_all(self):
        relay = HookRelay(node_id="llm-1")

        relay.on_tool_start("tool_a", "call-1", {"x": 1}, MagicMock())
        relay.on_tool_start("tool_b", "call-2", {"y": 2}, MagicMock())

        assert len(relay.collected_tool_calls) == 2
        assert relay.collected_tool_calls[0]["function"]["name"] == "tool_a"
        assert relay.collected_tool_calls[1]["function"]["name"] == "tool_b"

    def test_on_tool_complete_accumulates_tool_result(self):
        relay = HookRelay(node_id="llm-1")

        result = ToolResult(
            tool_call_id="call-abc123",
            name="web_search",
            content='{"results": ["data"]}',
            is_error=False,
            duration_ms=150.0,
        )
        relay.on_tool_complete(result, MagicMock())

        events, calls, results = _split_events(relay)
        assert len(results) == 1
        assert results[0]["data"]["role"] == "tool"
        assert results[0]["data"]["tool_call_id"] == "call-abc123"
        assert results[0]["data"]["status"] == "completed"
        assert results[0]["data"]["execution_time_ms"] == 150.0

    def test_on_tool_complete_error_sets_status_and_error(self):
        relay = HookRelay(node_id="llm-1")

        result = ToolResult(
            tool_call_id="call-xyz",
            name="fetch_data",
            content="",
            is_error=True,
            error="Connection refused",
            duration_ms=5000.0,
        )
        relay.on_tool_complete(result, MagicMock())

        events, calls, results = _split_events(relay)
        assert len(results) == 1
        assert results[0]["data"]["status"] == "error"
        assert results[0]["data"]["tool_error"] == "Connection refused"
        assert results[0]["data"]["execution_time_ms"] == 5000.0

    def test_get_collected_tool_data_for_yield_returns_both_types(self):
        relay = HookRelay(node_id="llm-1")

        relay.on_tool_start("search", "call-1", {"q": "hello"}, MagicMock())
        relay.on_tool_complete(
            ToolResult(tool_call_id="call-1", name="search", content="ok", duration_ms=50.0),
            MagicMock(),
        )

        events = relay.get_collected_tool_data_for_yield()
        assert len(events) == 2
        assert events[0]["type"] == "tool_call"
        assert events[0]["data"]["role"] == "assistant"
        assert len(events[0]["data"]["tool_calls"]) == 1
        assert events[0]["data"]["tool_calls"][0]["function"]["name"] == "search"

        assert events[1]["type"] == "tool_result"
        assert events[1]["data"]["role"] == "tool"
        assert events[1]["data"]["tool_call_id"] == "call-1"

    def test_yielded_events_match_insert_tool_messages_shape(self):
        """Verify the yielded event data matches insert_tool_messages() expectations.

        insert_tool_messages expects:
        - assistant msg: {'role': 'assistant', 'tool_calls': [{'id', 'type': 'function',
          'function': {'name', 'arguments'}}]}
        - tool msg: {'role': 'tool', 'tool_call_id', 'content', 'status', 'execution_time_ms'}
        """
        relay = HookRelay(node_id="llm-1")

        relay.on_tool_start("web_search", "call-1", {"query": "test"}, MagicMock())
        relay.on_tool_complete(
            ToolResult(tool_call_id="call-1", name="web_search", content='{"data": "ok"}', duration_ms=120.0),
            MagicMock(),
        )

        events = relay.get_collected_tool_data_for_yield()

        # Assistant tool_call message shape
        assert events[0]["data"]["role"] == "assistant"
        assert "tool_calls" in events[0]["data"]
        tc = events[0]["data"]["tool_calls"][0]
        assert tc["id"] == "call-1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"

        # Tool result message shape
        assert events[1]["data"]["role"] == "tool"
        assert events[1]["data"]["tool_call_id"] == "call-1"
        assert events[1]["data"]["status"] == "completed"
        assert events[1]["data"]["execution_time_ms"] == 120.0

    def test_empty_relay_returns_empty_lists(self):
        relay = HookRelay(node_id="llm-1")

        assert relay.collected_tool_calls == []
        assert relay.get_collected_tool_data_for_yield() == []


def _split_events(relay: HookRelay) -> tuple[list, list]:
    """Helper: split collected tool data into (calls, results)."""
    events = relay.get_collected_tool_data_for_yield()
    calls = [e for e in events if e["type"] == "tool_call"]
    results = [e for e in events if e["type"] == "tool_result"]
    return events, calls, results
