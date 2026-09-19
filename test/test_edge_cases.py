"""
Edge case tests for magic-agents.

Tests that don't need API keys run always.
Tests that need API keys are skipped gracefully when keys are missing.
"""
import json
import os

import pytest
import asyncio

from magic_agents import run_agent
from magic_agents.agt_flow import build, validate_graph
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from test_support import _is_placeholder_value, collect_all_from_generator


# Try to load API keys from environment or configured file path
_API_KEYS = None
_api_keys_file = os.environ.get("MAGIC_AGENTS_API_KEY_FILE", "")
_api_keys_env = os.environ.get("OPENAI_API_KEY", "")

if _api_keys_file and os.path.exists(_api_keys_file):
    try:
        with open(_api_keys_file, encoding="utf-8") as f:
            _API_KEYS = json.load(f)
    except (OSError, json.JSONDecodeError, KeyError):
        pass
elif _api_keys_env:
    _API_KEYS = {"openai_key": _api_keys_env}

_openai_key = (_API_KEYS or {}).get("openai_key", "")
_live_tests_enabled = os.environ.get("MAGIC_AGENTS_RUN_LIVE_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
_live_opt_in = pytest.mark.skipif(
    not _live_tests_enabled,
    reason="live provider smoke test; set MAGIC_AGENTS_RUN_LIVE_TESTS=1 to opt in",
)
_needs_api = pytest.mark.skipif(
    not _openai_key or _is_placeholder_value(_openai_key),
    reason="Real OpenAI API key required (placeholder test keys do not enable live tests)",
)


class TestEdgeCases:
    """Test suite for edge cases and error handling scenarios."""

    def setup_method(self):
        """Setup method to initialize common test data."""
        self.load_chat = lambda **kwargs: None
        self.api_keys = _API_KEYS

    # ─── No-API tests (always run) ──────────────────────────────────────

    def test_circular_reference_detection(self):
        """Test that circular references in the graph are handled during build."""
        # Graph with a cycle: A -> B -> A
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "node-a", "target": "node-b",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "node-b", "target": "node-a",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "node-a", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "A"}},
                {"id": "node-b", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "B"}},
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        # build() should handle cycles gracefully (networkx falls back to insertion order)
        graph = build(agt_data=agt, message='test', load_chat=self.load_chat)
        assert graph is not None
        # All nodes should be present including the void sentinel
        assert len(graph.nodes) >= 5  # 4 user nodes + void sentinel

    def test_malformed_json_in_parser_template(self):
        """Test that a parser with intentionally malformed JSON template builds fine."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "bad-parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "bad-parser", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "bad-parser", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {
                        "text": '{"message": "{{ handle_parser_input }}", "incomplete": '
                    }
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        # Build should succeed — the malformed JSON is in the template, not the graph
        graph = build(agt_data=agt, message='test', load_chat=self.load_chat)
        assert graph is not None
        parser_node = graph.nodes.get("bad-parser")
        assert parser_node is not None
        # The template text should be stored as-is (NodeParser stores it in .text)
        assert "incomplete" in parser_node.text

    def test_empty_loop_graph_builds(self):
        """Test that a graph with an empty list loop builds correctly."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "empty-list", "target": "loop-node",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e2", "source": "loop-node", "target": "final-parser",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "final-parser", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "empty-list", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "[]"}},
                {"id": "loop-node", "type": ModelAgentFlowTypesModel.LOOP, "data": {}},
                {
                    "id": "final-parser", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {"text": "Done: {{ handle_parser_input | length }} items"}
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        graph = build(agt_data=agt, message='', load_chat=self.load_chat)
        assert graph is not None
        # Loop node should be present
        loop_node = graph.nodes.get("loop-node")
        assert loop_node is not None
        assert loop_node.__class__.__name__ == "NodeLoop"

    @pytest.mark.asyncio
    async def test_timeout_simulation(self):
        """Test that a simple flow completes within a reasonable timeout.

        The old test misused asyncio.wait_for on a generator.
        This test properly collects all events and then checks timing.
        """
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "parser-node",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "parser-node", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "parser-node", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {"text": "Processed: {{ handle_parser_input }}"}
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        graph = build(agt_data=agt, message='Test timeout handling', load_chat=self.load_chat)
        assert graph is not None

        # Properly consume the async generator with a timeout
        async def _collect_with_timeout():
            return await asyncio.wait_for(
                collect_all_from_generator(run_agent(graph=graph)),
                timeout=2.0
            )

        events = await _collect_with_timeout()
        # Should have completed within timeout — at least got some events
        assert len(events) > 0, "Expected at least one event from execution"

    def test_validation_empty_nodes_list(self):
        """Test that validate_graph handles empty nodes list."""
        result = validate_graph([], [])
        assert result["valid"] is False
        assert len(result["errors"]) >= 1
        assert any("USER_INPUT" in e["error_message"] for e in result["errors"])

    def test_validation_empty_edges_list(self):
        """Test that validate_graph handles empty edges list (valid if nodes exist)."""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.END},
        ]
        result = validate_graph(nodes, [])
        # Should be valid — no duplicate edges
        assert result["valid"] is True

    # ─── Credential-free execution edge cases ───────────────────────────

    @pytest.mark.asyncio
    async def test_very_long_input_handling(self):
        """The real parser truncates long input without a provider call."""
        long_text = "This is a test. " * 100

        agt = {
            "type": "graph",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "truncator",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "truncator", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "truncator", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {
                        "text": """{% set max_length = 200 %}
{% if handle_parser_input | length > max_length %}
Input truncated (was {{ handle_parser_input | length }} chars): {{ handle_parser_input[:max_length] }}...
{% else %}
{{ handle_parser_input }}
{% endif %}"""
                    }
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        graph = build(agt_data=agt, message=long_text, load_chat=self.load_chat)
        await collect_all_from_generator(run_agent(graph=graph))

        response = graph.nodes["truncator"].response
        assert "truncated" in response.lower()
        assert f"{len(long_text)} chars" in response
        assert len(response) < len(long_text)

    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """The parser's Jinja environment escapes HTML and JSON locally."""
        agt = {
            "type": "graph",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "escaper",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "escaper", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "escaper", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {
                        "text": """Input with special handling:
Original: {{ handle_parser_input | e }}
JSON Safe: {{ handle_parser_input | tojson }}"""
                    }
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        special_input = 'Hello & "world" <script>alert("test")</script>'
        graph = build(agt_data=agt, message=special_input, load_chat=self.load_chat)
        await collect_all_from_generator(run_agent(graph=graph))

        response = graph.nodes["escaper"].response
        assert "&lt;script&gt;" in response
        assert '\\"world\\"' in response
        assert '<script>alert(\\"test\\")</script>' in response

    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """The real parser preserves Unicode without a provider round-trip."""
        agt = {
            "type": "graph",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "unicode-processor",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "unicode-processor", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "unicode-processor", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {
                        "text": """Unicode test:
Original: {{ handle_parser_input }}
Length: {{ handle_parser_input | length }} characters"""
                    }
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        unicode_input = "Hello 世界 🌍 مرحبا"
        graph = build(agt_data=agt, message=unicode_input, load_chat=self.load_chat)
        await collect_all_from_generator(run_agent(graph=graph))

        response = graph.nodes["unicode-processor"].response
        assert unicode_input in response
        assert f"Length: {len(unicode_input)} characters" in response

    @pytest.mark.asyncio
    async def test_missing_required_inputs(self):
        """A parser can select its real fallback input when user input is absent."""
        agt = {
            "type": "graph",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "default-text", "target": "input-checker",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_default"},
                {"id": "e2", "source": "input-checker", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "default-text", "type": ModelAgentFlowTypesModel.TEXT,
                    "data": {"text": "Default fallback message"}
                },
                {
                    "id": "input-checker", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {
                        "text": """{% if handle_user_message is defined %}
User input: {{ handle_user_message }}
{% else %}
No user input provided. Using default: {{ handle_default }}
{% endif %}"""
                    }
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        graph = build(agt_data=agt, message='Test message', load_chat=self.load_chat)
        # Verify the graph builds correctly with the conditional parser
        assert graph is not None
        checker_node = graph.nodes.get("input-checker")
        assert checker_node is not None
        # The parser template should contain the conditional logic
        assert "handle_default" in checker_node.text

        await collect_all_from_generator(run_agent(graph=graph))
        assert checker_node.response is not None
        assert "No user input provided" in checker_node.response
        assert "Default fallback message" in checker_node.response

    @pytest.mark.asyncio
    async def test_nested_json_parsing(self):
        """Test complex nested JSON parsing with fromjson filter.

        Verifies that a text node outputting a JSON string can be parsed
        by a downstream parser node using the | fromjson Jinja2 filter,
        even when the parser's safe_json_parse has already converted the
        string to a dict (idempotent fromjson).
        """
        agt = {
            "type": "graph",
            "debug": True,
            "edges": [
                {"id": "e0", "source": "input", "target": "complex-json",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e1", "source": "complex-json", "target": "json-navigator",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "json-navigator", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "complex-json", "type": ModelAgentFlowTypesModel.TEXT,
                    "data": {
                        "text": '{"user": {"name": "John", "preferences": {"theme": "dark", "notifications": {"email": true, "sms": false}}, "tags": ["developer", "python", "ai"]}}'
                    }
                },
                {
                    "id": "json-navigator", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {
                        "text": """{% set data = handle_parser_input | fromjson %}
User Profile:
- Name: {{ data.user.name }}
- Theme: {{ data.user.preferences.theme }}
- Email notifications: {{ data.user.preferences.notifications.email }}
- Tags: {{ data.user.tags | join(", ") }}
- Tag count: {{ data.user.tags | length }}"""
                    }
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        graph = build(agt_data=agt, message='test', load_chat=self.load_chat)
        assert graph is not None
        navigator = graph.nodes.get("json-navigator")
        assert navigator is not None
        assert "data.user.name" in navigator.text

        await collect_all_from_generator(run_agent(graph=graph))

        rendered = navigator.response
        assert rendered is not None
        assert "Name: John" in rendered
        assert "Theme: dark" in rendered
        assert "Tags: developer, python, ai" in rendered
        assert graph.nodes["end-node"].inputs["handle_flow_input"] == rendered

    @pytest.mark.needs_api
    @pytest.mark.credential_gated
    @pytest.mark.slow
    @_live_opt_in
    @_needs_api
    @pytest.mark.asyncio
    async def test_timeout_simulation_with_api(self):
        """Test handling of slow operations with real LLM (simulated timeout)."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "timer-parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "timer-parser", "target": "llm-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_user_message"},
                {"id": "e3", "source": "client-node", "target": "llm-node",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e4", "source": "llm-node", "target": "end-node",
                 "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input"},
            ],
            "nodes": [
                {"id": "user-input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "timer-parser", "type": ModelAgentFlowTypesModel.PARSER,
                    "data": {"text": "Processing request: {{ handle_parser_input }}"}
                },
                {
                    "id": "client-node", "type": ModelAgentFlowTypesModel.CLIENT,
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "llm-node", "type": ModelAgentFlowTypesModel.LLM,
                    "data": {"top_p": 1, "stream": True, "max_tokens": 50, "temperature": 0.7}
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        graph = build(agt_data=agt, message='Test timeout handling', load_chat=self.load_chat)
        assert graph is not None

        async def _collect_with_timeout():
            return await asyncio.wait_for(
                collect_all_from_generator(run_agent(graph=graph)),
                timeout=30.0
            )

        events = await _collect_with_timeout()
        # Should complete within timeout — at least got some events
        assert len(events) > 0, "Expected at least one event from execution"
