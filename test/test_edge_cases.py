"""
Edge case tests for magic-agents.

Tests that don't need API keys run always.
Tests that need API keys are skipped gracefully when keys are missing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import asyncio

from magic_agents import run_agent
from magic_agents.agt_flow import build, validate_graph
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from conftest import collect_all_from_generator


# Try to load API keys from environment or configured file path
_API_KEYS = None
_api_keys_file = os.environ.get("MAGIC_AGENTS_API_KEY_FILE", "")
_api_keys_env = os.environ.get("OPENAI_API_KEY", "")

if _api_keys_file and os.path.exists(_api_keys_file):
    try:
        with open(_api_keys_file) as f:
            _API_KEYS = json.load(f)
    except (json.JSONDecodeError, KeyError):
        pass
elif _api_keys_env:
    _API_KEYS = {"openai_key": _api_keys_env}

_needs_api = pytest.mark.skipif(
    _API_KEYS is None,
    reason="API keys not available (set OPENAI_API_KEY or MAGIC_AGENTS_API_KEY_FILE)"
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
                 "sourceHandle": "out", "targetHandle": "in"},
                {"id": "e2", "source": "node-b", "target": "node-a",
                 "sourceHandle": "out", "targetHandle": "in"},
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
                {"id": "e2", "source": "bad-parser", "target": "end-node"},
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
                {"id": "e3", "source": "final-parser", "target": "end-node"},
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
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle-5"},
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
                timeout=30.0
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

    # ─── API-dependent tests (skip when keys missing) ───────────────────

    @pytest.mark.needs_api
    @_needs_api
    @pytest.mark.asyncio
    async def test_very_long_input_handling(self):
        """Test handling of very long inputs."""
        long_text = "This is a test. " * 100

        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "truncator",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "truncator", "target": "llm-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_user_message"},
                {"id": "e3", "source": "client-node", "target": "llm-node",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e4", "source": "llm-node", "target": "end-node",
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle-5"},
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

        graph = build(agt_data=agt, message=long_text, load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        assert "truncated" in response.lower()

    @pytest.mark.needs_api
    @_needs_api
    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters and escaping."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "escaper",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "escaper", "target": "llm-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_user_message"},
                {"id": "e3", "source": "client-node", "target": "llm-node",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e4", "source": "llm-node", "target": "end-node",
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle-5"},
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
                    "data": {"top_p": 1, "stream": True, "max_tokens": 100, "temperature": 0.7}
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        special_input = 'Hello & "world" <script>alert("test")</script>'
        graph = build(agt_data=agt, message=special_input, load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        assert "&" in response or "amp" in response

    @pytest.mark.needs_api
    @_needs_api
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "user-input", "target": "unicode-processor",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "unicode-processor", "target": "llm-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_user_message"},
                {"id": "e3", "source": "client-node", "target": "llm-node",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e4", "source": "llm-node", "target": "end-node",
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle-5"},
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
                    "data": {"top_p": 1, "stream": True, "max_tokens": 100, "temperature": 0.7}
                },
                {"id": "end-node", "type": ModelAgentFlowTypesModel.END},
            ]
        }

        unicode_input = "Hello 世界 🌍 مرحبا"
        graph = build(agt_data=agt, message=unicode_input, load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        assert "世界" in response or "🌍" in response or "مرحبا" in response

    @pytest.mark.needs_api
    @_needs_api
    @pytest.mark.asyncio
    async def test_missing_required_inputs(self):
        """Test handling of missing required inputs."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {"id": "e1", "source": "default-text", "target": "input-checker",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_default"},
                {"id": "e2", "source": "input-checker", "target": "llm-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_user_message"},
                {"id": "e3", "source": "client-node", "target": "llm-node",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e4", "source": "llm-node", "target": "end-node",
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle-5"},
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

        graph = build(agt_data=agt, message='Test message', load_chat=self.load_chat)
        # Verify the graph builds correctly with the conditional parser
        assert graph is not None
        checker_node = graph.nodes.get("input-checker")
        assert checker_node is not None
        # The parser template should contain the conditional logic
        assert "handle_default" in checker_node.text

        # Run the agent — should complete without errors
        events = await collect_all_from_generator(run_agent(graph=graph))
        assert len(events) > 0, "Expected at least one event from execution"

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
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e1", "source": "complex-json", "target": "json-navigator",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "json-navigator", "target": "end-node",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle-5"},
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

        events = await collect_all_from_generator(run_agent(graph=graph))
        assert len(events) > 0, "Expected at least one event from execution"

        # Verify no execution errors occurred
        debug_errors = [
            e for e in events
            if e.get("type") == "debug" and e.get("content", {}).get("error")
        ]
        assert len(debug_errors) == 0, (
            f"Parser should not error on nested JSON. Errors: "
            f"{[e['content']['error'] for e in debug_errors]}"
        )

        # Verify the parser node executed successfully
        parser_debug = [
            e for e in events
            if e.get("type") == "debug"
            and e.get("content", {}).get("node_id") == "json-navigator"
            and e.get("content", {}).get("was_executed")
        ]
        assert len(parser_debug) > 0, "Parser node should have executed"

        # Verify content was produced (send_message or text output)
        content_events = [e for e in events if e.get("type") == "content"]
        assert len(content_events) > 0, "Expected content events from parser output"

    @pytest.mark.needs_api
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
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle-5"},
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
