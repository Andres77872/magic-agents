"""
Focused tests for the llm-tooling gap fixes identified by verification.

Tests cover:
1. FetchToolCallable __name__ for tool_functions registration
2. tool_parameters explicit schema support
3. handle-tool-calls output handle
4. Sync run_agent fallback via asyncio.to_thread
5. _assign_tool_handles only rewrites fetch edges when tool_mode=true
"""
import asyncio
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from magic_agents.agt_flow import _assign_tool_handles
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.node_system.NodeFetch import FetchToolCallable, NodeFetch
from magic_agents.node_system.NodeLLM import NodeLLM


# ─── Gap 1: FetchToolCallable __name__ for tool_functions registration ───────

class TestFetchToolCallableName:
    """FetchToolCallable must expose __name__ so _collect_tools registers it."""

    def test_fetch_tool_callable_has_name_property(self):
        """FetchToolCallable.__name__ returns the configured tool_name."""
        callable_tool = FetchToolCallable(
            url_template="https://api.example.com/search?q={{query}}",
            tool_name="search_api",
        )
        assert callable_tool.__name__ == "search_api"

    def test_fetch_tool_callable_default_name(self):
        """Default __name__ is 'fetch' when no tool_name provided."""
        callable_tool = FetchToolCallable(
            url_template="https://api.example.com/data",
        )
        assert callable_tool.__name__ == "fetch"

    def test_collect_tools_registers_fetch_by_name(self):
        """NodeLLM._collect_tools registers FetchToolCallable in tool_functions."""
        callable_tool = FetchToolCallable(
            url_template="https://api.example.com/search?q={{query}}",
            tool_name="search_api",
        )

        # Create a minimal NodeLLM mock with tool inputs
        node = MagicMock(spec=NodeLLM)
        node.INPUT_TOOL_PREFIX = 'handle-tool-'
        node.inputs = {
            'handle-tool-definition-0': callable_tool,
            'handle-client-provider': MagicMock(),
        }

        # Call the real _collect_tools method bound to our mock
        tools_schemas, tool_functions = NodeLLM._collect_tools(node)

        assert len(tools_schemas) == 1
        assert "search_api" in tool_functions
        assert tool_functions["search_api"] is callable_tool


# ─── Gap 2: tool_parameters explicit schema support ─────────────────────────

class TestToolParametersSchema:
    """FetchToolCallable must honor explicit tool_parameters over template extraction."""

    def test_explicit_tool_parameters_used_in_schema(self):
        """When tool_parameters provided, schema uses them (not template vars)."""
        callable_tool = FetchToolCallable(
            url_template="https://api.example.com/weather?city={{city}}",
            tool_name="weather",
            tool_parameters={
                "city": {"type": "string", "description": "City name", "required": True},
                "days": {"type": "integer", "description": "Number of days", "required": False},
            },
        )
        schema = callable_tool.tool_schema

        func = schema["function"]
        assert func["name"] == "weather"
        props = func["parameters"]["properties"]
        assert "city" in props
        assert "days" in props
        assert props["city"]["type"] == "string"
        assert props["days"]["type"] == "integer"
        # Only explicitly required params are in required list
        assert "city" in func["parameters"]["required"]

    def test_tool_parameters_override_template_vars(self):
        """tool_parameters takes precedence even when template vars exist."""
        callable_tool = FetchToolCallable(
            url_template="https://api.example.com/search?q={{query}}&limit={{limit}}",
            tool_parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results"},
            },
        )
        schema = callable_tool.tool_schema
        props = schema["function"]["parameters"]["properties"]
        # Should have query and max_results (from tool_parameters), NOT limit
        assert "query" in props
        assert "max_results" in props
        assert "limit" not in props

    def test_fallback_to_template_vars_when_no_tool_parameters(self):
        """Without tool_parameters, schema auto-generates from template vars."""
        callable_tool = FetchToolCallable(
            url_template="https://api.example.com/weather?city={{city}}&units={{units}}",
        )
        schema = callable_tool.tool_schema
        props = schema["function"]["parameters"]["properties"]
        assert "city" in props
        assert "units" in props
        assert props["city"]["type"] == "string"

    def test_node_fetch_passes_tool_parameters(self):
        """NodeFetch passes tool_parameters to FetchToolCallable."""
        mock_data = MagicMock()
        mock_data.method = "GET"
        mock_data.url = "https://api.example.com/search?q={{query}}"
        mock_data.headers = {}
        mock_data.data = None
        mock_data.params = None
        mock_data.json_data = None
        mock_data.tool_mode = True
        mock_data.tool_name = "search"
        mock_data.tool_parameters = {
            "query": {"type": "string", "description": "Search query"},
        }
        mock_data.debug = False

        node = NodeFetch(data=mock_data, node_id="fetch-1")
        assert node.tool_parameters == {"query": {"type": "string", "description": "Search query"}}


# ─── Gap 3: handle-tool-calls output handle ─────────────────────────────────

class TestHandleToolCalls:
    """NodeLLM must emit handle-tool-calls output."""

    def test_default_tool_calls_handle(self):
        """NodeLLM has DEFAULT_OUTPUT_TOOL_CALLS constant."""
        assert NodeLLM.DEFAULT_OUTPUT_TOOL_CALLS == 'handle-tool-calls'

    def test_tool_calls_handle_configurable(self):
        """handle-tool-calls can be overridden via JSON config."""
        mock_data = MagicMock()
        mock_data.stream = False
        mock_data.json_output = False
        mock_data.extra_data = {}
        mock_data.temperature = None
        mock_data.top_p = None
        mock_data.max_tokens = None

        node = NodeLLM(
            data=mock_data,
            node_id="llm-1",
            handles={'output_tool_calls': 'handle-custom-tool-calls'},
        )
        assert node.OUTPUT_HANDLE_TOOL_CALLS == 'handle-custom-tool-calls'

    def test_tool_calls_handle_default(self):
        """Default OUTPUT_HANDLE_TOOL_CALLS is handle-tool-calls."""
        mock_data = MagicMock()
        mock_data.stream = False
        mock_data.json_output = False
        mock_data.extra_data = {}
        mock_data.temperature = None
        mock_data.top_p = None
        mock_data.max_tokens = None

        node = NodeLLM(data=mock_data, node_id="llm-1")
        assert node.OUTPUT_HANDLE_TOOL_CALLS == 'handle-tool-calls'


# ─── Gap 4: Sync fallback via asyncio.to_thread ─────────────────────────────

class TestSyncFallback:
    """NodeLLM must fall back to sync run_agent when async unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_to_sync_run_agent_when_no_async(self):
        """When client lacks run_agent_async(), NodeLLM falls back to asyncio.to_thread."""
        from magic_llm.model import ModelChat
        from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel
        from magic_llm.model.ModelChatResponse import UsageModel

        # Mock client WITHOUT run_agent_async() but WITH run_agent()
        mock_llm_engine = MagicMock()
        mock_llm_engine.model = "mock-model"

        mock_usage = UsageModel(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        mock_response = MagicMock()
        mock_response.content = "sync fallback response"
        mock_response.usage = mock_usage
        mock_response.tool_calls = []
        
        mock_client = MagicMock()
        mock_client.llm = mock_llm_engine
        mock_client.run_agent = MagicMock(return_value=mock_response)
        # Explicitly remove run_agent_async to trigger fallback
        del mock_client.run_agent_async

        mock_data = MagicMock()
        mock_data.stream = False
        mock_data.json_output = False
        mock_data.extra_data = {}
        mock_data.temperature = None
        mock_data.top_p = None
        mock_data.max_tokens = None

        node = NodeLLM(data=mock_data, node_id="llm-1")
        node.inputs = {
            'handle-client-provider': mock_client,
            'handle_user_message': 'hello',
            'handle-tool-definition-0': MagicMock(
                tool_schema={"type": "function", "function": {"name": "test", "description": "test", "parameters": {}}},
                tool_callable=MagicMock(__name__="test"),
            ),
        }

        # Capture warnings
        with patch('magic_agents.node_system.NodeLLM.logger') as mock_logger:
            outputs = []
            async for output in node.process([]):
                outputs.append(output)

            # Verify warning was logged
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "async agent loop not available" in warning_msg
            assert "falling back to sync run_agent()" in warning_msg

            # Verify run_agent was called via asyncio.to_thread
            mock_client.run_agent.assert_called_once()


# ─── Gap 5: _assign_tool_handles only for tool_mode fetch ───────────────────

class TestAssignToolHandlesToolMode:
    """_assign_tool_handles must only rewrite fetch edges when tool_mode=true."""

    def test_tool_mode_false_fetch_not_rewritten(self):
        """Plain fetch (tool_mode=false) edge is NOT assigned a tool handle."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": False, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        # Edge should NOT have a tool handle assigned
        assert edges[0].get('targetHandle') is None

    def test_tool_mode_true_fetch_rewritten(self):
        """Fetch with tool_mode=true gets a tool handle."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    def test_tool_mode_absent_fetch_not_rewritten(self):
        """Fetch without tool_mode field (absent) is NOT assigned a tool handle."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0].get('targetHandle') is None

    def test_python_exec_always_rewritten(self):
        """python_exec edges are always assigned tool handles (no tool_mode check)."""
        nodes = [
            {"id": "py1", "type": "python_exec", "data": {}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    def test_mixed_tool_mode_and_plain_fetch(self):
        """Mixed graph: only tool_mode=true fetch gets tool handle, plain fetch does not."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com/tool"}},
            {"id": "f2", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": False, "url": "https://api.example.com/plain"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
            {"id": "e2", "source": "f2", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        # f1 (tool_mode=true) gets tool handle
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'
        # f2 (tool_mode=false) does NOT get tool handle
        assert edges[1].get('targetHandle') is None

    def test_user_handle_takes_precedence_over_tool_mode(self):
        """Edge with explicit targetHandle is not overwritten even for tool_mode=true."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1", "targetHandle": "handle-custom"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-custom'

    # ─── Slice 1: sourceHandle for tool_mode fetch edges ─────────────────────

    def test_sourceHandle_set_for_tool_mode_fetch_default(self):
        """tool_mode=true fetch→LLM edge gets sourceHandle='handle_fetch_output'."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle_fetch_output'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    def test_sourceHandle_set_for_tool_mode_fetch_custom_output(self):
        """tool_mode=true fetch with custom data.handles.output gets matching sourceHandle."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com",
                      "handles": {"output": "handle-custom-fetch"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-custom-fetch'

    def test_sourceHandle_set_for_tool_mode_fetch_custom_response(self):
        """tool_mode=true fetch with data.handles.response (legacy) gets matching sourceHandle."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com",
                      "handles": {"response": "handle-legacy-response"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-legacy-response'

    def test_sourceHandle_not_set_for_tool_mode_false(self):
        """Plain fetch (tool_mode=false) edge gets NO sourceHandle assigned."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": False, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0].get('targetHandle') is None
        assert edges[0].get('sourceHandle') is None

    def test_sourceHandle_not_overwritten_if_already_set(self):
        """If edge already has sourceHandle, it is not overwritten."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1", "sourceHandle": "handle-preset"},
        ]
        _assign_tool_handles(nodes, edges)

        # sourceHandle should remain as preset (setdefault behavior)
        assert edges[0]['sourceHandle'] == 'handle-preset'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    # ─── Slice 7: explicit targetHandle + missing sourceHandle backfill ──────

    def test_sourceHandle_backfilled_when_targetHandle_explicit_fetch(self):
        """Fetch edge with explicit targetHandle but no sourceHandle gets sourceHandle backfilled."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1", "targetHandle": "handle-custom-target"},
        ]
        _assign_tool_handles(nodes, edges)

        # targetHandle must be preserved (not overwritten)
        assert edges[0]['targetHandle'] == 'handle-custom-target'
        # sourceHandle must be backfilled even though targetHandle was explicit
        assert edges[0]['sourceHandle'] == 'handle_fetch_output'

    def test_sourceHandle_backfilled_when_targetHandle_explicit_fetch_custom_output(self):
        """Fetch with explicit targetHandle + custom handles.output gets matching sourceHandle."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com",
                      "handles": {"output": "handle-custom-fetch"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1", "targetHandle": "handle-custom-target"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-custom-target'
        assert edges[0]['sourceHandle'] == 'handle-custom-fetch'

    def test_sourceHandle_preserved_when_both_handles_explicit_fetch(self):
        """Fetch edge with both targetHandle AND sourceHandle explicit: neither is overwritten."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1",
             "targetHandle": "handle-custom-target", "sourceHandle": "handle-custom-source"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-custom-target'
        assert edges[0]['sourceHandle'] == 'handle-custom-source'

    def test_sourceHandle_not_backfilled_for_tool_mode_false_with_explicit_target(self):
        """Plain fetch (tool_mode=false) with explicit targetHandle gets NO sourceHandle backfill."""
        nodes = [
            {"id": "f1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": False, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "f1", "target": "llm-1", "targetHandle": "handle-custom"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-custom'
        assert edges[0].get('sourceHandle') is None


class TestAssignToolHandlesPythonExec:
    """_assign_tool_handles must set sourceHandle for python_exec edges."""

    def test_sourceHandle_set_for_python_exec_default(self):
        """python_exec→LLM edge gets sourceHandle='handle-tool-definition'."""
        nodes = [
            {"id": "py1", "type": "python_exec", "data": {}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-tool-definition'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    def test_sourceHandle_set_for_python_exec_custom_output(self):
        """python_exec with custom data.handles.output gets matching sourceHandle."""
        nodes = [
            {"id": "py1", "type": "python_exec",
             "data": {"handles": {"output": "handle-custom-exec"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-custom-exec'

    def test_sourceHandle_not_overwritten_if_already_set_python_exec(self):
        """If python_exec edge already has sourceHandle, it is not overwritten."""
        nodes = [
            {"id": "py1", "type": "python_exec", "data": {}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1", "sourceHandle": "handle-preset"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-preset'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    # ─── Slice 7: explicit targetHandle + missing sourceHandle backfill ──────

    def test_sourceHandle_backfilled_when_targetHandle_explicit_python_exec(self):
        """python_exec edge with explicit targetHandle but no sourceHandle gets sourceHandle backfilled."""
        nodes = [
            {"id": "py1", "type": "python_exec", "data": {}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1", "targetHandle": "handle-custom-target"},
        ]
        _assign_tool_handles(nodes, edges)

        # targetHandle must be preserved
        assert edges[0]['targetHandle'] == 'handle-custom-target'
        # sourceHandle must be backfilled even though targetHandle was explicit
        assert edges[0]['sourceHandle'] == 'handle-tool-definition'

    def test_sourceHandle_backfilled_when_targetHandle_explicit_python_exec_custom(self):
        """python_exec with explicit targetHandle + custom handles.output gets matching sourceHandle."""
        nodes = [
            {"id": "py1", "type": "python_exec",
             "data": {"handles": {"output": "handle-custom-exec"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1", "targetHandle": "handle-custom-target"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-custom-target'
        assert edges[0]['sourceHandle'] == 'handle-custom-exec'

    def test_sourceHandle_preserved_when_both_handles_explicit_python_exec(self):
        """python_exec edge with both handles explicit: neither is overwritten."""
        nodes = [
            {"id": "py1", "type": "python_exec", "data": {}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1",
             "targetHandle": "handle-custom-target", "sourceHandle": "handle-custom-source"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-custom-target'
        assert edges[0]['sourceHandle'] == 'handle-custom-source'

    # ─── Node-mode python_exec: skip tool handle assignment ──────────────

    def test_node_mode_python_exec_skips_tool_handle_assignment(self):
        """Node-mode python_exec (data.code set) edge is NOT modified by _assign_tool_handles."""
        nodes = [
            {"id": "py1", "type": "python_exec",
             "data": {"code": "def run(handler): return handler"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1",
             "sourceHandle": "custom-source", "targetHandle": "custom-target"},
        ]
        _assign_tool_handles(nodes, edges)

        # Neither sourceHandle nor targetHandle should be modified
        assert edges[0]['sourceHandle'] == 'custom-source'
        assert edges[0]['targetHandle'] == 'custom-target'

    def test_node_mode_python_exec_no_handles_not_modified(self):
        """Node-mode python_exec with no explicit handles: NOT auto-assigned."""
        nodes = [
            {"id": "py1", "type": "python_exec",
             "data": {"code": "def run(handler): return handler"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "py1", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        # No sourceHandle auto-assigned, no targetHandle auto-assigned
        assert 'sourceHandle' not in edges[0] or edges[0].get('sourceHandle') is None
        assert 'targetHandle' not in edges[0] or edges[0].get('targetHandle') is None

    def test_tool_mode_python_exec_still_gets_handles(self):
        """Tool-mode python_exec (no code) still gets tool handles assigned."""
        nodes = [
            {"id": "py1", "type": "python_exec", "data": {}},
            {"id": "py2", "type": "python_exec",
             "data": {"code": "def run(handler): return handler"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            # Tool-mode edge
            {"id": "e1", "source": "py1", "target": "llm-1"},
            # Node-mode edge
            {"id": "e2", "source": "py2", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        # Tool-mode edge gets handles
        assert edges[0]['sourceHandle'] == 'handle-tool-definition'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

        # Node-mode edge is NOT modified
        assert 'sourceHandle' not in edges[1] or edges[1].get('sourceHandle') is None
        assert 'targetHandle' not in edges[1] or edges[1].get('targetHandle') is None
