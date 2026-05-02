"""
Integration tests for the full tool path:
  graph build → propagation → NodeLLM._collect_tools

Tests cover:
- Slice 4: End-to-end tool_mode fetch → LLM tool collection
- Slice 5: Custom fetch output handle propagation
- Slice 6: Multiple tool inputs to single LLM node
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from magic_agents.agt_flow import build, _assign_tool_handles
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.node_system.NodeFetch import FetchToolCallable
from magic_agents.node_system.NodeLLM import NodeLLM


def _make_mock_llm_node(node_id: str = "llm-1"):
    """Create a minimal mock LLM node with inputs dict."""
    class MockLLM:
        def __init__(self):
            self.node_id = node_id
            self.inputs = {}
            self.outputs = {}
            self._response = None
            self.generated = ""
            self.INPUT_TOOL_PREFIX = NodeLLM.DEFAULT_INPUT_TOOL_PREFIX
        def mark_bypassed(self):
            pass
        def _collect_tools(self):
            """Use the real _collect_tools logic."""
            return NodeLLM._collect_tools(self)
    return MockLLM()


def _make_mock_fetch_node(node_id: str = "fetch-1", tool_mode: bool = True,
                          output_handle: str = "handle_fetch_output"):
    """Create a minimal mock fetch node that yields a FetchToolCallable."""
    class MockFetch:
        def __init__(self):
            self.node_id = node_id
            self.inputs = {}
            self._response = None
            self.generated = ""
            self._tool_mode = tool_mode
            self._output_handle = output_handle
            self._outputs = {}
        def mark_bypassed(self):
            pass
        @property
        def outputs(self):
            if self._tool_mode:
                callable_tool = FetchToolCallable(
                    url_template="https://api.example.com/search?q={{query}}",
                    tool_name="search_api",
                )
                return {self._output_handle: {"content": callable_tool}}
            return self._outputs
    return MockFetch()


class TestEndToEndToolCollection:
    """Slice 4: Full path from graph build to NodeLLM._collect_tools."""

    def test_assign_tool_handles_produces_correct_edges(self):
        """_assign_tool_handles produces edges with both sourceHandle and targetHandle."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [{"id": "e1", "source": "fetch-1", "target": "llm-1"}]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle_fetch_output'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'

    def test_propagation_populates_llm_inputs(self):
        """Dispatcher propagation correctly populates LLM node inputs from fetch output."""
        nodes = {"fetch-1": _make_mock_fetch_node(), "llm-1": _make_mock_llm_node()}
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle="handle_fetch_output",
                targetHandle="handle-tool-definition-0"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            outputs = nodes["fetch-1"].outputs
            await dispatcher.propagate_outputs("fetch-1", outputs)
            assert "handle-tool-definition-0" in nodes["llm-1"].inputs

        asyncio.get_event_loop().run_until_complete(_test())

    def test_collect_tools_returns_schema_and_callable(self):
        """After propagation, _collect_tools returns tool schema and callable."""
        nodes = {"fetch-1": _make_mock_fetch_node(), "llm-1": _make_mock_llm_node()}
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle="handle_fetch_output",
                targetHandle="handle-tool-definition-0"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            outputs = nodes["fetch-1"].outputs
            await dispatcher.propagate_outputs("fetch-1", outputs)

            tools_schemas, tool_functions = nodes["llm-1"]._collect_tools()

            assert len(tools_schemas) == 1
            assert tools_schemas[0]["type"] == "function"
            assert tools_schemas[0]["function"]["name"] == "search_api"
            assert "search_api" in tool_functions
            assert isinstance(tool_functions["search_api"], FetchToolCallable)

        asyncio.get_event_loop().run_until_complete(_test())


class TestCustomFetchOutputHandle:
    """Slice 5: Custom fetch output handle propagation."""

    def test_custom_output_handle_in_edge(self):
        """_assign_tool_handles uses custom data.handles.output for sourceHandle."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com",
                      "handles": {"output": "handle-custom-fetch"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [{"id": "e1", "source": "fetch-1", "target": "llm-1"}]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-custom-fetch'

    def test_custom_output_handle_propagation(self):
        """Custom output handle propagates correctly through dispatcher."""
        class MockFetchCustom:
            def __init__(self):
                self.node_id = "fetch-1"
                self.inputs = {}
                self._response = None
                self.generated = ""
            def mark_bypassed(self):
                pass
            @property
            def outputs(self):
                callable_tool = FetchToolCallable(
                    url_template="https://api.example.com/data",
                    tool_name="custom_fetch",
                )
                return {"handle-custom-fetch": {"content": callable_tool}}

        nodes = {"fetch-1": MockFetchCustom(), "llm-1": _make_mock_llm_node()}
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle="handle-custom-fetch",
                targetHandle="handle-tool-definition-0"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            outputs = nodes["fetch-1"].outputs
            await dispatcher.propagate_outputs("fetch-1", outputs)

            tools_schemas, tool_functions = nodes["llm-1"]._collect_tools()

            assert len(tools_schemas) == 1
            assert "custom_fetch" in tool_functions
            assert isinstance(tool_functions["custom_fetch"], FetchToolCallable)

        asyncio.get_event_loop().run_until_complete(_test())


class TestMultipleToolInputs:
    """Slice 6: Multiple tool inputs to single LLM node."""

    def test_multiple_fetch_tools_assign_unique_handles(self):
        """Two tool_mode fetch nodes → one LLM get unique targetHandles and correct sourceHandles."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com/tool1"}},
            {"id": "fetch-2", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com/tool2"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "fetch-1", "target": "llm-1"},
            {"id": "e2", "source": "fetch-2", "target": "llm-1"},
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'
        assert edges[0]['sourceHandle'] == 'handle_fetch_output'
        assert edges[1]['targetHandle'] == 'handle-tool-definition-1'
        assert edges[1]['sourceHandle'] == 'handle_fetch_output'

    def test_multiple_tools_collected_by_llm(self):
        """After propagation, _collect_tools returns both tool schemas and callables."""
        class MockFetch1:
            def __init__(self):
                self.node_id = "fetch-1"
                self.inputs = {}
                self._response = None
                self.generated = ""
            def mark_bypassed(self):
                pass
            @property
            def outputs(self):
                return {"handle_fetch_output": {"content": FetchToolCallable(
                    url_template="https://api.example.com/search?q={{query}}",
                    tool_name="search_api",
                )}}

        class MockFetch2:
            def __init__(self):
                self.node_id = "fetch-2"
                self.inputs = {}
                self._response = None
                self.generated = ""
            def mark_bypassed(self):
                pass
            @property
            def outputs(self):
                return {"handle_fetch_output": {"content": FetchToolCallable(
                    url_template="https://api.example.com/weather?city={{city}}",
                    tool_name="weather_api",
                )}}

        nodes = {
            "fetch-1": MockFetch1(),
            "fetch-2": MockFetch2(),
            "llm-1": _make_mock_llm_node(),
        }
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle="handle_fetch_output",
                targetHandle="handle-tool-definition-0"
            ),
            EdgeNodeModel(
                id="e2", source="fetch-2", target="llm-1",
                sourceHandle="handle_fetch_output",
                targetHandle="handle-tool-definition-1"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Propagate both fetch outputs
            await dispatcher.propagate_outputs("fetch-1", nodes["fetch-1"].outputs)
            await dispatcher.propagate_outputs("fetch-2", nodes["fetch-2"].outputs)

            tools_schemas, tool_functions = nodes["llm-1"]._collect_tools()

            assert len(tools_schemas) == 2
            assert len(tool_functions) == 2
            assert "search_api" in tool_functions
            assert "weather_api" in tool_functions
            assert isinstance(tool_functions["search_api"], FetchToolCallable)
            assert isinstance(tool_functions["weather_api"], FetchToolCallable)

        asyncio.get_event_loop().run_until_complete(_test())

    def test_mixed_fetch_and_python_exec_tools(self):
        """A fetch tool and a python_exec tool both collected by the same LLM."""
        from magic_llm.util.python_executor import PythonExecutor

        class MockFetch:
            def __init__(self):
                self.node_id = "fetch-1"
                self.inputs = {}
                self._response = None
                self.generated = ""
            def mark_bypassed(self):
                pass
            @property
            def outputs(self):
                return {"handle_fetch_output": {"content": FetchToolCallable(
                    url_template="https://api.example.com/search?q={{query}}",
                    tool_name="search_api",
                )}}

        class MockPythonExec:
            def __init__(self):
                self.node_id = "py-1"
                self.inputs = {}
                self._response = None
                self.generated = ""
                self.executor = PythonExecutor(safety_mode='subprocess')
            def mark_bypassed(self):
                pass
            @property
            def outputs(self):
                return {"handle-tool-definition": {"content": self.executor}}

        nodes = {
            "fetch-1": MockFetch(),
            "py-1": MockPythonExec(),
            "llm-1": _make_mock_llm_node(),
        }
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle="handle_fetch_output",
                targetHandle="handle-tool-definition-0"
            ),
            EdgeNodeModel(
                id="e2", source="py-1", target="llm-1",
                sourceHandle="handle-tool-definition",
                targetHandle="handle-tool-definition-1"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            await dispatcher.propagate_outputs("fetch-1", nodes["fetch-1"].outputs)
            await dispatcher.propagate_outputs("py-1", nodes["py-1"].outputs)

            tools_schemas, tool_functions = nodes["llm-1"]._collect_tools()

            assert len(tools_schemas) == 2
            assert len(tool_functions) == 2
            assert "search_api" in tool_functions
            assert "execute_python" in tool_functions

        asyncio.get_event_loop().run_until_complete(_test())


class TestAssignToolHandlesOverwritesWrongValues:
    """Regression: _assign_tool_handles MUST overwrite wrong frontend sourceHandle values."""

    def test_overwrites_wrong_fetch_sourcehandle(self):
        """Backend normalization is authoritative: wrong JSON sourceHandle is overwritten."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "fetch-1", "target": "llm-1",
             "sourceHandle": "handle-tool-definition"}  # WRONG! NodeFetch outputs on handle_fetch_output
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle_fetch_output'

    def test_preserves_correct_custom_fetch_sourcehandle(self):
        """When fetch node has custom handles.output, sourceHandle uses that value."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com",
                      "handles": {"output": "my-custom-handle"}}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [{"id": "e1", "source": "fetch-1", "target": "llm-1",
                  "sourceHandle": "wrong-handle"}]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'my-custom-handle'

    def test_python_exec_sourcehandle_is_handle_tool_definition(self):
        """Python_exec edges get sourceHandle=handle-tool-definition (correct default)."""
        nodes = [
            {"id": "py-1", "type": ModelAgentFlowTypesModel.PYTHON_EXEC,
             "data": {}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [{"id": "e1", "source": "py-1", "target": "llm-1",
                  "sourceHandle": "wrong-handle"}]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle-tool-definition'

    def test_overwrites_even_when_json_has_handle_tool_definition_for_fetch(self):
        """Critical regression: JSON may have handle-tool-definition for fetch, must be overwritten."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "fetch-2", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges = [
            {"id": "e1", "source": "fetch-1", "target": "llm-1",
             "sourceHandle": "handle-tool-definition"},  # WRONG - deep_research.json bug
            {"id": "e2", "source": "fetch-2", "target": "llm-1",
             "sourceHandle": "handle-tool-definition"},  # WRONG
        ]
        _assign_tool_handles(nodes, edges)

        assert edges[0]['sourceHandle'] == 'handle_fetch_output'
        assert edges[1]['sourceHandle'] == 'handle_fetch_output'
        assert edges[0]['targetHandle'] == 'handle-tool-definition-0'
        assert edges[1]['targetHandle'] == 'handle-tool-definition-1'

    def test_propagation_works_after_overwrite(self):
        """After _assign_tool_handles overwrites wrong sourceHandle, propagation succeeds."""
        nodes = [
            {"id": "fetch-1", "type": ModelAgentFlowTypesModel.FETCH,
             "data": {"tool_mode": True, "url": "https://api.example.com"}},
            {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM},
        ]
        edges_dict = [
            {"id": "e1", "source": "fetch-1", "target": "llm-1",
             "sourceHandle": "handle-tool-definition"}  # WRONG
        ]
        _assign_tool_handles(nodes, edges_dict)

        edges = [EdgeNodeModel(**edges_dict[0])]
        mock_nodes = {"fetch-1": _make_mock_fetch_node(), "llm-1": _make_mock_llm_node()}
        dispatcher = GraphEventDispatcher(mock_nodes, edges)

        async def _test():
            outputs = mock_nodes["fetch-1"].outputs
            await dispatcher.propagate_outputs("fetch-1", outputs)

            assert "handle-tool-definition-0" in mock_nodes["llm-1"].inputs

        asyncio.get_event_loop().run_until_complete(_test())


# ─── PythonExecToolWrapper: Dual-param tool schema ───────────────────


class TestPythonExecToolWrapper:
    """Tests for PythonExecToolWrapper dual-param tool schema."""

    def test_tool_schema_has_code_and_handler_params(self):
        """PythonExecToolWrapper.tool_schema contains both code and handler params."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        wrapper = PythonExecToolWrapper(executor=MagicMock())

        schema = wrapper.tool_schema
        params = schema['function']['parameters']['properties']

        assert 'code' in params, "Schema must have 'code' parameter"
        assert params['code']['type'] == 'string'
        assert 'handler' in params, "Schema must have 'handler' parameter"
        assert params['handler']['type'] == 'object'

    def test_tool_schema_description_contains_run_handler(self):
        """Tool schema description references run(handler) contract."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        wrapper = PythonExecToolWrapper(executor=MagicMock())

        description = wrapper.tool_schema['function']['description']
        assert 'run(handler)' in description

    @pytest.mark.asyncio
    async def test_call_with_code_legacy_path(self):
        """__call__(code='...') delegates to PythonExecutor (legacy path)."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper

        executor = AsyncMock()
        executor.side_effect = lambda code="", **kw: f"executed: {code}"
        wrapper = PythonExecToolWrapper(executor=executor)

        result = await wrapper(code="print('hello')")

        executor.assert_called_once_with(code="print('hello')")

    @pytest.mark.asyncio
    async def test_call_with_handler_uses_code_runner(self):
        """__call__(handler={'x': 1}) uses CodeRunner for execution."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        from magic_agents.node_system.python_code_runner import CodeRunner

        executor = MagicMock()
        code_runner = CodeRunner()
        wrapper = PythonExecToolWrapper(
            executor=executor,
            code_runner=code_runner,
            node_code="def run(handler): return handler['x'] * 2",
        )

        result = await wrapper(handler={"x": 5})

        import json
        parsed = json.loads(result)
        assert parsed == {"result": 10}

    @pytest.mark.asyncio
    async def test_call_both_code_and_handler_prefers_handler(self):
        """When both code and handler provided, handler takes precedence and warning logged."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        from magic_agents.node_system.python_code_runner import CodeRunner

        executor = MagicMock()
        code_runner = CodeRunner()
        wrapper = PythonExecToolWrapper(
            executor=executor,
            code_runner=code_runner,
            node_id='test-node',
        )

        with patch('magic_agents.node_system.NodePythonExec.logger') as mock_logger:
            result = await wrapper(code="print('ignored')", handler={"value": "test"})

            # Warning should be logged
            mock_logger.warning.assert_called_once()
            assert 'preferring handler' in mock_logger.warning.call_args[0][0]

        import json
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_tool_callable_property(self):
        """tool_callable property returns self for tool_functions registration."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        wrapper = PythonExecToolWrapper(executor=MagicMock())
        assert wrapper.tool_callable is wrapper

    def test_name_property(self):
        """__name__ returns 'execute_python' for tool registration."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        wrapper = PythonExecToolWrapper(executor=MagicMock())
        assert wrapper.__name__ == 'execute_python'
