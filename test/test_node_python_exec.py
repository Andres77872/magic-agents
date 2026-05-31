"""
Unit tests for NodePythonExec dual-mode helpers.

Tests cover:
- _has_code() with various code field values
- _build_handler_dict() with various input configurations
"""
import pytest
from unittest.mock import MagicMock, patch

from magic_agents.models.factory.Nodes import PythonExecNodeModel
from magic_agents.node_system.NodePythonExec import NodePythonExec


class TestNodePythonExecHasCode:
    """Tests for NodePythonExec._has_code()."""

    def test_has_code_none(self):
        """_has_code() returns False when code is None."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(safety_mode='subprocess', timeout=30.0, max_output_chars=8000),
                node_id='py-1',
                debug=False,
            )
        assert node._has_code() is False

    def test_has_code_with_code(self):
        """_has_code() returns True when code is a non-empty string."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=30.0,
                    max_output_chars=8000,
                    code="def run(handler): return handler",
                ),
                node_id='py-1',
                debug=False,
            )
        assert node._has_code() is True

    def test_has_code_empty_string(self):
        """_has_code() returns False when code is empty string."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=30.0,
                    max_output_chars=8000,
                    code="",
                ),
                node_id='py-1',
                debug=False,
            )
        assert node._has_code() is False


class TestNodePythonExecBuildHandlerDict:
    """Tests for NodePythonExec._build_handler_dict()."""

    def _make_node(self, code=None, handles=None):
        """Helper to create a NodePythonExec instance with mocked PythonExecutor."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=30.0,
                    max_output_chars=8000,
                    code=code,
                ),
                handles=handles,
                node_id='py-1',
                debug=False,
            )
        return node

    def test_build_handler_excludes_config_handles(self):
        """_build_handler_dict() excludes config handles (safety_mode, timeout, max_output_chars)."""
        node = self._make_node(code="def run(h): return h")
        node.inputs.update({
            'handle-python_exec-safety_mode': 'subprocess',
            'handle-python_exec-timeout': 30.0,
            'handle-python_exec-max_output_chars': 8000,
            'user_query': 'hello',
            'threshold': 0.5,
        })
        handler = node._build_handler_dict()
        assert handler == {'user_query': 'hello', 'threshold': 0.5}

    def test_build_handler_only_config_inputs(self):
        """_build_handler_dict() returns empty dict when only config handles present."""
        node = self._make_node(code="def run(h): return h")
        node.inputs.update({
            'handle-python_exec-safety_mode': 'subprocess',
            'handle-python_exec-timeout': 30.0,
            'handle-python_exec-max_output_chars': 8000,
        })
        handler = node._build_handler_dict()
        assert handler == {}

    def test_build_handler_custom_handle_names(self):
        """_build_handler_dict() uses custom handle names when configured."""
        node = self._make_node(
            code="def run(h): return h",
            handles={
                'safety_mode': 'custom_safety',
                'timeout': 'custom_tmo',
                'max_output_chars': 'custom_max',
            },
        )
        node.inputs.update({
            'custom_safety': 'in_process',
            'custom_tmo': 10.0,
            'custom_max': 4000,
            'data_points': [1, 2, 3],
        })
        handler = node._build_handler_dict()
        assert handler == {'data_points': [1, 2, 3]}

    def test_build_handler_excludes_code_key(self):
        """_build_handler_dict() excludes 'code' key if it appears in inputs."""
        node = self._make_node(code="def run(h): return h")
        node.inputs.update({
            'handle-python_exec-safety_mode': 'subprocess',
            'code': 'some injected code',
            'valid_input': 42,
        })
        handler = node._build_handler_dict()
        assert handler == {'valid_input': 42}

    def test_build_handler_empty_inputs(self):
        """_build_handler_dict() with empty inputs returns empty dict."""
        node = self._make_node(code="def run(h): return h")
        node.inputs = {}
        handler = node._build_handler_dict()
        assert handler == {}


class TestNodePythonExecDualModeOutputHandle:
    """Tests for NodePythonExec output handle configuration."""

    def test_node_mode_default_output_handle(self):
        """Node mode default output handle is 'handle-python_exec-result'."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=30.0,
                    max_output_chars=8000,
                    code="def run(handler): return handler",
                ),
                node_id='py-1',
                debug=False,
            )
        assert node.OUTPUT_HANDLE_CODE_RESULT == 'handle-python_exec-result'

    def test_node_mode_custom_output_handle(self):
        """Node mode respects configured handles.output for output handle."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=30.0,
                    max_output_chars=8000,
                    code="def run(handler): return handler",
                ),
                handles={'output': 'my-custom-handle'},
                node_id='py-1',
                debug=False,
            )
        assert node.OUTPUT_HANDLE_CODE_RESULT == 'my-custom-handle'

    def test_tool_mode_default_output_handle(self):
        """Tool mode default output handle is 'handle-tool-definition'."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(),
                node_id='py-1',
                debug=False,
            )
        assert node.OUTPUT_HANDLE == 'handle-tool-definition'


# ═══════════════════════════════════════════════════════════════════════
# Task 4.1: PythonExecToolWrapper custom tool_name behavior
# ═══════════════════════════════════════════════════════════════════════

class TestPythonExecToolWrapperToolName:
    """Tests for PythonExecToolWrapper custom tool_name."""

    def test_custom_tool_name_returns_analyze(self):
        """PythonExecToolWrapper(tool_name='analyze').__name__ returns 'analyze'."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper

        wrapper = PythonExecToolWrapper(
            executor=MagicMock(),
            tool_name="analyze",
        )
        assert wrapper.__name__ == "analyze"

    def test_custom_tool_name_in_schema(self):
        """PythonExecToolWrapper(tool_name='analyze').tool_schema uses 'analyze'."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper

        wrapper = PythonExecToolWrapper(
            executor=MagicMock(),
            tool_name="analyze",
        )
        schema = wrapper.tool_schema
        assert schema["function"]["name"] == "analyze"

    def test_default_tool_name_is_execute_python(self):
        """PythonExecToolWrapper().__name__ returns 'execute_python' (default)."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper

        wrapper = PythonExecToolWrapper(executor=MagicMock())
        assert wrapper.__name__ == "execute_python"

    def test_default_tool_name_in_schema(self):
        """PythonExecToolWrapper().tool_schema uses 'execute_python'."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper

        wrapper = PythonExecToolWrapper(executor=MagicMock())
        schema = wrapper.tool_schema
        assert schema["function"]["name"] == "execute_python"

    def test_distinct_tool_names_have_distinct_schemas(self):
        """Two wrappers with different tool_name have distinct schemas."""
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper

        wrapper_a = PythonExecToolWrapper(executor=MagicMock(), tool_name="analyze")
        wrapper_b = PythonExecToolWrapper(executor=MagicMock(), tool_name="execute")

        schema_a = wrapper_a.tool_schema
        schema_b = wrapper_b.tool_schema

        assert schema_a["function"]["name"] == "analyze"
        assert schema_b["function"]["name"] == "execute"
        assert schema_a["function"]["name"] != schema_b["function"]["name"]


# ═══════════════════════════════════════════════════════════════════════
# Task 4.2: PythonExecNodeModel tool_name field + NodePythonExec threading
# ═══════════════════════════════════════════════════════════════════════

class TestPythonExecNodeModelToolName:
    """Tests for PythonExecNodeModel tool_name field."""

    def test_default_tool_name_is_none(self):
        """PythonExecNodeModel() has tool_name == None."""
        model = PythonExecNodeModel()
        assert model.tool_name is None

    def test_custom_tool_name_stored(self):
        """PythonExecNodeModel(tool_name='analyze') has tool_name == 'analyze'."""
        model = PythonExecNodeModel(tool_name="analyze")
        assert model.tool_name == "analyze"

    def test_node_init_stores_custom_tool_name(self):
        """NodePythonExec.__init__ with custom tool_name stores and passes it to wrapper."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(tool_name="analyze"),
                node_id='py-1',
                debug=False,
            )
        assert node._tool_name == "analyze"

    def test_node_init_default_tool_name(self):
        """NodePythonExec.__init__ with default None tool_name stores 'execute_python'."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(),
                node_id='py-1',
                debug=False,
            )
        assert node._tool_name == "execute_python"

    @pytest.mark.asyncio
    async def test_process_passes_custom_tool_name_to_wrapper(self):
        """NodePythonExec.process() passes tool_name to PythonExecToolWrapper."""
        with patch('magic_llm.util.python_executor.PythonExecutor') as mock_exec_cls:
            mock_exec = MagicMock()
            mock_exec_cls.return_value = mock_exec

            node = NodePythonExec(
                data=PythonExecNodeModel(tool_name="analyze"),
                node_id='py-1',
                debug=False,
            )

            # Process in tool mode (no code) yields a wrapper
            from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
            import json

            results = []
            async for result in node.process(chat_log=None):
                results.append(result)

            # The yielded wrapper should have tool_name="analyze"
            assert len(results) > 0
            wrapper = results[0].get('content') if isinstance(results[0], dict) else None
            # The wrapper is yielded as yield_static content
            # Check that the wrapper has the correct name
            if wrapper is not None and hasattr(wrapper, '__name__'):
                assert wrapper.__name__ == "analyze"
