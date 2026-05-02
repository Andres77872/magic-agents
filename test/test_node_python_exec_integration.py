"""
Integration tests for NodePythonExec dual-mode process() execution.

Tests cover:
- Node mode execution with handler dict construction and result output
- Error handling: runtime exceptions, syntax errors, missing run, timeout
- None result handling
- Custom output handle configuration
- In-process safety warning
- Tool mode unchanged (backward compat)

These tests call process() directly on NodePythonExec instances with
mocked dependencies to avoid full graph execution.
NOTE: process() is wrapped by @magic_telemetry which adds NODE_START
and NODE_END content events. We filter for the actual result events.
"""
import asyncio
import logging
import pytest
from unittest.mock import patch, MagicMock

from magic_agents.models.factory.Nodes import PythonExecNodeModel
from magic_agents.node_system.NodePythonExec import NodePythonExec


def _find_result(results, expected_type):
    """Find the first result item with the given type (ignoring telemetry events)."""
    for item in results:
        if item.get('type') == expected_type:
            return item
    return None


@pytest.mark.asyncio
class TestNodeModeProcess:
    """Tests for NodePythonExec.process() in node mode."""

    async def _run_process(self, node, chat_log=None):
        """Run node.process() and collect all yielded items."""
        results = []
        async for item in node.process(chat_log or MagicMock()):
            results.append(item)
        return results

    async def test_node_mode_basic_execution(self):
        """Node mode executes user code and yields result via handle-python_exec-result."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): return {'sum': handler['a'] + handler['b']}",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'a': 3, 'b': 5})

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None, f"No handle-python_exec-result found in {results}"
        content = item.get('content', {})
        assert content.get('content') == {'sum': 8}

    async def test_node_mode_custom_output_handle(self):
        """Node mode with custom handles.output uses configured handle."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): return {'ok': True}",
                ),
                handles={"output": "my-custom-handle"},
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        results = await self._run_process(node)
        item = _find_result(results, 'my-custom-handle')

        assert item is not None, f"No my-custom-handle found in {results}"

    async def test_node_mode_handler_excludes_config(self):
        """Handler dict excludes config handles (safety_mode, timeout, max_output_chars)."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): return list(handler.keys())",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({
            'handle-python_exec-safety_mode': 'subprocess',
            'handle-python_exec-timeout': 30.0,
            'handle-python_exec-max_output_chars': 8000,
            'user_query': 'hello',
            'threshold': 0.5,
        })

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None
        content = item.get('content', {})
        keys = content.get('content', [])
        assert 'user_query' in keys
        assert 'threshold' in keys
        assert 'handle-python_exec-safety_mode' not in keys

    async def test_node_mode_exception_returns_error(self):
        """Runtime exception in user code yields error via process()."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): raise ValueError('bad input')",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None
        content = item.get('content', {})
        err = content.get('content', {})
        # ValueError is not available in restricted builtins, so the actual error
        # will be a NameError. The important thing is an error is caught and returned.
        assert 'error' in err

    async def test_node_mode_none_return(self):
        """Node mode with None return does NOT produce error."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): return None",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None
        content = item.get('content', {})
        # None should be yielded as-is (not wrapped in error)
        assert content.get('content') is None

    async def test_node_mode_syntax_error(self):
        """Syntax error in user code yields error dict."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): return 1/",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None
        content = item.get('content', {})
        err = content.get('content', {})
        assert 'error' in err
        assert 'Syntax error' in err['error']

    async def test_node_mode_missing_run(self):
        """Missing run() yields error dict."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="x = 1",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None
        content = item.get('content', {})
        err = content.get('content', {})
        assert 'error' in err
        assert 'must define' in err['error']

    async def test_node_mode_timeout(self):
        """Timeout enforcement yields timeout error via process()."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=0.01,
                    max_output_chars=8000,
                    code="def run(handler):\n"
                         "    total = 0\n"
                         "    for _ in range(10**8):\n"
                         "        total += 1\n"
                         "    return total",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        results = await self._run_process(node)
        item = _find_result(results, 'handle-python_exec-result')

        assert item is not None
        content = item.get('content', {})
        err = content.get('content', {})
        assert 'error' in err
        assert 'timed out' in err['error']

    async def test_node_mode_in_process_warning(self):
        """safety_mode='in_process' in node mode logs warning."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='in_process',
                    timeout=10.0,
                    max_output_chars=8000,
                    code="def run(handler): return {'ok': True}",
                ),
                node_id='py-1',
                debug=False,
            )
        node.inputs.update({'data': 42})

        with patch('magic_agents.node_system.NodePythonExec.logger') as mock_logger:
            results = await self._run_process(node)

        # Warning should be logged (from _log_in_process_warning in node mode branch)
        assert mock_logger.warning.call_count >= 1
        args_combined = ' '.join(str(a) for call in mock_logger.warning.call_args_list for a in call[0])
        assert 'in_process' in args_combined

        # Execution should still succeed
        item = _find_result(results, 'handle-python_exec-result')
        assert item is not None


@pytest.mark.asyncio
class TestToolModeProcess:
    """Tests for NodePythonExec.process() in tool mode (unchanged behavior)."""

    async def test_tool_mode_yields_executor(self):
        """Tool mode yields wrapped executor via handle-tool-definition."""
        with patch('magic_llm.util.python_executor.PythonExecutor'):
            node = NodePythonExec(
                data=PythonExecNodeModel(
                    safety_mode='subprocess',
                    timeout=30.0,
                    max_output_chars=8000,
                ),
                node_id='py-1',
                debug=False,
            )

        results = []
        async for item in node.process(MagicMock()):
            results.append(item)

        item = _find_result(results, 'handle-tool-definition')
        assert item is not None
        content = item.get('content', {})
        wrapped = content.get('content')
        from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
        assert isinstance(wrapped, PythonExecToolWrapper)
