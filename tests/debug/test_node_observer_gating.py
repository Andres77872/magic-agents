"""Focused tests: Node.__call__() observer gating for legacy debug tracking.

This proves that when an observer IS provided to Node.__call__(), the legacy
_start_debug_tracking() / _end_debug_tracking() calls are SKIPPED.  This
prevents the ``StreamExecutionError`` caused by NodeDebugInfo Pydantic
validation failing on unusual inputs (e.g. loop sub-nodes with None keys).

When observer is None (no observer), the legacy tracking still runs for
backward compatibility.
"""

import pytest
from unittest.mock import AsyncMock

from magic_agents.node_system.Node import Node
from magic_agents.debug.observer import DebugObserver
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


class _TestNode(Node):
    """Minimal concrete Node subclass for testing."""
    async def process(self, chat_log):
        yield self.yield_static("ok", content_type="end")


class _TestErrorNode(Node):
    """Node that raises during process."""
    async def process(self, chat_log):
        yield self.yield_static("before", content_type="end")
        raise ValueError("test node error")


class TestNodeObserverGating:
    """Prove legacy debug tracking is gated by observer presence."""

    @pytest.mark.asyncio
    async def test_legacy_tracking_called_when_no_observer(self):
        """When observer=None, _start_debug_tracking must be called."""
        node = _TestNode(node_id="t1", node_type="text", debug=True)
        chat_log = ModelAgentRunLog()
        mock_observer = None

        results = []

        async for item in node(chat_log, observer=mock_observer):
            results.append(item)

        # With no observer, legacy tracking SHOULD run -> _debug_info has start_time
        assert node._debug_info is not None, (
            "Legacy tracking should populate _debug_info when observer=None"
        )
        # _start_debug_tracking() sets start_time; _init_debug_info() alone
        # (from __init__) leaves it None
        assert node._debug_info.start_time is not None, (
            "_start_debug_tracking should set start_time when observer=None"
        )

    @pytest.mark.asyncio
    async def test_legacy_tracking_skipped_when_observer_provided(self):
        """When observer is provided, _start_debug_tracking must NOT be called."""
        node = _TestNode(node_id="t2", node_type="text", debug=True)
        chat_log = ModelAgentRunLog()
        mock_observer = AsyncMock(spec=DebugObserver)

        results = []

        async for item in node(chat_log, observer=mock_observer):
            results.append(item)

        # _debug_info may exist from __init__() which calls _init_debug_info()
        # when debug=True, but _start_debug_tracking() must NOT have run,
        # so start_time should remain None.
        assert node._debug_info is not None  # debug=True initializes it
        assert node._debug_info.start_time is None, (
            "_start_debug_tracking should NOT have set start_time when observer is provided"
        )
        # Observer must have been called
        mock_observer.on_node_start.assert_awaited_once()
        mock_observer.on_node_end.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_legacy_tracking_skipped_on_error_with_observer(self):
        """On error path, legacy _end_debug_tracking should not be called."""
        node = _TestErrorNode(node_id="t3", node_type="text", debug=True)
        chat_log = ModelAgentRunLog()
        mock_observer = AsyncMock(spec=DebugObserver)

        results = []

        with pytest.raises(ValueError, match="test node error"):
            async for item in node(chat_log, observer=mock_observer):
                results.append(item)

        # With observer, legacy tracking should be skipped even on error
        assert node._debug_info is not None  # debug=True initializes it
        assert node._debug_info.start_time is None, (
            "_start_debug_tracking should NOT have set start_time on error path"
        )
        mock_observer.on_node_start.assert_awaited_once()
        mock_observer.on_node_error.assert_awaited_once()
        mock_observer.on_node_end.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_tracking_isolation_no_observer_node_debug_false(self):
        """Without observer AND debug=False, legacy tracking should not run."""
        node = _TestNode(node_id="t4", node_type="text", debug=False)
        chat_log = ModelAgentRunLog()

        results = []

        async for item in node(chat_log, observer=None):
            results.append(item)

        # debug=False means _start_debug_tracking returns early, _debug_info stays None
        assert node._debug_info is None


class TestLegacyTrackingFailOpen:
    """Legacy debug bookkeeping must never fail the node it describes."""

    @pytest.mark.asyncio
    async def test_none_input_key_does_not_fail_node(self):
        """A None input key fails NodeDebugInfo validation; the node still runs."""
        node = _TestNode(node_id="t5", node_type="void", debug=True)
        node.inputs[None] = "terminal"

        results = [item async for item in node(ModelAgentRunLog(), observer=None)]

        assert results
        assert node._debug_info.was_executed is True
        assert node._debug_info.error is None

    @pytest.mark.asyncio
    async def test_tracking_exceptions_do_not_fail_successful_node(self, monkeypatch):
        node = _TestNode(node_id="t6", node_type="text", debug=True)

        def broken(**kwargs):
            raise RuntimeError("tracking broke")
        monkeypatch.setattr(node, "_start_debug_tracking", broken)
        monkeypatch.setattr(node, "_end_debug_tracking", broken)

        results = [item async for item in node(ModelAgentRunLog(), observer=None)]

        assert results

    @pytest.mark.asyncio
    async def test_tracking_exception_does_not_mask_process_error(self, monkeypatch):
        node = _TestErrorNode(node_id="t7", node_type="text", debug=True)

        def broken(**kwargs):
            raise RuntimeError("tracking broke")
        monkeypatch.setattr(node, "_end_debug_tracking", broken)

        with pytest.raises(ValueError, match="test node error"):
            async for _ in node(ModelAgentRunLog(), observer=None):
                pass
