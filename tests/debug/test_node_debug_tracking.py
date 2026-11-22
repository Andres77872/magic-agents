"""Legacy per-node debug tracking (``Node._start_debug_tracking`` / ``_end_debug_tracking``).

Bypass tracking (``mark_bypassed``) is covered in tests/nodes/test_node_base.py.
"""

from datetime import datetime

from magic_agents.models.debug_feedback import NodeDebugInfo
from magic_agents.node_system.Node import Node


class _SampleNode(Node):
    async def process(self, chat_log):
        yield self.yield_static("done")


def test_start_tracking_snapshots_identity_inputs_and_running_state():
    node = _SampleNode(node_id="node-1", node_type="sample", debug=True)
    node.inputs = {"handle_in": "value"}

    info = node._start_debug_tracking()

    assert isinstance(info, NodeDebugInfo) and info == node._debug_info
    assert (info.node_id, info.node_type, info.node_class) == ("node-1", "sample", "_SampleNode")
    assert info.inputs == {"handle_in": "value"}
    assert info.is_running is True
    assert info.was_executed is False and info.was_bypassed is False
    assert datetime.fromisoformat(info.start_time) and info.end_time is None


def test_end_tracking_marks_success_or_error():
    ok = _SampleNode(node_id="ok", node_type="sample", debug=True)
    ok._start_debug_tracking()
    ok._end_debug_tracking(error=None)

    failed = _SampleNode(node_id="failed", node_type="sample", debug=True)
    failed._start_debug_tracking()
    failed._end_debug_tracking(error="Connection timeout after 30s")

    assert (ok._debug_info.is_running, ok._debug_info.was_executed, ok._debug_info.error) == (False, True, None)
    assert ok._debug_info.end_time is not None
    assert failed._debug_info.is_running is False
    assert failed._debug_info.error == "Connection timeout after 30s"


def test_tracking_is_disabled_without_debug():
    node = _SampleNode(node_id="quiet", node_type="sample", debug=False)

    assert node._start_debug_tracking() is None
    assert node._debug_info is None
