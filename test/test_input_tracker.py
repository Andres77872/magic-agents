"""
Unit tests for NodeInputTracker.

Tests cover:
- Input readiness detection
- Bypass propagation
- Mixed bypass/real input states
- Reset for loop iteration
- Timeout behavior
"""
import asyncio
import pytest

from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo


class TestInputTrackerBasic:
    """Basic input tracking tests."""

    def test_tracker_no_inputs_is_ready(self):
        """Node with no inputs is immediately ready."""
        tracker = NodeInputTracker(node_id="test")
        assert tracker.is_ready is True
        assert tracker.should_execute is True
        assert tracker.is_bypassed is False

    @pytest.mark.asyncio
    async def test_tracker_single_input_received(self):
        """receive_input → is_ready=True, should_execute=True."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="src", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        assert tracker.is_ready is False

        result = await tracker.receive_input("in1", "hello")
        assert result is True  # Made ready
        assert tracker.is_ready is True
        assert tracker.should_execute is True
        assert tracker.is_bypassed is False
        assert tracker.received_handles == {"in1"}

    @pytest.mark.asyncio
    async def test_tracker_single_input_bypassed(self):
        """receive_bypass → is_ready=True, is_bypassed=True, should_execute=False."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="src", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        result = await tracker.receive_bypass("in1")
        assert result is True  # Made ready
        assert tracker.is_ready is True
        assert tracker.is_bypassed is True
        assert tracker.should_execute is False
        assert tracker.bypassed_handles == {"in1"}


class TestInputTrackerMultipleInputs:
    """Tests with multiple expected inputs."""

    @pytest.mark.asyncio
    async def test_tracker_multiple_inputs_all_received(self):
        """All inputs received → ready, should execute."""
        inputs = [
            InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1"),
            InputInfo(edge_id="e2", handle="in2", source_node="b", source_handle="out2"),
        ]
        tracker = NodeInputTracker(node_id="test", expected_inputs=inputs)

        await tracker.receive_input("in1", "data1")
        assert tracker.is_ready is False  # Still waiting for in2

        await tracker.receive_input("in2", "data2")
        assert tracker.is_ready is True
        assert tracker.should_execute is True
        assert tracker.is_bypassed is False

    @pytest.mark.asyncio
    async def test_tracker_multiple_inputs_mixed(self):
        """Some received, some bypassed → should_execute=True."""
        inputs = [
            InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1"),
            InputInfo(edge_id="e2", handle="in2", source_node="b", source_handle="out2"),
        ]
        tracker = NodeInputTracker(node_id="test", expected_inputs=inputs)

        await tracker.receive_input("in1", "data1")
        await tracker.receive_bypass("in2")

        assert tracker.is_ready is True
        assert tracker.should_execute is True  # At least one real input
        assert tracker.is_bypassed is False  # Not ALL bypassed
        assert "in1" in tracker.received_handles
        assert "in2" in tracker.bypassed_handles

    @pytest.mark.asyncio
    async def test_tracker_multiple_inputs_all_bypassed(self):
        """All inputs bypassed → is_bypassed=True."""
        inputs = [
            InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1"),
            InputInfo(edge_id="e2", handle="in2", source_node="b", source_handle="out2"),
        ]
        tracker = NodeInputTracker(node_id="test", expected_inputs=inputs)

        await tracker.receive_bypass("in1")
        await tracker.receive_bypass("in2")

        assert tracker.is_ready is True
        assert tracker.is_bypassed is True
        assert tracker.should_execute is False


class TestInputTrackerEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_tracker_unexpected_input_logged(self):
        """Unexpected input returns False, doesn't count."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        result = await tracker.receive_input("unexpected_handle", "data")
        assert result is False
        assert tracker.is_ready is False  # Still waiting for in1

    @pytest.mark.asyncio
    async def test_tracker_reset_clears_state(self):
        """After reset, all inputs unaccounted."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        await tracker.receive_input("in1", "data")
        assert tracker.is_ready is True

        tracker.reset()
        assert tracker.is_ready is False
        assert tracker.received_handles == set()
        assert tracker.bypassed_handles == set()

    @pytest.mark.asyncio
    async def test_tracker_wait_ready_timeout(self):
        """asyncio.TimeoutError raised when timeout reached."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        with pytest.raises(asyncio.TimeoutError):
            await tracker.wait_ready(timeout=0.01)

    @pytest.mark.asyncio
    async def test_tracker_wait_ready_succeeds(self):
        """wait_ready returns True when input arrives."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        async def provide_input():
            await asyncio.sleep(0.01)
            await tracker.receive_input("in1", "data")

        asyncio.create_task(provide_input())
        result = await tracker.wait_ready(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_tracker_get_input(self):
        """get_input returns stored content."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        await tracker.receive_input("in1", {"key": "value"})
        assert tracker.get_input("in1") == {"key": "value"}
        assert tracker.get_input("nonexistent") is None

    @pytest.mark.asyncio
    async def test_tracker_get_all_inputs(self):
        """get_all_inputs returns all received inputs."""
        inputs = [
            InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1"),
            InputInfo(edge_id="e2", handle="in2", source_node="b", source_handle="out2"),
        ]
        tracker = NodeInputTracker(node_id="test", expected_inputs=inputs)

        await tracker.receive_input("in1", "data1")
        await tracker.receive_input("in2", "data2")
        all_inputs = tracker.get_all_inputs()
        assert all_inputs == {"in1": "data1", "in2": "data2"}

    @pytest.mark.asyncio
    async def test_tracker_bulk_bypass(self):
        """receive_bypass(None) bypasses all remaining handles."""
        inputs = [
            InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1"),
            InputInfo(edge_id="e2", handle="in2", source_node="b", source_handle="out2"),
            InputInfo(edge_id="e3", handle="in3", source_node="c", source_handle="out3"),
        ]
        tracker = NodeInputTracker(node_id="test", expected_inputs=inputs)

        await tracker.receive_input("in1", "data")
        await tracker.receive_bypass()  # Bulk bypass remaining

        assert tracker.is_ready is True
        assert "in1" in tracker.received_handles
        assert "in2" in tracker.bypassed_handles
        assert "in3" in tracker.bypassed_handles
        assert tracker.should_execute is True  # Has real input

    @pytest.mark.asyncio
    async def test_tracker_bypass_does_not_override_received(self):
        """Bypass doesn't override an already received input."""
        info = InputInfo(edge_id="e1", handle="in1", source_node="a", source_handle="out1")
        tracker = NodeInputTracker(node_id="test", expected_inputs=[info])

        await tracker.receive_input("in1", "data")
        await tracker.receive_bypass("in1")  # Should not override

        assert tracker.received_handles == {"in1"}
        assert tracker.bypassed_handles == set()
        assert tracker.should_execute is True
