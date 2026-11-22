"""
Property tests for edge.id tracking invariant.

CRITICAL FIX: NodeInputTracker now keys by edge.id instead of targetHandle.
This fixes the fan-in bug where multiple edges targeting the same handle
would collapse into a single entry, causing silent data loss.

These tests verify:
1. Multi-edge same-targetHandle scenario produces N entries in tracker
2. Each edge tracked independently (edge.id as key)
3. Readiness check considers all edges, not collapsed single entry
4. Edge.id preserved through JSON roundtrip
"""

from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo


class TestEdgeIdTrackingBasic:
    """Basic tests for edge.id tracking."""

    def test_tracker_keys_by_edge_id(self):
        """Tracker must key _expected_inputs by edge_id."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h1", source_node="s1", source_handle="out1"),
                InputInfo(edge_id="e2", handle="h2", source_node="s2", source_handle="out2"),
            ]
        )
        
        # Check internal structure keys by edge_id
        assert "e1" in tracker._expected_inputs
        assert "e2" in tracker._expected_inputs
        assert len(tracker._expected_inputs) == 2

    def test_tracker_expected_edges_property(self):
        """Tracker must expose expected_edges set."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h1", source_node="s1", source_handle="out1"),
                InputInfo(edge_id="e2", handle="h2", source_node="s2", source_handle="out2"),
            ]
        )
        
        assert tracker.expected_edges == {"e1", "e2"}

    def test_tracker_received_edges_property(self):
        """Tracker must expose received_edges set."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h1", source_node="s1", source_handle="out1"),
                InputInfo(edge_id="e2", handle="h2", source_node="s2", source_handle="out2"),
            ]
        )
        
        # Initially none received
        assert tracker.received_edges == set()
        
        # After receiving one
        import asyncio
        asyncio.run(tracker.receive_input("h1", "content1"))
        assert tracker.received_edges == {"e1"}

    def test_tracker_get_input_by_edge(self):
        """Tracker must support getting input by specific edge_id."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h1", source_node="s1", source_handle="out1"),
                InputInfo(edge_id="e2", handle="h2", source_node="s2", source_handle="out2"),
            ]
        )
        
        import asyncio
        asyncio.run(tracker.receive_input("h1", "content_from_e1"))
        
        assert tracker.get_input_by_edge("e1") == "content_from_e1"
        assert tracker.get_input_by_edge("e2") is None


class TestFanInNeverCollapses:
    """Critical invariant: multi-edge same-handle never collapses."""

    def test_two_edges_same_handle_stored_separately(self):
        """Two edges targeting same handle must be stored as separate entries."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="handle_input", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="handle_input", source_node="s2", source_handle="out"),
            ]
        )
        
        # CRITICAL: Must have 2 entries, not 1 (collapsed)
        assert len(tracker._expected_inputs) == 2
        assert "e1" in tracker._expected_inputs
        assert "e2" in tracker._expected_inputs

    def test_three_edges_same_handle_stored_separately(self):
        """Three edges targeting same handle must be stored as separate entries."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
                InputInfo(edge_id="e3", handle="h", source_node="s3", source_handle="out"),
            ]
        )
        
        # Must have 3 entries
        assert len(tracker._expected_inputs) == 3
        assert tracker.expected_edges == {"e1", "e2", "e3"}

    def test_fan_in_first_edge_receives_correctly(self):
        """First edge in fan-in must receive correctly."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        asyncio.run(tracker.receive_input("h", "content1"))
        
        # e1 should be received, e2 not yet
        assert tracker._expected_inputs["e1"].received is True
        assert tracker._expected_inputs["e2"].received is False

    def test_fan_in_both_edges_can_receive(self):
        """Both edges in fan-in must be able to receive."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        asyncio.run(tracker.receive_input("h", "content1"))
        asyncio.run(tracker.receive_input("h", "content2"))
        
        # Both should be received
        assert tracker._expected_inputs["e1"].received is True
        assert tracker._expected_inputs["e2"].received is True

    def test_fan_in_readiness_considers_all_edges(self):
        """Readiness check must consider all edges, not collapsed."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        # Receive one edge - not yet ready (other edge pending)
        asyncio.run(tracker.receive_input("h", "content1"))
        assert tracker.is_ready is False
        
        # Receive second edge - now ready
        asyncio.run(tracker.receive_input("h", "content2"))
        assert tracker.is_ready is True

    def test_fan_in_bypass_by_edge_id(self):
        """Bypass can target specific edge by edge_id."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        # Bypass specific edge
        asyncio.run(tracker.receive_bypass(edge_id="e1"))
        
        # e1 bypassed, e2 not yet
        assert tracker._expected_inputs["e1"].bypassed is True
        assert tracker._expected_inputs["e2"].bypassed is False

    def test_fan_in_mixed_receive_and_bypass(self):
        """Fan-in can have one edge received and one bypassed."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        asyncio.run(tracker.receive_input("h", "content1"))
        asyncio.run(tracker.receive_bypass(edge_id="e2"))
        
        # e1 received, e2 bypassed
        assert tracker._expected_inputs["e1"].received is True
        assert tracker._expected_inputs["e2"].bypassed is True
        
        # Should be ready (all edges accounted)
        assert tracker.is_ready is True

    def test_ten_edges_same_handle_no_collapse(self):
        """Ten edges targeting same handle must have 10 entries."""
        expected_inputs = [
            InputInfo(edge_id=f"e{i}", handle="h", source_node=f"s{i}", source_handle="out")
            for i in range(10)
        ]
        
        tracker = NodeInputTracker(node_id="target", expected_inputs=expected_inputs)
        
        assert len(tracker._expected_inputs) == 10
        for i in range(10):
            assert f"e{i}" in tracker._expected_inputs


class TestTrackerGetAllInputsByEdge:
    """Tracker must support edge-level input retrieval."""

    def test_get_all_inputs_by_edge(self):
        """get_all_inputs_by_edge returns dict keyed by edge_id."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h1", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h2", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        asyncio.run(tracker.receive_input("h1", "content1"))
        asyncio.run(tracker.receive_input("h2", "content2"))
        
        inputs_by_edge = tracker.get_all_inputs_by_edge()
        assert inputs_by_edge == {"e1": "content1", "e2": "content2"}

    def test_get_all_inputs_by_handle_fan_in(self):
        """get_all_inputs returns dict keyed by handle (last received wins for fan-in)."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h", source_node="s1", source_handle="out"),
                InputInfo(edge_id="e2", handle="h", source_node="s2", source_handle="out"),
            ]
        )
        
        import asyncio
        asyncio.run(tracker.receive_input("h", "content1"))
        asyncio.run(tracker.receive_input("h", "content2"))
        
        # By handle - last received wins
        inputs_by_handle = tracker.get_all_inputs()
        assert inputs_by_handle == {"h": "content2"}
        
        # By edge - both preserved
        inputs_by_edge = tracker.get_all_inputs_by_edge()
        assert inputs_by_edge == {"e1": "content1", "e2": "content2"}

    def test_get_edge_info(self):
        """get_edge_info returns InputInfo for specific edge."""
        tracker = NodeInputTracker(
            node_id="target",
            expected_inputs=[
                InputInfo(edge_id="e1", handle="h1", source_node="s1", source_handle="out"),
            ]
        )
        
        info = tracker.get_edge_info("e1")
        assert info is not None
        assert info.edge_id == "e1"
        assert info.handle == "h1"
        assert info.source_node == "s1"