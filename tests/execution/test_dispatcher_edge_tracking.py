"""
Dispatcher-level tests for edge.id tracking and runtime propagation.

These tests verify the core runtime behavior of edge.id tracking:
1. GraphEventDispatcher builds trackers keyed by edge.id
2. Edge.id propagates from build() through dispatcher
3. Runtime propagation (dispatch_input, dispatch_bypass) uses edge.id correctly
4. Fan-in scenarios tracked per edge, not collapsed by handle

CRITICAL: These tests exercise the ACTUAL dispatcher, not mocked behavior.
"""
import asyncio

from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.agt_flow import build


class TestDispatcherEdgeIdTracking:
    """Tests for dispatcher's edge.id tracking behavior."""

    def test_dispatcher_builds_trackers_keyed_by_edge_id(self):
        """
        Dispatcher._build_trackers() must create trackers keyed by edge.id.
        
        This is the critical fix - each edge gets its own entry in the tracker,
        not collapsed by targetHandle.
        """
        # Create minimal nodes
        nodes = {
            'source': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'target': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
        }
        
        # Create edges with explicit IDs
        edges = [
            EdgeNodeModel(id='edge_alpha', source='source', target='target', targetHandle='handle_flow_input'),
            EdgeNodeModel(id='edge_beta', source='source', target='target', targetHandle='handle_flow_input'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        
        # Get tracker for target node
        tracker = dispatcher.get_tracker('target')
        
        # Tracker must have TWO entries (one per edge), not one (collapsed by handle)
        assert tracker is not None
        assert len(tracker._expected_inputs) == 2
        
        # Entries keyed by edge_id
        assert 'edge_alpha' in tracker._expected_inputs
        assert 'edge_beta' in tracker._expected_inputs
        
        # Both have same handle but different edge_id
        assert tracker._expected_inputs['edge_alpha'].handle == 'handle_flow_input'
        assert tracker._expected_inputs['edge_beta'].handle == 'handle_flow_input'

    def test_dispatcher_tracker_has_expected_edges_property(self):
        """
        Tracker.expected_edges must expose the set of expected edge_ids.
        """
        nodes = {
            'input': type('MockNode', (), {'node_type': 'user_input', 'inputs': {}})(),
            'text': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
        }
        
        edges = [
            EdgeNodeModel(id='e1', source='input', target='text', targetHandle='h1'),
            EdgeNodeModel(id='e2', source='input', target='text', targetHandle='h2'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        tracker = dispatcher.get_tracker('text')
        
        # Expected edges set matches edge IDs
        assert tracker.expected_edges == {'e1', 'e2'}

    def test_dispatcher_single_edge_becomes_ready_after_receive(self):
        """
        Tracker with single edge becomes ready after receiving that input.
        """
        nodes = {
            'source': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'target': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
        }
        
        edges = [
            EdgeNodeModel(id='e1', source='source', target='target', targetHandle='handle_flow_input'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        tracker = dispatcher.get_tracker('target')
        
        # Dispatch input
        asyncio.run(dispatcher.dispatch_input('target', 'handle_flow_input', 'test_content'))
        
        # Now ready
        assert tracker.is_ready
        
        # Received edges includes e1
        assert 'e1' in tracker.received_edges


class TestDispatcherFanInTracking:
    """Tests for dispatcher handling fan-in scenarios."""

    def test_dispatcher_fan_in_both_edges_required(self):
        """
        Fan-in to same handle requires BOTH edges to be received.
        
        Previously: tracker collapsed entries -> receiving one made ready
        Now: tracker keyed by edge_id -> both must be received
        """
        nodes = {
            'source1': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'source2': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'target': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
        }
        
        edges = [
            EdgeNodeModel(id='e1', source='source1', target='target', targetHandle='handle_flow_input'),
            EdgeNodeModel(id='e2', source='source2', target='target', targetHandle='handle_flow_input'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        tracker = dispatcher.get_tracker('target')
        
        # Initially not ready
        assert not tracker.is_ready
        
        # Receive first edge
        asyncio.run(dispatcher.dispatch_input('target', 'handle_flow_input', 'content1'))
        
        # Still not ready (second edge pending)
        assert not tracker.is_ready
        
        # Receive second edge
        asyncio.run(dispatcher.dispatch_input('target', 'handle_flow_input', 'content2'))
        
        # Now ready (both edges received)
        assert tracker.is_ready
        
        # Both edges in received set
        assert tracker.received_edges == {'e1', 'e2'}

    def test_dispatcher_fan_in_tracked_per_edge(self):
        """
        Fan-in tracking must preserve which edge delivered content.
        """
        nodes = {
            'source1': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'source2': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'target': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
        }
        
        edges = [
            EdgeNodeModel(id='edge_alpha', source='source1', target='target', targetHandle='handle_flow_input'),
            EdgeNodeModel(id='edge_beta', source='source2', target='target', targetHandle='handle_flow_input'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        tracker = dispatcher.get_tracker('target')
        
        # Receive from first edge
        asyncio.run(dispatcher.dispatch_input('target', 'handle_flow_input', 'from_alpha'))
        
        # Can query which edge delivered
        assert tracker.get_input_by_edge('edge_alpha') == 'from_alpha'
        assert tracker.get_input_by_edge('edge_beta') is None
        
        # Receive from second edge
        asyncio.run(dispatcher.dispatch_input('target', 'handle_flow_input', 'from_beta'))
        
        # Both tracked
        assert tracker.get_input_by_edge('edge_alpha') == 'from_alpha'
        assert tracker.get_input_by_edge('edge_beta') == 'from_beta'

    def test_dispatcher_different_handles_separate_entries(self):
        """
        Edges with different targetHandles create separate tracker entries.
        """
        nodes = {
            'source1': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'source2': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'llm': type('MockNode', (), {'node_type': 'llm', 'inputs': {}})(),
        }
        
        edges = [
            EdgeNodeModel(id='e1', source='source1', target='llm', targetHandle='handle_user_message'),
            EdgeNodeModel(id='e2', source='source2', target='llm', targetHandle='handle_client'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        tracker = dispatcher.get_tracker('llm')
        
        # Two separate entries (different handles)
        assert len(tracker._expected_inputs) == 2
        
        # Different handles
        assert tracker._expected_inputs['e1'].handle == 'handle_user_message'
        assert tracker._expected_inputs['e2'].handle == 'handle_client'


class TestDispatcherBypassPropagation:
    """Tests for bypass event propagation."""

    def test_dispatcher_bypass_marks_edge_bypassed(self):
        """
        dispatch_bypass must mark the corresponding edge as bypassed.
        """
        nodes = {
            'source': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
            'target': type('MockNode', (), {'node_type': 'text', 'inputs': {}})(),
        }
        
        edges = [
            EdgeNodeModel(id='e1', source='source', target='target', targetHandle='handle_flow_input'),
        ]
        
        dispatcher = GraphEventDispatcher(nodes=nodes, edges=edges)
        tracker = dispatcher.get_tracker('target')
        
        # Dispatch bypass
        asyncio.run(dispatcher.dispatch_bypass('target', 'handle_flow_input'))
        
        # Tracker should show bypassed
        assert tracker.is_bypassed
        assert len(tracker.bypassed_edges) > 0


class TestDispatcherFromBuild:
    """Tests for dispatcher behavior from actual build() path."""

    def test_build_then_dispatcher_tracker_keyed_by_edge_id(self):
        """
        After build(), dispatcher must use edge.id as tracker key.
        
        This tests the REAL build path, not mock construction.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'text', 'type': 'text', 'data': {'text': 'hello'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e_custom_1', 'source': 'input', 'target': 'text', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'},
                {'id': 'e_custom_2', 'source': 'text', 'target': 'end', 'sourceHandle': 'handle_text_output', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Build dispatcher from graph
        dispatcher = GraphEventDispatcher(nodes=graph.nodes, edges=graph.edges)
        
        # Get tracker for text node
        tracker = dispatcher.get_tracker('text')
        
        # Tracker must have expected edge
        assert 'e_custom_1' in tracker.expected_edges
        
        # Also check end node
        tracker_end = dispatcher.get_tracker('end')
        # Note: end node might have void edge, so we check at least one edge
        assert len(tracker_end._expected_inputs) >= 1

