"""
Tests for verifying text node → chat node system context flow.

This test suite ensures that:
1. Text node output properly connects to chat node's system context input
2. The flow works regardless of edge/node order in the JSON
3. The reactive executor properly waits for all inputs before executing
"""

import json
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from magic_agents.agt_flow import build, run_agent
from magic_agents.execution.event_dispatcher import GraphEventDispatcher


# Minimal graph with text → chat system context connection
MINIMAL_TEXT_TO_CHAT_GRAPH = {
    "type": "graph",
    "debug": True,
    "content": {
        "nodes": [
            {
                "id": "user_input_node",
                "type": "user_input",
                "position": {"x": 0, "y": 0},
                "data": {}
            },
            {
                "id": "text_node",
                "type": "text",
                "position": {"x": 100, "y": 0},
                "data": {"text": "Always respond in French"}
            },
            {
                "id": "chat_node",
                "type": "chat",
                "position": {"x": 200, "y": 0},
                "data": {"memory": {}}
            },
            {
                "id": "end_node",
                "type": "end",
                "position": {"x": 300, "y": 0},
                "data": {}
            }
        ],
        "edges": [
            {
                "id": "edge1",
                "source": "user_input_node",
                "target": "chat_node",
                "sourceHandle": "handle_user_message",
                "targetHandle": "handle_user_message"
            },
            {
                "id": "edge2",
                "source": "user_input_node",
                "target": "chat_node",
                "sourceHandle": "handle_user_images",
                "targetHandle": "handle_user_images"
            },
            {
                "id": "edge3",
                "source": "text_node",
                "target": "chat_node",
                "sourceHandle": "handle_text_output",
                "targetHandle": "handle-system-context"
            },
            {
                "id": "edge4",
                "source": "chat_node",
                "target": "end_node",
                "sourceHandle": "handle_chat_output",
                "targetHandle": "handle_generated_end"
            }
        ]
    }
}


class TestTextToChatSystemContext:
    """Test suite for text → chat system context flow."""
    
    def test_graph_builds_correctly(self):
        """Test that the graph builds without errors."""
        graph = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        # Verify nodes were created
        assert len(graph.nodes) >= 4  # At least 4 nodes + void node
        
        # Verify chat node exists
        chat_node = graph.nodes.get("chat_node")
        assert chat_node is not None
        
        # Verify text node exists
        text_node = graph.nodes.get("text_node")
        assert text_node is not None
    
    def test_input_tracker_expects_system_context(self):
        """Test that the input tracker for chat node expects system context."""
        graph = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        # Create dispatcher like the executor does
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
        
        # Get tracker for chat node
        tracker = dispatcher.get_tracker("chat_node")
        assert tracker is not None
        
        # Verify expected handles include system context
        expected = tracker.expected_handles
        assert "handle-system-context" in expected
        assert "handle_user_message" in expected
        assert "handle_user_images" in expected
    
    def test_reversed_edge_order_builds_same_graph(self):
        """Test that edge order doesn't affect graph building."""
        # Build with normal order
        graph1 = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        # Build with reversed edges
        reversed_config = json.loads(json.dumps(MINIMAL_TEXT_TO_CHAT_GRAPH))
        reversed_config["content"]["edges"] = list(reversed(reversed_config["content"]["edges"]))
        graph2 = build(reversed_config, "Hello", load_chat=None)
        
        # Both should have same input tracker expectations
        dispatcher1 = GraphEventDispatcher(graph1.nodes, graph1.edges)
        dispatcher2 = GraphEventDispatcher(graph2.nodes, graph2.edges)
        
        tracker1 = dispatcher1.get_tracker("chat_node")
        tracker2 = dispatcher2.get_tracker("chat_node")
        
        assert tracker1.expected_handles == tracker2.expected_handles
    
    def test_reversed_node_order_builds_same_graph(self):
        """Test that node order doesn't affect graph building."""
        # Build with normal order
        graph1 = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        # Build with reversed nodes
        reversed_config = json.loads(json.dumps(MINIMAL_TEXT_TO_CHAT_GRAPH))
        reversed_config["content"]["nodes"] = list(reversed(reversed_config["content"]["nodes"]))
        graph2 = build(reversed_config, "Hello", load_chat=None)
        
        # Both should have chat node with same handle configuration
        chat1 = graph1.nodes.get("chat_node")
        chat2 = graph2.nodes.get("chat_node")
        
        assert chat1.INPUT_HANDLER_SYSTEM_CONTEXT == chat2.INPUT_HANDLER_SYSTEM_CONTEXT
        assert chat1.INPUT_HANDLER_USER_MESSAGE == chat2.INPUT_HANDLER_USER_MESSAGE
    
    @pytest.mark.asyncio
    async def test_chat_node_receives_system_context_input(self):
        """Test that chat node actually receives system context during execution."""
        graph = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        # Use internal execution without LLM
        # We'll manually trigger the text node and check chat node inputs
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
        
        # Simulate text node output propagation
        text_node = graph.nodes["text_node"]
        text_output = {"handle_text_output": {"node": "NodeText", "content": "Always respond in French"}}
        await dispatcher.propagate_outputs("text_node", text_output)
        
        # Check chat node inputs
        chat_node = graph.nodes["chat_node"]
        assert "handle-system-context" in chat_node.inputs
        assert chat_node.inputs["handle-system-context"] == "Always respond in French"
    
    @pytest.mark.asyncio  
    async def test_chat_node_receives_all_inputs_before_ready(self):
        """Test that chat node waits for ALL inputs (including system context) before becoming ready."""
        graph = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
        tracker = dispatcher.get_tracker("chat_node")
        
        # Initially not ready (waiting for inputs)
        assert not tracker.is_ready
        
        # Send user_input outputs
        user_output = {
            "handle_user_message": {"node": "NodeUserInput", "content": "Hello"},
            "handle_user_images": {"node": "NodeUserInput", "content": None}
        }
        await dispatcher.propagate_outputs("user_input_node", user_output)
        
        # Still not ready (waiting for system context)
        assert not tracker.is_ready
        
        # Send text node output
        text_output = {"handle_text_output": {"node": "NodeText", "content": "Always respond in French"}}
        await dispatcher.propagate_outputs("text_node", text_output)
        
        # Now should be ready
        assert tracker.is_ready
        assert tracker.should_execute
    
    @pytest.mark.asyncio
    async def test_reversed_propagation_order_works(self):
        """Test that inputs can arrive in any order and still work correctly."""
        graph = build(MINIMAL_TEXT_TO_CHAT_GRAPH.copy(), "Hello", load_chat=None)
        
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
        tracker = dispatcher.get_tracker("chat_node")
        
        # Send text node output FIRST
        text_output = {"handle_text_output": {"node": "NodeText", "content": "Always respond in French"}}
        await dispatcher.propagate_outputs("text_node", text_output)
        
        # Still not ready (waiting for user inputs)
        assert not tracker.is_ready
        
        # Then send user_input outputs
        user_output = {
            "handle_user_message": {"node": "NodeUserInput", "content": "Hello"},
            "handle_user_images": {"node": "NodeUserInput", "content": None}
        }
        await dispatcher.propagate_outputs("user_input_node", user_output)
        
        # Now should be ready
        assert tracker.is_ready
        
        # Verify all inputs received
        chat_node = graph.nodes["chat_node"]
        assert "handle-system-context" in chat_node.inputs
        assert "handle_user_message" in chat_node.inputs
        assert chat_node.inputs["handle-system-context"] == "Always respond in French"


class TestImageJsonGraph:
    """Test suite specifically for the image.json example."""
    
    @pytest.fixture
    def image_json_config(self):
        """Load the image.json config."""
        import os
        config_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "examples", "json", "image.json"
        )
        with open(config_path, "r") as f:
            return json.load(f)
    
    def test_image_json_loads_correctly(self, image_json_config):
        """Test that image.json loads and builds correctly."""
        graph = build(image_json_config, "Hello", load_chat=None)
        assert graph is not None
        assert len(graph.nodes) > 0
    
    def test_image_json_has_text_to_chat_edge(self, image_json_config):
        """Test that image.json has the text → chat system context edge."""
        edges = image_json_config["content"]["edges"]
        
        # Find edge with handle-system-context target
        system_context_edges = [
            e for e in edges 
            if e.get("targetHandle") == "handle-system-context"
        ]
        
        assert len(system_context_edges) == 1
        edge = system_context_edges[0]
        assert edge["sourceHandle"] == "handle_text_output"
    
    def test_image_json_chat_node_tracker_expects_system_context(self, image_json_config):
        """Test that the chat node's input tracker expects system context."""
        graph = build(image_json_config, "Hello", load_chat=None)
        
        # Find chat node ID
        chat_node_id = None
        for node in image_json_config["content"]["nodes"]:
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break
        
        assert chat_node_id is not None
        
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
        tracker = dispatcher.get_tracker(chat_node_id)
        
        assert "handle-system-context" in tracker.expected_handles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
