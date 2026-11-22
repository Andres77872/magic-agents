"""
Tests for verifying text node → chat node system context flow.

This test suite ensures that:
1. Text node output properly connects to chat node's system context input
2. The flow works regardless of edge/node order in the JSON
3. The reactive executor properly waits for all inputs before executing
"""

import pytest

from magic_agents.agt_flow import build
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
