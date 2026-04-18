"""
Test for the image.json example.

This test verifies that:
1. The image.json example loads and builds correctly
2. The text node system context is properly passed to the chat node
3. The LLM receives the system prompt and responds accordingly
"""

import json
import os
import pytest
import asyncio

from magic_agents.agt_flow import build, run_agent
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from conftest import skip_if_no_api_keys


@pytest.fixture
def image_json_config():
    """Load the image.json config."""
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "..", "examples", "json", "image.json"
    )
    with open(config_path, "r") as f:
        return json.load(f)


class TestImageJsonExample:
    """Test suite for the image.json example."""
    
    def test_image_json_loads(self, image_json_config):
        """Test that image.json loads correctly."""
        assert image_json_config is not None
        assert "content" in image_json_config
        assert "nodes" in image_json_config["content"]
        assert "edges" in image_json_config["content"]
    
    def test_image_json_has_required_nodes(self, image_json_config):
        """Test that image.json has all required node types."""
        nodes = image_json_config["content"]["nodes"]
        node_types = [n["type"] for n in nodes]
        
        assert "user_input" in node_types
        assert "text" in node_types
        assert "chat" in node_types
        assert "llm" in node_types
        assert "client" in node_types
        assert "end" in node_types
    
    def test_image_json_has_system_context_edge(self, image_json_config):
        """Test that image.json has the text→chat system context edge."""
        edges = image_json_config["content"]["edges"]
        
        # Find the system context edge
        system_context_edge = None
        for edge in edges:
            if edge.get("targetHandle") == "handle-system-context":
                system_context_edge = edge
                break
        
        assert system_context_edge is not None
        assert system_context_edge["sourceHandle"] == "handle_text_output"
    
    def test_text_node_has_system_prompt(self, image_json_config):
        """Test that the text node contains the system prompt."""
        nodes = image_json_config["content"]["nodes"]
        
        text_node = None
        for node in nodes:
            if node["type"] == "text":
                text_node = node
                break
        
        assert text_node is not None
        assert "data" in text_node
        assert "text" in text_node["data"]
        assert text_node["data"]["text"] == "Write always on French"
    
    def test_graph_builds_successfully(self, image_json_config):
        """Test that the graph builds without errors."""
        graph = build(image_json_config, "Hello", load_chat=None)
        
        assert graph is not None
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
    
    def test_chat_node_input_tracker_configuration(self, image_json_config):
        """Test that chat node's input tracker expects system context."""
        graph = build(image_json_config, "Hello", load_chat=None)
        
        # Find chat node ID
        chat_node_id = None
        for node in image_json_config["content"]["nodes"]:
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break
        
        assert chat_node_id is not None
        
        # Check input tracker
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
        tracker = dispatcher.get_tracker(chat_node_id)
        
        assert tracker is not None
        assert "handle-system-context" in tracker.expected_handles
        assert "handle_user_message" in tracker.expected_handles
        assert "handle_user_images" in tracker.expected_handles
    
    @pytest.mark.asyncio
    async def test_chat_node_receives_system_context(self, image_json_config, mock_magic_llm):
        """Test that chat node receives system context during execution."""
        graph = build(image_json_config, "Test message", load_chat=None)

        # Find chat node
        chat_node_id = None
        for node in image_json_config["content"]["nodes"]:
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break

        chat_node = graph.nodes.get(chat_node_id)

        # Run the agent and collect events to verify clean execution
        events = []
        async for result in run_agent(graph):
            events.append(result)

        # Verify execution produced events (graph completed)
        assert len(events) > 0, "Expected events from graph execution"

        # Verify no execution errors occurred
        errors = [
            e for e in events
            if e.get("type") == "debug" and e.get("content", {}).get("error")
        ]
        assert len(errors) == 0, f"Unexpected errors: {[e['content']['error'] for e in errors]}"

        # Verify chat node received system context
        assert "handle-system-context" in chat_node.inputs
        assert chat_node.inputs["handle-system-context"] == "Write always on French"
    
    @pytest.mark.asyncio
    async def test_chat_messages_include_system_message(self, image_json_config, mock_magic_llm):
        """Test that chat messages include the system message."""
        graph = build(image_json_config, "Hi", load_chat=None)

        # Find chat node
        chat_node_id = None
        for node in image_json_config["content"]["nodes"]:
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break

        chat_node = graph.nodes.get(chat_node_id)

        # Run the agent and verify clean execution
        events = []
        async for result in run_agent(graph):
            events.append(result)

        assert len(events) > 0, "Expected events from graph execution"

        # Verify chat messages structure
        messages = chat_node.chat.messages
        assert len(messages) >= 2

        # First message should be system
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Write always on French"

        # Second message should be user
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"
    
    @pytest.mark.needs_api
    @pytest.mark.asyncio
    async def test_llm_responds_in_french(self, image_json_config):
        """Test that the LLM responds in French due to system prompt.

        Requires real API keys — skipped otherwise.
        """
        skip_if_no_api_keys()
        graph = build(image_json_config, "Hello", load_chat=None)
        
        # Collect the response
        response = ""
        async for result in run_agent(graph):
            if result.get("type") == "content":
                content = result.get("content")
                if hasattr(content, "choices") and content.choices:
                    delta = content.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        response += delta.content
        
        # The response should be in French
        # Common French words/phrases to check for
        french_indicators = [
            "bonjour", "salut", "comment", "puis-je", "aider",
            "aujourd", "bienvenue", "je", "vous", "merci"
        ]
        
        response_lower = response.lower()
        has_french = any(word in response_lower for word in french_indicators)
        
        assert has_french, f"Expected French response, got: {response}"
    
    @pytest.mark.asyncio
    async def test_debug_output_shows_system_context(self, image_json_config, mock_magic_llm):
        """Test that debug output correctly shows system context."""
        config = image_json_config.copy()
        config["debug"] = True
        
        graph = build(config, "Test", load_chat=None)
        
        # Find chat node ID
        chat_node_id = None
        for node in config["content"]["nodes"]:
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break
        
        # Collect debug info for chat node
        chat_debug = None
        async for result in run_agent(graph):
            if result.get("type") == "debug":
                content = result.get("content", {})
                if content.get("node_id") == chat_node_id:
                    chat_debug = content
                    break
        
        assert chat_debug is not None
        assert "inputs" in chat_debug
        assert "handle-system-context" in chat_debug["inputs"]
        assert chat_debug["inputs"]["handle-system-context"] == "Write always on French"
        
        # Verify has_system_message is True
        internal = chat_debug.get("internal_variables", {})
        assert internal.get("has_system_message") is True


class TestImageJsonEdgeOrderIndependence:
    """Test that image.json works regardless of edge/node order."""
    
    @pytest.mark.asyncio
    async def test_reversed_edges_still_works(self, image_json_config, mock_magic_llm):
        """Test that reversing edge order doesn't break execution."""
        config = json.loads(json.dumps(image_json_config))
        config["content"]["edges"] = list(reversed(config["content"]["edges"]))

        graph = build(config, "Hello", load_chat=None)

        # Find chat node
        chat_node_id = None
        for node in config["content"]["nodes"]:
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break

        chat_node = graph.nodes.get(chat_node_id)

        # Run and verify clean execution
        events = []
        async for result in run_agent(graph):
            events.append(result)

        assert len(events) > 0, "Expected events from graph execution"

        assert "handle-system-context" in chat_node.inputs
        assert chat_node.inputs["handle-system-context"] == "Write always on French"

    @pytest.mark.asyncio
    async def test_reversed_nodes_still_works(self, image_json_config, mock_magic_llm):
        """Test that reversing node order doesn't break execution."""
        config = json.loads(json.dumps(image_json_config))
        config["content"]["nodes"] = list(reversed(config["content"]["nodes"]))

        graph = build(config, "Hello", load_chat=None)

        # Find chat node
        chat_node_id = None
        for node in image_json_config["content"]["nodes"]:  # Use original to find ID
            if node["type"] == "chat":
                chat_node_id = node["id"]
                break

        chat_node = graph.nodes.get(chat_node_id)

        # Run and verify clean execution
        events = []
        async for result in run_agent(graph):
            events.append(result)

        assert len(events) > 0, "Expected events from graph execution"

        assert "handle-system-context" in chat_node.inputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
