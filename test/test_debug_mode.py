"""
Test debug mode functionality.
Demonstrates how to properly handle debug output from graph execution.
"""
import json
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from magic_agents import run_agent
from magic_agents.agt_flow import build


# Simple test graph with debug enabled
simple_debug_graph = {
    "type": "chat",
    "debug": True,  # Enable debug mode
    "nodes": [
        {
            "id": "user-input-1",
            "type": "user_input"
        },
        {
            "id": "text-1",
            "type": "text",
            "data": {
                "text": "Hello, {{handle_user_message}}!"
            }
        },
        {
            "id": "end-1",
            "type": "end"
        }
    ],
    "edges": [
        {
            "id": "edge-1",
            "source": "user-input-1",
            "target": "text-1",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_user_message"
        },
        {
            "id": "edge-2",
            "source": "text-1",
            "target": "end-1",
            "sourceHandle": "handle_text_output",
            "targetHandle": "input"
        }
    ]
}


@pytest.mark.asyncio
async def test_debug_mode_basic():
    """Test that debug mode returns debug information."""
    def load_chat(**kwargs):
        pass

    graph = build(
        agt_data=simple_debug_graph,
        message="World",
        load_chat=load_chat
    )

    content_chunks = []
    debug_info = None

    async for result in run_agent(graph=graph):
        # All results are dicts with "type" and "content" keys
        if result.get("type") == "content":
            # Content message - extract ChatCompletionModel
            chat_model = result["content"]
            content_chunks.append(chat_model)
        elif result.get("type") == "debug_summary":
            # Final debug summary
            debug_info = result["content"]
            print("\n=== DEBUG SUMMARY RECEIVED ===")
            print(f"Execution ID: {debug_info['execution_id']}")
            print(f"Total nodes: {debug_info['total_nodes']}")
            print(f"Executed nodes: {debug_info['executed_nodes']}")
            print(f"Bypassed nodes: {debug_info['bypassed_nodes']}")
            print(f"Failed nodes: {debug_info['failed_nodes']}")
            print(f"Total duration: {debug_info['total_duration_ms']:.2f}ms")

    # Verify debug info was received
    assert debug_info is not None, "Debug information should be returned when debug=True"
    assert debug_info['execution_id'] is not None
    assert debug_info['total_nodes'] > 0
    assert debug_info['executed_nodes'] > 0
    assert 'nodes' in debug_info
    assert len(debug_info['nodes']) > 0
    
    # Verify node debug details
    for node in debug_info['nodes']:
        print(f"\nNode: {node['node_id']} ({node['node_class']})")
        print(f"  Executed: {node['was_executed']}")
        print(f"  Bypassed: {node['was_bypassed']}")
        if node['execution_duration_ms']:
            print(f"  Duration: {node['execution_duration_ms']:.2f}ms")
        print(f"  Inputs: {list(node['inputs'].keys())}")
        print(f"  Outputs: {list(node['outputs'].keys())}")
    
    print("\n=== TEST PASSED ===")


# Conditional flow graph with debug
conditional_debug_graph = {
    "type": "chat",
    "debug": True,
    "nodes": [
        {
            "id": "user-input-1",
            "type": "user_input"
        },
        {
            "id": "conditional-1",
            "type": "conditional",
            "data": {
                "condition": "{{ 'handle_true' if (value|string|length) > 5 else 'handle_false' }}"
            }
        },
        {
            "id": "text-long",
            "type": "text",
            "data": {
                "text": "Long input received"
            }
        },
        {
            "id": "text-short",
            "type": "text",
            "data": {
                "text": "Short input received"
            }
        },
        {
            "id": "end-1",
            "type": "end"
        }
    ],
    "edges": [
        {
            "source": "user-input-1",
            "target": "conditional-1",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_input"
        },
        {
            "source": "conditional-1",
            "target": "text-long",
            "sourceHandle": "handle_true",
            "targetHandle": "handle_text_input"
        },
        {
            "source": "conditional-1",
            "target": "text-short",
            "sourceHandle": "handle_false",
            "targetHandle": "handle_text_input"
        },
        {
            "source": "text-long",
            "target": "end-1",
            "sourceHandle": "handle_text_output",
            "targetHandle": "input"
        },
        {
            "source": "text-short",
            "target": "end-1",
            "sourceHandle": "handle_text_output",
            "targetHandle": "input"
        }
    ]
}


@pytest.mark.asyncio
async def test_debug_mode_conditional():
    """Test debug mode with conditional flow to verify bypassed nodes."""
    def load_chat(**kwargs):
        pass

    # Test with long input - use deepcopy to prevent mutation
    graph = build(
        agt_data=deepcopy(conditional_debug_graph),
        message="Long message here",
        load_chat=load_chat
    )

    debug_info = None
    async for result in run_agent(graph=graph):
        if result.get("type") == "debug_summary":
            debug_info = result["content"]

    assert debug_info is not None
    
    # Check that we have both executed and bypassed nodes
    executed_count = sum(1 for n in debug_info['nodes'] if n['was_executed'])
    bypassed_count = sum(1 for n in debug_info['nodes'] if n['was_bypassed'])
    
    print(f"\nConditional flow debug info:")
    print(f"  Executed: {executed_count}")
    print(f"  Bypassed: {bypassed_count}")
    
    assert executed_count > 0, "Should have executed nodes"
    assert bypassed_count > 0, "Should have bypassed nodes in conditional flow"
    
    # Verify bypassed nodes have their debug info captured
    for node in debug_info['nodes']:
        if node['was_bypassed']:
            print(f"\nBypassed node: {node['node_id']}")
            # Bypassed nodes should still have their structure captured
            assert 'inputs' in node
            assert 'outputs' in node
            assert 'internal_variables' in node


@pytest.mark.asyncio 
async def test_proper_result_handling():
    """
    Demonstrates the proper way to handle mixed content and debug results.
    This is the recommended pattern for using debug mode.
    """
    def load_chat(**kwargs):
        pass
    
    # Demonstrate CORRECT handling of all message types
    print("\n=== Starting Conditional Flow Test ===")
    
    # Use deepcopy to prevent mutation from previous tests
    graph = build(
        agt_data=deepcopy(conditional_debug_graph),
        message="Short message",
        load_chat=load_chat
    )
    
    content_output = []
    per_node_debug = []
    final_debug = None
    
    async for result in run_agent(graph=graph):
        # Check message type to handle correctly
        if result.get("type") == "content":
            # Content message - extract ChatCompletionModel
            chat_model = result["content"]
            if hasattr(chat_model, 'choices') and chat_model.choices:
                delta = chat_model.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end='')
                    content_output.append(delta.content)
        
        elif result.get("type") == "debug":
            # Per-node debug info (yielded after each node)
            node_debug = result["content"]
            per_node_debug.append(node_debug)
            print(f"[DEBUG] Node {node_debug['node_id']} executed")
        
        elif result.get("type") == "debug_summary":
            # Final debug summary
            final_debug = result["content"]
            print(f"\n[DEBUG] Received summary with {final_debug.get('total_nodes', 0)} nodes")
    
    # Verify we got data
    assert final_debug is not None, "Should have received debug summary"
    print("\n=== Proper result handling test passed ===")


if __name__ == "__main__":
    import asyncio
    print("Test 1: Basic Debug Mode")
    print("=" * 60)
    asyncio.run(test_debug_mode_basic())
    
    print("\n" + "=" * 60)
    print("Test 2: Debug Mode with Conditional Flow")
    print("=" * 60)
    asyncio.run(test_debug_mode_conditional())
    
    print("\n" + "=" * 60)
    print("Test 3: Proper Result Handling")
    print("=" * 60)
    asyncio.run(test_proper_result_handling())
    
    print("\n\nAll tests passed!")
