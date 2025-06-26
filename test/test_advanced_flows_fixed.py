import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import asyncio

from magic_agents import run_agent
from magic_agents.agt_flow import build

# Load API keys from the specified JSON file
var_env = json.load(open('/home/andres/Documents/agents_key.json'))


class TestAdvancedFlowsFixed:
    """Fixed test suite for advanced agent flow patterns that properly handle node outputs."""
    
    def setup_method(self):
        """Setup method to initialize common test data."""
        self.load_chat = lambda **kwargs: print(f"Chat loaded: {kwargs}")
        self.api_keys = var_env
    
    @pytest.mark.asyncio
    async def test_send_message_with_extras(self):
        """Test SendMessage node with extras functionality - properly yields ChatCompletionModel."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-parser",
                    "source": "user-input",
                    "target": "metadata-parser",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "parser-to-send",
                    "source": "metadata-parser",
                    "target": "send-msg",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                {
                    "id": "send-to-end",
                    "source": "send-msg",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "user-input",
                    "type": "user_input"
                },
                {
                    "id": "metadata-parser",
                    "type": "parser",
                    "data": {
                        "text": '{"metadata": {"source": "user", "timestamp": "2024-01-01", "message": "{{ handle_parser_input }}"}}'
                    }
                },
                {
                    "id": "send-msg",
                    "type": "send_message",
                    "data": {
                        "json_extras": "Processing message with metadata"
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        graph = build(agt_data=agt, message='Hello with metadata', load_chat=self.load_chat)
        response = ""
        extras_found = False
        
        async for i in run_agent(graph=graph):
            # run_agent yields dictionaries with 'content' key
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content
                if hasattr(content, 'extras') and content.extras:
                    extras_found = True
                    print(f"\nExtras found: {content.extras}")
        
        print(f"\nSendMessage Test Response: {response}")
        assert "Processing message with metadata" in response
        assert extras_found or len(response) > 0  # Either extras or content should be present
    
    @pytest.mark.asyncio
    async def test_deeply_nested_inner_flows_fixed(self):
        """Test deeply nested inner flows with proper content generation using SendMessage nodes."""
        # Level 3 - innermost flow with SendMessage
        level3_flow = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "l3-user-to-parser",
                    "source": "l3-user",
                    "target": "l3-parser",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "l3-parser-to-send",
                    "source": "l3-parser",
                    "target": "l3-send",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                {
                    "id": "l3-send-to-end",
                    "source": "l3-send",
                    "target": "l3-end",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "l3-user",
                    "type": "user_input"
                },
                {
                    "id": "l3-parser",
                    "type": "parser",
                    "data": {
                        "text": "[L3: {{ handle_parser_input }}]"
                    }
                },
                {
                    "id": "l3-send",
                    "type": "send_message",
                    "data": {
                        "json_extras": "L3 processed"
                    }
                },
                {
                    "id": "l3-end",
                    "type": "end"
                }
            ]
        }
        
        # Level 2 - middle flow with SendMessage
        level2_flow = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "l2-user-to-inner",
                    "source": "l2-user",
                    "target": "l2-inner",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "l2-inner-to-parser",
                    "source": "l2-inner",
                    "target": "l2-parser",
                    "sourceHandle": "handle_execution_content",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "l2-parser-to-send",
                    "source": "l2-parser",
                    "target": "l2-send",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                {
                    "id": "l2-send-to-end",
                    "source": "l2-send",
                    "target": "l2-end",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "l2-user",
                    "type": "user_input"
                },
                {
                    "id": "l2-inner",
                    "type": "inner",
                    "data": {
                        "magic_flow": level3_flow
                    }
                },
                {
                    "id": "l2-parser",
                    "type": "parser",
                    "data": {
                        "text": "[L2: {{ handle_parser_input }}]"
                    }
                },
                {
                    "id": "l2-send",
                    "type": "send_message",
                    "data": {
                        "json_extras": "L2 processed"
                    }
                },
                {
                    "id": "l2-end",
                    "type": "end"
                }
            ]
        }
        
        # Level 1 - outermost flow with SendMessage
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-inner",
                    "source": "user-input",
                    "target": "inner-node",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "inner-to-parser",
                    "source": "inner-node",
                    "target": "final-parser",
                    "sourceHandle": "handle_execution_content",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "parser-to-send",
                    "source": "final-parser",
                    "target": "final-send",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                {
                    "id": "send-to-end",
                    "source": "final-send",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "user-input",
                    "type": "user_input"
                },
                {
                    "id": "inner-node",
                    "type": "inner",
                    "data": {
                        "magic_flow": level2_flow
                    }
                },
                {
                    "id": "final-parser",
                    "type": "parser",
                    "data": {
                        "text": "[L1: {{ handle_parser_input }}]"
                    }
                },
                {
                    "id": "final-send",
                    "type": "send_message",
                    "data": {
                        "json_extras": "L1 processed"
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        graph = build(agt_data=agt, message='nested test', load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content
        
        print(f"\nDeeply Nested Response: {response}")
        # With SendMessage nodes, we should see the processed messages
        assert "processed" in response
    
    @pytest.mark.asyncio
    async def test_parser_to_sendmessage_flow(self):
        """Test using SendMessage to display parser output."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-parser",
                    "source": "user-input",
                    "target": "transform-parser",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "parser-to-send",
                    "source": "transform-parser",
                    "target": "send-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                {
                    "id": "send-to-end",
                    "source": "send-node",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "user-input",
                    "type": "user_input"
                },
                {
                    "id": "transform-parser",
                    "type": "parser",
                    "data": {
                        "text": """Transform complete:
Original: {{ handle_parser_input }}
Uppercase: {{ handle_parser_input | upper }}
Length: {{ handle_parser_input | length }}
Reversed: {{ handle_parser_input | reverse }}"""
                    }
                },
                {
                    "id": "send-node",
                    "type": "send_message",
                    "data": {
                        "json_extras": "Transformation result:"
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        graph = build(agt_data=agt, message='hello world', load_chat=self.load_chat)
        response = ""
        extras_content = ""
        
        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                node_name = i.get('node', 'Unknown')
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content
                # Only capture extras from SendMessage nodes, not END nodes
                if node_name == 'NodeSendMessage' and hasattr(content, 'extras') and content.extras:
                    # Skip if it's just metadata
                    if not (isinstance(content.extras, dict) and list(content.extras.keys()) == ['meta']):
                        extras_content = str(content.extras)
        
        print(f"\nParser to SendMessage Response: {response}")
        print(f"Extras: {extras_content}")
        
        assert "Transformation result:" in response
        # The parser output should be in extras['text'] field
        assert "'text': 'Transform complete:" in extras_content or "HELLO WORLD" in extras_content
    
    @pytest.mark.asyncio
    async def test_loop_with_sendmessage_aggregation(self):
        """Test loop results displayed via SendMessage."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "items-to-loop",
                    "source": "items-text",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                {
                    "id": "loop-to-processor",
                    "source": "loop-node",
                    "target": "item-processor",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "processor-to-loop",
                    "source": "item-processor",
                    "target": "loop-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_loop"
                },
                {
                    "id": "loop-to-formatter",
                    "source": "loop-node",
                    "target": "result-formatter",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "formatter-to-send",
                    "source": "result-formatter",
                    "target": "send-results",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                {
                    "id": "send-to-end",
                    "source": "send-results",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "items-text",
                    "type": "text",
                    "data": {
                        "text": '["apple", "banana", "cherry"]'
                    }
                },
                {
                    "id": "loop-node",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "item-processor",
                    "type": "parser",
                    "data": {
                        "text": "Processed {{ handle_parser_input | upper }}"
                    }
                },
                {
                    "id": "result-formatter",
                    "type": "parser",
                    "data": {
                        "text": """Loop Results:
Total items: {{ handle_parser_input | length }}
{% for item in handle_parser_input %}
- {{ item }}
{% endfor %}"""
                    }
                },
                {
                    "id": "send-results",
                    "type": "send_message",
                    "data": {
                        "json_extras": "Processing Complete!"
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        graph = build(agt_data=agt, message='', load_chat=self.load_chat)
        response = ""
        extras_content = ""
        
        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                node_name = i.get('node', 'Unknown')
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content
                # Only capture extras from SendMessage nodes, not END nodes
                if node_name == 'NodeSendMessage' and hasattr(content, 'extras') and content.extras:
                    # Skip if it's just metadata
                    if not (isinstance(content.extras, dict) and list(content.extras.keys()) == ['meta']):
                        extras_content = str(content.extras)
        
        print(f"\nLoop with SendMessage Response: {response}")
        print(f"Extras: {extras_content}")
        
        assert "Processing Complete!" in response
        # Look for the loop results in extras - might be in 'text' field
        assert "'text': 'Loop Results:" in extras_content or "Total items: 3" in extras_content


def run_fixed_advanced_tests():
    """Helper function to run all fixed advanced tests."""
    import asyncio
    
    test_suite = TestAdvancedFlowsFixed()
    test_suite.setup_method()
    
    tests = [
        test_suite.test_send_message_with_extras(),
        test_suite.test_deeply_nested_inner_flows_fixed(),
        test_suite.test_parser_to_sendmessage_flow(),
        test_suite.test_loop_with_sendmessage_aggregation()
    ]
    
    async def run_tests():
        for i, test in enumerate(tests, 1):
            print(f"\n{'='*60}")
            print(f"Running Fixed Advanced Test {i}")
            print(f"{'='*60}")
            try:
                await test
                print(f"✓ Fixed Advanced Test {i} passed")
            except Exception as e:
                print(f"✗ Fixed Advanced Test {i} failed: {e}")
                import traceback
                traceback.print_exc()
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    run_fixed_advanced_tests() 