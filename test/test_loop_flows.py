import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build

# Load API keys from the specified JSON file
var_env = json.load(open('/home/andres/Documents/agents_key.json'))


class TestLoopFlows:
    """Comprehensive test suite for loop node functionality."""

    def setup_method(self):
        """Setup method to initialize common test data."""
        self.load_chat = lambda **kwargs: print(f"Chat loaded: {kwargs}")
        self.api_keys = var_env

    @pytest.mark.asyncio
    async def test_basic_loop_with_text_processing(self):
        """Test basic loop functionality that processes a list of items through an LLM."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # User input provides system context to the item processor LLM
                {
                    "id": "user-to-llm",
                    "source": "user-input",
                    "target": "item-processor",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle-system-context"
                },
                # Client connection to item processor LLM
                {
                    "id": "client-to-item-llm",
                    "source": "client-node",
                    "target": "item-processor",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                # Text list to loop
                {
                    "id": "text-to-loop",
                    "source": "list-text",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                # Loop items to item processor
                {
                    "id": "loop-to-processor",
                    "source": "loop-node",
                    "target": "item-processor",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_user_message"
                },
                # Processor results back to loop
                {
                    "id": "processor-to-loop",
                    "source": "item-processor",
                    "target": "loop-node",
                    "sourceHandle": "handle_generated_content",
                    "targetHandle": "handle_loop"
                },
                # Final results to end
                {
                    "id": "loop-to-end",
                    "source": "loop-node",
                    "target": "end-node",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "user-input",
                    "type": "user_input"
                },
                {
                    "id": "list-text",
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
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "model": "gpt-4o-mini",
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        }
                    }
                },
                {
                    "id": "item-processor",
                    "type": "llm",
                    "data": {
                        "stream": True,
                        "iterate": True,
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }

        graph = build(agt_data=agt, message='Process each fruit name into a short description',
                      load_chat=self.load_chat)
        response = ""

        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        print(f"\nBasic Loop Response: {response}")
        # Should contain references to fruits being processed
        assert any(fruit in response.lower() for fruit in ['apple', 'banana', 'cherry'])

    @pytest.mark.asyncio
    async def test_loop_with_aggregation_processing(self):
        """Test loop with final aggregation processing like the provided example."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # User input to item processor system context
                {
                    "id": "user-to-item-processor",
                    "source": "user-input",
                    "target": "item-processor",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle-system-context"
                },
                # Client to both LLMs
                {
                    "id": "client-to-item-processor",
                    "source": "client-node",
                    "target": "item-processor",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "client-to-aggregator",
                    "source": "client-node",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                # Text list to loop
                {
                    "id": "list-to-loop",
                    "source": "numbers-text",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                # Loop items to processor
                {
                    "id": "loop-item-to-processor",
                    "source": "loop-node",
                    "target": "item-processor",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_user_message"
                },
                # Processor back to loop
                {
                    "id": "processor-to-loop",
                    "source": "item-processor",
                    "target": "loop-node",
                    "sourceHandle": "handle_generated_content",
                    "targetHandle": "handle_loop"
                },
                # Aggregation context to final LLM
                {
                    "id": "agg-context-to-llm",
                    "source": "aggregation-context",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle-system-context"
                },
                # Loop end to aggregator
                {
                    "id": "loop-to-aggregator",
                    "source": "loop-node",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_user_message"
                },
                # Aggregator to end
                {
                    "id": "aggregator-to-end",
                    "source": "aggregator-llm",
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
                    "id": "numbers-text",
                    "type": "text",
                    "data": {
                        "text": "[1, 2, 3]"
                    }
                },
                {
                    "id": "loop-node",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "model": "gpt-4o-mini",
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        }
                    }
                },
                {
                    "id": "item-processor",
                    "type": "llm",
                    "data": {
                        "stream": True,
                        "iterate": True,
                        "max_tokens": 50,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "aggregation-context",
                    "type": "text",
                    "data": {
                        "text": "Summarize the content"
                    }
                },
                {
                    "id": "aggregator-llm",
                    "type": "llm",
                    "data": {
                        "stream": True,
                        "iterate": False,
                        "max_tokens": 200,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }

        graph = build(agt_data=agt, message='Describe each number', load_chat=self.load_chat)
        response = ""

        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        print(f"\nLoop with Aggregation Response: {response}")
        # Should contain some summary of processing numbers
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_loop_exact_example_pattern(self):
        """Test the exact pattern from the provided loop example."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # User input to item processor system context
                {
                    "id": "xy-edge__0handle_user_message-bb3c62ae-396c-4743-9f61-71064f19b65fhandle-system-context",
                    "source": "user-input",
                    "target": "item-processor-llm",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle-system-context"
                },
                # Client to both LLMs
                {
                    "id": "xy-edge__client-to-item-processor",
                    "source": "client-provider",
                    "target": "item-processor-llm",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "xy-edge__client-to-aggregator",
                    "source": "client-provider",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                # Text list to loop
                {
                    "id": "xy-edge__text-to-loop",
                    "source": "numbers-text",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                # Item processor results back to loop
                {
                    "id": "xy-edge__processor-to-loop",
                    "source": "item-processor-llm",
                    "target": "loop-node",
                    "sourceHandle": "handle_generated_content",
                    "targetHandle": "handle_loop"
                },
                # Aggregation context to final LLM
                {
                    "id": "xy-edge__context-to-aggregator",
                    "source": "aggregation-context",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle-system-context"
                },
                # Loop items to item processor
                {
                    "id": "xy-edge__loop-item-to-processor",
                    "source": "loop-node",
                    "target": "item-processor-llm",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_user_message"
                },
                # Loop end to aggregator
                {
                    "id": "xy-edge__loop-end-to-aggregator",
                    "source": "loop-node",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_user_message"
                },
                # Aggregator to end
                {
                    "id": "xy-edge__aggregator-to-end",
                    "source": "aggregator-llm",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle_generated_end"
                }
            ],
            "nodes": [
                {
                    "id": "user-input",
                    "type": "user_input"
                },
                {
                    "id": "end-node",
                    "type": "end"
                },
                {
                    "id": "loop-node",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "client-provider",
                    "type": "client",
                    "data": {
                        "model": "gpt-4o-mini",
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        }
                    }
                },
                {
                    "id": "item-processor-llm",
                    "type": "llm",
                    "data": {
                        "stream": True,
                        "iterate": True,
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "numbers-text",
                    "type": "text",
                    "data": {
                        "text": "[1,2,3]"
                    }
                },
                {
                    "id": "aggregator-llm",
                    "type": "llm",
                    "data": {
                        "stream": True,
                        "iterate": False,
                        "max_tokens": 200,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "aggregation-context",
                    "type": "text",
                    "data": {
                        "text": "Describe the content"
                    }
                }
            ],
            "master": "1"
        }

        graph = build(agt_data=agt, message='Process each number and describe it', load_chat=self.load_chat)
        response = ""

        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        print(f"\nExact Example Pattern Response: {response}")
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_loop_with_parser_transformation(self):
        """Test loop with parser nodes for data transformation."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # Text list to loop
                {
                    "id": "text-to-loop",
                    "source": "data-text",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                # Loop items to parser
                {
                    "id": "loop-to-parser",
                    "source": "loop-node",
                    "target": "item-transformer",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_parser_input"
                },
                # Parser back to loop
                {
                    "id": "parser-to-loop",
                    "source": "item-transformer",
                    "target": "loop-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_loop"
                },
                # Loop end to final parser
                {
                    "id": "loop-to-final-parser",
                    "source": "loop-node",
                    "target": "result-formatter",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_parser_input"
                },
                # Final parser to send message
                {
                    "id": "parser-to-send",
                    "source": "result-formatter",
                    "target": "send-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_send_extra"
                },
                # Send to end
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
                    "id": "data-text",
                    "type": "text",
                    "data": {
                        "text": '["red", "blue", "green"]'
                    }
                },
                {
                    "id": "loop-node",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "item-transformer",
                    "type": "parser",
                    "data": {
                        "text": "Color: {{ handle_parser_input | upper }}"
                    }
                },
                {
                    "id": "result-formatter",
                    "type": "parser",
                    "data": {
                        "text": """Loop Processing Results:
Total colors processed: {{ handle_parser_input | length }}
{% for item in handle_parser_input %}
- {{ item }}
{% endfor %}"""
                    }
                },
                {
                    "id": "send-node",
                    "type": "send_message",
                    "data": {
                        "json_extras": "Color processing complete!"
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

        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        print(f"\nLoop with Parser Response: {response}")
        assert "Color processing complete!" in response

    @pytest.mark.asyncio
    async def test_loop_with_mixed_data_types(self):
        """Test loop with mixed data types in the list."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # Text list to loop
                {
                    "id": "mixed-to-loop",
                    "source": "mixed-data",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                # Loop items to parser
                {
                    "id": "loop-to-parser",
                    "source": "loop-node",
                    "target": "type-analyzer",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_parser_input"
                },
                # Parser back to loop
                {
                    "id": "parser-to-loop",
                    "source": "type-analyzer",
                    "target": "loop-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_loop"
                },
                # Loop end to send
                {
                    "id": "loop-to-send",
                    "source": "loop-node",
                    "target": "send-results",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_send_extra"
                },
                # Send to end
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
                    "id": "mixed-data",
                    "type": "text",
                    "data": {
                        "text": '[42, "hello", true, 3.14]'
                    }
                },
                {
                    "id": "loop-node",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "type-analyzer",
                    "type": "parser",
                    "data": {
                        "text": "Value: {{ handle_parser_input }} (processed)"
                    }
                },
                {
                    "id": "send-results",
                    "type": "send_message",
                    "data": {
                        "json_extras": "Mixed data processing complete!"
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

        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        print(f"\nMixed Data Types Response: {response}")
        assert "Mixed data processing complete!" in response

    @pytest.mark.asyncio
    async def test_empty_loop_handling(self):
        """Test loop behavior with empty list."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # Empty list to loop
                {
                    "id": "empty-to-loop",
                    "source": "empty-list",
                    "target": "loop-node",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_list"
                },
                # Loop items to parser (shouldn't execute)
                {
                    "id": "loop-to-parser",
                    "source": "loop-node",
                    "target": "item-processor",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_parser_input"
                },
                # Parser back to loop
                {
                    "id": "parser-to-loop",
                    "source": "item-processor",
                    "target": "loop-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_loop"
                },
                # Loop end to send
                {
                    "id": "loop-to-send",
                    "source": "loop-node",
                    "target": "send-empty",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_send_extra"
                },
                # Send to end
                {
                    "id": "send-to-end",
                    "source": "send-empty",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "empty-list",
                    "type": "text",
                    "data": {
                        "text": "[]"
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
                        "text": "Should not execute: {{ handle_parser_input }}"
                    }
                },
                {
                    "id": "send-empty",
                    "type": "send_message",
                    "data": {
                        "json_extras": "Empty loop completed!"
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

        async for i in run_agent(graph=graph):
            if isinstance(i, dict) and 'content' in i:
                content = i['content']
                if hasattr(content, 'choices') and content.choices and content.choices[0].delta.content:
                    response += content.choices[0].delta.content

        print(f"\nEmpty Loop Response: {response}")
        assert "Empty loop completed!" in response
        assert "Should not execute" not in response


def run_loop_tests():
    """Helper function to run all loop tests."""
    import asyncio

    test_suite = TestLoopFlows()
    test_suite.setup_method()

    tests = [
        ("Basic Loop with Text Processing", test_suite.test_basic_loop_with_text_processing()),
        ("Loop with Aggregation Processing", test_suite.test_loop_with_aggregation_processing()),
        ("Exact Example Pattern", test_suite.test_loop_exact_example_pattern()),
        ("Loop with Parser Transformation", test_suite.test_loop_with_parser_transformation()),
        ("Loop with Mixed Data Types", test_suite.test_loop_with_mixed_data_types()),
        ("Empty Loop Handling", test_suite.test_empty_loop_handling())
    ]

    async def run_tests():
        for i, (test_name, test_coro) in enumerate(tests, 1):
            print(f"\n{'=' * 70}")
            print(f"Running Loop Test {i}: {test_name}")
            print(f"{'=' * 70}")
            try:
                await test_coro
                print(f"✓ Loop Test {i} ({test_name}) passed")
            except Exception as e:
                print(f"✗ Loop Test {i} ({test_name}) failed: {e}")
                import traceback
                traceback.print_exc()

    asyncio.run(run_tests())


if __name__ == "__main__":
    run_loop_tests()
