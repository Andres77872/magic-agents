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


class TestAdvancedFlows:
    """Test suite for advanced agent flow patterns and edge cases."""
    
    def setup_method(self):
        """Setup method to initialize common test data."""
        self.load_chat = lambda **kwargs: print(f"Chat loaded: {kwargs}")
        self.api_keys = var_env
    
    @pytest.mark.asyncio
    async def test_send_message_with_extras(self):
        """Test SendMessage node with extras functionality."""
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
                    "id": "send-to-llm",
                    "source": "send-msg",
                    "target": "llm-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "client-to-llm",
                    "source": "client-node",
                    "target": "llm-node",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "llm-to-end",
                    "source": "llm-node",
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
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "llm-node",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": True,
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
        
        graph = build(agt_data=agt, message='Hello with metadata', load_chat=self.load_chat)
        response = ""
        extras_found = False
        
        async for i in run_agent(graph=graph):
            if i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
            if hasattr(i['content'], 'extras') and i['content'].extras:
                extras_found = True
                print(f"\nExtras found: {i['content'].extras}")
        
        print(f"\nSendMessage Test Response: {response}")
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_deeply_nested_inner_flows(self):
        """Test deeply nested inner flows (3 levels)."""
        # Level 3 - innermost flow
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
                    "id": "l3-parser-to-end",
                    "source": "l3-parser",
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
                    "id": "l3-end",
                    "type": "end"
                }
            ]
        }
        
        # Level 2 - middle flow
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
                    "id": "l2-parser-to-end",
                    "source": "l2-parser",
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
                    "id": "l2-end",
                    "type": "end"
                }
            ]
        }
        
        # Level 1 - outermost flow
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
                    "id": "parser-to-end",
                    "source": "final-parser",
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
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        graph = build(agt_data=agt, message='nested test', load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if hasattr(i['content'], 'choices') and i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
        
        print(f"\nDeeply Nested Response: {response}")
        assert "[L1:" in response
        assert "[L2:" in response
        assert "[L3:" in response
    
    @pytest.mark.asyncio
    async def test_complex_loop_with_conditional_exit(self):
        """Test loop with conditional early exit."""
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
                    "id": "processor-to-llm",
                    "source": "item-processor",
                    "target": "llm-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "client-to-llm",
                    "source": "client-node",
                    "target": "llm-node",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "llm-to-loop",
                    "source": "llm-node",
                    "target": "loop-node",
                    "sourceHandle": "handle_generated_content",
                    "targetHandle": "handle_loop"
                },
                {
                    "id": "loop-to-final",
                    "source": "loop-node",
                    "target": "final-parser",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "final-to-end",
                    "source": "final-parser",
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
                        "text": '["task1", "task2", "urgent_task", "task4", "task5"]'
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
                        "text": """Process this task: {{ handle_parser_input }}
{% if 'urgent' in handle_parser_input %}
URGENT: This requires immediate attention!
{% endif %}"""
                    }
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "llm-node",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": False,
                        "max_tokens": 50,
                        "temperature": 0.5,
                        "iterate": True
                    }
                },
                {
                    "id": "final-parser",
                    "type": "parser",
                    "data": {
                        "text": """Task Processing Summary:
Total tasks processed: {{ handle_parser_input | length }}
{% for result in handle_parser_input %}
Task {{ loop.index }}: {{ result | truncate(50) }}
{% endfor %}"""
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
            if hasattr(i['content'], 'choices') and i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
        
        print(f"\nConditional Loop Response: {response}")
        assert "Total tasks processed: 5" in response
        assert "URGENT" in response or "urgent" in response
    
    @pytest.mark.asyncio
    async def test_parallel_fetch_aggregation(self):
        """Test parallel fetching with result aggregation."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # Query preparation
                {
                    "id": "user-to-queries",
                    "source": "user-input",
                    "target": "query-generator",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "queries-to-loop",
                    "source": "query-generator",
                    "target": "query-loop",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_list"
                },
                # Loop processing
                {
                    "id": "loop-to-fetch",
                    "source": "query-loop",
                    "target": "fetch-node",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_fetch_input"
                },
                {
                    "id": "fetch-to-parser",
                    "source": "fetch-node",
                    "target": "result-parser",
                    "sourceHandle": "handle_response_json",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "parser-to-loop",
                    "source": "result-parser",
                    "target": "query-loop",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_loop"
                },
                # Final aggregation
                {
                    "id": "loop-to-aggregator",
                    "source": "query-loop",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "context-to-aggregator",
                    "source": "aggregator-context",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle-system-context"
                },
                {
                    "id": "client-to-aggregator",
                    "source": "client-node",
                    "target": "aggregator-llm",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
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
                    "id": "query-generator",
                    "type": "parser",
                    "data": {
                        "text": '["{{ handle_parser_input }} news", "{{ handle_parser_input }} latest updates", "{{ handle_parser_input }} 2024"]'
                    }
                },
                {
                    "id": "query-loop",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "fetch-node",
                    "type": "fetch",
                    "data": {
                        "url": "https://google.serper.dev/search",
                        "method": "POST",
                        "headers": {
                            "Content-Type": "application/json",
                            "X-API-KEY": self.api_keys['serper_key']
                        },
                        "json_data": {
                            "q": "{{handle_fetch_input}}",
                            "num": 2
                        }
                    }
                },
                {
                    "id": "result-parser",
                    "type": "parser",
                    "data": {
                        "text": """{% if handle_parser_input.organic %}Top results:
{% for item in handle_parser_input.organic[:2] %}
- {{ item.title }}
{% endfor %}{% else %}No results found{% endif %}"""
                    }
                },
                {
                    "id": "aggregator-context",
                    "type": "text",
                    "data": {
                        "text": "Summarize all the search results into a coherent response"
                    }
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "aggregator-llm",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": True,
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
        
        graph = build(agt_data=agt, message='AI', load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
        
        print(f"\nParallel Fetch Response: {response}")
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_dynamic_flow_construction(self):
        """Test dynamic flow construction based on input."""
        router_template = """
        {% set topic = handle_parser_input | lower %}
        {% if 'code' in topic or 'programming' in topic %}
        {"flow_type": "technical", "system": "You are a programming expert", "temperature": 0.3}
        {% elif 'creative' in topic or 'story' in topic %}
        {"flow_type": "creative", "system": "You are a creative writer", "temperature": 0.9}
        {% else %}
        {"flow_type": "general", "system": "You are a helpful assistant", "temperature": 0.7}
        {% endif %}
        """
        
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-router",
                    "source": "user-input",
                    "target": "flow-router",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "router-to-config",
                    "source": "flow-router",
                    "target": "config-parser",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "config-to-llm",
                    "source": "config-parser",
                    "target": "dynamic-llm",
                    "sourceHandle": "handle_system",
                    "targetHandle": "handle-system-context"
                },
                {
                    "id": "user-to-llm",
                    "source": "user-input",
                    "target": "dynamic-llm",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "client-to-llm",
                    "source": "client-node",
                    "target": "dynamic-llm",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "llm-to-end",
                    "source": "dynamic-llm",
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
                    "id": "flow-router",
                    "type": "parser",
                    "data": {
                        "text": router_template
                    }
                },
                {
                    "id": "config-parser",
                    "type": "parser",
                    "data": {
                        "text": """{% set config = handle_parser_input | fromjson %}{{ config.system }}"""
                    }
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "dynamic-llm",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": True,
                        "max_tokens": 150,
                        "temperature": 0.7  # This would ideally be dynamic based on config
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        # Test different flow types
        test_inputs = [
            "Write code to sort an array",
            "Tell me a creative story",
            "What's the weather like?"
        ]
        
        for test_input in test_inputs:
            graph = build(agt_data=agt, message=test_input, load_chat=self.load_chat)
            response = ""
            async for i in run_agent(graph=graph):
                if i['content'].choices[0].delta.content:
                    response += i['content'].choices[0].delta.content
            
            print(f"\nDynamic Flow '{test_input}': {response[:100]}...")
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_multi_modal_flow_with_images(self):
        """Test flow with image inputs (simulated)."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-image-parser",
                    "source": "user-input",
                    "target": "image-parser",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "user-images-to-parser",
                    "source": "user-input",
                    "target": "image-parser",
                    "sourceHandle": "handle_images",
                    "targetHandle": "handle_images"
                },
                {
                    "id": "parser-to-llm",
                    "source": "image-parser",
                    "target": "llm-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "client-to-llm",
                    "source": "client-node",
                    "target": "llm-node",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "llm-to-end",
                    "source": "llm-node",
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
                    "id": "image-parser",
                    "type": "parser",
                    "data": {
                        "text": """User message: {{ handle_parser_input }}
{% if handle_images %}
Images provided: {{ handle_images | length }} image(s)
Please describe what you would see in these images based on the user's question.
{% else %}
No images provided.
{% endif %}"""
                    }
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "llm-node",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": True,
                        "max_tokens": 150,
                        "temperature": 0.7
                    }
                },
                {
                    "id": "end-node",
                    "type": "end"
                }
            ]
        }
        
        # Test with simulated image paths
        graph = build(
            agt_data=agt,
            message='What do you see in these images?',
            images=['image1.jpg', 'image2.png'],
            load_chat=self.load_chat
        )
        response = ""
        async for i in run_agent(graph=graph):
            if i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
        
        print(f"\nMulti-modal Response: {response}")
        assert "2 image" in response or "images" in response
    
    @pytest.mark.asyncio
    async def test_state_management_across_nodes(self):
        """Test state passing and transformation across multiple nodes."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-state-init",
                    "source": "user-input",
                    "target": "state-initializer",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "init-to-transform1",
                    "source": "state-initializer",
                    "target": "transform1",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "transform1-to-transform2",
                    "source": "transform1",
                    "target": "transform2",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "transform2-to-final",
                    "source": "transform2",
                    "target": "final-state",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "final-to-llm",
                    "source": "final-state",
                    "target": "llm-node",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "client-to-llm",
                    "source": "client-node",
                    "target": "llm-node",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "llm-to-end",
                    "source": "llm-node",
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
                    "id": "state-initializer",
                    "type": "parser",
                    "data": {
                        "text": '{"original": "{{ handle_parser_input }}", "step": 1, "transformations": []}'
                    }
                },
                {
                    "id": "transform1",
                    "type": "parser",
                    "data": {
                        "text": """{% set state = handle_parser_input | fromjson %}
{% set _ = state.transformations.append("uppercase") %}
{% set _ = state.update({"step": 2, "text": state.original | upper}) %}
{{ state | tojson }}"""
                    }
                },
                {
                    "id": "transform2",
                    "type": "parser",
                    "data": {
                        "text": """{% set state = handle_parser_input | fromjson %}
{% set _ = state.transformations.append("reverse") %}
{% set _ = state.update({"step": 3, "text": state.text | reverse}) %}
{{ state | tojson }}"""
                    }
                },
                {
                    "id": "final-state",
                    "type": "parser",
                    "data": {
                        "text": """{% set state = handle_parser_input | fromjson %}
State transformation complete:
Original: {{ state.original }}
Final: {{ state.text }}
Steps: {{ state.transformations | join(" → ") }}"""
                    }
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "llm-node",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": True,
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
        
        graph = build(agt_data=agt, message='hello world', load_chat=self.load_chat)
        response = ""
        async for i in run_agent(graph=graph):
            if i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
        
        print(f"\nState Management Response: {response}")
        assert "uppercase → reverse" in response
    
    @pytest.mark.asyncio
    async def test_recursive_summarization(self):
        """Test recursive summarization pattern."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "text-to-splitter",
                    "source": "long-text",
                    "target": "text-splitter",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle_parser_input"
                },
                {
                    "id": "splitter-to-loop",
                    "source": "text-splitter",
                    "target": "summarize-loop",
                    "sourceHandle": "handle_parser_output",
                    "targetHandle": "handle_list"
                },
                {
                    "id": "loop-to-llm",
                    "source": "summarize-loop",
                    "target": "chunk-summarizer",
                    "sourceHandle": "handle_item",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "prompt-to-llm",
                    "source": "summarize-prompt",
                    "target": "chunk-summarizer",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle-system-context"
                },
                {
                    "id": "client-to-chunk",
                    "source": "client-node",
                    "target": "chunk-summarizer",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "chunk-to-loop",
                    "source": "chunk-summarizer",
                    "target": "summarize-loop",
                    "sourceHandle": "handle_generated_content",
                    "targetHandle": "handle_loop"
                },
                {
                    "id": "loop-to-final",
                    "source": "summarize-loop",
                    "target": "final-summarizer",
                    "sourceHandle": "handle_end",
                    "targetHandle": "handle_user_message"
                },
                {
                    "id": "final-prompt-to-llm",
                    "source": "final-prompt",
                    "target": "final-summarizer",
                    "sourceHandle": "handle_text_output",
                    "targetHandle": "handle-system-context"
                },
                {
                    "id": "client-to-final",
                    "source": "client-node",
                    "target": "final-summarizer",
                    "sourceHandle": "handle-client-provider",
                    "targetHandle": "handle-client-provider"
                },
                {
                    "id": "final-to-end",
                    "source": "final-summarizer",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                }
            ],
            "nodes": [
                {
                    "id": "long-text",
                    "type": "text",
                    "data": {
                        "text": """Machine learning is a subset of artificial intelligence. 
                        It focuses on the development of algorithms that can learn from data. 
                        Deep learning is a subset of machine learning. 
                        Neural networks are the foundation of deep learning. 
                        These technologies are transforming industries worldwide."""
                    }
                },
                {
                    "id": "text-splitter",
                    "type": "parser",
                    "data": {
                        "text": """{% set sentences = handle_parser_input.split('.') %}
[{% for s in sentences if s.strip() %}"{{ s.strip() }}"{% if not loop.last %},{% endif %}{% endfor %}]"""
                    }
                },
                {
                    "id": "summarize-loop",
                    "type": "loop",
                    "data": {}
                },
                {
                    "id": "summarize-prompt",
                    "type": "text",
                    "data": {
                        "text": "Summarize this sentence in 5 words or less:"
                    }
                },
                {
                    "id": "client-node",
                    "type": "client",
                    "data": {
                        "engine": "openai",
                        "api_info": {
                            "api_key": self.api_keys['openai_key'],
                            "base_url": "https://api.openai.com/v1"
                        },
                        "model": "gpt-4.1-mini-2025-04-14"
                    }
                },
                {
                    "id": "chunk-summarizer",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": False,
                        "max_tokens": 20,
                        "temperature": 0.3,
                        "iterate": True
                    }
                },
                {
                    "id": "final-prompt",
                    "type": "text",
                    "data": {
                        "text": "Create a single coherent summary from these individual summaries:"
                    }
                },
                {
                    "id": "final-summarizer",
                    "type": "llm",
                    "data": {
                        "top_p": 1,
                        "stream": True,
                        "max_tokens": 100,
                        "temperature": 0.5
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
            if i['content'].choices[0].delta.content:
                response += i['content'].choices[0].delta.content
        
        print(f"\nRecursive Summarization: {response}")
        assert len(response) > 0
        assert any(term in response.lower() for term in ["machine", "learning", "ai", "artificial"])


def run_advanced_tests():
    """Helper function to run all advanced tests."""
    import asyncio
    
    test_suite = TestAdvancedFlows()
    test_suite.setup_method()
    
    tests = [
        test_suite.test_send_message_with_extras(),
        test_suite.test_deeply_nested_inner_flows(),
        test_suite.test_complex_loop_with_conditional_exit(),
        test_suite.test_parallel_fetch_aggregation(),
        test_suite.test_dynamic_flow_construction(),
        test_suite.test_multi_modal_flow_with_images(),
        test_suite.test_state_management_across_nodes(),
        test_suite.test_recursive_summarization()
    ]
    
    async def run_tests():
        for i, test in enumerate(tests, 1):
            print(f"\n{'='*60}")
            print(f"Running Advanced Test {i}")
            print(f"{'='*60}")
            try:
                await test
                print(f"✓ Advanced Test {i} passed")
            except Exception as e:
                print(f"✗ Advanced Test {i} failed: {e}")
                import traceback
                traceback.print_exc()
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    run_advanced_tests() 