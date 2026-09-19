import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build


async def _collect_events(graph):
    events = []
    async for event in run_agent(graph=graph):
        events.append(event)
    return events


def _stream_text(events):
    chunks = []
    for event in events:
        content = event.get("content") if isinstance(event, dict) else None
        choices = getattr(content, "choices", None)
        if not choices:
            continue
        text = getattr(getattr(choices[0], "delta", None), "content", None)
        if text:
            chunks.append(text)
    return "".join(chunks)


def _debug_errors(events):
    return [
        event["content"]
        for event in events
        if isinstance(event, dict)
        and event.get("type") == "debug"
        and isinstance(event.get("content"), dict)
        and (event["content"].get("error_type") or event["content"].get("error"))
    ]


def _payload_extras(events):
    payloads = []
    for event in events:
        content = event.get("content") if isinstance(event, dict) else None
        extras = getattr(content, "extras", None)
        if not isinstance(extras, dict):
            continue
        payload = {key: value for key, value in extras.items() if key != "meta"}
        if payload:
            payloads.append(payload)
    return payloads


class TestAdvancedFlowsFixed:
    """Fixed test suite for advanced agent flow patterns that properly handle node outputs."""
    
    def setup_method(self):
        """Setup method to initialize common test data."""
        self.load_chat = lambda **kwargs: None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
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
                    "sourceHandle": "handle_message_output",
                    "targetHandle": "handle_flow_input"
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
        assert not graph._validation_errors

        events = await _collect_events(graph)

        assert not _debug_errors(events)
        assert _stream_text(events) == "Processing message with metadata"
        assert {
            "metadata": {
                "source": "user",
                "timestamp": "2024-01-01",
                "message": "Hello with metadata",
            }
        } in _payload_extras(events)
        assert graph.nodes["end-node"].response is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
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
                    "sourceHandle": "handle_message_output",
                    "targetHandle": "handle_flow_input"
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
                    "sourceHandle": "handle_message_output",
                    "targetHandle": "handle_flow_input"
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
                    "sourceHandle": "handle_message_output",
                    "targetHandle": "handle_flow_input"
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
        assert not graph._validation_errors

        events = await _collect_events(graph)

        assert not _debug_errors(events)
        assert _stream_text(events) == "L3 processedL2 processedL1 processed"
        completed_subgraphs = {
            event["content"].get("node_id")
            for event in events
            if event.get("type") == "debug"
            and isinstance(event.get("content"), dict)
            and event["content"].get("event_type") == "SUBGRAPH_END"
            and event["content"].get("status") == "completed"
        }
        assert {"l2-inner", "inner-node"}.issubset(completed_subgraphs)
        assert graph.nodes["end-node"].response is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
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
                    "sourceHandle": "handle_message_output",
                    "targetHandle": "handle_flow_input"
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
        assert not graph._validation_errors

        events = await _collect_events(graph)

        assert not _debug_errors(events)
        assert _stream_text(events) == "Transformation result:"
        assert {
            "text": (
                "Transform complete:\n"
                "Original: hello world\n"
                "Uppercase: HELLO WORLD\n"
                "Length: 11\n"
                "Reversed: dlrow olleh"
            )
        } in _payload_extras(events)
        assert graph.nodes["end-node"].response is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_loop_with_sendmessage_aggregation(self):
        """Test loop results displayed via SendMessage."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                # Feed the static list node from the graph's required user input.
                {
                    "id": "ui-to-items",
                    "source": "user-input",
                    "target": "items-text",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_flow_input"
                },
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
                    "sourceHandle": "handle_message_output",
                    "targetHandle": "handle_flow_input"
                }
            ],
            "nodes": [
                {
                    "id": "user-input",
                    "type": "user_input"
                },
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
        assert not graph._validation_errors

        events = await _collect_events(graph)

        assert not _debug_errors(events)
        assert _stream_text(events) == "Processing Complete!"
        payloads = _payload_extras(events)
        loop_results = next(payload["text"] for payload in payloads if "text" in payload)
        assert "Total items: 3" in loop_results
        assert "Processed APPLE" in loop_results
        assert "Processed BANANA" in loop_results
        assert "Processed CHERRY" in loop_results
        assert graph.nodes["end-node"].response is not None


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
