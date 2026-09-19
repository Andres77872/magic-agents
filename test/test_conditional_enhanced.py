"""
Integration tests for enhanced NodeConditional features.

Tests the new conditional features including:
- default_handle fallback
- Fan-out execution
- Multi-input scenarios with namespaced strategy
- Merge collision detection
- BYPASS_ALL signal handling

NOTE: NodeText nodes store output for routing but don't stream to user.
To test actual streamed output, we use NodeSendMessage which has OUTPUT_HANDLE_CONTENT.
For routing lifecycle assertions, direct runtimes collect typed observer events
through ``debug_callback``.
"""

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType


def extract_streamed_content(item):
    """
    Extract streamed content from NodeSendMessage or NodeLLM output.
    
    These nodes use OUTPUT_HANDLE_CONTENT='content' which streams to user.
    The content is a ChatCompletionModel with choices[0].delta.content.
    """
    if not isinstance(item, dict):
        return ""
    
    # Only content type items are actual streamed content
    if item.get("type") != "content":
        return ""
    
    content = item.get("content")
    if content is None:
        return ""
    
    # ChatCompletionModel with choices (SendMessage/LLM output)
    if hasattr(content, 'choices') and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, 'content') and delta.content:
            return delta.content
    
    return ""


class DebugCapture(list):
    """Async callback collector for the direct run_agent debug contract."""

    async def __call__(self, event):
        self.append(event)


def get_executed_nodes(debug_events) -> set:
    """Extract node IDs with a successful NODE_END lifecycle event."""
    return {
        event.node_id
        for event in debug_events
        if event.event_type == DebugEventType.NODE_END
    }


def get_bypassed_nodes(debug_events) -> set:
    """Extract node IDs with a NODE_BYPASS lifecycle event."""
    return {
        event.node_id
        for event in debug_events
        if event.event_type == DebugEventType.NODE_BYPASS
    }


class TestConditionalDefaultHandle:
    """Tests for default_handle fallback behavior."""
    
    @pytest.mark.asyncio
    async def test_default_handle_fallback_on_empty(self):
        """Test default_handle is used when condition returns empty."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ '' }}",  # Always empty
                    "default_handle": "fallback",
                    "output_handles": ["normal", "fallback"]
                }},
                {"id": "send_normal", "type": "send_message", "data": {"message": "", "json_extras": "NORMAL_PATH"}},
                {"id": "send_fallback", "type": "send_message", "data": {"message": "", "json_extras": "FALLBACK_PATH"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_normal", 
                 "sourceHandle": "normal", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_fallback", 
                 "sourceHandle": "fallback", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send_normal", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_fallback", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # Should have used fallback path
        content_str = "".join(content_output)
        assert "FALLBACK_PATH" in content_str
        assert "NORMAL_PATH" not in content_str
        
        # Verify via typed lifecycle events.
        executed = get_executed_nodes(debug_events)
        assert "send_fallback" in executed
        # send_normal should be bypassed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_normal" in bypassed
    
    @pytest.mark.asyncio
    async def test_condition_takes_precedence_over_default(self):
        """Test that valid condition result takes precedence over default."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'normal' }}",  # Valid result
                    "default_handle": "fallback",
                    "output_handles": ["normal", "fallback"]
                }},
                {"id": "send_normal", "type": "send_message", "data": {"message": "", "json_extras": "NORMAL_PATH"}},
                {"id": "send_fallback", "type": "send_message", "data": {"message": "", "json_extras": "FALLBACK_PATH"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_normal", 
                 "sourceHandle": "normal", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_fallback", 
                 "sourceHandle": "fallback", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send_normal", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_fallback", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # Should have used normal path, not fallback
        content_str = "".join(content_output)
        assert "NORMAL_PATH" in content_str
        assert "FALLBACK_PATH" not in content_str
        
        # Verify via typed lifecycle events.
        executed = get_executed_nodes(debug_events)
        assert "send_normal" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_fallback" in bypassed


class TestConditionalFanOut:
    """Tests for fan-out execution patterns."""
    
    @pytest.mark.asyncio
    async def test_fan_out_all_targets_execute(self):
        """Test that all targets of selected handle execute in parallel."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'multi' }}",
                    "output_handles": ["multi", "single"]
                }},
                {"id": "send_1", "type": "send_message", "data": {"message": "", "json_extras": "TARGET_1"}},
                {"id": "send_2", "type": "send_message", "data": {"message": "", "json_extras": "TARGET_2"}},
                {"id": "send_3", "type": "send_message", "data": {"message": "", "json_extras": "TARGET_3"}},
                {"id": "send_single", "type": "send_message", "data": {"message": "", "json_extras": "SINGLE"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                # Fan-out: same handle to 3 targets
                {"id": "e2", "source": "cond", "target": "send_1", 
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_2", 
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "cond", "target": "send_3", 
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                # Single target on other handle
                {"id": "e5", "source": "cond", "target": "send_single", 
                 "sourceHandle": "single", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "send_1", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e7", "source": "send_2", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_3", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e9", "source": "send_single", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # All three multi targets should execute
        content_str = "".join(content_output)
        assert "TARGET_1" in content_str
        assert "TARGET_2" in content_str
        assert "TARGET_3" in content_str
        # Single should be bypassed
        assert "SINGLE" not in content_str
        
        # Verify via typed lifecycle events.
        executed = get_executed_nodes(debug_events)
        assert "send_1" in executed
        assert "send_2" in executed
        assert "send_3" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_single" in bypassed
    
    @pytest.mark.asyncio
    async def test_fan_out_non_selected_bypassed(self):
        """Test that non-selected branch targets are bypassed."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'single' }}",  # Select single, not multi
                    "output_handles": ["multi", "single"]
                }},
                {"id": "send_multi_1", "type": "send_message", "data": {"message": "", "json_extras": "MULTI_1"}},
                {"id": "send_multi_2", "type": "send_message", "data": {"message": "", "json_extras": "MULTI_2"}},
                {"id": "send_single", "type": "send_message", "data": {"message": "", "json_extras": "SINGLE"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                # Fan-out on multi (should be bypassed)
                {"id": "e2", "source": "cond", "target": "send_multi_1", 
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_multi_2", 
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                # Single target (should execute)
                {"id": "e4", "source": "cond", "target": "send_single", 
                 "sourceHandle": "single", "targetHandle": "handle_send_extra"},
                {"id": "e5", "source": "send_multi_1", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e6", "source": "send_multi_2", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e7", "source": "send_single", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # Single should execute
        content_str = "".join(content_output)
        assert "SINGLE" in content_str
        # Multi targets should be bypassed
        assert "MULTI_1" not in content_str
        assert "MULTI_2" not in content_str
        
        # Verify via typed lifecycle events.
        executed = get_executed_nodes(debug_events)
        assert "send_single" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_multi_1" in bypassed
        assert "send_multi_2" in bypassed


class TestConditionalMultiInput:
    """Tests for multi-input conditional scenarios."""
    
    @pytest.mark.asyncio
    async def test_flat_merge_simple(self):
        """Test flat merge with non-colliding keys."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser1", "type": "parser", "data": {
                    "text": '{"status": "ok"}'
                }},
                {"id": "parser2", "type": "parser", "data": {
                    "text": '{"score": 95}'
                }},
                {"id": "cond", "type": "conditional", "data": {
                    # Should have access to both status and score
                    "condition": "{{ 'pass' if status == 'ok' and score > 90 else 'fail' }}",
                    "merge_strategy": "flat",
                    "output_handles": ["pass", "fail"]
                }},
                {"id": "send_pass", "type": "send_message", "data": {"message": "", "json_extras": "PASSED"}},
                {"id": "send_fail", "type": "send_message", "data": {"message": "", "json_extras": "FAILED"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser1", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser2", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser1", "target": "cond", 
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_0"},
                {"id": "e4", "source": "parser2", "target": "cond", 
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_1"},
                {"id": "e5", "source": "cond", "target": "send_pass", 
                 "sourceHandle": "pass", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "cond", "target": "send_fail", 
                 "sourceHandle": "fail", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send_pass", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_fail", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # Should pass since status=ok and score=95>90
        content_str = "".join(content_output)
        assert "PASSED" in content_str
        assert "FAILED" not in content_str
        
        # Verify via typed lifecycle events.
        executed = get_executed_nodes(debug_events)
        assert "send_pass" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_fail" in bypassed
    
    @pytest.mark.asyncio
    async def test_namespaced_merge(self):
        """Test namespaced merge keeps inputs separate."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser1", "type": "parser", "data": {
                    "text": '{"status": "ok"}'
                }},
                {"id": "parser2", "type": "parser", "data": {
                    "text": '{"status": "error"}'  # Same key, different value
                }},
                {"id": "cond", "type": "conditional", "data": {
                    # Access via namespaced handles
                    "condition": "{{ 'branch_a' if handle_input_0.status == 'ok' else 'branch_b' }}",
                    "merge_strategy": "namespaced",
                    "output_handles": ["branch_a", "branch_b"]
                }},
                {"id": "send_a", "type": "send_message", "data": {"message": "", "json_extras": "BRANCH_A"}},
                {"id": "send_b", "type": "send_message", "data": {"message": "", "json_extras": "BRANCH_B"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser1", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser2", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser1", "target": "cond", 
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_0"},
                {"id": "e4", "source": "parser2", "target": "cond", 
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_1"},
                {"id": "e5", "source": "cond", "target": "send_a", 
                 "sourceHandle": "branch_a", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "cond", "target": "send_b", 
                 "sourceHandle": "branch_b", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send_a", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_b", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # Should select branch_a since handle_input_0.status == 'ok'
        content_str = "".join(content_output)
        assert "BRANCH_A" in content_str
        assert "BRANCH_B" not in content_str
        
        # Verify via typed lifecycle events.
        executed = get_executed_nodes(debug_events)
        assert "send_a" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_b" in bypassed


class TestConditionalErrorHandling:
    """Tests for conditional error handling."""
    
    @pytest.mark.asyncio
    async def test_undefined_variable_error(self):
        """Test error handling for undefined variable."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    # Nested access raises under Jinja's default Undefined policy.
                    "condition": "{{ undefined_var.required_attribute }}",
                    "output_handles": ["yes", "no"]
                }},
                {"id": "send_yes", "type": "send_message", "data": {"message": "", "json_extras": "YES"}},
                {"id": "send_no", "type": "send_message", "data": {"message": "", "json_extras": "NO"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_yes", 
                 "sourceHandle": "yes", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_no", 
                 "sourceHandle": "no", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send_yes", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_no", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        content_output = []
        debug_items = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
            if (
                isinstance(item, dict)
                and item.get("type") == "debug"
                and item.get("content", {}).get("error_type") == "TemplateError"
            ):
                debug_items.append(item)
        
        # Should see debug error and neither branch should execute
        content_str = "".join(content_output)
        assert "YES" not in content_str
        assert "NO" not in content_str
        assert {"send_yes", "send_no"} <= get_bypassed_nodes(debug_events)
        # Should have debug info about the error
        assert len(debug_items) == 1


class TestGraphValidation:
    """Tests for graph validation of conditional nodes."""
    
    def test_validation_missing_edge_warning(self):
        """Test that missing edges for declared handles produces warning."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'yes' }}",
                    "output_handles": ["yes", "no", "maybe"]  # 'maybe' has no edge
                }},
                {"id": "send_yes", "type": "send_message", "data": {"message": "", "json_extras": "YES"}},
                {"id": "send_no", "type": "send_message", "data": {"message": "", "json_extras": "NO"}},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_yes", 
                 "sourceHandle": "yes", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_no", 
                 "sourceHandle": "no", "targetHandle": "handle_send_extra"},
                # Note: no edge for 'maybe'
                {"id": "e4", "source": "send_yes", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_no", "target": "end", 
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"}
            ]
        }
        
        graph = build(agt, message="test")
        errors = [
            error for error in graph._validation_errors or []
            if error.get("type") == "MissingConditionalEdge"
        ]
        assert len(errors) == 1
        assert errors[0]["missing_handles"] == ["maybe"]
    
    def test_validation_invalid_jinja2_at_build_time(self):
        """Test that invalid Jinja2 is caught at build time."""
        agt = {
            "type": "chat",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ if broken }}"  # Invalid syntax
                }},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond", 
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "end", 
                 "sourceHandle": "yes", "targetHandle": "handle_flow_input"}
            ]
        }
        
        # Node configuration is Pydantic-validated during build, so malformed
        # templates fail immediately instead of creating an executable stub.
        with pytest.raises(ValueError, match="Invalid Jinja2 syntax in condition"):
            build(agt, message="test")
