"""
Protocol conformance tests for ConditionalRouting.

Tests verify that NodeConditional properly implements the ConditionalRouting
protocol and that the protocol correctly identifies conditional-like nodes.
"""

import pytest

from magic_agents.node_system import NodeConditional
from magic_agents.execution.conditional_routing import ConditionalRouting


class TestConditionalRoutingProtocol:
    """Test ConditionalRouting protocol conformance."""

    def test_node_conditional_implements_protocol_after_execution(self):
        """NodeConditional implements ConditionalRouting after process() sets selected_handle."""
        from unittest.mock import MagicMock
        import asyncio

        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ 'handle_yes' if value else 'handle_no' }}",
        )
        cond.inputs = {"handle_input": '{"value": true}'}

        # Before execution: selected_handle not set, protocol check fails
        assert not isinstance(cond, ConditionalRouting), \
            "Protocol check should fail before selected_handle is set"

        # Execute the conditional
        async def run():
            chat_log = MagicMock()
            async for _ in cond(chat_log):
                pass

        asyncio.get_event_loop().run_until_complete(run())

        # After execution: selected_handle is set, protocol check passes
        assert isinstance(cond, ConditionalRouting), \
            "NodeConditional should implement ConditionalRouting after execution"
        assert cond.selected_handle == "handle_yes"

    def test_node_conditional_has_required_attributes(self):
        """NodeConditional has all attributes required by the protocol."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
            default_handle="handle_no",
            output_handles=["handle_yes", "handle_no"],
        )

        # Has condition_template (pre-execution detection)
        assert hasattr(cond, 'condition_template')
        assert cond.condition_template == "{{ 'handle_yes' }}"

        # Has default_handle (optional protocol attribute)
        assert hasattr(cond, 'default_handle')
        assert cond.default_handle == "handle_no"

        # Has output_handles (optional protocol attribute)
        assert hasattr(cond, 'output_handles')
        assert cond.output_handles == ["handle_yes", "handle_no"]

    def test_non_conditional_node_does_not_implement_protocol(self):
        """Non-conditional nodes do not implement ConditionalRouting."""
        from magic_agents.node_system.Node import Node

        # Create a simple non-conditional node
        class SimpleNode(Node):
            async def process(self, chat_log):
                yield self.yield_static("hello")

        simple_node = SimpleNode(node_id="simple-test", node_type="simple")

        assert not isinstance(simple_node, ConditionalRouting), \
            "SimpleNode should not implement ConditionalRouting"
        assert not hasattr(simple_node, 'condition_template'), \
            "SimpleNode should not have condition_template"

    def test_protocol_detects_conditional_by_hasattr(self):
        """hasattr check for condition_template works for pre-execution detection."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
        )

        # Pre-execution detection via hasattr
        assert hasattr(cond, 'condition_template'), \
            "Should detect conditional-like node via condition_template attribute"
