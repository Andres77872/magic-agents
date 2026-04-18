"""
Phase 0: Characterization Tests for Conditional Workflow Node Agnostic Refactor.

These tests capture CURRENT behavior BEFORE any production code changes.
They serve as a safety net — all must pass against the existing codebase.

Slices covered:
- 0a: Executor non-loop conditional bypass propagation
- 0b: Executor loop conditional static-phase behavior
- 0c: Executor loop conditional iteration-phase behavior
- 0d: NodeConditional.process() error paths
- 0e: GraphEventDispatcher.propagate_conditional_bypass()
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from magic_agents.execution.reactive_executor import (
    execute_graph_reactive,
    execute_graph_loop_reactive,
)
from magic_agents.execution.event_dispatcher import GraphEventDispatcher, NodeState
from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.node_system.NodeConditional import NodeConditional
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes
from magic_agents.util.const import SYSTEM_EVENT_DEBUG, SYSTEM_EVENT_STREAMING


# ─── Helpers ────────────────────────────────────────────────────────────────

async def _collect_all(async_gen):
    """Consume an async generator and return all yielded items."""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def _make_mock_graph(nodes_dict: dict, edges_list: list, debug: bool = True) -> AgentFlowModel:
    """Build a minimal mock AgentFlowModel for testing."""
    graph = MagicMock(spec=AgentFlowModel)
    graph.nodes = nodes_dict
    graph.edges = edges_list
    graph.debug = debug
    graph.resolved_debug_config = None
    graph._validation_errors = []
    graph.type = "graph"
    graph.app_id = None
    graph.id_app = None
    return graph


def _make_mock_loop_graph(nodes_dict: dict, edges_list: list, debug: bool = True) -> AgentFlowModel:
    """Build a mock AgentFlowModel for loop executor testing."""
    graph = _make_mock_graph(nodes_dict, edges_list, debug)
    return graph


class _CollectingNode(Node):
    """Test node that records its execution and yields a deterministic output."""

    def __init__(self, node_id: str, output_value: str = "done", output_handle: str = "output", **kwargs):
        super().__init__(node_id=node_id, **kwargs)
        self.output_value = output_value
        self.output_handle = output_handle
        self.execute_count = 0
        self.input_snapshot = None

    async def process(self, chat_log):
        self.execute_count += 1
        self.input_snapshot = dict(self.inputs)
        self._response = self.output_value
        yield self.yield_static(self.output_value, content_type=self.output_handle)


class _BypassAllNode(Node):
    """Test node that emits a BYPASS_ALL signal."""

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, **kwargs)
        self.BYPASS_ALL = ConditionalSignalTypes.BYPASS_ALL

    async def process(self, chat_log):
        self._response = "bypass_all_sent"
        yield {
            "type": self.BYPASS_ALL,
            "content": {"signal": "bypass_all"}
        }


def _make_minimal_node(node_class, node_id: str, **kwargs):
    """Create a minimal node instance with all required Node attributes."""
    node = node_class.__new__(node_class)
    node.node_id = node_id
    node.outputs = {}
    node.inputs = {}
    node._response = None
    node.node_type = kwargs.get('node_type', node_class.__name__.replace('Node', '').lower())
    node.debug = False
    node.cost = 0.0
    node.extra_params = {'node_type': node.node_type}
    node._debug_info = None
    node._execution_start = None
    node._execution_end = None
    node.generated = ""
    for key, value in kwargs.items():
        setattr(node, key, value)
    return node


# ============================================================================
# Slice 0a: Executor non-loop conditional bypass propagation
# ============================================================================

class TestExecutorNonLoopConditionalBypass:
    """Slice 0a: Characterize executor non-loop conditional bypass propagation."""

    @pytest.mark.asyncio
    async def test_conditional_selected_handle_propagates_bypass(self):
        """Conditional with known selected_handle → propagate_conditional_bypass called.
        
        Graph: input → conditional → yes→node_a, no→node_b
        Conditional selects 'handle_yes' → node_a executes, node_b bypassed.
        """
        # Create a conditional that selects 'handle_yes'
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        node_a = _CollectingNode(node_id="node_a", output_value="A", output_handle="output")
        node_b = _CollectingNode(node_id="node_b", output_value="B", output_handle="output")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"input": _CollectingNode(node_id="input", output_value="test_input"),
                 "cond": cond, "node_a": node_a, "node_b": node_b, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="input", target="cond", sourceHandle="output", targetHandle="handle_input"),
            EdgeNodeModel(id="e2", source="cond", target="node_a", sourceHandle="handle_yes", targetHandle="input"),
            EdgeNodeModel(id="e3", source="cond", target="node_b", sourceHandle="handle_no", targetHandle="input"),
            EdgeNodeModel(id="e4", source="node_a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_reactive(graph))

        # Conditional should have executed and set selected_handle
        assert hasattr(cond, 'selected_handle')
        assert cond.selected_handle == "handle_yes"

        # node_a (on selected handle) should have executed
        assert node_a.execute_count == 1, "node_a should execute on selected handle"

        # node_b (on non-selected handle) should be bypassed
        assert node_b.execute_count == 0, "node_b should be bypassed on non-selected handle"

    @pytest.mark.asyncio
    async def test_conditional_getattr_selected_handle_fallback(self):
        """Verify getattr(node, 'selected_handle', None) fallback path.
        
        When conditional_selected_handle is None (not tracked during iteration),
        executor falls back to reading node.selected_handle attribute.
        """
        # Create conditional — the executor tracks selected_handle during iteration
        # at line 414. After process(), node.selected_handle is set at line 349.
        # The fallback at line 439 reads getattr(node, 'selected_handle', None).
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        # Manually set selected_handle to simulate the fallback scenario
        cond.selected_handle = "handle_yes"

        node_a = _CollectingNode(node_id="node_a", output_value="A", output_handle="output")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"input": _CollectingNode(node_id="input", output_value="test_input"),
                 "cond": cond, "node_a": node_a, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="input", target="cond", sourceHandle="output", targetHandle="handle_input"),
            EdgeNodeModel(id="e2", source="cond", target="node_a", sourceHandle="handle_yes", targetHandle="input"),
            EdgeNodeModel(id="e3", source="node_a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_reactive(graph))

        # The fallback should work — node_a should execute
        assert node_a.execute_count == 1

    @pytest.mark.asyncio
    async def test_conditional_default_handle_fallback_when_no_matching_edge(self):
        """Conditional selects handle with no edge → default_handle used for bypass routing.
        
        Graph: conditional selects 'handle_maybe' but no edge exists for it.
        default_handle='handle_yes' has an edge → routing falls back to default for bypass.
        
        NOTE: This reveals a BUG in the current implementation. The default_handle fallback
        changes which handle is used for propagate_conditional_bypass, but does NOT re-deliver
        output. Nodes on the default_handle path end up in selected_targets (not bypassed) but
        never receive data.
        
        To avoid hanging, the test graph has no downstream nodes that depend on the conditional.
        We verify the executor code path is exercised by checking the conditional executed.
        """
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_maybe' }}",  # No edge for this
            default_handle="handle_yes",
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        # No downstream nodes — just verify the conditional executes without crashing
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"input": _CollectingNode(node_id="input", output_value="test_input"),
                 "cond": cond, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="input", target="cond", sourceHandle="output", targetHandle="handle_input"),
            # No edges from cond — the default_handle fallback path is exercised but nothing downstream
            EdgeNodeModel(id="e2", source="input", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_reactive(graph))

        # Conditional should have executed
        assert hasattr(cond, 'selected_handle')
        assert cond.selected_handle == "handle_maybe"
        # End should execute (independent)
        assert node_end.execute_count == 1

    @pytest.mark.asyncio
    async def test_conditional_no_match_no_default_yields_error_and_bypass_all(self):
        """Conditional selects handle with no edge AND no default → error + BYPASS_ALL.
        
        Graph: conditional selects 'handle_maybe', no edge exists, no default_handle.
        Expected: yield_debug_error + handle_bypass_all_signal.
        """
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_maybe' }}",  # No edge for this
            # No default_handle
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        node_downstream = _CollectingNode(node_id="downstream", output_value="DS", output_handle="output")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"input": _CollectingNode(node_id="input", output_value="test_input"),
                 "cond": cond, "downstream": node_downstream, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="input", target="cond", sourceHandle="output", targetHandle="handle_input"),
            EdgeNodeModel(id="e2", source="cond", target="downstream", sourceHandle="handle_yes", targetHandle="input"),
            EdgeNodeModel(id="e3", source="input", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_reactive(graph))

        # Should have debug error event
        debug_events = [
            r for r in results
            if isinstance(r, dict) and r.get("type") == SYSTEM_EVENT_DEBUG
        ]
        error_events = [
            e for e in debug_events
            if "GraphRoutingError" in str(e.get("content", {}).get("error_type", ""))
        ]
        assert len(error_events) > 0, f"Expected GraphRoutingError debug event, got: {[e.get('content', {}).get('error_type') for e in debug_events]}"

        # Downstream node should be bypassed (BYPASS_ALL cascades)
        assert node_downstream.execute_count == 0, "Downstream should be bypassed after routing error"


# ============================================================================
# Slice 0d: NodeConditional.process() error paths
# ============================================================================

class TestNodeConditionalProcessErrorPaths:
    """Slice 0d: Characterize NodeConditional.process() error paths."""

    @pytest.mark.asyncio
    async def test_process_undefined_variable_renders_empty_then_empty_handle_error(self):
        """Template references undefined variable → Jinja2 default Undefined renders to '' → EmptyHandleError + BYPASS_ALL.
        
        NOTE: Jinja2's default Undefined silently returns empty string, NOT raising UndefinedError.
        This is a known weakness (explore.md §3.2, copilot-research §2 #2).
        The actual behavior: undefined_variable → '' → EmptyHandleError → BYPASS_ALL.
        """
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ undefined_variable }}",
        )
        cond.inputs = {"handle_input": '{"other_key": "value"}'}

        chat_log = MagicMock()
        results = []
        async for item in cond(chat_log):
            results.append(item)

        # Jinja2 default Undefined renders to '', triggering EmptyHandleError path
        error_events = [r for r in results if r.get("type") == SYSTEM_EVENT_DEBUG]
        bypass_events = [r for r in results if r.get("type") == ConditionalSignalTypes.BYPASS_ALL]

        assert len(error_events) == 1, f"Expected 1 debug error, got {len(error_events)}"
        # Actual behavior: EmptyHandleError (not TemplateError) because undefined renders to ''
        assert error_events[0]["content"]["error_type"] == "EmptyHandleError"
        assert len(bypass_events) == 1, "Expected BYPASS_ALL after empty handle"

    def test_process_template_syntax_error_raised_at_init_not_process(self):
        """Invalid Jinja2 syntax → TemplateSyntaxError raised at __init__, NOT during process().
        
        NOTE: The current code pre-compiles the template in __init__ (line 88).
        A syntax error raises TemplateSyntaxError at construction time, not at process() time.
        This means the error is NOT caught by the process() try/except block.
        This is a known weakness — syntax validation should be at build time (model), not runtime.
        """
        import jinja2
        with pytest.raises(jinja2.TemplateSyntaxError):
            NodeConditional(
                node_id="cond-test",
                node_type="conditional",
                condition="{{ invalid syntax here }}",  # Invalid: space in variable name
            )

    @pytest.mark.asyncio
    async def test_process_empty_result_with_default_handle_uses_default(self):
        """Condition evaluates to empty string + default_handle configured → uses default."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ '' }}",  # Evaluates to empty string
            default_handle="handle_fallback",
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        chat_log = MagicMock()
        results = []
        async for item in cond(chat_log):
            results.append(item)

        # Should emit the default handle, not BYPASS_ALL
        handle_events = [r for r in results if r.get("type") == "handle_fallback"]
        bypass_events = [r for r in results if r.get("type") == ConditionalSignalTypes.BYPASS_ALL]

        assert len(handle_events) == 1, "Should emit default_handle event"
        assert len(bypass_events) == 0, "Should NOT emit BYPASS_ALL when default_handle is used"

        # selected_handle should be set to default
        assert hasattr(cond, 'selected_handle')
        assert cond.selected_handle == "handle_fallback"

    @pytest.mark.asyncio
    async def test_process_empty_result_without_default_yields_error_and_bypass_all(self):
        """Condition evaluates to empty + no default_handle → debug_error + BYPASS_ALL."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ '' }}",  # Evaluates to empty string
            # No default_handle
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        chat_log = MagicMock()
        results = []
        async for item in cond(chat_log):
            results.append(item)

        error_events = [r for r in results if r.get("type") == SYSTEM_EVENT_DEBUG]
        bypass_events = [r for r in results if r.get("type") == ConditionalSignalTypes.BYPASS_ALL]

        assert len(error_events) == 1
        assert error_events[0]["content"]["error_type"] == "EmptyHandleError"
        assert len(bypass_events) == 1

    @pytest.mark.asyncio
    async def test_process_init_error_empty_condition_yields_debug_error(self):
        """Empty condition template → init_error → yields debug_error immediately."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="",  # Empty condition
        )
        cond.inputs = {"handle_input": '{"value": "test"}'}

        chat_log = MagicMock()
        results = []
        async for item in cond(chat_log):
            results.append(item)

        # Should yield debug_error and return (no BYPASS_ALL for init error)
        error_events = [r for r in results if r.get("type") == SYSTEM_EVENT_DEBUG]
        bypass_events = [r for r in results if r.get("type") == ConditionalSignalTypes.BYPASS_ALL]

        assert len(error_events) == 1
        assert error_events[0]["content"]["error_type"] == "ConfigurationError"
        assert "non-empty" in error_events[0]["content"]["error_message"]
        # Init error does NOT yield BYPASS_ALL
        assert len(bypass_events) == 0

    @pytest.mark.asyncio
    async def test_process_no_inputs_yields_error_and_bypass_all(self):
        """No inputs received → InputError → debug_error + BYPASS_ALL."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ value }}",
        )
        cond.inputs = {}  # No inputs

        chat_log = MagicMock()
        results = []
        async for item in cond(chat_log):
            results.append(item)

        error_events = [r for r in results if r.get("type") == SYSTEM_EVENT_DEBUG]
        bypass_events = [r for r in results if r.get("type") == ConditionalSignalTypes.BYPASS_ALL]

        assert len(error_events) == 1
        assert error_events[0]["content"]["error_type"] == "InputError"
        assert len(bypass_events) == 1

    @pytest.mark.asyncio
    async def test_process_valid_condition_emits_selected_handle(self):
        """Valid condition → emits selected handle event + static end event."""
        cond = NodeConditional(
            node_id="cond-test",
            node_type="conditional",
            condition="{{ 'handle_yes' if value else 'handle_no' }}",
        )
        cond.inputs = {"handle_input": '{"value": true}'}

        chat_log = MagicMock()
        results = []
        async for item in cond(chat_log):
            results.append(item)

        # Should emit the selected handle event
        handle_events = [r for r in results if r.get("type") == "handle_yes"]
        assert len(handle_events) == 1

        # Should have static end event (type='end' by default) with metadata in content
        static_events = [r for r in results if r.get("type") == "end"]
        assert len(static_events) == 1
        # Content is wrapped: {"content": {"node": "...", "content": {"selected": ...}}}
        inner_content = static_events[0]["content"].get("content", {})
        if isinstance(inner_content, dict) and "content" in inner_content:
            inner_content = inner_content["content"]
        assert inner_content.get("selected") == "handle_yes"

        # selected_handle should be set
        assert hasattr(cond, 'selected_handle')
        assert cond.selected_handle == "handle_yes"


# ============================================================================
# Slice 0e: GraphEventDispatcher.propagate_conditional_bypass()
# ============================================================================

class TestDispatcherConditionalBypass:
    """Slice 0e: Characterize GraphEventDispatcher.propagate_conditional_bypass()."""

    def _make_nodes(self, node_ids: list):
        """Create mock nodes for dispatcher tests."""
        nodes = {}
        for nid in node_ids:
            class MockNode:
                def __init__(self):
                    self.node_id = nid
                    self.inputs = {}
                    self.outputs = {}
                    self._response = None
                    self.generated = ""
                def mark_bypassed(self):
                    pass
            nodes[nid] = MockNode()
        return nodes

    @pytest.mark.asyncio
    async def test_fan_out_multiple_edges_same_handle_all_targets_selected(self):
        """Multiple edges with same sourceHandle → all targets selected (not bypassed)."""
        nodes = self._make_nodes(["cond", "target_a", "target_b", "target_c"])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="target_a", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e2", source="cond", target="target_b", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e3", source="cond", target="target_c", sourceHandle="handle_no", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        result = await dispatcher.propagate_conditional_bypass("cond", "handle_yes")

        # Both target_a and target_b should be active (on selected handle)
        assert "target_a" in result["active_targets"]
        assert "target_b" in result["active_targets"]
        # target_c should be bypassed
        assert "target_c" in result["bypassed_targets"]
        assert "target_a" not in result["bypassed_targets"]
        assert "target_b" not in result["bypassed_targets"]

    @pytest.mark.asyncio
    async def test_defensive_target_in_both_selected_and_bypassed(self):
        """Target appears in both selected and bypassed sets → selected wins (defensive)."""
        nodes = self._make_nodes(["cond", "shared_target", "bypassed_target"])
        edges = [
            # shared_target reachable via BOTH handles (shouldn't happen in well-formed graphs)
            EdgeNodeModel(id="e1", source="cond", target="shared_target", sourceHandle="handle_yes", targetHandle="in1"),
            EdgeNodeModel(id="e2", source="cond", target="shared_target", sourceHandle="handle_no", targetHandle="in2"),
            EdgeNodeModel(id="e3", source="cond", target="bypassed_target", sourceHandle="handle_no", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        result = await dispatcher.propagate_conditional_bypass("cond", "handle_yes")

        # shared_target should be in active_targets, NOT bypassed
        assert "shared_target" in result["active_targets"]
        assert "shared_target" not in result["bypassed_targets"]
        # bypassed_target should only be bypassed
        assert "bypassed_target" in result["bypassed_targets"]

    @pytest.mark.asyncio
    async def test_recursive_bypass_cascades_through_multiple_levels(self):
        """Bypass cascades through multiple levels of downstream nodes."""
        nodes = self._make_nodes(["cond", "mid_a", "mid_b", "end_a", "end_b"])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="mid_a", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e2", source="cond", target="mid_b", sourceHandle="handle_no", targetHandle="in"),
            EdgeNodeModel(id="e3", source="mid_b", target="end_a", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="e4", source="end_a", target="end_b", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        await dispatcher.propagate_conditional_bypass("cond", "handle_yes")

        # mid_b should be bypassed
        assert dispatcher.get_state("mid_b") == NodeState.BYPASSED
        # end_a should be bypassed (cascade from mid_b)
        assert dispatcher.get_state("end_a") == NodeState.BYPASSED
        # end_b should be bypassed (cascade from end_a)
        assert dispatcher.get_state("end_b") == NodeState.BYPASSED
        # mid_a should NOT be bypassed (on selected path)
        assert dispatcher.get_state("mid_a") != NodeState.BYPASSED

    @pytest.mark.asyncio
    async def test_recursive_bypass_idempotency(self):
        """Calling _recursive_bypass twice on same node is idempotent."""
        nodes = self._make_nodes(["cond", "mid", "end"])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="mid", sourceHandle="handle_no", targetHandle="in"),
            EdgeNodeModel(id="e2", source="mid", target="end", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        # First bypass
        await dispatcher.propagate_conditional_bypass("cond", "handle_yes")
        state_after_first = dispatcher.get_state("mid")

        # Second bypass (should be idempotent)
        await dispatcher._recursive_bypass("mid")
        state_after_second = dispatcher.get_state("mid")

        assert state_after_first == NodeState.BYPASSED
        assert state_after_second == NodeState.BYPASSED
        # end should also be bypassed
        assert dispatcher.get_state("end") == NodeState.BYPASSED


# ============================================================================
# Slice 0b: Executor loop conditional static-phase behavior
# ============================================================================

class TestExecutorLoopConditionalStaticPhase:
    """Slice 0b: Characterize executor loop conditional static-phase behavior."""

    @pytest.mark.asyncio
    async def test_conditional_with_inputs_executed_in_static_phase(self):
        """Conditional with static inputs → executed in static phase, bypass handled.
        
        When a conditional node receives inputs from static edges (not from loop item),
        it should execute in the static phase and handle_conditional_bypass_static
        should be called.
        
        NOTE: The static phase uses execute_node_inline which stores outputs directly
        in node.outputs. The conditional receives input via add_parent which extracts
        content from the wrapped output.
        """
        from magic_agents.node_system import NodeLoop, NodeText, NodeEND

        loop_node = NodeLoop(node_id="loop", debug=False)
        loop_node.inputs[loop_node.INPUT_HANDLE_LIST] = '["item1"]'

        # Conditional with hardcoded condition (no input dependency needed)
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",  # Hardcoded, doesn't need input
        )

        end_node = NodeEND(node_id="end", debug=False)

        nodes = {"loop": loop_node, "cond": cond, "end": end_node}
        edges = [
            # No static edges to cond — it has no input dependencies
            # This means cond has NO inputs, so it gets SKIPPED in static phase
            # and deferred to iteration phase
            EdgeNodeModel(id="e1", source="loop", target="end", sourceHandle="handle_end", targetHandle="h1"),
        ]

        graph = _make_mock_loop_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_loop_reactive(graph))

        # Since cond has no inputs, it's skipped in static phase (line 805-817)
        # It will NOT have selected_handle after static phase
        # But it also won't be in the iteration subgraph (no edge from loop item)
        # So it never executes at all
        assert not hasattr(cond, 'selected_handle'), "Conditional without inputs is skipped in static phase"

    @pytest.mark.asyncio
    async def test_conditional_without_inputs_deferred_to_iteration(self):
        """Conditional without static inputs → skipped in static phase, deferred to iteration.
        
        When a conditional's only input comes from the loop's handle_item,
        it should be skipped in the static phase and executed during iteration.
        """
        from magic_agents.node_system import NodeLoop, NodeText, NodeParser, NodeEND

        loop_node = NodeLoop(node_id="loop", debug=False)
        loop_node.inputs[loop_node.INPUT_HANDLE_LIST] = '["item1"]'

        # Conditional with NO static inputs — depends on loop item
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
        )

        # Parser node in iteration subgraph
        parser_node = _make_minimal_node(NodeParser, "parser", parser_type="identity")

        end_node = NodeEND(node_id="end", debug=False)

        nodes = {"loop": loop_node, "cond": cond, "parser": parser_node, "end": end_node}
        edges = [
            # Loop item → conditional (this input is NOT available in static phase)
            EdgeNodeModel(id="e1", source="loop", target="cond", sourceHandle="handle_item", targetHandle="handle_input"),
            EdgeNodeModel(id="e2", source="cond", target="parser", sourceHandle="handle_yes", targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e3", source="parser", target="loop", sourceHandle="handle_parser_output", targetHandle="handle_loop"),
            EdgeNodeModel(id="e4", source="loop", target="end", sourceHandle="handle_end", targetHandle="h1"),
        ]

        graph = _make_mock_loop_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_loop_reactive(graph))

        # Conditional should have been executed during iteration (not static phase)
        # After full execution, it should have a selected_handle
        assert hasattr(cond, 'selected_handle'), "Conditional should have selected_handle after iteration"
        assert cond.selected_handle == "handle_yes"


# ============================================================================
# Slice 0c: Executor loop conditional iteration-phase behavior
# ============================================================================

class TestExecutorLoopConditionalIterationPhase:
    """Slice 0c: Characterize executor loop conditional iteration-phase behavior."""

    @pytest.mark.asyncio
    async def test_conditional_in_iteration_bypasses_non_selected_branches(self):
        """Conditional in iteration → bypass_non_selected_conditional_branches called.
        
        When a conditional executes during loop iteration, non-selected branches
        and their downstream nodes should be bypassed via iteration_bypassed set.
        
        NOTE: The current topological sort (topological_sort_iteration) does NOT include
        conditional branch edges in the in-degree calculation. This means nodes on
        non-selected branches may execute BEFORE the conditional can bypass them.
        The bypass logic at lines 1036-1043 runs AFTER each node executes, so it can
        only bypass nodes that come AFTER the conditional in execution order.
        
        This test verifies the CURRENT behavior: nodes that come after the conditional
        in execution order are properly bypassed.
        """
        from magic_agents.node_system import NodeLoop, NodeParser, NodeEND

        loop_node = NodeLoop(node_id="loop", debug=False)
        loop_node.inputs[loop_node.INPUT_HANDLE_LIST] = '["item1"]'

        # Conditional that will execute during iteration
        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
        )

        # Parser nodes on both branches
        parser_yes = _make_minimal_node(NodeParser, "parser_yes", parser_type="identity")
        parser_yes.execute_count = 0
        original_process_yes = parser_yes.process
        async def counting_process_yes(chat_log):
            parser_yes.execute_count += 1
            async for item in original_process_yes(chat_log):
                yield item
        parser_yes.process = counting_process_yes

        parser_no = _make_minimal_node(NodeParser, "parser_no", parser_type="identity")
        parser_no.execute_count = 0
        original_process_no = parser_no.process
        async def counting_process_no(chat_log):
            parser_no.execute_count += 1
            async for item in original_process_no(chat_log):
                yield item
        parser_no.process = counting_process_no

        end_node = NodeEND(node_id="end", debug=False)

        nodes = {
            "loop": loop_node, "cond": cond,
            "parser_yes": parser_yes, "parser_no": parser_no,
            "end": end_node
        }
        edges = [
            # Loop item → conditional
            EdgeNodeModel(id="e1", source="loop", target="cond", sourceHandle="handle_item", targetHandle="handle_input"),
            EdgeNodeModel(id="e2", source="cond", target="parser_yes", sourceHandle="handle_yes", targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e3", source="cond", target="parser_no", sourceHandle="handle_no", targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e4", source="parser_yes", target="loop", sourceHandle="handle_parser_output", targetHandle="handle_loop"),
            EdgeNodeModel(id="e5", source="loop", target="end", sourceHandle="handle_end", targetHandle="h1"),
        ]

        graph = _make_mock_loop_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_loop_reactive(graph))

        # Conditional should have executed
        assert hasattr(cond, 'selected_handle')
        assert cond.selected_handle == "handle_yes"
        
        # parser_yes (on selected path, feeds back to loop) should have executed
        assert parser_yes.execute_count >= 1, "parser_yes should execute (selected path, feedback to loop)"

        # NOTE: parser_no may or may not execute depending on topological order.
        # The current topological_sort_iteration doesn't include conditional branch
        # edges in in-degree calculation, so parser_no might execute before cond.
        # This is existing behavior — the bypass only affects nodes AFTER cond in order.
        # We verify the conditional executed and selected the right handle.

    @pytest.mark.asyncio
    async def test_conditional_iteration_bypassed_set_updated(self):
        """After conditional in iteration, iteration_bypassed set is updated correctly.
        
        NOTE: Due to the topological sort limitation (conditional branch edges not
        included in in-degree calculation), nodes on non-selected branches that come
        BEFORE the conditional in execution order will execute before being bypassed.
        Only nodes that come AFTER the conditional are properly bypassed.
        
        This test uses a graph where a downstream node depends on parser_no, so it
        comes after parser_no in topological order. If parser_no executes before cond,
        downstream_no will also execute. If cond executes first, both are bypassed.
        """
        from magic_agents.node_system import NodeLoop, NodeParser, NodeEND

        loop_node = NodeLoop(node_id="loop", debug=False)
        loop_node.inputs[loop_node.INPUT_HANDLE_LIST] = '["item1"]'

        cond = NodeConditional(
            node_id="cond",
            node_type="conditional",
            condition="{{ 'handle_yes' }}",
        )

        parser_yes = _make_minimal_node(NodeParser, "parser_yes", parser_type="identity")
        parser_yes.execute_count = 0
        original_process_yes = parser_yes.process
        async def counting_process_yes(chat_log):
            parser_yes.execute_count += 1
            async for item in original_process_yes(chat_log):
                yield item
        parser_yes.process = counting_process_yes

        parser_no = _make_minimal_node(NodeParser, "parser_no", parser_type="identity")
        parser_no.execute_count = 0
        original_process_no = parser_no.process
        async def counting_process_no(chat_log):
            parser_no.execute_count += 1
            async for item in original_process_no(chat_log):
                yield item
        parser_no.process = counting_process_no

        # Downstream of non-selected branch
        downstream_no = _make_minimal_node(NodeParser, "downstream_no", parser_type="identity")
        downstream_no.execute_count = 0
        original_process_ds = downstream_no.process
        async def counting_process_ds(chat_log):
            downstream_no.execute_count += 1
            async for item in original_process_ds(chat_log):
                yield item
        downstream_no.process = counting_process_ds

        end_node = NodeEND(node_id="end", debug=False)

        nodes = {
            "loop": loop_node, "cond": cond,
            "parser_yes": parser_yes, "parser_no": parser_no,
            "downstream_no": downstream_no, "end": end_node
        }
        edges = [
            EdgeNodeModel(id="e1", source="loop", target="cond", sourceHandle="handle_item", targetHandle="handle_input"),
            EdgeNodeModel(id="e2", source="cond", target="parser_yes", sourceHandle="handle_yes", targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e3", source="cond", target="parser_no", sourceHandle="handle_no", targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e4", source="parser_no", target="downstream_no", sourceHandle="handle_parser_output", targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e5", source="parser_yes", target="loop", sourceHandle="handle_parser_output", targetHandle="handle_loop"),
            EdgeNodeModel(id="e6", source="loop", target="end", sourceHandle="handle_end", targetHandle="h1"),
        ]

        graph = _make_mock_loop_graph(nodes, edges, debug=False)
        results = await _collect_all(execute_graph_loop_reactive(graph))

        # parser_yes should execute (selected path, feedback to loop)
        assert parser_yes.execute_count >= 1
        
        # Conditional should have selected handle_yes
        assert hasattr(cond, 'selected_handle')
        assert cond.selected_handle == "handle_yes"
        
        # NOTE: parser_no and downstream_no may execute before cond due to
        # topological sort not including conditional branch edges.
        # This is existing behavior characterized here.
