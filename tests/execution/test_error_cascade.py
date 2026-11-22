"""
Integration Tests: Error Cascade Bypass Propagation (Task 9.7).

Tests that:
1. Graph with failing upstream node bypasses downstream nodes
2. Downstream nodes are marked BYPASSED (not ERROR, not PENDING)
3. on_node_bypass hook fires for each bypassed node
4. Bypass reason includes "upstream_error" with source node ID
5. Nodes NOT downstream of the error still execute normally
6. Multiple downstream levels are all bypassed

NOTE: This test required a production fix in Node.__call__():
  The except block was changed from `return` to `raise` to let the
  exception propagate to execute_single_node, which then marks the
  node as ERROR and propagates the error cascade bypass.
"""
import asyncio
import pytest
from typing import Any, AsyncGenerator, Dict

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.node_system.Node import Node
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.util.aio import wait_for


# ─── Test Node Types ───────────────────────────────────────────────────────────

class FailingNode(Node):
    """Node that raises an exception during process()."""

    __test__ = False

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, node_type="failing", **kwargs)

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "debug", "content": {"node": self.node_id, "event": "before_failure"}}
        raise ValueError("Intentional upstream failure for error cascade test")


class BypassTestNode(Node):
    """Simple node that passes input through (not collected by pytest)."""

    __test__ = False

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, node_type="bypass_test", **kwargs)

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "output", "content": {"message": "complete"}}


# ─── Recording Hook ────────────────────────────────────────────────────────────

class RecordingBypassHook:
    """FlowHooks that records on_node_bypass invocations.

    Only implements the methods that will be invoked during error cascade.
    Other FlowHooks methods are omitted (getattr will return None, which
    HookRegistry.invoke skips gracefully).
    """

    def __init__(self):
        self.bypass_calls: list[dict] = []

    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        self.bypass_calls.append({
            "node_id": context.node_id,
            "reason": reason,
            "metadata": context.metadata,
            "context": context,
        })

    async def on_node_start(self, context: HookContext) -> None:
        pass

    async def on_node_end(self, context: HookContext) -> None:
        pass

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        pass

    async def on_graph_start(self, context: HookContext) -> None:
        pass

    async def on_graph_end(self, context: HookContext) -> None:
        pass


# ─── Test Helpers ──────────────────────────────────────────────────────────────

def make_graph(nodes: Dict[str, Node], edges: list = None) -> AgentFlowModel:
    """Create an AgentFlowModel graph with specified nodes and edges."""
    if edges is None:
        edges = []
    return AgentFlowModel(nodes=nodes, edges=edges, debug=True)


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestErrorCascadeBasic:
    """9.7.2: Graph with failing upstream node cascades bypass."""

    @pytest.mark.asyncio
    async def test_upstream_error_bypasses_downstream(self):
        """
        When upstream node fails, downstream nodes must be bypassed.
        Nodes should NOT hang indefinitely waiting for failed node output.
        """
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "downstream": BypassTestNode(node_id="downstream"),
                "end": BypassTestNode(node_id="end"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="downstream"),
                EdgeNodeModel(source="downstream", target="end"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        assert len(bypass_hook.bypass_calls) >= 1, (
            "Expected at least one on_node_bypass invocation"
        )

        bypassed_ids = [c["node_id"] for c in bypass_hook.bypass_calls]
        assert "downstream" in bypassed_ids, (
            f"Downstream node should be bypassed, got: {bypassed_ids}"
        )

    @pytest.mark.asyncio
    async def test_bypass_reason_is_upstream_error(self):
        """Bypass reason must be 'upstream_error' for error cascades."""
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "downstream": BypassTestNode(node_id="downstream"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="downstream"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        assert [(call["node_id"], call["reason"]) for call in bypass_hook.bypass_calls] == [
            ("downstream", "upstream_error"),
        ]

    @pytest.mark.asyncio
    async def test_bypass_metadata_includes_error_source(self):
        """Bypass metadata must include the source upstream error node ID."""
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "downstream": BypassTestNode(node_id="downstream"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="downstream"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        assert [call["metadata"].get("upstream_error_node") for call in bypass_hook.bypass_calls] == ["failing"]


class TestErrorCascadeDepth:
    """9.7.3: Error cascade propagates through multiple depths."""

    @pytest.mark.asyncio
    async def test_cascade_propagates_to_multiple_levels(self):
        """
        Error cascade must propagate through multiple levels:
        failing → level1 → level2 → level3. All must be bypassed.
        """
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "level1": BypassTestNode(node_id="level1"),
                "level2": BypassTestNode(node_id="level2"),
                "level3": BypassTestNode(node_id="level3"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="level1"),
                EdgeNodeModel(source="level1", target="level2"),
                EdgeNodeModel(source="level2", target="level3"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        bypassed_ids = [c["node_id"] for c in bypass_hook.bypass_calls]
        for level in ["level1", "level2", "level3"]:
            assert level in bypassed_ids, (
                f"Node '{level}' should be bypassed in cascade, got: {bypassed_ids}"
            )

    @pytest.mark.asyncio
    async def test_independent_nodes_still_execute(self):
        """
        Nodes NOT downstream of the failing node must still execute normally.
        Only the error branch should be bypassed.
        """
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "downstream": BypassTestNode(node_id="downstream"),
                "independent": BypassTestNode(node_id="independent"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="downstream"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        bypassed_ids = [c["node_id"] for c in bypass_hook.bypass_calls]
        assert "downstream" in bypassed_ids, "Downstream must be bypassed"
        assert "independent" not in bypassed_ids, (
            "Independent node should NOT be bypassed"
        )


class TestErrorCascadeHookInvocation:
    """9.7.4/9.7.5: on_node_bypass fires per bypassed node with correct context."""

    @pytest.mark.asyncio
    async def test_on_node_bypass_fires_per_bypassed_node(self):
        """Each bypassed node must get its own on_node_bypass invocation."""
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "a": BypassTestNode(node_id="a"),
                "b": BypassTestNode(node_id="b"),
                "c": BypassTestNode(node_id="c"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="a"),
                EdgeNodeModel(source="a", target="b"),
                EdgeNodeModel(source="b", target="c"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        bypassed_ids = [c["node_id"] for c in bypass_hook.bypass_calls]
        from collections import Counter
        counts = Counter(bypassed_ids)
        for nid in ["a", "b", "c"]:
            assert counts.get(nid, 0) == 1, (
                f"Node '{nid}' should have exactly 1 bypass call, got {counts.get(nid, 0)}"
            )

    @pytest.mark.asyncio
    async def test_bypass_hook_receives_correct_context(self):
        """Bypass hook must receive context with the bypassed node's ID."""
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)

        graph = make_graph(
            nodes={
                "failing": FailingNode(node_id="failing"),
                "downstream": BypassTestNode(node_id="downstream"),
            },
            edges=[
                EdgeNodeModel(source="failing", target="downstream"),
            ],
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        assert len(bypass_hook.bypass_calls) >= 1
        call = bypass_hook.bypass_calls[0]
        assert call["context"].node_id == "downstream", (
            f"Expected context.node_id='downstream', got '{call['context'].node_id}'"
        )


# ─── Edge-scoped cascade: fan-in keeps its in-flight inputs ───────────────────

class TimedNode(Node):
    """Waits ``delay`` seconds, then emits ``value`` on ``out`` or raises."""

    __test__ = False

    def __init__(self, node_id: str, value: str, delay: float, fail: bool = False, **kwargs):
        super().__init__(node_id=node_id, node_type="timed", **kwargs)
        self.value, self.delay, self.fail = value, delay, fail

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ValueError(f"{self.node_id} failed")
        yield self.yield_static(self.value, content_type="out")


class FanInNode(Node):
    """Records every run's inputs and emits them joined."""

    __test__ = False

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, node_type="fan_in", **kwargs)
        self.runs: list = []

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        self.runs.append(dict(self.inputs))
        yield self.yield_static("|".join(f"{k}={v}" for k, v in sorted(self.inputs.items())), content_type="out")


def fan_in_graph(delays: Dict[str, float]):
    """a, b (fails), c -> fan; b -> solo -> after (b's only consumers)."""
    nodes = {
        "a": TimedNode("a", "AAA", delays["a"]),
        "b": TimedNode("b", "BBB", delays["b"], fail=True),
        "c": TimedNode("c", "CCC", delays["c"]),
        "fan": FanInNode("fan"),
        "solo": FanInNode("solo"),
        "after": FanInNode("after"),
    }
    edges = [
        EdgeNodeModel(id="a-fan", source="a", target="fan", sourceHandle="out", targetHandle="a"),
        EdgeNodeModel(id="b-fan", source="b", target="fan", sourceHandle="out", targetHandle="b"),
        EdgeNodeModel(id="c-fan", source="c", target="fan", sourceHandle="out", targetHandle="c"),
        EdgeNodeModel(id="b-solo", source="b", target="solo", sourceHandle="out", targetHandle="b"),
        EdgeNodeModel(id="solo-after", source="solo", target="after", sourceHandle="out", targetHandle="x"),
    ]
    return AgentFlowModel(nodes=nodes, edges=edges, debug=True, timeout=5)


class TestEdgeScopedErrorCascade:
    """A failure bypasses only its own edges; the outcome never depends on arrival order."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("delays", [
        {"a": 0.0, "b": 0.05, "c": 0.15},   # a, then the failure, then slow c
        {"a": 0.1, "b": 0.0, "c": 0.15},    # the failure resolves first
        {"a": 0.0, "b": 0.15, "c": 0.05},   # the failure resolves last
    ], ids=["fail-middle", "fail-first", "fail-last"])
    async def test_fan_in_waits_for_in_flight_siblings_and_runs_once(self, delays):
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)
        graph = fan_in_graph(delays)
        result = {}
        await wait_for(
            _drain(execute_graph_reactive(graph, hooks=registry, result=result)), 3)

        assert graph.nodes["fan"].runs == [{"a": "AAA", "c": "CCC"}]
        assert graph.nodes["solo"].runs == [] and graph.nodes["after"].runs == []
        bypassed = {(c["node_id"], c["reason"]) for c in bypass_hook.bypass_calls}
        assert bypassed == {("solo", "upstream_error"), ("after", "upstream_error")}
        assert result["failed_nodes"] == ["b"]

    @pytest.mark.asyncio
    async def test_fan_in_whose_inputs_all_fail_is_bypassed(self):
        bypass_hook = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_graph(bypass_hook)
        nodes = {
            "a": TimedNode("a", "AAA", 0.0, fail=True),
            "b": TimedNode("b", "BBB", 0.05, fail=True),
            "fan": FanInNode("fan"),
        }
        edges = [
            EdgeNodeModel(id="a-fan", source="a", target="fan", sourceHandle="out", targetHandle="a"),
            EdgeNodeModel(id="b-fan", source="b", target="fan", sourceHandle="out", targetHandle="b"),
        ]
        graph = AgentFlowModel(nodes=nodes, edges=edges, debug=True, timeout=5)
        await wait_for(_drain(execute_graph_reactive(graph, hooks=registry)), 3)
        assert graph.nodes["fan"].runs == []
        assert [(c["node_id"], c["reason"]) for c in bypass_hook.bypass_calls] == [("fan", "upstream_error")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("order", ["fail-first", "fail-last"])
    async def test_fan_in_of_inner_flows_keeps_the_slower_sibling(self, order, monkeypatch):
        """Inner Flow producers (a failed sub-flow is a node error) behave the same."""
        from magic_agents.agt_flow import build, run_agent
        from magic_agents.node_system.NodeParser import NodeParser

        delays = {"a-p": 0.0, "b-p": 0.0, "c-p": 0.15} if order == "fail-first" else {"a-p": 0.0, "b-p": 0.15, "c-p": 0.05}
        original = NodeParser.process

        async def delayed(self, chat_log):
            await asyncio.sleep(delays.get(self.node_id, 0))
            async for item in original(self, chat_log):
                yield item

        monkeypatch.setattr(NodeParser, "process", delayed)

        def sub(prefix, template):
            return {"type": "graph", "nodes": [
                {"id": f"{prefix}-in", "type": "user_input"},
                {"id": f"{prefix}-p", "type": "parser", "data": {"text": template}},
                {"id": f"{prefix}-end", "type": "end"},
            ], "edges": [
                {"id": f"{prefix}1", "source": f"{prefix}-in", "target": f"{prefix}-p",
                 "sourceHandle": "handle_user_message", "targetHandle": "value"},
                {"id": f"{prefix}2", "source": f"{prefix}-p", "target": f"{prefix}-end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ]}

        nodes = [{"id": "input", "type": "user_input"},
                 {"id": "fmt", "type": "parser", "data": {"text": "A={{ a }}|B={{ b }}|C={{ c }}"}},
                 {"id": "end", "type": "end"}]
        edges = [{"id": "fmt-end", "source": "fmt", "target": "end",
                  "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"}]
        for name, template in (("a", "AAA"), ("b", "{{ missing.invalid() }}"), ("c", "CCC")):
            nodes.append({"id": f"inner-{name}", "type": "inner", "data": {"magic_flow": sub(name, template)}})
            edges += [{"id": f"in-{name}", "source": "input", "target": f"inner-{name}",
                       "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                      {"id": f"{name}-fmt", "source": f"inner-{name}", "target": "fmt",
                       "sourceHandle": "handle_execution_content", "targetHandle": name}]
        graph = build({"type": "graph", "timeout": 5, "nodes": nodes, "edges": edges}, message="go")
        await wait_for(_drain(run_agent(graph)), 5)
        assert graph.nodes["fmt"].outputs["handle_parser_output"]["content"] == "A=AAA|B=|C=CCC"


class OutputThenBypassAllNode(Node):
    """Yields a value, then signals failure with BYPASS_ALL."""

    __test__ = False

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, node_type="partial", **kwargs)

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        yield self.yield_static("PARTIAL", content_type="out")
        yield {"type": "__bypass_all__", "content": None}


@pytest.mark.asyncio
async def test_bypass_all_node_delivers_no_output_downstream():
    """A node that signals BYPASS_ALL failed: nothing it yielded is a result."""
    nodes = {"partial": OutputThenBypassAllNode("partial"), "after": FanInNode("after")}
    edges = [EdgeNodeModel(id="p-after", source="partial", target="after", sourceHandle="out", targetHandle="x")]
    graph = AgentFlowModel(nodes=nodes, edges=edges, debug=True, timeout=5)
    result = {}
    await wait_for(_drain(execute_graph_reactive(graph, result=result)), 3)
    assert graph.nodes["after"].runs == []
    assert result["failed_nodes"] == ["partial"]


async def _drain(generator):
    return [event async for event in generator]
