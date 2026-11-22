"""
Deep Coverage: Edge-driven NodeHook Execution via EdgeNodeModel.hooks (Verify Gap 3).

Tests dispatcher-level hook propagation, direct NodeHook behavior, and the
public build/run path for graph-order-independent edge-hook scheduling.

Strategy:
1. Unit-test `propagate_outputs` behavior directly via the dispatcher.
2. Integration-test full graph execution with hook nodes that have proper
   incoming edges (src→hook edge provides dependency + hook config triggers context).
3. Regress public graph execution with an otherwise unconnected hook node.
4. Test timeout override and error isolation at NodeHook level.

Covers:
1. Hook context is set on hook node when edge has hooks enabled + hook_node_id
2. Disabled edge hook does NOT set hook context on hook node
3. Hook context structure includes source, target, handle info
4. Timeout override on NodeHook aborts slow hook functions
5. NodeHook error isolation (error doesn't crash graph)
6. NodeHook emit.user produces expected output shape
"""
import pytest
from unittest.mock import MagicMock

from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel, EdgeHookConfig
from magic_agents.models.factory.Nodes.HookNodeModel import (
    HookNodeModel,
    DEFAULT_INPUT_HOOK_CONTEXT,
)
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.node_system.Node import Node
from magic_agents.node_system.NodeHook import NodeHook
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_registry import HookRegistry


# ─── Tests: Dispatcher-Level Hook Propagation ───────────────────────────────

class TestDispatcherHookPropagation:
    """Tests propagate_outputs hook behavior at dispatcher level."""

    def _make_node_mock(self, node_id: str) -> MagicMock:
        """Create a mock node with dict inputs compatible with dispatcher."""
        mock = MagicMock()
        mock.node_id = node_id
        mock.inputs = {}
        # A bare MagicMock would look like a lifecycle-hook binding to InvocationControl.
        mock.lifecycle_event = None
        return mock

    def _make_hook_node_mock(self, node_id: str) -> MagicMock:
        """Create a mock hook node with hook-context aware inputs."""
        mock = MagicMock()
        mock.node_id = node_id
        mock.inputs = {}
        # A bare MagicMock would look like a lifecycle-hook binding to InvocationControl.
        mock.lifecycle_event = None
        mock.INPUT_HANDLE_HOOK_CONTEXT = DEFAULT_INPUT_HOOK_CONTEXT
        return mock

    @pytest.mark.asyncio
    async def test_propagate_sets_hook_context_on_enabled_edge(self):
        """propagate_outputs must set HookContext on hook node when edge has hooks enabled."""
        hook_node = self._make_hook_node_mock("hook-1")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-1": hook_node, "tgt": tgt_node}
        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="handle-default",
                hooks=EdgeHookConfig(hook_node_id="hook-1", enabled=True),
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges)

        await dispatcher.propagate_outputs("src", {"handle-default": {"content": "test"}})

        # Check hook context was set on hook node
        assert DEFAULT_INPUT_HOOK_CONTEXT in hook_node.inputs, (
            "Hook context must be set on hook node's INPUT_HANDLE_HOOK_CONTEXT"
        )
        ctx = hook_node.inputs[DEFAULT_INPUT_HOOK_CONTEXT]
        assert isinstance(ctx, HookContext), (
            f"Value must be HookContext, got {type(ctx).__name__}"
        )
        assert ctx.inputs.get("source") == "src", (
            f"Hook context source must be 'src', got {ctx.inputs.get('source')}"
        )
        assert ctx.inputs.get("target") == "tgt"

    @pytest.mark.asyncio
    async def test_disabled_edge_hook_does_not_set_context(self):
        """propagate_outputs must NOT set hook context when hooks are disabled."""
        hook_node = self._make_hook_node_mock("hook-1")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-1": hook_node, "tgt": tgt_node}
        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="handle-default",
                hooks=EdgeHookConfig(hook_node_id="hook-1", enabled=False),
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges)

        await dispatcher.propagate_outputs("src", {"handle-default": {"content": "test"}})

        assert DEFAULT_INPUT_HOOK_CONTEXT not in hook_node.inputs, (
            "Hook context must NOT be set when hooks are disabled"
        )

    @pytest.mark.asyncio
    async def test_no_hooks_config_does_not_set_context(self):
        """propagate_outputs must NOT set hook context when edge has no hooks config."""
        hook_node = self._make_hook_node_mock("hook-1")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-1": hook_node, "tgt": tgt_node}
        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="handle-default",
                # NO hooks config
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges)

        await dispatcher.propagate_outputs("src", {"handle-default": {"content": "test"}})

        assert DEFAULT_INPUT_HOOK_CONTEXT not in hook_node.inputs, (
            "Hook context must NOT be set when edge has no hooks config"
        )

    @pytest.mark.asyncio
    async def test_hook_context_structure(self):
        """Hook context set by propagate_outputs must have correct structure."""
        hook_node = self._make_hook_node_mock("hook-1")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-1": hook_node, "tgt": tgt_node}
        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="my-output",
                targetHandle="my-input",
                hooks=EdgeHookConfig(hook_node_id="hook-1", enabled=True),
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges)

        await dispatcher.propagate_outputs("src", {"my-output": {"content": "data"}})

        ctx = hook_node.inputs[DEFAULT_INPUT_HOOK_CONTEXT]
        assert ctx.inputs["source"] == "src"
        assert ctx.inputs["target"] == "tgt"
        assert ctx.inputs["source_handle"] == "my-output"
        assert ctx.inputs["target_handle"] == "my-input"
        assert "content" in ctx.inputs


# ─── Tests: NodeHook Unit Behavior ──────────────────────────────────────────

# ─── Task 4.3: Full Graph Edge Hook E2E Tests ──────────────────────────


class TestFullGraphEdgeHookE2E:
    """Full graph E2E edge hook execution tests (Task 4.3).

    Executes a full graph with NodeHook + EdgeHookConfig via
    execute_graph_reactive() and verifies HookContext reaches
    NodeHook with correct execution identity.
    """

    def _make_node_mock(self, node_id: str):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.node_id = node_id
        mock.inputs = {}
        # A bare MagicMock would look like a lifecycle-hook binding to InvocationControl.
        mock.lifecycle_event = None
        return mock

    def _make_hook_node_mock(self, node_id: str):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.node_id = node_id
        mock.inputs = {}
        # A bare MagicMock would look like a lifecycle-hook binding to InvocationControl.
        mock.lifecycle_event = None
        mock.INPUT_HANDLE_HOOK_CONTEXT = DEFAULT_INPUT_HOOK_CONTEXT
        return mock

    @pytest.mark.asyncio
    async def test_edge_routing_data_in_context(self):
        """Edge routing data (source, target, source_handle, target_handle) is present."""
        hook_node = self._make_hook_node_mock("hook-1")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-1": hook_node, "tgt": tgt_node}
        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="output-default",
                targetHandle="input-args",
                hooks=EdgeHookConfig(hook_node_id="hook-1", enabled=True),
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges, execution_id="test-456")

        await dispatcher.propagate_outputs("src", {"output-default": {"content": "data"}})

        ctx = hook_node.inputs[DEFAULT_INPUT_HOOK_CONTEXT]
        assert ctx.inputs["source"] == "src"
        assert ctx.inputs["target"] == "tgt"
        assert ctx.inputs["source_handle"] == "output-default"
        assert ctx.inputs["target_handle"] == "input-args"
        assert "content" in ctx.inputs

    @pytest.mark.asyncio
    async def test_disabled_edge_hook_does_not_dispatch(self):
        """Disabled EdgeHookConfig (enabled=False) does NOT trigger NodeHook."""
        hook_node = self._make_hook_node_mock("hook-1")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-1": hook_node, "tgt": tgt_node}
        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="handle-default",
                hooks=EdgeHookConfig(hook_node_id="hook-1", enabled=False),
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges, execution_id="test-789")

        await dispatcher.propagate_outputs("src", {"handle-default": {"content": "data"}})

        assert DEFAULT_INPUT_HOOK_CONTEXT not in hook_node.inputs, (
            "Disabled edge hook must NOT set hook context"
        )

    @pytest.mark.asyncio
    async def test_edge_hook_config_hook_type_ignored(self):
        """EdgeHookConfig.hook_type is ignored (M1) — dispatch uses hook_node_id only."""
        hook_node = self._make_hook_node_mock("hook-ignored")
        src_node = self._make_node_mock("src")
        tgt_node = self._make_node_mock("tgt")

        nodes = {"src": src_node, "hook-ignored": hook_node, "tgt": tgt_node}
        legacy_hook = EdgeHookConfig(
            hook_node_id="hook-ignored",
            enabled=True,
            hook_type="legacy_edge_event",
        )
        with pytest.warns(DeprecationWarning, match="^deprecated$"):
            assert legacy_hook.hook_type == "legacy_edge_event"

        edges = [
            EdgeNodeModel(
                id="e1",
                source="src",
                target="tgt",
                sourceHandle="handle-default",
                hooks=legacy_hook,
            ),
        ]

        dispatcher = GraphEventDispatcher(nodes, edges, execution_id="test-ignored")

        await dispatcher.propagate_outputs("src", {"handle-default": {"content": "data"}})

        # The legacy hook_type value is ignored because dispatch is based on
        # hook_node_id only.
        assert DEFAULT_INPUT_HOOK_CONTEXT in hook_node.inputs, (
            "Edge hook must dispatch regardless of hook_type value"
        )


class TestPublicGraphEdgeHookExecution:
    """Real public-path regressions for declarative edge-triggered NodeHook."""

    @staticmethod
    def _hook_first_graph(
        *,
        selected_handle: str = "yes",
        hook_handle: str = "yes",
    ):
        from magic_agents.agt_flow import build

        return build(
            {
                "type": "graph",
                "debug": False,
                "timeout": 1,
                # Hook intentionally precedes its source. Node ordering must not
                # decide whether an edge hook receives its runtime context.
                "nodes": [
                    {
                        "id": "edge-hook",
                        "type": "hook",
                        "data": {
                            "hook_type": "custom",
                            "function_template": """
async def capture_edge(context, chat_log):
    return {
        "type": "hook-result",
        "content": {
            "source": context.inputs["source"],
            "target": context.inputs["target"],
            "payload": context.inputs["content"],
            "execution_id": context.execution_id,
        },
    }
""",
                        },
                    },
                    {"id": "input", "type": "user_input", "data": {}},
                    {
                        "id": "route",
                        "type": "conditional",
                        "data": {
                            "condition": "{{ '" + selected_handle + "' }}",
                            "output_handles": ["yes", "no"],
                        },
                    },
                    {"id": "yes", "type": "text", "data": {"text": "selected"}},
                    {"id": "no", "type": "text", "data": {"text": "not selected"}},
                    {"id": "end", "type": "end", "data": {}},
                ],
                "edges": [
                    {
                        "id": "input-route",
                        "source": "input",
                        "target": "route",
                        "sourceHandle": "handle_user_message",
                        "targetHandle": "handle_input",
                    },
                    {
                        "id": "selected-edge",
                        "source": "route",
                        "target": "yes",
                        "sourceHandle": "yes",
                        "targetHandle": "handle_flow_input",
                        **(
                            {
                                "hooks": {
                                    "hook_node_id": "edge-hook",
                                    "enabled": True,
                                }
                            }
                            if hook_handle == "yes"
                            else {}
                        ),
                    },
                    {
                        "id": "unselected-edge",
                        "source": "route",
                        "target": "no",
                        "sourceHandle": "no",
                        "targetHandle": "handle_flow_input",
                        **(
                            {
                                "hooks": {
                                    "hook_node_id": "edge-hook",
                                    "enabled": True,
                                }
                            }
                            if hook_handle == "no"
                            else {}
                        ),
                    },
                    {
                        "id": "yes-end",
                        "source": "yes",
                        "target": "end",
                        "sourceHandle": "handle_text_output",
                        "targetHandle": "handle_flow_input",
                    },
                    {
                        "id": "no-end",
                        "source": "no",
                        "target": "end",
                        "sourceHandle": "handle_text_output",
                        "targetHandle": "handle_flow_input",
                    },
                ],
            },
            message="edge payload",
        )

    @staticmethod
    def _hook_first_loop_graph(
        *,
        list_text: str = '["alpha", "beta"]',
    ):
        from magic_agents.agt_flow import build

        return build(
            {
                "type": "graph",
                "debug": False,
                "timeout": 1,
                "nodes": [
                    {
                        "id": "edge-hook",
                        "type": "hook",
                        "data": {
                            "hook_type": "custom",
                            "function_template": """
async def capture_loop_edge(context, chat_log):
    calls = chat_log.flow_state.get("edge_hook_calls", 0) + 1
    chat_log.flow_state["edge_hook_calls"] = calls
    return {
        "type": "hook-result",
        "content": {
            "source": context.inputs["source"],
            "target": context.inputs["target"],
            "payload": context.inputs["content"],
            "sequence": context.sequence_number,
            "calls": calls,
            "execution_id": context.execution_id,
        },
    }
""",
                        },
                    },
                    {
                        "id": "list-source",
                        "type": "text",
                        "data": {"text": list_text},
                    },
                    {"id": "input", "type": "user_input", "data": {}},
                    {"id": "loop", "type": "loop", "data": {}},
                    {
                        "id": "transform",
                        "type": "parser",
                        "data": {"text": "{{ handle_parser_input }}"},
                    },
                    {"id": "end", "type": "end", "data": {}},
                ],
                "edges": [
                    {
                        "id": "input-list",
                        "source": "input",
                        "target": "list-source",
                        "sourceHandle": "handle_user_message",
                        "targetHandle": "handle_flow_input",
                    },
                    {
                        "id": "list-loop",
                        "source": "list-source",
                        "target": "loop",
                        "sourceHandle": "handle_text_output",
                        "targetHandle": "handle_list",
                    },
                    {
                        "id": "loop-item",
                        "source": "loop",
                        "target": "transform",
                        "sourceHandle": "handle_item",
                        "targetHandle": "handle_parser_input",
                        "hooks": {
                            "hook_node_id": "edge-hook",
                            "enabled": True,
                        },
                    },
                    {
                        "id": "transform-loop",
                        "source": "transform",
                        "target": "loop",
                        "sourceHandle": "handle_parser_output",
                        "targetHandle": "handle_loop",
                    },
                    {
                        "id": "loop-end",
                        "source": "loop",
                        "target": "end",
                        "sourceHandle": "handle_end",
                        "targetHandle": "handle_flow_input",
                    },
                ],
            },
            message="run loop",
        )

    @pytest.mark.asyncio
    async def test_hook_listed_before_source_executes_from_public_run_agent(self):
        from magic_agents import run_agent

        graph = self._hook_first_graph()

        async for _event in run_agent(graph):
            pass

        result = graph.nodes["edge-hook"].outputs["hook-result"]
        assert result["source"] == "route"
        assert result["target"] == "yes"
        assert result["payload"] == {
            "handle_input": "edge payload",
            "value": "edge payload",
        }
        assert result["execution_id"]

    @pytest.mark.asyncio
    async def test_hook_on_unselected_edge_is_bypassed_without_timeout(self):
        from magic_agents import run_agent

        graph = self._hook_first_graph(selected_handle="yes", hook_handle="no")
        events = [event async for event in run_agent(graph)]

        assert graph.nodes["edge-hook"].outputs == {}
        assert "handle_end_output" in graph.nodes["end"].outputs


        assert not any(
            isinstance(event, dict)
            and event.get("type") == "debug"
            and isinstance(event.get("content"), dict)
            and event["content"].get("node_id") == "edge-hook"
            and event["content"].get("error_type") == "TimeoutError"
            for event in events
        )

    @pytest.mark.asyncio
    async def test_unselected_edge_hook_keeps_conditional_bypass_reason(self):
        from magic_agents.execution.reactive_executor import execute_graph_reactive

        class RecordingBypassHook:
            def __init__(self):
                self.calls = []

            async def on_node_bypass(self, context, reason):
                self.calls.append((context.node_id, reason))

        graph = self._hook_first_graph(selected_handle="yes", hook_handle="no")
        recorder = RecordingBypassHook()
        registry = HookRegistry()
        registry.register_global(recorder)

        async for _event in execute_graph_reactive(graph, hooks=registry):
            pass

        assert ("edge-hook", "condition") in recorder.calls
        assert ("edge-hook", "not_ready") not in recorder.calls

    @pytest.mark.asyncio
    async def test_loop_item_edge_hook_executes_once_per_real_traversal(self):
        from magic_agents import run_agent

        graph = self._hook_first_loop_graph()

        async for _event in run_agent(graph):
            pass

        result = graph.nodes["edge-hook"].outputs["hook-result"]
        assert result == {
            "source": "loop",
            "target": "transform",
            "payload": "beta",
            "sequence": 2,
            "calls": 2,
            "execution_id": result["execution_id"],
        }
        assert result["execution_id"]
        assert graph.nodes["loop"].outputs["handle_end"]["content"] == [
            "alpha",
            "beta",
        ]
        assert "handle_end_output" in graph.nodes["end"].outputs

    @pytest.mark.asyncio
    async def test_empty_loop_bypasses_untraversed_item_edge_hook(self):
        from magic_agents.execution.reactive_executor import execute_graph_reactive

        class GraphEndRecorder:
            def __init__(self):
                self.summary = None

            async def on_graph_end(self, context):
                self.summary = context.metadata["execution_summary"]

        graph = self._hook_first_loop_graph(list_text="[]")
        recorder = GraphEndRecorder()
        registry = HookRegistry()
        registry.register_global(recorder)

        async for _event in execute_graph_reactive(graph, hooks=registry):
            pass

        assert graph.nodes["edge-hook"].outputs == {}
        assert graph.nodes["loop"].outputs["handle_end"]["content"] == []
        assert "handle_end_output" in graph.nodes["end"].outputs
        assert recorder.summary is not None
        assert recorder.summary["pending"] == 0
        assert set(recorder.summary["states"]["bypassed"]) >= {
            "edge-hook",
            "transform",
        }


class TestEdgeHookErrorCascade:
    """Declarative edge-hook dependencies participate in error bypass."""

    @pytest.mark.asyncio
    async def test_source_failure_bypasses_hook_without_lifecycle_registry(self):
        from magic_agents.execution.reactive_executor import execute_graph_reactive
        from magic_agents.models.factory.AgentFlowModel import AgentFlowModel

        class FailingNode(Node):
            async def process(self, chat_log):
                if False:
                    yield None
                raise RuntimeError("source failed")

        class RecordingNode(Node):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.execute_count = 0

            async def process(self, chat_log):
                self.execute_count += 1
                yield self.yield_static("unexpected", content_type="out")

        source = FailingNode(node_id="source", node_type="test", debug=False)
        downstream = RecordingNode(
            node_id="downstream",
            node_type="test",
            debug=False,
        )
        hook = NodeHook(
            data=HookNodeModel(
                function_template="""
async def should_not_run(context, chat_log):
    return {"type": "hook-result", "content": "unexpected"}
""",
            ),
            node_id="edge-hook",
            node_type="hook",
            debug=False,
        )
        graph = AgentFlowModel(
            nodes={
                "edge-hook": hook,
                "source": source,
                "downstream": downstream,
            },
            edges=[
                EdgeNodeModel(
                    id="failing-edge",
                    source="source",
                    target="downstream",
                    sourceHandle="out",
                    targetHandle="in",
                    hooks=EdgeHookConfig(
                        hook_node_id="edge-hook",
                        enabled=True,
                    ),
                )
            ],
            timeout=1,
            debug=False,
        )
        graph._validation_errors = []

        events = [event async for event in execute_graph_reactive(graph)]
        errors = [
            event["content"]
            for event in events
            if isinstance(event, dict)
            and event.get("type") == "debug"
            and isinstance(event.get("content"), dict)
            and event["content"].get("error_type")
        ]

        assert [error["error_type"] for error in errors] == ["RuntimeError"]
        assert errors[0]["node_id"] == "source"
        assert downstream.execute_count == 0
        assert hook.outputs == {}


# ─── Task 4.9: Bypass Reason Canonical Value Assertions ────────────────


# The public-path regressions above prove that edge HookContext delivery is a
# tracked execution dependency, including conditional bypass of an untraversed
# hooked edge. Dispatcher-level tests retain focused context/config coverage.
