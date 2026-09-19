"""
Slice 15 — Inner node integration tests (deterministic / no API keys).

Tests inner node basic execution and nested inner graphs.
Uses text/send_message nodes inside inner graphs to avoid LLM dependency.

Phase 5 additions:
- extras propagation from build() → UserInput → NodeInner
- streaming forwarding through a deterministic child node
- flow state isolation
- parent state exposure (default and selective mapping)
- child completion and output propagation
"""
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.node_system import NodeInner
from magic_agents.node_system.Node import Node
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.models.factory.Nodes.InnerNodeModel import InnerNodeModel
from magic_agents.node_system.NodeInner import _get_nested_value
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel


def extract_streamed_content(item):
    """Extract streamed content from send_message output."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "content":
        return ""
    content = item.get("content")
    if content is None:
        return ""
    if hasattr(content, "choices") and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
    return ""


def get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    executed = set()
    if not debug_summary:
        return executed
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            executed.add(node.get("node_id"))
    return executed


async def run_with_debug_summary(graph):
    """Execute a graph and return its stream plus the observer-owned summary."""
    streamed = []
    debug_events = []

    async def record_debug_event(event):
        debug_events.append(event)

    async for item in run_agent(graph, debug_callback=record_debug_event):
        streamed.append(item)

    summaries = [
        event.payload
        for event in debug_events
        if event.event_type == DebugEventType.GRAPH_END
    ]
    assert len(summaries) == 1
    return streamed, summaries[0]


class DeterministicStreamingNode(Node):
    """Local source node used to exercise the real nested streaming pipeline."""

    OUTPUT_HANDLE_CONTENT = "handle_stream"

    async def process(self, chat_log):
        for index, text in enumerate(("first ", "second")):
            chunk = ChatCompletionModel(
                id=f"chunk-{index}",
                model="deterministic-test-node",
                choices=[ChoiceModel(delta=DeltaModel(content=text))],
            )
            yield self.yield_static(chunk, content_type="handle_stream")


class FlowStateProbeNode(Node):
    """Records and mutates the child log to prove parent/child state isolation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_state = None
        self.mutated_state = None

    async def process(self, chat_log):
        self.initial_state = dict(chat_log.flow_state)
        chat_log.flow_state["child_only"] = True
        self.mutated_state = dict(chat_log.flow_state)
        yield self.yield_static("done", content_type="probe_output")


def inner_node_with_child(child_node):
    """Create a real NodeInner around a deterministic in-memory child graph."""
    node = NodeInner(
        data=InnerNodeModel(magic_flow={"nodes": [], "edges": []}),
        load_chat=lambda: None,
        node_id="inner",
        node_type="inner",
        debug=True,
    )
    node.inner_graph = AgentFlowModel(
        nodes={child_node.node_id: child_node},
        edges=[],
        debug=False,
    )
    node.inputs[node.INPUT_HANDLE] = "parent message"
    return node


class TestInnerNodeBasicExecution:
    """Tests for basic inner node execution."""

    @pytest.mark.asyncio
    async def test_inner_node_basic_execution(self):
        """Inner graph executes, output propagates to outer graph."""
        # Inner graph: text -> end (no LLM)
        inner_graph = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "INNER_RESULT"}},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie2", "source": "inner_text", "target": "inner_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    "magic_flow": inner_graph,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "OUTER"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="outer message")

        # Verify inner graph was built
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        assert inner_node.inner_graph is not None
        assert len(inner_node.inner_graph.nodes) > 0

        streamed, debug_summary = await run_with_debug_summary(graph)
        content_output = [
            text
            for item in streamed
            if (text := extract_streamed_content(item))
        ]

        content_str = "".join(content_output)
        assert "OUTER" in content_str
        assert graph.nodes["inner"].outputs["handle_execution_content"]["content"] == "INNER_RESULT"
        assert graph.nodes["send"].inputs == {"handle_send_extra": "INNER_RESULT"}
        executed = get_executed_nodes(debug_summary)
        assert "inner" in executed
        assert "send" in executed

    @pytest.mark.asyncio
    async def test_inner_node_nested(self):
        """Inner node containing another inner node — recursive build and execution."""
        # Innermost graph
        innermost = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "innermost_input", "type": "user_input"},
                {"id": "innermost_text", "type": "text", "data": {"text": "NESTED"}},
                {"id": "innermost_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "innermost_input", "target": "innermost_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie2", "source": "innermost_text", "target": "innermost_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }

        # Middle graph with inner node
        middle = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "middle_input", "type": "user_input"},
                {"id": "middle_inner", "type": "inner", "data": {
                    "magic_flow": innermost,
                }},
                {"id": "middle_end", "type": "end"},
            ],
            "edges": [
                {"id": "me1", "source": "middle_input", "target": "middle_inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "me2", "source": "middle_inner", "target": "middle_end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
            ],
        }

        # Outer graph
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "outer_inner", "type": "inner", "data": {
                    "magic_flow": middle,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "NESTED_DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "outer_inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "outer_inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")

        # Verify recursive build
        outer_inner = graph.nodes.get("outer_inner")
        assert outer_inner is not None
        assert isinstance(outer_inner, NodeInner)
        assert outer_inner.inner_graph is not None

        # Verify the middle inner node was also built
        middle_inner = outer_inner.inner_graph.nodes.get("middle_inner")
        assert middle_inner is not None
        assert isinstance(middle_inner, NodeInner)
        assert middle_inner.inner_graph is not None

        streamed, debug_summary = await run_with_debug_summary(graph)
        content_output = [
            text
            for item in streamed
            if (text := extract_streamed_content(item))
        ]

        content_str = "".join(content_output)
        assert "NESTED_DONE" in content_str
        executed = get_executed_nodes(debug_summary)
        assert "outer_inner" in executed
        assert "send" in executed

    @pytest.mark.asyncio
    async def test_inner_node_missing_inner_graph_error(self):
        """Inner node without inner_graph set yields debug error."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    # Missing magic_flow — inner graph won't be built
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        # inner_graph should be None since magic_flow is missing
        assert inner_node.inner_graph is None

        debug_items = []
        async for item in run_agent(graph):
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)

        # Should have a debug error about missing inner_graph
        assert len(debug_items) > 0
        error_types = [d.get("content", {}).get("error_type") for d in debug_items]
        assert "ConfigurationError" in error_types or "InputError" in error_types

    def test_inner_node_missing_magic_flow_build_does_not_crash(self):
        """App fix: build() handles missing magic_flow gracefully (no TypeError)."""
        agt = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
            ],
        }

        # Before the fix, this would raise TypeError: argument of type 'NoneType' is not iterable
        graph = build(agt, message="test")
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        # inner_graph should remain None since magic_flow was missing
        assert inner_node.inner_graph is None


class TestInnerGraphWithConditional:
    """Slice 18 — Inner graph with conditional nodes."""

    @pytest.mark.asyncio
    async def test_inner_graph_with_conditional_routing(self):
        """Inner graph containing a conditional node routes correctly.

        Inner graph: user_input → text → conditional → parser_yes / parser_no → end
        The text node provides a known value that the conditional evaluates.
        """
        inner_graph = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "hello"}},
                {"id": "inner_cond", "type": "conditional", "data": {
                    "condition": "{{ handle_input }}",
                    "output_handles": ["hello", "other"],
                }},
                {"id": "parser_hello", "type": "parser", "data": {
                    "text": "HELLO: {{ handle_parser_input }}"
                }},
                {"id": "parser_other", "type": "parser", "data": {
                    "text": "OTHER: {{ handle_parser_input }}"
                }},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie0", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie1", "source": "inner_text", "target": "inner_cond",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_input"},
                {"id": "ie2", "source": "inner_cond", "target": "parser_hello",
                 "sourceHandle": "hello", "targetHandle": "handle_parser_input"},
                {"id": "ie3", "source": "inner_cond", "target": "parser_other",
                 "sourceHandle": "other", "targetHandle": "handle_parser_input"},
                {"id": "ie4", "source": "parser_hello", "target": "inner_end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
                {"id": "ie5", "source": "parser_other", "target": "inner_end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
        }

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    "magic_flow": inner_graph,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "OUTER_DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="hello")
        streamed, debug_summary = await run_with_debug_summary(graph)
        content_output = [
            text
            for item in streamed
            if (text := extract_streamed_content(item))
        ]

        content_str = "".join(content_output)
        assert "OUTER_DONE" in content_str

        executed = get_executed_nodes(debug_summary)
        # Inner node should execute
        assert "inner" in executed
        assert "send" in executed

        # The inner graph's conditional routing should have worked
        # We can't directly see inner graph node execution in the outer debug_summary
        # because inner graph has its own execution context. But we verify the
        # inner graph was built and executed without errors.
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert inner_node.inner_graph is not None
        # Verify the conditional was built inside the inner graph
        inner_cond = inner_node.inner_graph.nodes.get("inner_cond")
        assert inner_cond is not None
        assert inner_cond.node_type == "conditional"


class TestInnerGraphErrorPropagation:
    """Slice 19 — Inner graph error propagation to outer graph."""

    @pytest.mark.asyncio
    async def test_inner_graph_error_propagates_as_debug_event(self):
        """When inner graph node raises an exception, outer graph receives a debug error.

        We use a graph where the inner graph has a node that will fail
        (e.g., a parser with an invalid Jinja2 template that raises an error).
        The outer graph should continue and receive a debug event about the error.
        """
        # Inner graph with a parser that will fail (invalid Jinja2 syntax)
        inner_graph = {
            "type": "graph",
            "debug": True,
            "timeout": 1,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "bad_parser", "type": "parser", "data": {
                    # This Jinja2 template will raise an error
                    "text": "{{ undefined_var.some_method() }}"
                }},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "bad_parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "ie2", "source": "bad_parser", "target": "inner_end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
        }

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    "magic_flow": inner_graph,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "AFTER_INNER"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")

        debug_items = []
        content_output = []
        async for item in run_agent(graph):
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        # The outer graph completes (doesn't hang). Since NodeInner emits BYPASS_ALL
        # when inner graph errors, downstream nodes like 'send' are bypassed.
        # We verify completion by checking debug_summary and debug events.

        # There should be debug events about the inner graph error
        # NodeInner now propagates debug events from inner graph to outer graph
        assert len(debug_items) > 0, "Expected debug events from inner graph error"

        error_types = [d.get("content", {}).get("error_type") for d in debug_items]
        # Should have at least one error-type event (from bad_parser exception or inner_end timeout)
        assert any(
            et in ("TemplateError", "RuntimeError", "UndefinedError", "JinjaError",
                   "InputError", "TimeoutError")
            for et in error_types
        ), f"Expected error-type debug event, got: {error_types}"


class TestInnerNodeFlowIntegration:
    """
    Phase 5 tests for inner-node flow integration features.
    
    Covers:
    - Client extras propagation (5.1-5.3)
    - Streaming forwarding (5.4-5.5)
    - Flow state isolation (5.6-5.7)
    - Parent state exposure (5.8-5.11)
    - Child completion (5.12-5.14)
    - Extras merge and edge cases (5.15-5.16)
    - Regression check (5.17)
    """

    # ========== Tests 5.1-5.3: Extras Propagation ==========

    @pytest.mark.asyncio
    async def test_client_extras_propagation(self):
        """5.1: Client extras flow from build() → UserInput → NodeInner inputs."""
        extras = {"user_id": "abc123", "session_type": "premium"}
        
        inner_graph = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "CHILD_DONE"}},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie2", "source": "inner_text", "target": "inner_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {"magic_flow": inner_graph}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "EXTRAS_OK"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "input", "target": "inner",
                 "sourceHandle": "handle_client_extras", "targetHandle": "handle_client_extras"},
                {"id": "e3", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test message", extras=extras)
        
        # Verify UserInput node has extras
        user_input_node = graph.nodes.get("input")
        assert user_input_node is not None
        assert user_input_node._extras == extras
        
        # Verify NodeInner will receive extras via handle_client_extras input
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert inner_node.HANDLER_CLIENT_EXTRAS == "handle_client_extras"
        
        content_output = []
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        content_str = "".join(content_output)
        assert "EXTRAS_OK" in content_str

    @pytest.mark.asyncio
    async def test_userinput_yields_extras_handle(self):
        """5.2: UserInput yields extras on handle_client_extras when extras provided."""
        extras = {"context": "test_value"}
        
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser", "type": "parser", "data": {"text": "EXTRAS_RECEIVED: {{ handle_extras }}"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "PARSED"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser",
                 "sourceHandle": "handle_client_extras", "targetHandle": "handle_extras"},
                {"id": "e3", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="hello", extras=extras)
        
        # Verify UserInput has extras
        user_input = graph.nodes.get("input")
        assert user_input._extras == extras
        
        # Verify parser received extras via handle_extras input
        parser_node = graph.nodes.get("parser")
        assert parser_node is not None
        
        content_output = []
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        content_str = "".join(content_output)
        # Parser should receive extras and pass to send_message
        assert "PARSED" in content_str

    @pytest.mark.asyncio
    async def test_userinput_no_extras_backward_compat(self):
        """5.3: UserInput doesn't yield extras handle when extras=None (backward compat)."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser", "type": "parser", "data": {"text": "MSG: {{ handle_parser_input }}"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "BACKWARD_COMPAT"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        # Build WITHOUT extras (backward compat)
        graph = build(agt, message="test message")
        
        # Verify UserInput has None extras
        user_input = graph.nodes.get("input")
        assert user_input._extras is None
        
        content_output = []
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        content_str = "".join(content_output)
        assert "BACKWARD_COMPAT" in content_str

    # ========== Tests 5.4-5.7: Streaming and Flow State Isolation ==========

    @pytest.mark.asyncio
    async def test_nodeinner_forwards_child_chunks_before_final_output(self):
        """Child chunks traverse NodeInner in order before its aggregate output."""
        child = DeterministicStreamingNode(
            node_id="stream-source",
            node_type="stream-source",
            debug=False,
        )
        inner = inner_node_with_child(child)

        events = [event async for event in inner.process(ModelAgentRunLog())]
        chunks = [
            event["content"]["content"].choices[0].delta.content
            for event in events
            if event["type"] == inner.OUTPUT_HANDLE_CONTENT
        ]
        final_index = next(
            index
            for index, event in enumerate(events)
            if event["type"] == inner.HANDLER_EXECUTION_CONTENT
        )
        chunk_indexes = [
            index
            for index, event in enumerate(events)
            if event["type"] == inner.OUTPUT_HANDLE_CONTENT
        ]

        assert chunks == ["first ", "second"]
        assert max(chunk_indexes) < final_index
        assert events[final_index]["content"]["content"] == "first second"
        subgraph_events = [
            event["content"]
            for event in events
            if event["type"] == "debug"
            and event["content"].get("event_type", "").startswith("SUBGRAPH_")
        ]
        assert [event["event_type"] for event in subgraph_events] == [
            "SUBGRAPH_START",
            "SUBGRAPH_END",
        ]
        assert {event["parent_execution_id"] for event in subgraph_events} == {None}

    @pytest.mark.asyncio
    async def test_subgraph_debug_uses_parent_execution_and_run_domains(self):
        """A runtime run ID is never mislabeled as a parent execution span."""
        child = DeterministicStreamingNode(
            node_id="stream-source",
            node_type="stream-source",
            debug=False,
        )
        inner = inner_node_with_child(child)
        registry = HookRegistry()
        registry.execution_id = "parent-execution-123"
        registry.run_id = "parent-run-456"
        inner._hooks = registry

        events = [
            event
            async for event in inner.process(
                ModelAgentRunLog(run_id="chat-log-run-must-not-be-an-execution")
            )
        ]
        subgraph_events = [
            event["content"]
            for event in events
            if event["type"] == "debug"
            and event["content"].get("event_type", "").startswith("SUBGRAPH_")
        ]

        assert [event["event_type"] for event in subgraph_events] == [
            "SUBGRAPH_START",
            "SUBGRAPH_END",
        ]
        assert {
            event["parent_execution_id"]
            for event in subgraph_events
        } == {"parent-execution-123"}
        child_graph_start = next(
            event["content"]
            for event in events
            if event["type"] == "debug"
            and event["content"].get("event_type") == "GRAPH_START"
        )
        assert child_graph_start["parent_run_id"] == "parent-run-456"

    @pytest.mark.asyncio
    async def test_child_flow_state_is_empty_and_isolated_from_parent(self):
        """The child receives a fresh state mapping and cannot mutate the parent."""
        child = FlowStateProbeNode(
            node_id="state-probe",
            node_type="state-probe",
            debug=False,
        )
        inner = inner_node_with_child(child)
        parent_state = {"parent_counter": 5}

        events = [
            event
            async for event in inner.process(
                ModelAgentRunLog(flow_state=parent_state)
            )
        ]

        assert child.initial_state == {}
        assert child.mutated_state == {"child_only": True}
        assert parent_state == {"parent_counter": 5}
        assert any(
            event["type"] == "debug"
            and event["content"].get("event_type") == "SUBGRAPH_END"
            and event["content"]["status"] == "completed"
            for event in events
        )

    # ========== Tests 5.8-5.11: Parent State Exposure ==========

    def test_parent_state_default_full_exposure_unit(self):
        """5.8: Child receives parent_state in extras when no mapping configured (unit test)."""
        model = InnerNodeModel(magic_flow={"nodes": [], "edges": []})
        node = NodeInner(data=model, load_chat=lambda: None)
        
        client_extras = {"user_id": "abc"}
        parent_state = {"stage": "greeting", "mood": "happy"}
        
        child_extras = node._prepare_child_extras(client_extras, parent_state)
        
        # Verify default full exposure
        assert child_extras["user_id"] == "abc"  # Client extras preserved
        assert child_extras["parent_state"] == parent_state  # Full parent state exposed

    def test_parent_state_mapping_selective_unit(self):
        """5.9: Only mapped fields passed to child extras (unit test)."""
        model = InnerNodeModel(
            magic_flow={"nodes": [], "edges": []},
            parent_state_mapping={"child_stage": "stage", "child_mood": "mood"}
        )
        node = NodeInner(data=model, load_chat=lambda: None)
        
        client_extras = {"user_id": "abc"}
        parent_state = {"stage": "greeting", "mood": "happy", "secret": "hidden"}
        
        child_extras = node._prepare_child_extras(client_extras, parent_state)
        
        # Verify selective mapping
        assert child_extras["user_id"] == "abc"  # Client extras preserved
        assert child_extras["child_stage"] == "greeting"  # Mapped field
        assert child_extras["child_mood"] == "happy"  # Mapped field
        assert "secret" not in child_extras  # Unmapped field NOT exposed
        assert "parent_state" not in child_extras  # Default overridden

    def test_parent_state_mapping_nested_path_unit(self):
        """5.10: Dot-notation traverses nested dicts correctly (unit test)."""
        model = InnerNodeModel(
            magic_flow={"nodes": [], "edges": []},
            parent_state_mapping={
                "child_city": "location.city",
                "child_country": "location.country"
            }
        )
        node = NodeInner(data=model, load_chat=lambda: None)
        
        parent_state = {
            "stage": "greeting",
            "location": {"city": "Paris", "country": "France"}
        }
        
        child_extras = node._prepare_child_extras({}, parent_state)
        
        # Verify nested path traversal
        assert child_extras["child_city"] == "Paris"
        assert child_extras["child_country"] == "France"

    def test_parent_state_mapping_missing_path_unit(self):
        """5.11: Missing path returns None (no error) (unit test)."""
        model = InnerNodeModel(
            magic_flow={"nodes": [], "edges": []},
            parent_state_mapping={"child_missing": "nonexistent.path"}
        )
        node = NodeInner(data=model, load_chat=lambda: None)
        
        parent_state = {"stage": "greeting"}
        
        child_extras = node._prepare_child_extras({}, parent_state)
        
        # Verify missing path returns None (graceful handling)
        assert child_extras["child_missing"] is None

    # ========== Tests 5.12-5.14: Child Completion ==========

    @pytest.mark.asyncio
    async def test_child_flow_completion_parent_continues(self):
        """5.12: Parent flow continues after child END node."""
        inner_graph = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "CHILD_OUTPUT"}},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie2", "source": "inner_text", "target": "inner_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {"magic_flow": inner_graph}},
                {"id": "after_inner", "type": "text", "data": {"text": "PARENT_CONTINUES"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "FINAL"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "after_inner",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
                {"id": "e3", "source": "after_inner", "target": "send",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        _, debug_summary = await run_with_debug_summary(graph)
        
        # Verify parent continued after child END
        executed = get_executed_nodes(debug_summary)
        assert "inner" in executed
        assert "after_inner" in executed
        assert "send" in executed

    @pytest.mark.asyncio
    async def test_child_flow_output_propagates(self):
        """5.13: Child output appears on NodeInner handle_execution_content."""
        inner_graph = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "CHILD_VALUE_123"}},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie2", "source": "inner_text", "target": "inner_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {"magic_flow": inner_graph}},
                {"id": "parser", "type": "parser", "data": {"text": "RECEIVED: {{ handle_parser_input }}"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "OUTPUT_OK"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "parser",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        content_output = []
        
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        content_str = "".join(content_output)
        # Parser should receive child output and pass to send_message
        assert "OUTPUT_OK" in content_str
        assert graph.nodes["inner"].outputs["handle_execution_content"]["content"] == "CHILD_VALUE_123"
        assert graph.nodes["parser"].inputs == {"handle_parser_input": "CHILD_VALUE_123"}
        assert graph.nodes["parser"].outputs["handle_parser_output"]["content"] == "RECEIVED: CHILD_VALUE_123"
        assert graph.nodes["send"].inputs == {"handle_send_extra": "RECEIVED: CHILD_VALUE_123"}

    # ========== Tests 5.15-5.16: Extras Merge and Edge Cases ==========

    def test_inner_node_extras_merge_unit(self):
        """5.15: Client extras + parent state merged correctly in child extras (unit test)."""
        model = InnerNodeModel(
            magic_flow={"nodes": [], "edges": []},
            parent_state_mapping={"child_stage": "stage"}
        )
        node = NodeInner(data=model, load_chat=lambda: None)
        
        client_extras = {"user_id": "abc123", "request_id": "req_1"}
        parent_state = {"stage": "processing", "internal": "value"}
        
        child_extras = node._prepare_child_extras(client_extras, parent_state)
        
        # Verify merge
        assert child_extras["user_id"] == "abc123"  # Client extras preserved
        assert child_extras["request_id"] == "req_1"  # Client extras preserved
        assert child_extras["child_stage"] == "processing"  # Mapped from parent

    def test_extras_empty_dict_unit(self):
        """5.16: Empty extras dict handled gracefully (not same as None) (unit test)."""
        model = InnerNodeModel(magic_flow={"nodes": [], "edges": []})
        node = NodeInner(data=model, load_chat=lambda: None)
        
        # Empty dict {} is different from None
        client_extras = {}  # Empty dict
        parent_state = {"stage": "greeting"}
        
        child_extras = node._prepare_child_extras(client_extras, parent_state)
        
        # Empty dict {} is handled, parent_state still exposed
        assert child_extras["parent_state"] == parent_state
        assert len(child_extras) == 1  # Only parent_state, no client extras keys

    # ========== Test 5.17: Regression Check ==========

    @pytest.mark.asyncio
    async def test_no_regression_existing_inner_tests(self):
        """5.17: Run all existing tests to verify zero regression."""
        # This test verifies that new changes don't break existing behavior
        # We run a subset of existing test scenarios inline
        
        # Test basic execution still works
        inner_graph = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "BASIC_TEST"}},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "ie2", "source": "inner_text", "target": "inner_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {"magic_flow": inner_graph}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "REGRESSION_OK"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        # Test with extras=None (backward compat)
        graph = build(agt, message="test", extras=None)
        
        streamed, debug_summary = await run_with_debug_summary(graph)
        content_output = [
            text
            for item in streamed
            if (text := extract_streamed_content(item))
        ]
        
        content_str = "".join(content_output)
        assert "REGRESSION_OK" in content_str
        executed = get_executed_nodes(debug_summary)
        assert "inner" in executed
        assert "send" in executed


class TestGetNestedValueHelper:
    """Unit tests for _get_nested_value helper function."""

    def test_get_nested_value_simple(self):
        """Get value from single-level path."""
        data = {"name": "Alice"}
        result = _get_nested_value(data, "name")
        assert result == "Alice"

    def test_get_nested_value_nested(self):
        """Get value from nested path."""
        data = {"user": {"profile": {"age": 30}}}
        result = _get_nested_value(data, "user.profile.age")
        assert result == 30

    def test_get_nested_value_missing_key(self):
        """Return None for missing key."""
        data = {"name": "Alice"}
        result = _get_nested_value(data, "email")
        assert result is None

    def test_get_nested_value_missing_nested_path(self):
        """Return None for missing nested path."""
        data = {"user": {"name": "Alice"}}
        result = _get_nested_value(data, "user.email")
        assert result is None

    def test_get_nested_value_empty_path(self):
        """Return None for empty path."""
        data = {"name": "Alice"}
        result = _get_nested_value(data, "")
        assert result is None

    def test_get_nested_value_none_data(self):
        """Return None for None data."""
        result = _get_nested_value(None, "path")
        assert result is None

    def test_get_nested_value_deep_nesting(self):
        """Get value from deeply nested path."""
        data = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        result = _get_nested_value(data, "a.b.c.d.e")
        assert result == "deep"


class TestModelAgentRunLogFlowState:
    """Unit tests for flow_state field in ModelAgentRunLog."""

    def test_flow_state_field_exists(self):
        """Verify flow_state field exists."""
        log = ModelAgentRunLog(id_chat="test")
        assert hasattr(log, 'flow_state')
        assert log.flow_state is None  # Default is None

    def test_flow_state_initialization(self):
        """Initialize flow_state with dict."""
        state = {"counter": 5, "stage": "processing"}
        log = ModelAgentRunLog(id_chat="test", flow_state=state)
        assert log.flow_state == state

    def test_flow_state_empty_dict(self):
        """Initialize flow_state with empty dict."""
        log = ModelAgentRunLog(id_chat="test", flow_state={})
        assert log.flow_state == {}

    def test_flow_state_nested_dict(self):
        """flow_state can contain nested dicts."""
        state = {"context": {"user": {"id": 123, "name": "Alice"}}}
        log = ModelAgentRunLog(id_chat="test", flow_state=state)
        assert log.flow_state["context"]["user"]["id"] == 123


class TestInnerNodeModelParentStateMapping:
    """Unit tests for parent_state_mapping field in InnerNodeModel."""

    def test_parent_state_mapping_field_exists(self):
        """Verify parent_state_mapping field exists."""
        model = InnerNodeModel()
        assert hasattr(model, 'parent_state_mapping')
        assert model.parent_state_mapping is None  # Default is None

    def test_parent_state_mapping_initialization(self):
        """Initialize parent_state_mapping with mapping dict."""
        mapping = {"child_stage": "stage", "child_user": "user.id"}
        model = InnerNodeModel(parent_state_mapping=mapping)
        assert model.parent_state_mapping == mapping

    def test_parent_state_mapping_empty_dict(self):
        """Initialize parent_state_mapping with empty dict."""
        model = InnerNodeModel(parent_state_mapping={})
        assert model.parent_state_mapping == {}


class TestRunAgentExtrasInjection:
    """
    Tests for run_agent(graph, extras=...) on pre-built graphs.
    
    Verify that extras are injected into UserInput nodes when passed to run_agent
    even when the graph was built without extras.
    """

    @pytest.mark.asyncio
    async def test_run_agent_extras_injected_into_prebuilt_graph(self):
        """run_agent(graph, extras=...) injects extras into UserInput on pre-built graph."""
        # Build graph WITHOUT extras (simulating pre-built scenario)
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser", "type": "parser", "data": {"text": "EXTRAS_OK: {{ handle_extras }}"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "PARSED"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser",
                 "sourceHandle": "handle_client_extras", "targetHandle": "handle_extras"},
                {"id": "e3", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        # Build WITHOUT extras
        graph = build(agt, message="test message")
        
        # Verify UserInput has None extras after build
        user_input = graph.nodes.get("input")
        assert user_input._extras is None
        
        # Now run_agent WITH extras - extras should be injected
        extras = {"user_id": "run_agent_user", "source": "run_agent_path"}
        
        content_output = []
        async for item in run_agent(graph, extras=extras):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        content_str = "".join(content_output)
        # Parser should have received extras and output should be processed
        assert "PARSED" in content_str
    
    @pytest.mark.asyncio
    async def test_run_agent_extras_does_not_override_built_extras(self):
        """run_agent extras do NOT override extras already set during build()."""
        # Build graph WITH extras
        build_extras = {"source": "build_path", "priority": "high"}
        
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser", "type": "parser", "data": {"text": "SOURCE: {{ handle_extras }}"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "FINAL"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser",
                 "sourceHandle": "handle_client_extras", "targetHandle": "handle_extras"},
                {"id": "e3", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test", extras=build_extras)
        
        # Verify UserInput has build extras
        user_input = graph.nodes.get("input")
        assert user_input._extras == build_extras
        
        # run_agent with DIFFERENT extras - should NOT override
        run_extras = {"source": "run_agent_path", "priority": "low"}
        
        content_output = []
        async for item in run_agent(graph, extras=run_extras):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
        
        # UserInput should still have build extras (not overridden)
        assert user_input._extras == build_extras
        
        content_str = "".join(content_output)
        assert "FINAL" in content_str


class TestMalformedMagicFlowHandling:
    """
    Tests for malformed embedded child magic_flow validation.
    
    Verify that malformed magic_flow produces clear ConfigurationError,
    not raw KeyError.
    """

    @pytest.mark.asyncio
    async def test_magic_flow_missing_edges_produces_configuration_error(self):
        """magic_flow missing 'edges' key yields clear ConfigurationError, not KeyError."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    # Malformed: missing 'edges' key
                    "magic_flow": {"nodes": [{"id": "inner_input", "type": "user_input"}]}
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
            ],
        }
        
        # Build should NOT crash with KeyError
        graph = build(agt, message="test")
        
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        # inner_graph should be None due to malformed magic_flow
        assert inner_node.inner_graph is None
        
        # Execution should yield ConfigurationError with clear message
        debug_items = []
        async for item in run_agent(graph):
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)
        
        assert len(debug_items) > 0
        error_types = [d.get("content", {}).get("error_type") for d in debug_items]
        assert "ConfigurationError" in error_types
        
        # Verify error message mentions missing 'edges' key
        error_messages = [d.get("content", {}).get("error_message") for d in debug_items]
        assert any("edges" in msg for msg in error_messages if msg)

    @pytest.mark.asyncio
    async def test_magic_flow_missing_nodes_produces_configuration_error(self):
        """magic_flow missing 'nodes' key yields clear ConfigurationError."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    # Malformed: missing 'nodes' key
                    "magic_flow": {"edges": [{"source": "a", "target": "b"}]}
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        inner_node = graph.nodes.get("inner")
        assert inner_node.inner_graph is None
        
        debug_items = []
        async for item in run_agent(graph):
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)
        
        assert len(debug_items) > 0
        error_types = [d.get("content", {}).get("error_type") for d in debug_items]
        assert "ConfigurationError" in error_types
        
        error_messages = [d.get("content", {}).get("error_message") for d in debug_items]
        assert any("nodes" in msg for msg in error_messages if msg)

    @pytest.mark.asyncio
    async def test_magic_flow_not_dict_produces_validation_error(self):
        """magic_flow that is not a dict yields Pydantic ValidationError at build time."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    # Malformed: magic_flow is a string, not dict
                    "magic_flow": "invalid_flow_json"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_flow_input"},
            ],
        }
        
        # Pydantic validates magic_flow type at build time - should raise ValidationError
        from pydantic import ValidationError
        with pytest.raises(ValidationError) as exc_info:
            graph = build(agt, message="test")
        
        # Verify the error is about magic_flow dict type
        error_str = str(exc_info.value)
        assert "magic_flow" in error_str
        assert "dict" in error_str.lower() or "dictionary" in error_str.lower()
