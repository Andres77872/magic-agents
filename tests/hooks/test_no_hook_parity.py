"""Registering hooks must not change what a graph emits.

The same graph runs with no registry, an empty registry and a registry holding
a hook that implements every method as a no-op; the runtime event sequence and
the node outputs must be identical.
"""

import pytest

from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.hooks import HookRegistry, RuntimeConfig
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.node_system.Node import Node
from tests.support import RecordingHook


class EchoNode(Node):
    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type="test", debug=True)

    async def process(self, chat_log):
        received = self.inputs.get("in")
        yield self.yield_static(f"{self.node_id}<-{received}" if received else self.node_id)


def _graph():
    return AgentFlowModel(
        nodes={"a": EchoNode("a"), "b": EchoNode("b"), "c": EchoNode("c")},
        edges=[
            EdgeNodeModel(id="ab", source="a", target="b", sourceHandle="end", targetHandle="in"),
            EdgeNodeModel(id="bc", source="b", target="c", sourceHandle="end", targetHandle="in"),
        ],
        debug=True,
    )


def _shape(event):
    content = event.get("content")
    detail = content.get("event_type") or content.get("error_type") if isinstance(content, dict) else None
    return event.get("type"), detail


@pytest.fixture(autouse=True)
def no_global_hooks():
    RuntimeConfig.clear_global_hooks()
    yield
    RuntimeConfig.clear_global_hooks()


async def test_hooks_do_not_change_events_or_outputs():
    runs = {}
    recording = RecordingHook()
    for label, registry in [("none", None), ("empty", HookRegistry()), ("noop", HookRegistry())]:
        if label == "noop":
            registry.register_graph(recording)
        graph = _graph()
        events = [event async for event in execute_graph_reactive(graph, hooks=registry)]
        runs[label] = ([_shape(e) for e in events], {nid: node.outputs for nid, node in graph.nodes.items()})

    assert runs["none"] == runs["empty"] == runs["noop"]
    assert runs["none"][1]["c"]["end"]["content"] == "c<-b<-a"
    # The no-op hook really was invoked, so the parity is not vacuous.
    assert recording.call_count("on_node_end") == 3


def test_graph_model_has_no_hooks_by_default():
    assert _graph().hooks is None
