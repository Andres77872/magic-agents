"""A graph with ``debug_config: {"enabled": false}`` must still run.

That config keeps ``graph.debug`` True but disables the observer registry, so
every node takes the legacy ``_start_debug_tracking()`` path. The void node
``build()`` adds used to receive ``inputs[None]`` from the END -> void edges,
failing NodeDebugInfo validation outside the node's try block, which failed
the node and with it the whole run.
"""

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.node_system import NodeVoid
from magic_agents.util.const import HANDLE_VOID


class _Outcome(FlowHooks):
    def __init__(self):
        self.graph_errors = []
        self.node_errors = []
        self.graph_ends = 0

    async def on_node_error(self, context, error):
        self.node_errors.append((context.node_id, repr(error)))

    async def on_graph_error(self, context, error):
        self.graph_errors.append(repr(error))

    async def on_graph_end(self, context):
        self.graph_ends += 1


def _graph(debug_config):
    return {
        'type': 'chat',
        'debug': True,
        'debug_config': debug_config,
        'nodes': [
            {'id': 'user-1', 'type': 'user_input', 'data': {}},
            {'id': 'text-1', 'type': 'text', 'data': {'text': 'hello'}},
            {'id': 'send-1', 'type': 'send_message', 'data': {'json_extras': 'DONE'}},
            {'id': 'end-1', 'type': 'end'},
        ],
        'edges': [
            {'id': 'e1', 'source': 'user-1', 'target': 'text-1',
             'sourceHandle': 'handle_user_message', 'targetHandle': 'input'},
            {'id': 'e2', 'source': 'text-1', 'target': 'send-1',
             'sourceHandle': 'handle_text_output', 'targetHandle': 'handle_send_extra'},
            {'id': 'e3', 'source': 'send-1', 'target': 'end-1',
             'sourceHandle': 'handle_message_output', 'targetHandle': 'input'},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('debug_config', [{'enabled': False}, {'enabled': True}])
async def test_run_completes_whether_debug_config_is_enabled_or_not(debug_config):
    graph = build(_graph(debug_config), message='hi')
    assert graph.debug is True
    assert graph.resolved_debug_config.enabled is debug_config['enabled']

    outcome = _Outcome()
    config = RuntimeConfig()
    config.register_graph_hook(outcome)
    events = [event async for event in run_agent(graph, hooks=config)]

    assert outcome.node_errors == []
    assert outcome.graph_errors == []
    assert outcome.graph_ends == 1
    assert [event for event in events if event.get('type') == 'error'] == []

    void = next(node for node in graph.nodes.values() if isinstance(node, NodeVoid))
    assert set(void.inputs) == {HANDLE_VOID}
