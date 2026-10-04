"""Ordinary builder/executor owns shared messaging; no host or publication seam."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from magic_llm import MagicLLM
from magic_agents.agt_flow import build, run_agent
from magic_agents.coordination.invocation import invocation_runtime, owned_thread
from magic_agents.coordination.service import CoordinationError
from test.test_coordination_graph import Provider, tool_call


def definition(stream=False):
    return {'type': 'graph', 'coordination': {'enabled': True,
        'allowedParticipants': ['research', 'review'], 'limits': {
            'maxModelTurns': 12, 'maxInputTokens': 50000, 'maxOutputTokens': 12000,
            'maxToolCalls': 30, 'maxConcurrentModelTurns': 1,
            'maxWakeupsPerActor': 2, 'maxWakeupsPerWorkgroup': 3,
            'maxGroupLifetimeSeconds': 10}},
        'nodes': [{'id': 'input', 'type': 'user_input'},
            *[{'id': 'client-'+name, 'type': 'client', 'data': {'engine': 'openai', 'model': name}} for name in ('research','review')],
            *[{'id': name, 'type': 'llm', 'data': {'stream': stream, 'max_tokens': 500,
                 'max_input_tokens': 12000, 'agent_config': {'max_iterations': 5}}} for name in ('research','review')],
            {'id': 'messages', 'type': 'hook', 'data': {'hook_mode': 'messages',
                'target_node_ids': ['research','review'], 'messaging_by_target': {
                    'research': {'enabled': True, 'role': 'research', 'peers': ['review']},
                    'review': {'enabled': True, 'role': 'review', 'peers': ['research']}}}},
            {'id': 'python', 'type': 'hook', 'data': {'target_node_ids': ['research','review'],
                'lifecycle_event': 'onFinish', 'function_template':
                "def audit(context, chat_log):\n    context.emit.debug({'checked': context.event['node_id']})\n    return {'action': 'pass'}"}},
            {'id': 'skills', 'type': 'skills', 'data': {'schema_version': 1, 'skills': [{
                'id': 'review-guide', 'name': 'Review guide', 'description': 'Review the request',
                'prompt': 'PRIVATE_REVIEW_GUIDE'}]}},
            {'id': 'end', 'type': 'end'}],
        'edges': [*({'id': 'input-'+name, 'source': 'input', 'target': name,
                     'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'} for name in ('research','review')),
            *({'id': 'client-'+name, 'source': 'client-'+name, 'target': name,
                'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'} for name in ('research','review')),
            {'id': 'skill', 'source': 'skills', 'target': 'review', 'sourceHandle': 'handle-skills', 'targetHandle': 'handle-skills'},
            {'id': 'answer', 'source': 'research', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'}]}


def tool_results(chat):
    return {message['tool_call_id']: json.loads(message['content']) for message in chat.messages
            if message.get('role') == 'tool'}


def inbox_request(chat):
    for message in chat.messages:
        content = message.get('content')
        if not isinstance(content, str): continue
        try: data = json.loads(content.split(':\n', 1)[-1])
        except ValueError: continue
        if isinstance(data, dict) and data.get('kind') == 'request': return data['requestId']
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_normal_shared_hook_with_python_skills_and_participant_final_output(monkeypatch, stream):
    providers = {}
    async def research(provider, chat):
        results = tool_results(chat)
        if 'send' not in results:
            return tool_call('sendMessageToAgent', {'agentRef': 'review', 'message': 'Review this answer',
                'options': {'expectReply': True}}, 'send')
        if 'wait' not in results:
            return tool_call('waitForMessage', {'requestId': results['send']['receipt']['requestId'], 'timeoutSeconds': 2}, 'wait')
        assert 'review approved' in repr(chat.messages)
        return {'content': 'Final answer from the participant'}
    async def review(provider, chat):
        results = tool_results(chat)
        if 'skill' not in results:
            return tool_call('skills_load', {'skill_ids': ['review-guide']}, 'skill')
        assert 'PRIVATE_REVIEW_GUIDE' in repr(chat.messages)
        if 'reply' not in results:
            request = inbox_request(chat)
            assert request is not None
            return tool_call('replyToAgent', {'requestId': request, 'message': 'review approved'}, 'reply')
        return {'content': 'Review complete'}
    def client(**kwargs):
        name = kwargs['model']; result = object.__new__(MagicLLM)
        result.llm, result._task_executor = Provider(research if name == 'research' else review), None
        providers[name] = result.llm
        return result
    import importlib
    monkeypatch.setattr(importlib.import_module('magic_agents.node_system.NodeClientLLM'), 'MagicLLM', client)
    raw = definition(stream); before = copy.deepcopy(raw)
    graph = build(raw, message='Hello')
    assert raw == before and not graph._validation_errors
    events = await asyncio.wait_for(_collect(graph), 8)
    errors = [event for event in events if event.get('type') == 'debug' and event.get('content', {}).get('error_type')]
    assert not errors, errors
    texts = [choice.delta.content for event in events if event.get('type') == 'content'
             for choice in event['content'].choices]
    assert 'Final answer from the participant' in texts
    assert 'PRIVATE_REVIEW_GUIDE' not in repr(events)
    assert len(providers['research'].calls) == len(providers['review'].calls) == 3
    assert graph.nodes['end'].outputs
    records = graph.nodes['research']._invocation_control.records
    checked = [entry for entry in records if entry.get('node_id') == 'python']
    assert len(checked) == 2
    assert graph.nodes['research']._messages_hook_id == graph.nodes['review']._messages_hook_id == 'messages'


async def _collect(graph):
    return [event async for event in run_agent(graph)]


def test_inner_only_policy_creates_one_invocation_without_author_or_cost():
    from magic_agents.models.coordination import CoordinationPolicy
    child = SimpleNamespace(coordination=CoordinationPolicy.model_validate(definition()['coordination']), nodes={})
    root = SimpleNamespace(coordination=None, nodes={'inner': SimpleNamespace(inner_graph=child)})
    runtime = invocation_runtime(root)
    assert runtime.public_source is None and runtime.budget.invocation
    assert runtime.server_limits.max_cost is None and runtime.server_limits.max_image_jobs is None


@pytest.mark.parametrize('field,value', [('maxCost', {'amount': '1', 'currency': 'USD'}), ('maxImageJobs', 1)])
def test_unsupported_spending_policy_is_not_silently_ignored(field, value):
    from magic_agents.models.coordination import CoordinationPolicy
    policy = definition()['coordination']; policy['limits'][field] = value
    graph = SimpleNamespace(coordination=CoordinationPolicy.model_validate(policy), nodes={})
    with pytest.raises(CoordinationError) as caught: invocation_runtime(graph)
    assert caught.value.code == 'unsupported_invocation_policy'


def local_scope():
    from magic_agents.coordination.invocation import InvocationRuntime
    from magic_agents.models.coordination import CoordinationLimits
    runtime = InvocationRuntime(CoordinationLimits(maxToolCalls=8))
    return runtime.enter_graph(SimpleNamespace(coordination=None, nodes={}, edges=[]))


@pytest.mark.asyncio
async def test_invocation_fetch_reuses_normal_request_transport(monkeypatch):
    import magic_agents.node_system.fetch_request as fetch
    request, seen = object(), []
    async def normal(value):
        seen.append(value)
        return {'ordinary': True}
    def forbidden(*args):
        raise AssertionError('Messaging must not select the separate admitted HTTP profile')
    monkeypatch.setattr(fetch, 'send_step_request', normal)
    monkeypatch.setattr(fetch, 'admitted_http_options', forbidden)
    scope = local_scope()
    assert await fetch.coordinated_fetch(scope, 'fetch', request) == {'ordinary': True}
    assert seen == [request]
    assert (await scope.budget.snapshot())['spent']['tool_calls'] == 1


@pytest.mark.asyncio
async def test_invocation_mcp_preserves_stdio_transport_and_allowlist(monkeypatch):
    import importlib
    module = importlib.import_module('magic_agents.node_system.NodeMcp')
    from magic_agents.models.factory.Nodes.McpNodeModel import McpNodeModel
    sessions = []
    class Session:
        def __init__(self, config, node_id, debug):
            assert config.transport == 'stdio'
            self.server_key, self.server_instructions = 'native-stdio', None
            self._config, self._node_id, self.node_id, self.closed = config, node_id, node_id, False
            sessions.append(self)
        async def connect(self): pass
        async def cleanup(self): self.closed = True
        async def list_tools(self, cursor=None):
            return {'tools': [{'name': name, 'inputSchema': {'type': 'object'}}
                              for name in ('allowed', 'denied')]}
    monkeypatch.setattr(module, 'MCPSessionManager', Session)
    node = module.NodeMcp(McpNodeModel(servers=[{'transport': 'stdio', 'command': 'native-command',
                          'tool_allowlist': ['allowed']}]), node_id='mcp')
    events = [event async for event in node.process(SimpleNamespace(coordination=local_scope()))]
    assert sessions[0].closed
    assert len(node._bundle.tool_schemas) == 1
    assert node._bundle.tool_schemas[0]['function']['name'].endswith('allowed')
    assert events


@pytest.mark.asyncio
async def test_python_hook_timeout_does_not_join_sync_callback():
    import threading
    from magic_agents.node_system.NodeHook import NodeHook
    started, finished, release = threading.Event(), threading.Event(), threading.Event()
    def callback(context, chat_log):
        started.set()
        try: release.wait(1)
        finally: finished.set()
        return {'action': 'pass'}
    task = asyncio.create_task(NodeHook._execute_hook_function(
        object.__new__(NodeHook), callback, None, SimpleNamespace(coordination=local_scope())))
    try:
        for _ in range(100):
            if started.is_set(): break
            await asyncio.sleep(.005)
        assert started.is_set()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=.03)
        assert not finished.is_set(), 'Hook timeout must not wait for the callback thread'
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 1)
        await asyncio.gather(task, return_exceptions=True)
    assert finished.is_set()


@pytest.mark.asyncio
async def test_local_hook_child_keeps_journal_charge_and_deadline_without_pricing():
    from magic_agents.hooks.invocation_control import InvocationControl
    from magic_agents.models.factory.Nodes.ParserNodeModel import ParserNodeModel
    from magic_agents.node_system.NodeParser import NodeParser
    scope = local_scope()
    node = NodeParser(ParserNodeModel(text='Reviewed {{ content }}'), node_id='parser', node_type='parser')
    scope.graph.nodes['parser'] = node
    control = InvocationControl(scope.graph.nodes, [])
    log = SimpleNamespace(coordination=scope)
    calls = []
    async def invoke():
        calls.append('parser')
        return await control._operation('parser', 'hello', log, target_handle='content')
    result = await scope.run_child_operation('parser', 'first', 'hello', invoke)
    assert 'Reviewed hello' in repr(result)
    assert await scope.run_child_operation('parser', 'first', 'hello', invoke) == result
    assert calls == ['parser']
    assert (await scope.budget.snapshot())['spent']['tool_calls'] == 1
    scope.budget.deadline = scope.budget.clock() + .03
    async def waiting():
        await asyncio.Event().wait()
    with pytest.raises(asyncio.TimeoutError):
        await scope.run_child_operation('parser', 'second', 'wait', waiting)
    assert scope.child_operations['second']['state'] == 'unknown'
    assert (await scope.budget.snapshot())['spent']['tool_calls'] == 2
