"""Actual scheduler/NodeLLM/native-loop integration with bounded fake providers."""
import asyncio
import copy
import json
from decimal import Decimal

import pytest

from magic_llm import MagicLLM
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.model import ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel, UsageModel
from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.context import CoordinationRuntime
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
from magic_agents.models.coordination import CoordinationLimits, CoordinationPolicy, MessagingConfig
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.coordination.service import CoordinationError


class Provider:
    engine, engine_name, model = 'openai', 'openai', 'fake'

    def __init__(self, handler):
        self.handler, self.calls, self.options = handler, [], []
        self.kwargs, self.fallback, self.retry_config = {}, None, RetryConfig(1, 0)

    async def _execute_callback(self, *args): pass

    async def response(self, chat, options):
        self.calls.append(copy.deepcopy(chat.messages))
        self.options.append(options)
        message = await self.handler(self, chat)
        return ModelChatResponse(id=f'fake-{len(self.calls)}', model='fake', object='chat.completion', created=0,
            choices=[{'index': 0, 'message': {'role': 'assistant', **message},
                      'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}],
            usage=UsageModel(prompt_tokens=3, completion_tokens=2, total_tokens=5))

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat, **kwargs):
        return await self.response(chat, kwargs)

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat, **kwargs):
        response = await self.response(chat, kwargs)
        message = response.choices[0].message
        tools = None
        if message.tool_calls:
            tools = [{'index': i, **item.model_dump(exclude_none=True)} if hasattr(item, 'model_dump')
                     else {'index': i, **item} for i, item in enumerate(message.tool_calls)]
        yield ChatCompletionModel(id=response.id, model='fake', choices=[ChoiceModel(index=0,
            delta=DeltaModel(content=message.content, tool_calls=tools), finish_reason=response.choices[0].finish_reason)],
            usage=response.usage)


def limits(**extra):
    values = dict(maxModelTurns=20, maxInputTokens=50000, maxOutputTokens=10000,
        maxToolCalls=100, maxImageJobs=5, maxCost={'amount': '10', 'currency': 'USD'},
        maxWakeupsPerActor=2, maxWakeupsPerWorkgroup=3, maxConcurrentModelTurns=2)
    return CoordinationLimits(**(values | extra))


def runtime(server_limits=None, **extra):
    return CoordinationRuntime(server_limits=server_limits or limits(), authorize=lambda: None,
        estimate=lambda path, a: UsageBound(input_tokens=1000, output_tokens=10, cost=Decimal('.1')),
        usage=lambda path, a, o: UsageBound(input_tokens=3, output_tokens=2, cost=Decimal('.01')),
        authorize_attempt=lambda path, a: None, **extra)


def node(name, handler, *, peer=None, stream=False):
    messaging = MessagingConfig(enabled=True, role=name, peers=[peer], canWakePeers=[peer], wakeOnMessage=True) if peer else None
    llm = NodeLLM(LlmNodeModel(stream=stream, messaging=messaging), node_id=name, node_type='llm')
    client = object.__new__(MagicLLM)
    client.llm, client._task_executor = Provider(handler), None
    llm.inputs[llm.INPUT_HANDLER_CLIENT_PROVIDER] = client
    llm.inputs[llm.INPUT_HANDLER_USER_MESSAGE] = f'Original {name} instructions'
    return llm, client.llm


def graph(nodes):
    return AgentFlowModel(type='graph', nodes=nodes, coordination=CoordinationPolicy(
        enabled=True, allowedParticipants=['research', 'images'], limits=limits()), edges=[
        EdgeNodeModel(id='research-author', source='research', target='author',
                      sourceHandle=nodes['research'].OUTPUT_HANDLE_GENERATED, targetHandle=nodes['author'].INPUT_HANDLER_SYSTEM_CONTEXT),
        EdgeNodeModel(id='images-author', source='images', target='author',
                      sourceHandle=nodes['images'].OUTPUT_HANDLE_GENERATED, targetHandle=nodes['author'].INPUT_HANDLER_USER_MESSAGE)])


def send_call(*, wake=False):
    return {'content': None, 'tool_calls': [{'id': 'targeted-request', 'type': 'function', 'function': {
        'name': 'sendMessageToAgent', 'arguments': json.dumps({'agentRef': 'images', 'message': 'targeted image for slide 4',
            'options': {'expectReply': False, 'wakeIfCompleted': wake}})}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_graph_wake_resumes_canonical_actor_and_author_sees_only_sealed_assets(stream):
    rt = runtime()
    async def research(provider, chat):
        if len(provider.calls) == 1:
            scope = rt.scopes[0]
            async with scope.service._condition:
                while scope.service._actors[scope.service._roles['images']].state != 'quiescent':
                    await scope.service._condition.wait()
            return send_call(wake=True)
        return {'content': 'research complete'}
    async def images(provider, chat):
        if len(provider.calls) == 1: return {'content': 'generic asset'}
        assert any(m.get('content') == 'generic asset' for m in chat.messages)
        assert 'targeted image for slide 4' in chat.messages[-1]['content']
        return {'content': 'generic asset + targeted asset'}
    async def author(provider, chat):
        assert rt.scopes[0].service._state == 'sealed'
        assert 'generic asset + targeted asset' in repr(chat.messages)
        return {'content': 'public deck'}
    a, pa = node('research', research, peer='images', stream=stream)
    b, pb = node('images', images, peer='research', stream=stream)
    c, pc = node('author', author, stream=stream)
    outcome = {}
    messages = []
    async def run():
        async for item in execute_graph_reactive(graph({'research': a, 'images': b, 'author': c}),
            runtime_config=RuntimeConfig(coordination=rt), result=outcome):
            messages.append(item)
    await asyncio.wait_for(run(), 5)
    assert outcome['has_errors'] is False, messages
    assert len(pa.calls) == len(pb.calls) == 2
    assert len(pc.calls) == 1
    assert (await rt.budget.snapshot())['modelTurns'] == 5
    assert all(item.get('source_node') not in ('research', 'images') for item in messages if item.get('type') == 'content')
    assert not any('sendMessageToAgent' in repr(options.get('tools')) for options in pc.options)
    checkpoint = rt.scopes[0].service._actors[rt.scopes[0].service._roles['images']].checkpoint
    assert checkpoint['step'] == 2 and len(checkpoint['consumed_message_ids']) == 1


@pytest.mark.asyncio
async def test_runtime_context_is_not_serialized_as_run_log_or_shared_hook_state():
    from magic_agents.models.model_agent_run_log import ModelAgentRunLog
    rt = runtime()
    assert 'coordination' not in ModelAgentRunLog(coordination=rt).model_dump()
    assert RuntimeConfig(coordination=rt).coordination is rt
    assert RuntimeConfig().coordination is None


async def collect(g, rt):
    outcome, messages = {}, []
    async def run():
        async for item in execute_graph_reactive(g, runtime_config=RuntimeConfig(coordination=rt), result=outcome):
            messages.append(item)
    await asyncio.wait_for(run(), 3)
    return outcome, messages


@pytest.mark.asyncio
async def test_fail_group_cancels_inflight_peer_and_prevents_author_dispatch():
    rt = runtime()
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def research(provider, chat):
        started.set()
        try: await asyncio.Event().wait()
        finally: cancelled.set()
    async def images(provider, chat):
        await started.wait()
        raise RuntimeError('image operation failed')
    async def author(provider, chat):
        raise AssertionError('Author must not receive failed-group output')
    a, _ = node('research', research, peer='images')
    b, _ = node('images', images, peer='research')
    c, pc = node('author', author)
    outcome, _ = await collect(graph({'research': a, 'images': b, 'author': c}), rt)
    assert outcome['has_errors'] and cancelled.is_set()
    assert not pc.calls
    assert rt.scopes[0].service._state == 'failed'
    budget = await rt.budget.snapshot()
    assert budget['activeModels'] == 0 and budget['uncertain']['input_tokens'] > 0


@pytest.mark.asyncio
async def test_declared_partial_failure_reaches_author_with_failure_manifest_once():
    rt = runtime()
    async def research(provider, chat): return {'content': 'usable research'}
    async def images(provider, chat): raise RuntimeError('image unavailable')
    async def author(provider, chat):
        assert 'usable research' in repr(chat.messages)
        assert 'coordinationFailure' in repr(chat.messages)
        assert rt.scopes[0].service._state == 'sealed'
        return {'content': 'deck with an explicit missing-image warning'}
    a, _ = node('research', research, peer='images')
    b, _ = node('images', images, peer='research')
    c, pc = node('author', author)
    g = graph({'research': a, 'images': b, 'author': c})
    g.coordination.failure_policy = 'publish_partial'
    outcome, messages = await collect(g, rt)
    assert not outcome['has_errors'], messages
    assert len(pc.calls) == 1
    assert rt.scopes[0].service._publication['images']['failure']['reason'] == 'PROVIDER_ATTEMPTS_EXHAUSTED'
    assert c.outputs[c.OUTPUT_HANDLE_GENERATED]['content'] == 'deck with an explicit missing-image warning'


def tool_call(name, args, ident):
    return {'content': None, 'tool_calls': [{'id': ident, 'type': 'function', 'function': {
        'name': name, 'arguments': json.dumps(args)}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_request_reply_wait_uses_real_tools_with_one_model_slot(stream):
    rt = runtime(server_limits=limits(maxConcurrentModelTurns=1))
    async def research(provider, chat):
        if len(provider.calls) == 1:
            return tool_call('sendMessageToAgent', {'agentRef': 'images', 'message': 'specific request',
                'options': {'expectReply': True, 'wakeIfCompleted': True}}, 'send-1')
        if len(provider.calls) == 2:
            request = next(m for m in rt.scopes[0].service._messages.values() if m.kind == 'request')
            return tool_call('waitForMessage', {'requestId': request.request_id, 'timeoutSeconds': 1}, 'wait-1')
        assert 'targeted-result' in repr(chat.messages)
        return {'content': 'research using targeted-result'}
    async def images(provider, chat):
        text = repr(chat.messages)
        if 'specific request' in text and not any(m.get('role') == 'tool' and m.get('name') == 'replyToAgent' for m in chat.messages):
            # Matching tool result may omit name on the provider wire; identify
            # the already emitted canonical call by its stable ID instead.
            if not any(m.get('tool_call_id') == 'reply-1' for m in chat.messages):
                request = next(m for m in rt.scopes[0].service._messages.values() if m.kind == 'request')
                return tool_call('replyToAgent', {'requestId': request.request_id, 'message': 'targeted-result'}, 'reply-1')
        return {'content': 'assets ready'}
    async def author(provider, chat): return {'content': 'public final'}
    a, pa = node('research', research, peer='images', stream=stream)
    b, pb = node('images', images, peer='research', stream=stream)
    c, pc = node('author', author, stream=stream)
    outcome, messages = await collect(graph({'research': a, 'images': b, 'author': c}), rt)
    assert not outcome['has_errors'], messages
    assert len(pa.calls) == 3 and len(pc.calls) == 1
    assert any('specific request' in repr(call) for call in pb.calls)
    assert rt.scopes[0].service._state == 'sealed'
    assert (await rt.budget.snapshot())['spent']['tool_calls'] == 3


@pytest.mark.asyncio
async def test_two_real_inner_scopes_share_budget_and_preserve_distinct_outputs():
    from magic_agents.node_system.Node import Node
    from magic_agents.node_system.NodeInner import NodeInner
    from magic_agents.models.factory.Nodes.InnerNodeModel import InnerNodeModel
    class Seed(Node):
        def __init__(self, values):
            super().__init__(node_id='seed', node_type='test')
            self.values = values
        async def process(self, chat_log):
            for handle, value in self.values.items(): yield self.yield_static(value, content_type=handle)
    providers = []
    def inner(name):
        async def peer(provider, chat):
            assert name in repr(chat.messages)
            return {'content': name + ' private assets'}
        async def author(provider, chat):
            assert name + ' private assets' in repr(chat.messages)
            return {'content': name + ' public output'}
        a, pa = node('research', peer, peer='images')
        b, pb = node('images', peer, peer='research')
        c, pc = node('author', author)
        providers.extend([pa, pb, pc])
        g = graph({'research': a, 'images': b, 'author': c})
        values = {}
        for child in (a, b, c):
            for index, (handle, value) in enumerate(child.inputs.items()):
                source = f'{child.node_id}-{index}'
                values[source] = name + ' instructions' if handle == child.INPUT_HANDLER_USER_MESSAGE else value
                g.edges.append(EdgeNodeModel(id=source, source='seed', target=child.node_id,
                                             sourceHandle=source, targetHandle=handle))
        g.nodes['seed'] = Seed(values)
        n = NodeInner(InnerNodeModel(), load_chat=lambda *a: None, node_id=name, node_type='inner')
        n.inner_graph = g
        n.inputs[n.INPUT_HANDLE] = name + ' request'
        return n
    rt = runtime()
    root = AgentFlowModel(type='graph', nodes={'one': inner('one'), 'two': inner('two')}, edges=[])
    outcome, messages = await collect(root, rt)
    assert not outcome['has_errors'], messages
    assert len(rt.scopes) == 3 and all(len(p.calls) == 1 for p in providers)
    groups = [scope for scope in rt.scopes if scope.service is not None]
    assert {scope.path for scope in groups} == {('one',), ('two',)}
    assert len({scope.scope_id for scope in groups}) == 2
    assert {scope.service.workgroup_id for scope in groups} == {rt.workgroup_id}
    assert {scope.service.epoch_id for scope in groups} == {rt.epoch_id}
    assert (await rt.budget.snapshot())['modelTurns'] == 6
    for scope in groups:
        author = scope.graph.nodes['author']
        assert author.outputs[author.OUTPUT_HANDLE_GENERATED]['content'] == scope.path[0] + ' public output'
