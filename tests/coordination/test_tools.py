"""Real ToolExecutor and native-loop checks; no network, DB, or providers."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from magic_agents.coordination.control import ActorLoopControl
from magic_agents.coordination.service import CoordinationError, CoordinationService
from magic_agents.coordination.tools import CoordinationTools, reserve_builtin_names
from magic_agents.models.coordination import BUILTIN_NAMES, CoordinationLimits, CoordinationPolicy, MessagingConfig
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.engine.tooling import map_request_tools
from magic_llm.model import ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel

pytestmark = pytest.mark.asyncio


def service(*, max_tools=100, enabled=None, max_inline=16384):
    limits = CoordinationLimits(maxModelTurns=20, maxInputTokens=10000, maxOutputTokens=10000,
        maxToolCalls=max_tools, maxImageJobs=10, maxCost={'amount': '10', 'currency': 'USD'},
        maxWakeupsPerActor=2, maxWakeupsPerWorkgroup=3, maxInlineMessageBytes=max_inline)
    members = {role: ((role,), MessagingConfig(enabled=True, role=role, peers=[peer],
        description=role + ' description', canWakePeers=[peer], wakeOnMessage=True,
        tools=enabled if enabled is not None else list(BUILTIN_NAMES)))
        for role, peer in [('research', 'images'), ('images', 'research')]}
    return CoordinationService(CoordinationPolicy(enabled=True, allowedParticipants=list(members), limits=limits),
        members, server_limits=limits, authorize=lambda: None)


async def setup(*, max_tools=100, max_output=50000, enabled=None, max_inline=16384):
    svc = service(max_tools=max_tools, enabled=enabled, max_inline=max_inline)
    caller, peer = await svc.activate('research'), await svc.activate('images')
    bundle = CoordinationTools(caller)
    executor = ToolExecutor(enable_dedup=True, max_content_size=max_output)
    bundle.configure_executor(executor)
    for name, function in bundle.tool_functions.items():
        executor.register(name, function)
    return svc, caller, peer, bundle, executor


async def call(executor, name, identifier='call', **arguments):
    result = await executor.execute_async(CanonicalToolCall(identifier, name, arguments))
    assert not result.is_deduplicated
    assert '[TRUNCATED]' not in result.content
    return json.loads(result.content)


async def test_registered_schema_has_only_bound_capabilities_and_no_identity_arguments():
    svc, caller, peer, bundle, executor = await setup()
    assert set(bundle.tool_functions) == set(BUILTIN_NAMES)
    for spec in bundle.tools:
        parameters = spec['function']['parameters']
        assert parameters['additionalProperties'] is False
        serialized = json.dumps(parameters)
        assert 'actorId' not in serialized and 'senderId' not in serialized and 'budget' not in serialized
    result = await call(executor, 'listAgents')
    assert [item['role'] for item in result['peers']] == ['images']
    assert result['peers'][0]['agentRef'] == peer.actor_id
    assert result['peers'][0]['description'] == 'images description'
    assert 'remainingBudget' in result['peers'][0]


@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'google'])
async def test_tool_schemas_map_without_recursive_json_expansion(provider):
    svc, caller, peer, bundle, executor = await setup()
    mapped = map_request_tools(provider, bundle.tools, 'auto')
    encoded = json.dumps(mapped.tools)
    assert 'sendMessageToAgent' in encoded and 'waitForAgent' in encoded
    assert 'JsonValue' not in encoded
    if provider == 'google': assert '$ref' not in encoded


async def test_inspection_never_uses_fingerprint_cache():
    svc, caller, peer, bundle, executor = await setup()
    before = await call(executor, 'inspectAgent', 'first', agentRef='images')
    await svc.checkpoint(peer, {}, [])
    await svc.finish(peer, 'retained')
    after = await call(executor, 'inspectAgent', 'second', agentRef='images')
    assert before['state'] == 'running'
    assert after['state'] == 'quiescent' and not after['outputPublished']
    assert executor._dedup_cache == {}


async def test_same_call_id_replays_but_identical_new_call_creates_new_message():
    svc, caller, peer, bundle, executor = await setup()
    args = {'agentRef': 'images', 'message': 'same', 'options': {'expectReply': False}}
    first = await call(executor, 'sendMessageToAgent', 'first', **args)
    replay = await call(executor, 'sendMessageToAgent', 'first', **args)
    second = await call(executor, 'sendMessageToAgent', 'second', **args)
    assert first['receipt'] == replay['receipt']
    assert second['receipt']['messageId'] != first['receipt']['messageId']
    assert len(await svc.inbox(peer)) == 2
    assert (await svc.budget.snapshot())['spent']['tool_calls'] == 2
    assert first['receipt']['expiresAt'].endswith('Z')
    assert 'requestId' not in first['receipt']


async def test_explicit_domain_key_does_not_waive_new_physical_tool_call_charge():
    svc, caller, peer, bundle, executor = await setup()
    args = {'agentRef': 'images', 'message': 'same',
            'options': {'expectReply': False, 'idempotencyKey': 'business-operation'}}
    first = await call(executor, 'sendMessageToAgent', 'attempt-one', **args)
    replay = await call(executor, 'sendMessageToAgent', 'attempt-two', **args)
    assert first['receipt'] == replay['receipt']
    assert len(await svc.inbox(peer)) == 1
    assert (await svc.budget.snapshot())['spent']['tool_calls'] == 2
    conflict = await call(executor, 'sendMessageToAgent', 'attempt-three', **dict(args, message='different'))
    assert conflict['error']['code'] == 'idempotency_conflict'


async def test_stale_owner_cannot_charge_new_physical_call_but_original_receipt_replays():
    svc, caller, peer, bundle, executor = await setup()
    args = {'agentRef': 'images', 'message': 'accepted',
            'options': {'expectReply': False, 'idempotencyKey': 'business-key'}}
    first = await call(executor, 'sendMessageToAgent', 'original-call', **args)
    await svc.checkpoint(caller, {}, [])
    await svc.finish(caller, 'done')
    replay = await call(executor, 'sendMessageToAgent', 'original-call', **args)
    assert replay['receipt'] == first['receipt']
    rejected = await call(executor, 'sendMessageToAgent', 'new-call-old-key', **args)
    assert rejected['error']['code'] == 'stale_activation'
    assert (await svc.budget.snapshot())['spent']['tool_calls'] == 1
    assert len(await svc.inbox(peer)) == 1


async def test_cancelled_scope_tool_replay_does_not_charge_or_reopen_work():
    svc, caller, peer, bundle, executor = await setup()
    args = {'agentRef': 'images', 'message': 'accepted', 'options': {'expectReply': False}}
    first = await call(executor, 'sendMessageToAgent', 'original-call', **args)
    await svc.cancel()
    replay = await call(executor, 'sendMessageToAgent', 'original-call', **args)
    assert replay['receipt'] == first['receipt']
    rejected = await call(executor, 'sendMessageToAgent', 'new-call', **args)
    assert rejected['error']['code'] == 'cancelled'
    assert (await svc.budget.snapshot())['spent']['tool_calls'] == 1


async def test_missing_runtime_identity_never_mutates_even_with_explicit_key():
    svc, caller, peer, bundle, executor = await setup()
    result = await bundle.tool_functions['sendMessageToAgent'](agentRef='images', message='no context',
        options={'idempotencyKey': 'explicit'})
    assert result['error']['code'] == 'missing_operation_identity'
    assert await svc.inbox(peer) == []
    assert (await svc.budget.snapshot())['spent']['tool_calls'] == 0


async def test_tiny_shared_tool_budget_rejects_before_message_acceptance():
    svc, caller, peer, bundle, executor = await setup(max_tools=1)
    await call(executor, 'inspectAgent', 'one', agentRef='images')
    denied = await call(executor, 'sendMessageToAgent', 'two', agentRef='images', message='not accepted')
    assert denied['error']['code'] == 'budget_exhausted'
    assert await svc.inbox(peer) == []


@pytest.mark.parametrize('bad', [{'senderId': 'fake'}, {'actorId': 'fake'}, {'budget': {}},
    {'options': {'senderId': 'fake'}}, {'options': {'expectedEpochId': 1}}, {'options': {'idempotencyKey': ''}}])
async def test_identity_budget_and_invalid_options_are_not_model_authority(bad):
    svc, caller, peer, bundle, executor = await setup()
    args = {'agentRef': 'images', 'message': 'must reject', **bad}
    rejected = await call(executor, 'sendMessageToAgent', **args)
    assert rejected['error']['code'] == 'invalid_arguments'
    assert await svc.inbox(peer) == []


async def test_actor_and_scope_isolation_and_bounded_errors():
    svc, caller, peer, bundle, executor = await setup()
    foreign = service()
    foreign_caller = await foreign.activate('research')
    foreign_peer = await foreign.activate('images')
    denied = await call(executor, 'sendMessageToAgent', 'foreign', agentRef=foreign_peer.actor_id, message='no')
    assert denied['error']['code'] == 'unknown_ref'
    assert await foreign.inbox(foreign_peer) == []
    peer_bundle = CoordinationTools(peer)
    other_executor = ToolExecutor(enable_dedup=True)
    peer_bundle.configure_executor(other_executor)
    for name, function in peer_bundle.tool_functions.items(): other_executor.register(name, function)
    sent = await call(executor, 'sendMessageToAgent', 'same-id', agentRef='images', message='a', options={'expectReply': False})
    other = await call(other_executor, 'sendMessageToAgent', 'same-id', agentRef='research', message='b', options={'expectReply': False})
    assert sent['receipt']['messageId'] != other['receipt']['messageId']
    with pytest.raises(CoordinationError):
        CoordinationTools(type(caller)(svc, foreign_caller.actor_id, caller.activation_id, caller._owner_token))


async def test_reply_resolves_original_destination_and_error_code_without_rewriting_payload():
    svc, caller, peer, bundle, executor = await setup()
    request = await svc.send(peer, 'research', 'make asset', key='incoming')
    request_id = request['receipt']['requestId']
    receipt = await call(executor, 'replyToAgent', requestId=request_id, message='failed', payload={'nullable': None},
        options={'outcome': 'error', 'errorCode': 'asset_unavailable'})
    assert receipt['receipt']['requestOutcome'] == 'error'
    assert receipt['receipt']['errorCode'] == 'asset_unavailable'
    delivered = await svc.wait_message(peer, request_id=request_id, timeout=0.01)
    assert delivered['message']['payload'] == {'nullable': None}
    assert delivered['message']['errorCode'] == 'asset_unavailable'


async def test_reserved_names_reject_collisions_before_registration_and_cover_disabled_names():
    svc, caller, peer, bundle, executor = await setup(enabled=['inspectAgent'])
    assert list(bundle.tool_functions) == ['inspectAgent']
    with pytest.raises(CoordinationError, match='collision'):
        CoordinationTools(caller, existing_names=['sendMessageToAgent'])
    other = ToolExecutor()
    other.register('replyToAgent', lambda: 'foreign')
    with pytest.raises(CoordinationError, match='collision'):
        bundle.configure_executor(other)
    assert other.registered_names() == {'replyToAgent'}
    assert executor._serial_tools == set()
    reserve_builtin_names(['ordinaryTool'])


async def test_configure_preserves_parent_executor_and_short_mutation_barriers():
    svc, caller, peer, bundle, executor = await setup()
    parent = ToolExecutor(enable_dedup=True)
    child = parent.fork()
    bundle.configure_executor(child)
    assert child._serial_tools == {'sendMessageToAgent', 'replyToAgent'}
    assert not parent._serial_tools
    assert 'waitForMessage' not in child._serial_tools
    with pytest.raises(CoordinationError, match='2048-character'):
        bundle.configure_executor(ToolExecutor(max_content_size=10))


async def test_real_executor_mutations_are_ordered_around_fresh_inspection():
    svc, caller, peer, bundle, executor = await setup()
    results = await executor.execute_parallel_async([
        CanonicalToolCall('send-1', 'sendMessageToAgent', {'agentRef': 'images', 'message': 'one', 'options': {'expectReply': False}}),
        CanonicalToolCall('inspect', 'inspectAgent', {'agentRef': 'images'}),
        CanonicalToolCall('send-2', 'sendMessageToAgent', {'agentRef': 'images', 'message': 'two', 'options': {'expectReply': False}}),
    ])
    first, view, last = [json.loads(result.content) for result in results]
    assert first['receipt']['recipientSequence'] == 1
    assert view['pendingMessageCount'] == 1
    assert last['receipt']['recipientSequence'] == 2


async def test_targeted_wait_yields_for_unrelated_input_then_keeps_exact_reply_obligation():
    svc, caller, peer, bundle, executor = await setup()
    sent = await call(executor, 'sendMessageToAgent', 'send', agentRef='images', message='specific')
    request_id = sent['receipt']['requestId']
    await svc.send(peer, 'research', 'another input', key='event', kind='event', expect_reply=False)
    interrupted = await call(executor, 'waitForMessage', 'wait-1', requestId=request_id, timeoutSeconds=0.01)
    assert interrupted['outcome'] == 'inbox_ready'
    assert request_id in svc._actors[caller.actor_id].requests
    await svc.reply(peer, request_id, 'specific reply', key='reply')
    resolved = await call(executor, 'waitForMessage', 'wait-2', requestId=request_id, timeoutSeconds=0.01)
    assert resolved['outcome'] == 'reply'
    assert resolved['requestId'] == request_id
    assert resolved['message']['message'] == 'specific reply'


async def test_oversized_wait_is_complete_json_and_does_not_consume_message():
    svc, caller, peer, bundle, executor = await setup(max_output=2048)
    await svc.send(peer, 'research', 'x' * 6000, {'null': None}, key='large', expect_reply=False)
    result = await call(executor, 'waitForMessage', timeoutSeconds=0.01)
    assert result['error']['code'] == 'tool_output_limit'
    assert len(await svc.inbox(caller)) == 1


async def test_exact_wait_guards_and_timeout_are_structured_results():
    svc, caller, peer, bundle, executor = await setup()
    invalid = await call(executor, 'waitForAgent', 'invalid', agentRef='images',
        options={'until': 'activation_finished', 'timeoutSeconds': 0.01})
    assert invalid['error']['code'] == 'invalid_arguments'
    timeout = await call(executor, 'waitForAgent', 'timeout', agentRef='images',
        options={'until': 'activation_finished', 'activationId': peer.activation_id, 'timeoutSeconds': 0.001})
    assert timeout['ok'] and timeout['outcome'] == 'timeout'
    assert timeout['state'] == 'running' and timeout['outputRevision'] == 0
    await svc.checkpoint(peer, {}, []); await svc.finish(peer, 'done')
    complete = await call(executor, 'waitForAgent', 'finished', agentRef='images',
        options={'until': 'activation_finished', 'activationId': peer.activation_id, 'timeoutSeconds': 0.01})
    assert complete['outcome'] == 'finished' and complete['outputRevision'] == 1


async def test_wait_cancellation_propagates_and_restores_actor_state():
    svc, caller, peer, bundle, executor = await setup()
    active = asyncio.create_task(call(executor, 'waitForMessage', timeoutSeconds=10.0))
    for _ in range(10):
        await asyncio.sleep(0)
        if svc._actors[caller.actor_id].state == 'awaiting_reply': break
    assert svc._actors[caller.actor_id].state == 'awaiting_reply'
    active.cancel()
    with pytest.raises(asyncio.CancelledError): await active
    assert svc._actors[caller.actor_id].state == 'running'


@pytest.mark.parametrize('stream', [False, True])
async def test_native_wait_result_precedes_canonical_mailbox_delivery(stream):
    svc, caller, peer, bundle, executor = await setup()
    waiting = asyncio.Event()
    original_wait = svc.wait_message

    async def observed_wait(*args, **kwargs):
        waiting.set()
        return await original_wait(*args, **kwargs)

    svc.wait_message = observed_wait

    class Provider:
        engine, model, fallback = 'openai', 'fake', None
        retry_config, kwargs = RetryConfig(1, 0), {}

        def __init__(self): self.requests = []

        async def _execute_callback(self, *args): pass

        def response(self, chat):
            self.requests.append(copy.deepcopy(chat.messages))
            tool = {'id': 'wait-call', 'type': 'function', 'function': {
                'name': 'waitForMessage', 'arguments': json.dumps({'timeoutSeconds': 1.0})}}
            return ModelChatResponse(id='fake', object='chat.completion', created=0, model='fake', choices=[{
                'index': 0, 'message': {'role': 'assistant', 'content': 'done' if len(self.requests) > 1 else None,
                                      'tool_calls': [tool] if len(self.requests) == 1 else None},
                'finish_reason': 'tool_calls' if len(self.requests) == 1 else 'stop'}])

        @BaseChat.async_intercept_generate
        async def async_generate(self, chat, **kwargs): return self.response(chat)

        @BaseChat.async_intercept_stream_generate
        async def async_stream_generate(self, chat, **kwargs):
            response = self.response(chat)
            calls = [dict(item.model_dump(), index=0) for item in response.tool_calls or []]
            yield ChatCompletionModel(id='fake', model='fake', choices=[{'index': 0, 'delta': {
                'content': response.content, 'tool_calls': calls or None}, 'finish_reason': response.finish_reason}])

    class Admission:
        async def before_attempt(self, attempt): pass
        async def after_attempt(self, attempt, outcome): pass

    provider = Provider()
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), tools=bundle.tools, tool_functions=bundle.tool_functions,
        tool_executor=executor, builtin_todo_tools=False, control=ActorLoopControl(caller), provider_attempt_control=Admission())

    async def run():
        if stream: return [item async for item in loop.stream('task')]
        return await loop.run('task')

    task = asyncio.create_task(run())
    await waiting.wait()
    await svc.send(peer, 'research', 'arrived during wait', key='late', expect_reply=False)
    await task
    messages = provider.requests[1]
    assert messages[-2]['role'] == 'tool' and messages[-2]['tool_call_id'] == 'wait-call'
    assert json.loads(messages[-2]['content'])['outcome'] == 'message'
    assert messages[-1]['role'] == 'user' and 'arrived during wait' in messages[-1]['content']
    assert await svc.inbox(caller) == []
