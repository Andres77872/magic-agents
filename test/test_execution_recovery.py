"""Native confirmed-response recovery with the real async loop, no external IO."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.types import AgentBudget
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.model.ModelChatResponse import ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel
from magic_agents.execution.recorder import CoreRecorder, CoreScope, NodeInvocation, _scope, _node
from magic_agents.execution.llm_storage import CoreAttemptControl, CoreLoopControl, _tool
from magic_agents.execution.recovery import inspect_recovery
from magic_agents.execution.storage import ExecutionState
from test.test_execution_recorder import Store, config


class Cut(BaseException):
    pass


class CutStore(Store):
    limits = SimpleNamespace(max_state_bytes=1_000_000)

    def __init__(self, boundary):
        super().__init__()
        self.boundary = boundary

    async def commit(self, lease, transition):
        result = await super().commit(lease, transition)
        if self.boundary and any(item.kind == self.boundary for item in transition.events):
            self.boundary = None
            raise Cut()
        return result


def answer(calls=None):
    return ModelChatResponse(id='response', object='chat.completion', created=0, model='fake',
        choices=[{'index': 0, 'message': {'role': 'assistant', 'content': None if calls else 'done', 'tool_calls': calls},
                  'finish_reason': 'tool_calls' if calls else 'stop'}],
        usage=UsageModel(prompt_tokens=3, completion_tokens=2, total_tokens=5))


class Provider:
    engine, model = 'openai', 'fake'
    retry_config, fallback, kwargs = RetryConfig(1, 0), None, {}

    def __init__(self):
        self.calls = 0

    async def _execute_callback(self, *args):
        pass

    def response(self):
        self.calls += 1
        return answer([{'id': 'original-tool', 'type': 'function', 'function': {'name': 'count', 'arguments': '{}'}}] if self.calls == 1 else None)

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat, **kwargs):
        return self.response()

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat, **kwargs):
        response = self.response()
        calls = [dict(item.model_dump(), index=index) for index, item in enumerate(response.tool_calls or [])]
        yield ChatCompletionModel(id=response.id, model='fake', usage=response.usage,
            choices=[{'index': 0, 'delta': {'content': response.content, 'tool_calls': calls or None},
                      'finish_reason': response.finish_reason}])


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('boundary', ['llm_response', 'tool_end'])
async def test_confirmed_response_and_tool_cut_resume_without_repeating_effects(stream, boundary):
    store, provider = CutStore(boundary), Provider()
    snapshot = config(store).execution_snapshot
    physical_tools = []
    async def count():
        physical_tools.append(True)
        return {'count': 1}
    tool_schema = {'type': 'function', 'function': {'name': 'count', 'description': 'count',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}

    async def run(restore=None):
        recorder = CoreRecorder(store, snapshot, restore=restore)
        await recorder.start()
        scope = CoreScope(recorder, SimpleNamespace(nodes={}), (), snapshot.identity.run_id,
                          snapshot.identity.root_execution_id, 'original-scope', True)
        saved = (restore.state.checkpoint['graphs']['original-scope']['nodes']['llm'] if restore else None)
        node = NodeInvocation('llm', 'original-node', snapshot.cause, saved=saved)
        st, nt = _scope.set(scope), _node.set(node)
        try:
            if restore is None:
                await scope.record('graph_start', {'status': 'running'}, status='running')
                await scope.record('node_start', {'node_type': 'NodeLLM', 'execution_id': node.execution_id}, node_id='llm')
            control = CoreLoopControl(scope, node)
            loop = AsyncAgentLoop(SimpleNamespace(llm=provider), tools=[tool_schema],
                tool_functions={'count': _tool(scope, node, 'count', count)},
                control=control, provider_attempt_control=CoreAttemptControl(scope, node),
                budget=AgentBudget(max_iterations=3), builtin_todo_tools=False)
            options = {'continuation': saved['llm_response']['checkpoint']} if restore else {'user_input': 'count once'}
            if stream:
                result = [chunk async for chunk in loop.stream(**options)]
            else:
                result = await loop.run(**options)
            return loop.checkpoint, result
        finally:
            await recorder.close()
            _node.reset(nt); _scope.reset(st)

    with pytest.raises(Cut):
        await run()
    original = store.record
    assert inspect_recovery(original).recoverable
    assert provider.calls == 1
    checkpoint, result = await run(original)
    assert provider.calls == 2 and physical_tools == [True]
    assert checkpoint.step == 2
    assert checkpoint.started_at == original.state.checkpoint['graphs']['original-scope']['nodes']['llm']['llm_response']['checkpoint']['started_at']
    assert not any(effect.status in {'prepared', 'dispatched', 'unknown'} for effect in store.record.state.effects)


def test_failed_and_unconfirmed_provider_outcomes_do_not_offer_continue():
    store = Store()
    snapshot = config(store).execution_snapshot
    from magic_agents.execution.storage import ExecutionRecord
    node = {'node_start': {'node_type': 'NodeLLM'}, 'llm_checkpoint': {'boundary': 'input'},
            'provider_attempts': {'attempt': 'completed'}}
    state = ExecutionState(status='blocked', checkpoint={'schema_version': 1, 'graphs': {'scope': {
        'node_path': [], 'status': 'blocked', 'nodes': {'llm': node}}}})
    record = ExecutionRecord(snapshot=snapshot, state=state, version=3, fence=1)
    assert inspect_recovery(record).reason == 'provider_response_unconfirmed'
    assert inspect_recovery(record.model_copy(update={'state': state.model_copy(update={'status': 'failed'})})).reason == 'execution_terminal'


@pytest.mark.asyncio
@pytest.mark.parametrize('cut', ['between_iterations', 'within_iteration', 'post_loop'])
async def test_real_loop_restores_cursor_without_repeating_committed_parsers(cut):
    from magic_agents.agt_flow import build
    from magic_agents.execution.reactive_executor import execute_graph_reactive
    class Interrupted(Store):
        armed = True
        pending_cut = False
        async def commit(self, lease, transition):
            if self.pending_cut:
                self.pending_cut = False
                raise asyncio.CancelledError()
            event = transition.events[0]
            boundary = (event.kind == 'loop_checkpoint' and event.data['next_index'] == 1
                        if cut == 'between_iterations' else
                        event.kind == 'node_start' and event.data['node_id'] == ('tail' if cut == 'within_iteration' else 'final')
                        and event.data.get('execution_phase') == ('loop:iteration:1' if cut == 'within_iteration' else 'loop:post'))
            if self.armed and boundary and cut != 'between_iterations':
                self.armed = False
                raise asyncio.CancelledError()
            receipt = await super().commit(lease, transition)
            if self.armed and boundary:
                self.armed = False
                self.pending_cut = True
            return receipt

    raw = {'type': 'graph', 'nodes': [
        {'id': 'input', 'type': 'user_input'},
        {'id': 'items', 'type': 'text', 'data': {'text': '["a", "b", "c"]'}},
        {'id': 'loop', 'type': 'loop', 'data': {}},
        {'id': 'transform', 'type': 'parser', 'data': {'text': '{{ handle_parser_input }}!'}},
        {'id': 'tail', 'type': 'parser', 'data': {'text': '{{ handle_parser_input }}?'}},
        {'id': 'final', 'type': 'parser', 'data': {'text': "{{ handle_parser_input | join(',') }}"}},
    ], 'edges': [
        {'id': 'enter', 'source': 'input', 'sourceHandle': 'handle_user_message', 'target': 'items', 'targetHandle': 'handle_flow_input'},
        {'id': 'list', 'source': 'items', 'sourceHandle': 'handle_text_output', 'target': 'loop', 'targetHandle': 'handle_list'},
        {'id': 'item', 'source': 'loop', 'sourceHandle': 'handle_item', 'target': 'transform', 'targetHandle': 'handle_parser_input'},
        {'id': 'next', 'source': 'transform', 'sourceHandle': 'handle_parser_output', 'target': 'tail', 'targetHandle': 'handle_parser_input'},
        {'id': 'feedback', 'source': 'tail', 'sourceHandle': 'handle_parser_output', 'target': 'loop', 'targetHandle': 'handle_loop'},
        {'id': 'finish', 'source': 'loop', 'sourceHandle': 'handle_end', 'target': 'final', 'targetHandle': 'handle_parser_input'},
    ]}
    store = Interrupted()
    runtime = config(store)
    from magic_agents.execution.storage import canonical_bytes
    import hashlib
    runtime.execution_snapshot = runtime.execution_snapshot.model_copy(update={
        'graph_definition': raw, 'graph_digest': hashlib.sha256(canonical_bytes(raw)).hexdigest()})
    calls = []
    def graph():
        value = build(raw, message='unused')
        for key in ('items', 'transform', 'tail', 'final'):
            node = value.nodes[key]
            original = node.process
            async def observed(chat_log, *, original=original, node=node, key=key):
                calls.append((key, copy.deepcopy(node.inputs)))
                async for item in original(chat_log):
                    yield item
            node.process = observed
        return value
    with pytest.raises(asyncio.CancelledError):
        async for _ in execute_graph_reactive(graph(), runtime_config=runtime):
            pass
    saved = store.record
    inspection = inspect_recovery(saved)
    assert inspection.recoverable, inspection
    original_loop = next(iter(saved.state.checkpoint['graphs'].values()))['loop']
    runtime.execution_restore = saved
    restored_graph = graph()
    async for _ in execute_graph_reactive(restored_graph, runtime_config=runtime):
        pass
    loop = next(iter(store.record.state.checkpoint['graphs'].values()))['loop']
    assert loop['next_index'] == 3 and loop['lifecycle'] == 'ended'
    assert (loop['start_time'], loop['started_at'], loop['execution_id']) == (
        original_loop['start_time'], original_loop['started_at'], original_loop['execution_id'])
    assert restored_graph.nodes['final'].outputs['handle_parser_output']['content'] == 'a!?,b!?,c!?'
    assert [key for key, _ in calls].count('items') == 1
    assert [key for key, _ in calls].count('transform') == 3
    assert [key for key, _ in calls].count('tail') == 3
    assert [key for key, _ in calls].count('final') == 1
    assert store.created == 1 and store.record.state.status == 'completed'


def test_loop_unfinished_lifecycle_does_not_reexecute_unknown_hook():
    from magic_agents.execution.storage import ExecutionRecord
    store = Store()
    state = ExecutionState(status='blocked', checkpoint={'schema_version': 1, 'graphs': {
        'scope': {'node_path': [], 'status': 'blocked', 'nodes': {},
                  'loop': {'schema_version': 1, 'lifecycle_pending': 'end'}}}})
    record = ExecutionRecord(snapshot=config(store).execution_snapshot, state=state, version=3, fence=1)
    assert inspect_recovery(record).reason == 'loop_lifecycle_unresolved'


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['order', 'value', 'handle', 'list_order', 'new_phase'])
async def test_interrupted_parser_restores_exact_input_order_and_client_reference(change):
    from magic_agents.models.factory.Nodes import ClientNodeModel, ParserNodeModel
    from magic_agents.models.model_agent_run_log import ModelAgentRunLog
    from magic_agents.node_system.NodeClientLLM import NodeClientLLM
    from magic_agents.node_system.NodeParser import NodeParser
    from magic_agents.execution.storage import ExecutionStorageError, canonical_bytes

    class ExactSpanStore(Store):
        async def commit(self, lease, transition):
            for event in transition.events:
                if event.kind == 'node_start' and event.data.get('resumed'):
                    previous = self.record.state.checkpoint['graphs']['scope']['nodes']['parser']['node_start']
                    assert event.data['execution_id'] == previous['execution_id']
                    assert canonical_bytes(event.data['inputs']) == canonical_bytes(previous['inputs'])
            return await super().commit(lease, transition)

    store = ExactSpanStore()
    snapshot = config(store).execution_snapshot
    def nodes():
        client = NodeClientLLM(ClientNodeModel(engine='openai', model='gpt-4o',
            api_info={'api_key': 'test-only'}), node_id='client')
        assert client.init_error is None
        parser = NodeParser(ParserNodeModel(text='{{ system.label }}:{{ user }}'),
            node_id='parser', node_type='parser')
        return client, parser

    client, parser = nodes()
    recorder = CoreRecorder(store, snapshot)
    await recorder.start()
    scope = CoreScope(recorder, SimpleNamespace(nodes={'client': client, 'parser': parser}), (),
        snapshot.identity.run_id, snapshot.identity.root_execution_id, 'scope', True)
    try:
        await scope.record('graph_start', {'status': 'running'}, status='running')
        recorder.encode({'content': client.client}, resource={'node_path': ['client'], 'handle': client.OUTPUT_HANDLE})
        original = {'client': client.client, 'system': {'label': 'original', 'order': [1, 2]}, 'user': 'question'}
        await scope.record('node_start', {'inputs': recorder.encode(original), 'node_type': 'NodeParser',
            'execution_id': 'original-parser-span', 'parent_execution_id': scope.execution_id,
            'activation_id': None, 'execution_phase': None, 'resumed': False}, node_id='parser')
    finally:
        await recorder.close()
    retained = store.record

    rebuilt_client, rebuilt_parser = nodes()
    rebuilt_parser.inputs = {'user': 'question', 'client': rebuilt_client.client,
                            'system': {'order': [1, 2], 'label': 'original'}}
    if change in {'value', 'new_phase'}:
        rebuilt_parser.inputs['user'] = 'changed'
    elif change == 'handle':
        rebuilt_parser.inputs['different'] = rebuilt_parser.inputs.pop('user')
    elif change == 'list_order':
        rebuilt_parser.inputs['system']['order'] = [2, 1]
    recorder = CoreRecorder(store, snapshot, restore=retained)
    await recorder.start()
    scope = CoreScope(recorder, SimpleNamespace(nodes={'client': rebuilt_client, 'parser': rebuilt_parser}), (),
        snapshot.identity.run_id, snapshot.identity.root_execution_id, 'scope', True,
        execution_phase='loop:iteration:1' if change == 'new_phase' else None)
    recorder.encode({'content': rebuilt_client.client}, resource={'node_path': ['client'], 'handle': rebuilt_client.OUTPUT_HANDLE})
    token = _scope.set(scope)
    try:
        async def consume():
            return [item async for item in rebuilt_parser(ModelAgentRunLog(run_id=scope.run_id))]
        if change in {'value', 'handle', 'list_order'}:
            with pytest.raises(ExecutionStorageError) as failure:
                await consume()
            assert failure.value.code == 'execution_identity_conflict'
            assert not any(event.kind == 'node_output' for event in store.events)
        else:
            items = await consume()
            expected = 'original:changed' if change == 'new_phase' else 'original:question'
            assert any(item.get('type') == rebuilt_parser.OUTPUT_HANDLE and
                       item['content']['content'] == expected for item in items)
            start = next(event for event in reversed(store.events) if event.kind == 'node_start')
            assert start.data['resumed'] is (change == 'order')
            if change == 'order':
                assert list(rebuilt_parser.inputs) == ['client', 'system', 'user']
                assert list(rebuilt_parser.inputs['system']) == ['label', 'order']
                assert rebuilt_parser.inputs['client'] is rebuilt_client.client
                assert rebuilt_client.client is not client.client
            else:
                assert start.data['execution_id'] != 'original-parser-span'
                assert list(rebuilt_parser.inputs) == ['user', 'client', 'system']
    finally:
        _scope.reset(token)
        await recorder.close()


def test_child_effect_does_not_reenter_an_unfinished_python_hook():
    import hashlib
    from magic_agents.execution.storage import ExecutionRecord, ExecutionCause, EffectRecord, canonical_bytes
    snapshot = config(Store()).execution_snapshot
    value = {'answer': 'confirmed'}
    effect = EffectRecord(effect_id='child-effect', attempt_id='child-effect', kind='hook_child',
        status='succeeded', request_digest='a' * 64,
        result=value, result_digest=hashlib.sha256(canonical_bytes(value)).hexdigest(),
        cause=ExecutionCause(kind='normal', run_id=snapshot.identity.run_id,
            execution_id='owner-span', node_path=('child',), actor_id='actor', activation_id='activation'))
    node = {'node_start': {'node_type': 'NodeLLM', 'execution_id': 'owner-span'},
            'llm_checkpoint': {'boundary': 'candidate'}}
    graph = {'node_path': [], 'run_id': snapshot.identity.run_id, 'status': 'blocked', 'nodes': {'owner': node}}
    state = ExecutionState(status='blocked', effects=(effect,), checkpoint={'schema_version': 1, 'graphs': {'scope': graph}})
    record = ExecutionRecord(snapshot=snapshot, state=state, version=3, fence=1)
    assert inspect_recovery(record).reason == 'hook_lifecycle_unresolved'
    for status in ('prepared', 'dispatched', 'unknown'):
        unresolved = effect.model_copy(update={'status': status, 'result': None, 'result_digest': None})
        candidate = record.model_copy(update={'state': state.model_copy(update={'effects': (unresolved,)})})
        assert inspect_recovery(candidate).reason == 'hook_child_operation_unresolved'
    node['coordination_extensions'] = {'coordination_lifecycle': {
        'invocation': {'outcome': {'status': 'success'}}, 'childOperations': [{
            'state': 'completed', 'node_id': 'child', 'ownerActorId': 'actor', 'ownerActivationId': 'activation',
            'resultRef': {'version': 1, 'effect_id': effect.effect_id, 'attempt_id': effect.attempt_id,
                          'request_digest': effect.request_digest, 'result_digest': effect.result_digest}}]}}
    # Models detach JSON, so bind the updated checkpoint explicitly.
    state = state.model_copy(update={'checkpoint': {'schema_version': 1, 'graphs': {'scope': graph}}})
    assert inspect_recovery(record.model_copy(update={'state': state})).recoverable
    node['coordination_extensions']['coordination_lifecycle']['childOperations'][0]['resultRef']['request_digest'] = 'b' * 64
    assert inspect_recovery(record.model_copy(update={'state': state})).reason == 'hook_lifecycle_unresolved'
