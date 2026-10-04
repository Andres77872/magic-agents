"""Real actor wakes retain private Skills instructions and reconstructed guards."""
import copy
import json
from types import SimpleNamespace

import pytest

from magic_agents.coordination.service import CoordinationError
from magic_agents.coordination.skills import CHECKPOINT_KEY, PinnedSkills, split_checkpoint
from magic_agents.models.factory.Nodes.SkillsNodeModel import SkillsNodeModel
from magic_agents.skills import SkillPromptBundle, create_request_guard
from magic_llm.agent.request import AgentRequestContext
from magic_llm.model import ModelChat
from test.test_coordination_graph import collect, graph, node, runtime, send_call

PRIVATE = 'PINNED_PRIVATE_SKILL_INSTRUCTIONS'


def skills(prompt=PRIVATE):
    return SkillPromptBundle.from_model('skills', SkillsNodeModel(schema_version=1, skills=[{
        'id': 'illustrate', 'name': 'Illustrate', 'description': 'Draw a matching image',
        'prompt': prompt, 'props': {'style': 'private-pinned-style'}}]))


def skill_call():
    return {'content': None, 'tool_calls': [{'id': 'load-pinned-skill', 'type': 'function',
        'function': {'name': 'skills_load', 'arguments': '{"skill_ids":["illustrate"]}'}}]}


async def scenario(stream=False, change=None):
    authorities = []
    def authorize(path, manifest):
        assert path == ('images',)
        assert PRIVATE not in repr(manifest)
        authorities.append(copy.deepcopy(manifest))
        if change == 'revoke' and len(pb.calls) >= 2:
            raise CoordinationError('unauthorized', 'Skills resource access revoked')
    rt = runtime(authorize_skills=authorize)
    async def research(provider, chat):
        if len(provider.calls) == 1:
            scope = rt.scopes[0]
            async with scope.service._condition:
                while scope.service._actors[scope.service._roles['images']].state != 'quiescent':
                    await scope.service._condition.wait()
            if change == 'source':
                b.inputs[b.INPUT_HANDLER_SKILLS] = skills('CHANGED_UNADMITTED_INSTRUCTIONS')
            if change in ('missing', 'version', 'guard', 'manifest'):
                saved = scope.service._actors[scope.service._roles['images']].checkpoint
                if change == 'missing': saved.pop(CHECKPOINT_KEY)
                elif change == 'version': saved[CHECKPOINT_KEY]['schemaVersion'] = 100
                elif change == 'guard': saved[CHECKPOINT_KEY]['guard']['loaderCallIds'] = []
                else: saved[CHECKPOINT_KEY]['manifest']['skills'][0]['definition']['prompt'] = 'tampered'
            return send_call(wake=True)
        return {'content': 'research complete'}
    async def images(provider, chat):
        if len(provider.calls) == 1: return skill_call()
        assert PRIVATE in repr(chat.messages)
        assert 'CHANGED_UNADMITTED_INSTRUCTIONS' not in repr(chat.messages)
        assert len([m for m in chat.messages if m.get('role') == 'tool' and m.get('tool_call_id') == 'load-pinned-skill']) == 1
        assert PRIVATE not in repr(chat.observer_projection().messages)
        return {'content': 'generic asset' if len(provider.calls) == 2 else 'generic asset + targeted asset'}
    async def author(provider, chat):
        assert PRIVATE not in repr(chat.messages)
        return {'content': 'public deck'}
    a, pa = node('research', research, peer='images', stream=stream)
    b, pb = node('images', images, peer='research', stream=stream)
    b.inputs[b.INPUT_HANDLER_SKILLS] = skills()
    b._skills_source_node_ids = ('skills',)
    c, pc = node('author', author, stream=stream)
    outcome, events = await collect(graph({'research': a, 'images': b, 'author': c}), rt)
    return rt, (pa, pb, pc), outcome, events, authorities


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('change', [None, 'source'])
async def test_actual_actor_wake_retains_pinned_skills_and_private_projection(stream, change):
    rt, (pa, pb, pc), result, events, authorities = await scenario(stream, change)
    assert not result['has_errors'], result
    assert (len(pa.calls), len(pb.calls), len(pc.calls)) == (2, 3, 1)
    assert PRIVATE not in repr(events)
    assert len({item['manifestDigest'] for item in authorities}) == 1
    saved = rt.scopes[0].service._actors[rt.scopes[0].service._roles['images']].checkpoint
    loop, data = split_checkpoint(saved)
    assert PRIVATE in repr(loop.messages) and loop.step == 3
    assert data['guard']['loaderCallIds'] == ['load-pinned-skill']
    restored = PinnedSkills.restore(json.loads(json.dumps(data)), loop)
    assert restored.authorization_manifest == authorities[0]
    budget = await rt.budget.snapshot()
    assert budget['modelTurns'] == 6 and budget['spent']['tool_calls'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['missing', 'version', 'guard', 'manifest'])
async def test_invalid_private_skill_checkpoint_refuses_before_resumed_dispatch(change):
    _, (_, pb, pc), result, _, _ = await scenario(change=change)
    assert result['has_errors']
    assert len(pb.calls) == 2 and pc.calls == []


@pytest.mark.asyncio
async def test_revoked_skill_authority_stops_continuation_and_author():
    _, (_, pb, pc), result, _, _ = await scenario(change='revoke')
    assert result['has_errors']
    assert len(pb.calls) == 2 and pc.calls == []


@pytest.mark.asyncio
async def test_missing_host_skill_authorizer_rejects_before_any_graph_provider_work():
    async def never(*args): raise AssertionError('No provider dispatch expected')
    a, pa = node('research', never, peer='images')
    b, pb = node('images', never, peer='research')
    c, pc = node('author', never)
    b.inputs[b.INPUT_HANDLER_SKILLS] = skills()
    with pytest.raises(CoordinationError, match='pinned-resource authorizer'):
        await collect(graph({'research': a, 'images': b, 'author': c}), runtime())
    assert not pa.calls and not pb.calls and not pc.calls


def test_guard_resume_preserves_projection_and_finite_token_limit():
    chat = ModelChat()
    chat.messages = [{'role': 'user', 'content': 'initial'}]
    guard = create_request_guard(5000, source_node_id='skills')
    ctx = AgentRequestContext(chat=chat, tools=[], tool_choice=None, generation_options={}, provider='openai', model='fake')
    guard(ctx)
    chat.messages.extend([{'role': 'assistant', **skill_call()}, {'role': 'tool',
        'tool_call_id': 'load-pinned-skill', 'content': PRIVATE}])
    saved = guard.checkpoint_state(chat.messages)
    restored = create_request_guard(5000, source_node_id='skills', resume_state=json.loads(json.dumps(saved)))
    restored(ctx)
    assert PRIVATE in repr(chat.messages)
    assert PRIVATE not in repr(chat.observer_projection().messages)
    chat.messages.append({'role': 'user', 'content': 'x' * 5000})
    from magic_agents.hooks.invocation_control import OperationFailure, error_code
    with pytest.raises(OperationFailure) as denied:
        restored(ctx)
    assert error_code(denied.value) == 'SKILLS_CONTEXT_LIMIT'


def test_pinned_skill_guard_cap_can_narrow_but_not_expand():
    state = PinnedSkills(skills(), 5000)
    target = SimpleNamespace(_max_input_tokens=4000)
    state.attach_guard(target)
    target._max_input_tokens = 9000
    state.attach_guard(target)
    assert state.cap == 4000
