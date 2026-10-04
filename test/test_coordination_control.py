import asyncio
import copy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.types import AgentBudget
from magic_llm.engine.attempt_control import ProviderAttemptControlError
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.model import ModelChat, ModelChatResponse
from magic_llm.model.ModelChatStream import UsageModel

from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.control import ActorLoopControl, BudgetAttemptControl
from magic_agents.coordination.service import CoordinationError, CoordinationService
from magic_agents.models.coordination import CoordinationLimits, CoordinationPolicy, MessagingConfig


def service():
    limits = CoordinationLimits(maxModelTurns=20, maxInputTokens=10000, maxOutputTokens=10000,
        maxToolCalls=100, maxImageJobs=10, maxCost={"amount": "10", "currency": "USD"},
        maxWakeupsPerActor=2, maxWakeupsPerWorkgroup=3)
    policy = CoordinationPolicy(enabled=True, allowedParticipants=["research", "images"], limits=limits)
    members = {role: ((role,), MessagingConfig(enabled=True, role=role, peers=[peer], canWakePeers=[peer], wakeOnMessage=True))
               for role, peer in (("research", "images"), ("images", "research"))}
    return CoordinationService(policy, members, server_limits=limits, authorize=lambda: None)


async def actors(s):
    return await s.activate("research"), await s.activate("images")


class Provider:
    engine, model = "openai", "fake"

    def __init__(self, responses, retries=1):
        self.responses = responses
        self.calls = []
        self.kwargs, self.fallback, self.retry_config = {}, None, RetryConfig(retries, 0)

    async def _execute_callback(self, *args): pass

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat, **kwargs):
        self.calls.append(copy.deepcopy(chat.messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception): raise response
        return ModelChatResponse(id="fake", model="fake", object="chat.completion", created=0,
            choices=[{"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": "stop"}],
            usage=UsageModel(prompt_tokens=3, completion_tokens=2, total_tokens=5))


def admission(s, *, unknown=False, bound=100):
    return BudgetAttemptControl(s.budget, estimate=lambda a: UsageBound(input_tokens=bound, output_tokens=10, cost=Decimal("0.1")),
        usage=lambda a, o: None if unknown else UsageBound(input_tokens=o.usage.prompt_tokens, output_tokens=o.usage.completion_tokens, cost=Decimal("0.01")),
        authorize=lambda a: s.authorize())


@pytest.mark.asyncio
async def test_real_loop_consumes_mail_and_resumes_retained_canonical_state_on_wake():
    s = service(); a, b = await actors(s)
    sent = await s.send(a, "images", "targeted slide 4", key="image", expect_reply=False)
    provider = Provider(["first asset manifest", "updated asset manifest"])
    control = ActorLoopControl(b)
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), control=control, provider_attempt_control=admission(s),
                         builtin_todo_tools=False, budget=AgentBudget(max_iterations=3))
    result = await loop.run("Create visuals")
    assert result.content == "first asset manifest"
    assert control.retained.consumed_message_ids == [sent["receipt"]["messageId"]]
    assert "targeted slide 4" in provider.calls[0][-1]["content"]
    assert await s.finish(b, control.retained.output_candidate) == "quiescent"
    await s.send(a, "images", "revise slide 4", key="wake", wake=True, expect_reply=False)
    fresh = await s.activate("images")
    resumed_control = ActorLoopControl(fresh)
    resumed = AsyncAgentLoop(SimpleNamespace(llm=provider), control=resumed_control, provider_attempt_control=admission(s),
                             builtin_todo_tools=False, budget=AgentBudget(max_iterations=3))
    result = await resumed.run(continuation=control.retained)
    assert result.content == "updated asset manifest"
    assert resumed_control.retained.step == 2
    assert any(m.get("content") == "first asset manifest" for m in provider.calls[1])
    assert (await s.budget.snapshot())["modelTurns"] == 2
    assert (await s.budget.snapshot())["spent"]["input_tokens"] == 6


@pytest.mark.asyncio
async def test_failed_input_checkpoint_stops_before_provider_dispatch(monkeypatch):
    s = service(); a, b = await actors(s)
    await s.send(a, "images", "must be acknowledged", key="1", expect_reply=False)
    async def fail(*args): raise CoordinationError("checkpoint_unavailable", "Mandatory state store failed")
    monkeypatch.setattr(s, "checkpoint", fail)
    provider = Provider(["never"])
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), control=ActorLoopControl(b), provider_attempt_control=admission(s), builtin_todo_tools=False)
    with pytest.raises(Exception, match="Mandatory state store failed"):
        await loop.run("task")
    assert provider.calls == []
    assert (await s.inspect(a, "images"))["pendingMessageCount"] == 1


@pytest.mark.asyncio
async def test_unknown_failed_attempt_blocks_retry_before_second_paid_dispatch():
    s = service(); provider = Provider([ConnectionError("lost"), "would retry"], retries=2)
    with pytest.raises(ProviderAttemptControlError, match="input_tokens"):
        await provider.async_generate(ModelChat(), provider_attempt_control=admission(s, unknown=True, bound=6000))
    assert len(provider.calls) == 1
    snap = await s.budget.snapshot()
    assert snap["uncertain"]["input_tokens"] == 6000 and snap["activeModels"] == 0


@pytest.mark.asyncio
async def test_schema_only_author_spends_same_lineage_as_native_peer():
    s = service(); _, b = await actors(s)
    provider = Provider(["peer", "author"])
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), control=ActorLoopControl(b), provider_attempt_control=admission(s), builtin_todo_tools=False)
    await loop.run("peer task")
    await provider.async_generate(ModelChat(), provider_attempt_control=admission(s))
    snap = await s.budget.snapshot()
    assert snap["modelTurns"] == 2 and snap["spent"]["input_tokens"] == 6


@pytest.mark.asyncio
async def test_growth_exhaustion_before_input_commit_settles_message_and_never_dispatches_provider():
    s = service(); s._checkpoint_bytes = 2500; a, b = await actors(s)
    await s.checkpoint(b, {'retained': 'last valid'}, [])
    receipt = await s.send(a, 'images', 'accepted before local history growth', key='capacity')
    provider = Provider(['must not dispatch'])
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), control=ActorLoopControl(b),
        provider_attempt_control=admission(s), builtin_todo_tools=False)
    with pytest.raises(Exception, match='can no longer fit'):
        await loop.run('x' * 2000)
    assert provider.calls == []
    actor = s._actors[b.actor_id]
    assert actor.checkpoint == {'retained': 'last valid'} and not actor.consumed
    assert s._messages[receipt['receipt']['messageId']].state == 'failed'
    budget = await s.budget.snapshot()
    assert budget['modelTurns'] == 0 and budget['reserved']['messages'] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
async def test_cancelling_real_provider_attempt_settles_uncertain_usage_and_releases_permit(stream):
    class BlockingProvider(Provider):
        def __init__(self):
            super().__init__([])
            self.entered = asyncio.Event()

        @BaseChat.async_intercept_generate
        async def async_generate(self, chat, **kwargs):
            self.entered.set()
            await asyncio.Future()

        @BaseChat.async_intercept_stream_generate
        async def async_stream_generate(self, chat, **kwargs):
            self.entered.set()
            await asyncio.Future()
            yield  # keep this a native asynchronous generator

    s = service(); provider = BlockingProvider()
    control = admission(s, unknown=True)
    async def run():
        if stream:
            async for item in provider.async_stream_generate(ModelChat(), provider_attempt_control=control):
                pass
        else:
            await provider.async_generate(ModelChat(), provider_attempt_control=control)
    task = asyncio.create_task(run())
    await provider.entered.wait()
    assert (await s.budget.snapshot())['activeModels'] == 1
    await s.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    snapshot = await s.budget.snapshot()
    assert snapshot['activeModels'] == 0 and snapshot['reserved']['input_tokens'] == 0
    assert snapshot['uncertain']['input_tokens'] == 100
    assert snapshot['modelTurns'] == 1
