import asyncio
from decimal import Decimal

import pytest

from magic_agents.coordination.budget import BudgetError, UsageBound, WorkgroupBudget
from magic_agents.models.coordination import CoordinationLimits


def limits(**changes):
    return CoordinationLimits(**dict(maxModelTurns=4, maxInputTokens=100, maxOutputTokens=100,
        maxToolCalls=10, maxImageJobs=2, maxCost={"amount": "1.00", "currency": "USD"}, **changes))


@pytest.mark.asyncio
async def test_parallel_siblings_share_one_atomic_spending_ceiling():
    root = WorkgroupBudget(limits())
    first, second = root.child(CoordinationLimits()), root.child(CoordinationLimits())
    bound = UsageBound(input_tokens=60, output_tokens=10, cost=Decimal("0.60"))
    results = await asyncio.gather(first.reserve("1", bound), second.reserve("2", bound), return_exceptions=True)
    assert sum(isinstance(r, BudgetError) for r in results) == 1
    snapshot = await root.snapshot()
    assert snapshot["modelTurns"] == 1
    assert snapshot["reserved"]["input_tokens"] == 60
    assert snapshot["reserved"]["cost"] == "0.60"


@pytest.mark.asyncio
async def test_child_cannot_reset_root_model_attempts_or_deadline():
    now = [10.0]
    root = WorkgroupBudget(limits(), absolute_deadline=20.0, clock=lambda: now[0])
    for index in range(4):
        child = root.child(CoordinationLimits(maxModelTurns=999))
        await child.reserve(str(index), UsageBound())
        await child.settle(str(index), UsageBound())
        assert child.deadline == root.deadline
    with pytest.raises(BudgetError, match="attempt allowance"):
        await root.child(CoordinationLimits()).reserve("new-identity", UsageBound())
    assert (await root.snapshot())["modelTurns"] == 4


@pytest.mark.asyncio
async def test_unknown_failure_releases_compute_without_refunding_exposure():
    root = WorkgroupBudget(limits())
    upper = UsageBound(input_tokens=80, output_tokens=30, cost=Decimal("0.8"))
    await root.reserve("failed", upper)
    await root.settle("failed", None, uncertain=True)
    snap = await root.snapshot()
    assert snap["activeModels"] == 0 and snap["uncertain"]["input_tokens"] == 80
    with pytest.raises(BudgetError, match="input_tokens"):
        await root.reserve("retry", UsageBound(input_tokens=21))
    await root.reconcile("failed", UsageBound(input_tokens=50, output_tokens=10, cost=Decimal("0.5")))
    await root.reserve("admitted-retry", UsageBound(input_tokens=50, output_tokens=10, cost=Decimal("0.5")))
    assert (await root.snapshot())["spent"]["input_tokens"] == 50


@pytest.mark.asyncio
async def test_duplicate_settlement_is_idempotent_and_conflicting_charge_rejected():
    root = WorkgroupBudget(limits()); estimate = UsageBound(input_tokens=20)
    await root.reserve("op", estimate)
    await root.settle("op", UsageBound(input_tokens=10))
    await root.settle("op", UsageBound(input_tokens=10))
    assert (await root.snapshot())["spent"]["input_tokens"] == 10
    with pytest.raises(BudgetError, match="different recorded"):
        await root.settle("op", UsageBound(input_tokens=5))
    with pytest.raises(BudgetError, match="cannot dispatch again"):
        await root.reserve("op", estimate)


@pytest.mark.asyncio
async def test_waiting_compute_rechecks_guard_and_cancellation_before_dispatch():
    root = WorkgroupBudget(limits(maxConcurrentModelTurns=1))
    await root.reserve("held", UsageBound())
    allowed = True
    def guard():
        if not allowed: raise PermissionError("ownership lost")
    blocked = asyncio.create_task(root.reserve("blocked", UsageBound(), guard=guard))
    await asyncio.sleep(0)
    assert not blocked.done()
    allowed = False
    await root.settle("held", UsageBound())
    with pytest.raises(PermissionError, match="ownership lost"): await blocked
    await root.reserve("held-again", UsageBound())
    blocked = asyncio.create_task(root.reserve("cancelled", UsageBound()))
    await asyncio.sleep(0); await root.cancel()
    with pytest.raises(BudgetError, match="cancelled"): await blocked
    await root.settle("held-again", None, uncertain=True)
    assert (await root.snapshot())["activeModels"] == 0


@pytest.mark.asyncio
async def test_observed_overage_is_retained_and_stops_further_admission():
    root = WorkgroupBudget(limits())
    await root.reserve("op", UsageBound(input_tokens=10))
    with pytest.raises(BudgetError, match="exceeded"):
        await root.settle("op", UsageBound(input_tokens=15))
    assert (await root.snapshot())["spent"]["input_tokens"] == 15
    with pytest.raises(BudgetError): await root.reserve("next", UsageBound())


@pytest.mark.asyncio
async def test_unknown_with_partial_usage_does_not_double_count_reconciliation():
    root = WorkgroupBudget(limits())
    await root.reserve("op", UsageBound(input_tokens=20, cost=Decimal("0.5")))
    await root.settle("op", UsageBound(input_tokens=5, cost=Decimal("0.1")), uncertain=True)
    snapshot = await root.snapshot()
    assert snapshot["spent"]["input_tokens"] == 5 and snapshot["uncertain"]["input_tokens"] == 15
    await root.reconcile("op", UsageBound(input_tokens=12, cost=Decimal("0.3")))
    await root.reconcile("op", UsageBound(input_tokens=12, cost=Decimal("0.3")))
    snapshot = await root.snapshot()
    assert snapshot["spent"]["input_tokens"] == 12 and snapshot["uncertain"]["input_tokens"] == 0
    assert Decimal(snapshot["spent"]["cost"]) == Decimal("0.3")


@pytest.mark.parametrize("kwargs", [{"input_tokens": True}, {"output_tokens": -1}, {"cost": 1.0}, {"cost": Decimal("NaN")}])
def test_usage_bound_is_strict_and_finite(kwargs):
    with pytest.raises(ValueError): UsageBound(**kwargs)


@pytest.mark.asyncio
async def test_tool_retries_keep_identity_and_siblings_share_tool_allowance():
    root = WorkgroupBudget(limits())
    first, second = root.child(CoordinationLimits()), root.child(CoordinationLimits())
    for index in range(10):
        scope = first if index % 2 else second
        await scope.charge_tool_call(f"tool-{index}")
        await scope.charge_tool_call(f"tool-{index}")
    assert (await root.snapshot())["spent"]["tool_calls"] == 10
    with pytest.raises(BudgetError, match="tool_calls"):
        await root.child(CoordinationLimits()).charge_tool_call("new-scope-new-id")


@pytest.mark.asyncio
async def test_tool_charge_replay_checks_access_and_new_call_checks_owner_before_debit():
    root = WorkgroupBudget(limits())
    allowed = True
    owner = True
    def access():
        if not allowed: raise PermissionError('read denied')
    def new_guard():
        if not owner: raise PermissionError('stale owner')
    await root.charge_tool_call('original', guard=access, new_guard=new_guard)
    owner = False
    await root.charge_tool_call('original', guard=access, new_guard=new_guard)
    with pytest.raises(PermissionError, match='stale owner'):
        await root.charge_tool_call('fresh', guard=access, new_guard=new_guard)
    assert (await root.snapshot())['spent']['tool_calls'] == 1
    allowed = False
    with pytest.raises(PermissionError, match='read denied'):
        await root.charge_tool_call('original', guard=access, new_guard=new_guard)


@pytest.mark.asyncio
async def test_child_defaults_never_reset_active_parent_wake_or_other_limits():
    root = WorkgroupBudget(limits(maxWakeupsPerActor=3, maxWakeupsPerWorkgroup=7,
        maxAcceptedMessages=100, maxSingleWaitSeconds=50))
    child = root.child(CoordinationLimits(maxInputTokens=50))
    assert child._limits.max_wakeups_per_actor == 3
    assert child._limits.max_wakeups_per_workgroup == 7
    assert child._limits.max_accepted_messages == 100
    assert child._limits.max_single_wait_seconds == 50
    assert child._limits.max_input_tokens == 50
    assert child.child(CoordinationLimits(maxWakeupsPerWorkgroup=0))._limits.max_wakeups_per_workgroup == 0
