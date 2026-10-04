import asyncio
import json

import pytest

from magic_agents.coordination.service import CoordinationError, CoordinationService, native_inbox_message
from magic_agents.models.coordination import CoordinationLimits, CoordinationPolicy, MessagingConfig
from magic_agents.coordination.budget import WorkgroupBudget


def service(*, max_pending=8, max_messages=32, depth=8, clock=None, authorize=lambda: None):
    limits = dict(maxPendingMessagesPerActor=max_pending, maxAcceptedMessages=max_messages,
                  maxWakeupsPerActor=2, maxWakeupsPerWorkgroup=3, maxRequestDepth=depth,
                  maxModelTurns=20, maxInputTokens=10000, maxOutputTokens=10000,
                  maxToolCalls=100, maxImageJobs=10, maxCost={"amount": "10", "currency": "USD"})
    policy = CoordinationPolicy(enabled=True, allowedParticipants=["research", "images"], limits=limits)
    members = {role: (("inner", role), MessagingConfig(enabled=True, role=role, peers=[peer],
               canWakePeers=[peer], wakeOnMessage=True)) for role, peer in (("research", "images"), ("images", "research"))}
    return CoordinationService(policy, members, server_limits=CoordinationLimits(**limits),
                               authorize=authorize, **({"clock": clock} if clock else {}))


async def actors(s):
    return await s.activate("research"), await s.activate("images")


async def consume(s, caller):
    messages = await s.inbox(caller)
    await s.checkpoint(caller, {"messages": messages}, [m["messageId"] for m in messages])
    return messages


@pytest.mark.asyncio
async def test_request_delivery_checkpoint_and_correlated_reply():
    s = service(); a, b = await actors(s)
    receipt = await s.send(a, "images", "targeted", {"slide": 4}, key="call-1")
    request_id = receipt["receipt"]["requestId"]
    assert receipt["receipt"]["stage"] == "accepted"
    offered = await s.inbox(b)
    assert offered == await s.inbox(b)  # delivery is not acknowledgement
    assert (await s.inspect(a, "images"))["pendingMessageCount"] == 1
    await s.checkpoint(b, {"canonical": offered}, [offered[0]["messageId"]])
    assert await s.inbox(b) == []
    assert await s.finish(a, "candidate") == "awaiting_reply"
    reply = await s.reply(b, request_id, "asset ready", {"assetId": "owned-asset"}, key="reply-1")
    result = await s.wait_message(a, request_id=request_id, timeout=1)
    assert result["outcome"] == "reply"
    assert result["message"]["messageId"] == reply["receipt"]["messageId"]
    assert result["message"]["payload"] == {"assetId": "owned-asset"}


@pytest.mark.asyncio
async def test_unknown_scope_refs_and_other_actors_requests_are_private():
    s = service(); a, b = await actors(s)
    foreign = service(); other_a, other_b = await actors(foreign)
    with pytest.raises(CoordinationError, match="not visible"):
        await s.inspect(a, other_b.actor_id)
    with pytest.raises(CoordinationError, match="Unknown actor"):
        await s.inbox(other_a)
    receipt = await s.send(a, "images", "request", key="1")
    with pytest.raises(CoordinationError, match="not addressed"):
        await s.reply(a, receipt["receipt"]["requestId"], "spoof", key="r")
    with pytest.raises(CoordinationError, match="does not own"):
        await s.wait_message(b, request_id=receipt["receipt"]["requestId"], timeout=1)


@pytest.mark.asyncio
async def test_arrival_wins_finalization_without_losing_accepted_message():
    s = service(); a, b = await actors(s)
    await s.checkpoint(b, {"candidate": "generic"}, [])
    sent = await s.send(a, "images", "arrived before finish", key="arrival", expect_reply=False)
    assert await s.finish(b, "generic") == "continue"
    assert (await s.inbox(b))[0]["messageId"] == sent["receipt"]["messageId"]
    await consume(s, b)
    assert await s.finish(b, "generic plus requested") == "quiescent"


@pytest.mark.asyncio
async def test_finalization_wins_then_wake_is_coalesced_and_checkpoint_retained():
    s = service(); a, b = await actors(s)
    await s.checkpoint(b, {"paid_assets": ["existing"]}, [])
    assert await s.finish(b, "generic") == "quiescent"
    with pytest.raises(CoordinationError) as exc:
        await s.send(a, "images", "missing wake", key="denied", expect_reply=False)
    assert exc.value.code == "wake_required"
    results = await asyncio.gather(*(s.send(a, "images", f"wake {i}", key=f"wake-{i}", wake=True, expect_reply=False) for i in range(2)))
    assert sum(r["receipt"]["wakeScheduled"] for r in results) == 1
    fresh = await s.activate("images")
    assert fresh.actor_id == b.actor_id and fresh.activation_id != b.activation_id
    assert await s.retained_checkpoint(fresh) == {"paid_assets": ["existing"]}
    with pytest.raises(CoordinationError) as exc:
        await s.inbox(b)
    assert exc.value.code == "stale_activation"


@pytest.mark.asyncio
async def test_seal_requires_all_actors_and_snapshot_is_immutable():
    s = service(); a, b = await actors(s)
    await s.checkpoint(a, {}, []); await s.finish(a, {"research": "ready"})
    assert await s.seal() is None
    await s.checkpoint(b, {}, []); await s.finish(b, {"images": ["one"]})
    sealed = await s.seal()
    sealed["images"]["output"]["images"].append("tampered")
    assert s._actors[b.actor_id].output == {"images": ["one"]}
    with pytest.raises(CoordinationError) as exc:
        await s.send(a, "images", "after seal", key="late", wake=True)
    # New work fails even when the caller owns a historical actor handle.
    assert exc.value.code in ("stale_activation", "epoch_closed")


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancel", "seal"])
async def test_idempotent_retry_precedes_mutable_admission_checks(terminal):
    s = service(); a, b = await actors(s)
    original = await s.send(a, "images", "same", key="stable", expect_reply=False)
    await consume(s, b)
    if terminal == "cancel": await s.cancel()
    else:
        await s.checkpoint(a, {}, [])
        await s.finish(a, "a"); await s.finish(b, "b"); await s.seal()
    retried = await s.send(a, "images", "same", key="stable", expect_reply=False)
    assert retried["receipt"] == original["receipt"]
    assert s._accepted == 1
    with pytest.raises(CoordinationError) as exc:
        await s.send(a, "images", "different", key="stable", expect_reply=False)
    assert exc.value.code == "idempotency_conflict"


@pytest.mark.asyncio
async def test_retry_still_checks_current_authorization():
    revoked = False
    def authorize():
        if revoked: raise CoordinationError("not_authorized", "Access revoked")
    s = service(authorize=authorize); a, b = await actors(s)
    await s.send(a, "images", "message", key="same")
    revoked = True
    with pytest.raises(CoordinationError) as exc:
        await s.send(a, "images", "message", key="same")
    assert exc.value.code == "not_authorized"


@pytest.mark.asyncio
async def test_reply_capacity_survives_general_inbox_and_message_saturation():
    s = service(max_pending=1, max_messages=2); a, b = await actors(s)
    sent = await s.send(a, "images", "request", key="r")
    with pytest.raises(CoordinationError) as exc:
        await s.send(b, "research", "flood", key="flood", kind="event", expect_reply=False)
    assert exc.value.code == "message_budget_exhausted"
    reply = await s.reply(b, sent["receipt"]["requestId"], "reserved outcome", key="reply")
    assert reply["ok"] and s._accepted == 2 and s._reserved_replies == 0
    assert len(await s.inbox(a)) == 1


@pytest.mark.asyncio
async def test_wait_yields_to_own_inbox_instead_of_deadlocking_peer():
    s = service(); a, b = await actors(s)
    request = await s.send(a, "images", "need image", key="a1")
    waiting = asyncio.create_task(s.wait_message(a, request_id=request["receipt"]["requestId"], timeout=1))
    await asyncio.sleep(0)
    inbound = await s.send(b, "research", "need caption", key="b1")
    result = await waiting
    assert result["outcome"] == "inbox_ready"
    assert result["message"]["messageId"] == inbound["receipt"]["messageId"]
    assert (await s.inspect(b, "research"))["pendingMessageCount"] == 1


@pytest.mark.asyncio
async def test_timeout_does_not_cancel_request_and_cancel_releases_wait():
    s = service(); a, b = await actors(s)
    sent = await s.send(a, "images", "request", key="1")
    request_id = sent["receipt"]["requestId"]
    assert (await s.wait_message(a, request_id=request_id, timeout=0.01))["outcome"] == "timeout"
    assert request_id in s._actors[a.actor_id].requests
    waiter = asyncio.create_task(s.wait_message(a, request_id=request_id, timeout=1))
    await asyncio.sleep(0); await s.cancel()
    assert (await waiter)["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_expiry_preserves_receipt_without_renewing_obligation():
    now = [100.0]
    s = service(clock=lambda: now[0]); a, b = await actors(s)
    sent = await s.send(a, "images", "request", key="r", expires_in=5)
    now[0] = 106
    replay = await s.send(a, "images", "request", key="r", expires_in=5)
    assert replay["receipt"] == sent["receipt"]
    assert replay["currentDisposition"]["state"] == "expired"
    assert s._reserved_replies == 0
    assert (await s.wait_message(a, request_id=sent["receipt"]["requestId"], timeout=1))["outcome"] == "request_failed"


@pytest.mark.asyncio
async def test_new_ids_do_not_reset_causal_depth():
    s = service(depth=1); a, b = await actors(s)
    await s.send(a, "images", "depth one", key="1")
    await consume(s, b)
    with pytest.raises(CoordinationError) as exc:
        await s.send(b, "research", "new delegated request", key="brand-new")
    assert exc.value.code == "request_depth_exceeded"


@pytest.mark.asyncio
async def test_checkpoint_requires_offered_ids_and_detaches_mutable_data():
    s = service(); a, b = await actors(s)
    sent = await s.send(a, "images", "request", key="1")
    with pytest.raises(CoordinationError, match="undelivered"):
        await s.checkpoint(b, {}, [sent["receipt"]["messageId"]])
    snapshot = {"history": [{"content": "private"}]}
    await s.checkpoint(b, snapshot, [])
    snapshot["history"][0]["content"] = "changed"
    retained = await s.retained_checkpoint(b)
    assert retained["history"][0]["content"] == "private"
    retained["history"].clear()
    assert await s.retained_checkpoint(b) == {"history": [{"content": "private"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, -1, 0, 31])
async def test_wait_requires_finite_bounded_timeout(value):
    s = service(); a, _ = await actors(s)
    with pytest.raises(CoordinationError, match="finite"):
        await s.wait_message(a, timeout=value)


@pytest.mark.asyncio
async def test_payload_rejects_nonfinite_and_invalid_unicode_before_acceptance():
    s = service(); a, _ = await actors(s)
    for payload in (float("nan"), {"text": "\ud800"}, {1: "non-string-key"}):
        with pytest.raises(CoordinationError):
            await s.send(a, "images", "message", payload, key="invalid")
    assert s._accepted == 0


@pytest.mark.asyncio
async def test_scoped_services_share_lineage_message_allowance_and_cancellation():
    base = service(max_messages=2)
    first = CoordinationService(base.policy, {a.config.role: (a.path, a.config) for a in base._actors.values()},
                                server_limits=base.limits, authorize=lambda: None, budget=base.budget)
    second = CoordinationService(base.policy, {a.config.role: (a.path, a.config) for a in base._actors.values()},
                                 server_limits=base.limits, authorize=lambda: None, budget=base.budget)
    a1, b1 = await actors(first); a2, b2 = await actors(second)
    await first.send(a1, "images", "uses message and outcome slots", key="root-budget")
    with pytest.raises(CoordinationError, match="messages allowance"):
        await second.send(a2, "images", "new scope is not new allowance", key="new-id", expect_reply=False)
    await first.cancel()
    with pytest.raises(CoordinationError) as exc:
        await second.inbox(b2)
    assert exc.value.code == "cancelled"


@pytest.mark.asyncio
async def test_inbox_delivery_is_bounded_by_bytes_as_well_as_message_count():
    s = service(); a, b = await actors(s)
    for index in range(5):
        await s.send(a, "images", "x" * 16000, key=f"large-{index}", kind="event", expect_reply=False)
    first = await s.inbox(b)
    assert len(first) == 4
    assert sum(len(native_inbox_message(m).render().encode('utf-8')) for m in first) <= 65536
    await s.checkpoint(b, {}, [message["messageId"] for message in first])
    assert len(await s.inbox(b)) == 1


@pytest.mark.asyncio
async def test_scheduler_wake_waiter_reuses_actor_then_seals_atomically_and_idempotently():
    s = service(); a, b = await actors(s)
    await s.checkpoint(b, {'retained': 'initial'}, [])
    await s.finish(b, 'old output')
    waiter = asyncio.create_task(s.wait_for_activation('images'))
    await asyncio.sleep(0)
    sent = await s.send(a, 'images', 'wake', key='wake', wake=True, expect_reply=False)
    fresh = await waiter
    assert fresh.actor_id == b.actor_id and fresh.activation_id != b.activation_id
    assert await s.retained_checkpoint(fresh) == {'retained': 'initial'}
    assert (await consume(s, fresh))[0]['messageId'] == sent['receipt']['messageId']
    await s.finish(fresh, 'new output')
    waiter = asyncio.create_task(s.wait_for_activation('images'))
    await asyncio.sleep(0)
    await s.checkpoint(a, {}, []); await s.finish(a, 'research')
    assert await waiter is None
    assert await s.wait_for_activation('research') is None
    published = await s.seal()
    assert published['images'] == {'output': 'new output', 'revision': 2}
    published['images']['output'] = 'tampered'
    assert (await s.seal())['images']['output'] == 'new output'


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['failed', 'bypassed', 'timed_out'])
async def test_failure_before_initial_activation_closes_group_and_request_reservations(state):
    s = service(); a = await s.activate('research')
    sent = await s.send(a, 'images', 'waiting on readiness', key='request')
    waiter = asyncio.create_task(s.candidate_decision(a))
    await asyncio.sleep(0)
    await s.terminate('images', state=state, reason='readiness_failed')
    with pytest.raises(CoordinationError) as error:
        await waiter
    assert error.value.code == 'workgroup_failed'
    assert s._reserved_replies == 0
    assert (await s.budget.snapshot())['reserved']['messages'] == 0
    assert all(not actor.requests for actor in s._actors.values())
    replay = await s.send(a, 'images', 'waiting on readiness', key='request')
    assert replay['receipt'] == sent['receipt'] and replay['currentDisposition']['state'] == 'failed'
    with pytest.raises(CoordinationError, match='no longer admits'):
        await s.wait_for_activation('images')


@pytest.mark.asyncio
async def test_partial_failure_settles_incoming_and_outgoing_once_and_discards_old_output():
    s = service(); s.policy.failure_policy = 'publish_partial'; a, b = await actors(s)
    await s.checkpoint(b, {}, []); await s.finish(b, 'stale successful output')
    request = await s.send(a, 'images', 'revise', key='wake', wake=True)
    b = await s.activate('images')
    await s.send(b, 'research', 'dependent', key='reverse')
    await consume(s, a)
    waiter = asyncio.create_task(s.wait_agent(a, 'images', until='activation_finished',
        activation_id=b.activation_id, timeout=1))
    await asyncio.sleep(0)
    await s.terminate('images', state='failed', reason='asset_failed')
    await s.terminate('images', state='failed', reason='duplicate_failure')
    assert (await waiter)['outcome'] == 'target_failed'
    assert s._reserved_replies == 0
    assert (await s.budget.snapshot())['reserved']['messages'] == 0
    result = await s.wait_message(a, request_id=request['receipt']['requestId'], timeout=1)
    assert result['outcome'] == 'request_failed' and result['reason'] == 'asset_failed'
    notices = await consume(s, a)
    assert len(notices) == 1 and notices[0]['senderRole'] == 'runtime'
    await s.finish(a, 'partial result with warning')
    sealed = await s.seal()
    assert sealed['images']['output'] is None
    assert sealed['images']['failure'] == {'state': 'failed', 'reason': 'asset_failed'}
    assert sealed['research']['output'] == 'partial result with warning'
    await s.terminate('images', state='failed', reason='late_cleanup')
    assert await s.seal() == sealed


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal', ['cancel', 'deadline', 'overage'])
async def test_epoch_terminal_settlement_releases_reserved_outcomes_and_wake_waiters(terminal):
    now = [100.0]; s = service(clock=lambda: now[0]); a, b = await actors(s)
    await s.send(a, 'images', 'pending', key='request')
    if terminal == 'cancel': await s.cancel()
    elif terminal == 'deadline': now[0] = s.deadline
    else: s.budget._shared.exceeded = True
    with pytest.raises(CoordinationError): await s.guard(a)
    assert s._reserved_replies == 0
    assert (await s.budget.snapshot())['reserved']['messages'] == 0
    assert not any(actor.requests for actor in s._actors.values())
    assert not any(message.state == 'accepted' for message in s._messages.values())


@pytest.mark.asyncio
@pytest.mark.parametrize('unit', ['x', '🙂', '\\"\n'])
async def test_admission_and_largest_batch_use_exact_rendered_unicode_and_escaped_bytes(unit):
    s = service(); s.limits.max_inline_message_bytes = 131072
    a, b = await actors(s)
    probe = await s.send(a, 'images', '', key='probe', kind='event', expect_reply=False)
    view = (await s.inbox(b))[0]
    await consume(s, b)
    empty_size = len(native_inbox_message(view).render().encode('utf-8'))
    view['message'] = unit
    increment = len(native_inbox_message(view).render().encode('utf-8')) - empty_size
    count = (65536 - empty_size) // increment
    fits = unit * count
    sent = await s.send(a, 'images', fits, key='fits', kind='event', expect_reply=False)
    before = await s.budget.snapshot()
    with pytest.raises(CoordinationError) as error:
        await s.send(a, 'images', fits + unit, key='too-big', kind='event', expect_reply=False)
    assert error.value.code == 'payload_too_large'
    assert await s.budget.snapshot() == before
    offered = await s.inbox(b)
    assert [m['messageId'] for m in offered] == [sent['receipt']['messageId']]
    assert len(native_inbox_message(offered[0]).render().encode('utf-8')) <= 65536


@pytest.mark.asyncio
async def test_native_count_bound_is_32_with_ordered_unconsumed_remainder():
    s = service(max_pending=40, max_messages=50); a, b = await actors(s)
    receipts = [await s.send(a, 'images', str(i), key=str(i), kind='event', expect_reply=False) for i in range(33)]
    first = await s.inbox(b)
    assert [m['messageId'] for m in first] == [r['receipt']['messageId'] for r in receipts[:32]]
    assert len(s._pending(s._actors[b.actor_id])) == 33
    await s.checkpoint(b, {}, [m['messageId'] for m in first])
    assert (await s.inbox(b))[0]['messageId'] == receipts[-1]['receipt']['messageId']


@pytest.mark.asyncio
async def test_checkpoint_exhaustion_preserves_previous_commit_and_terminally_settles_unacked_input():
    s = service(); s._checkpoint_bytes = 2500; a, b = await actors(s)
    previous = {'messages': [{'role': 'user', 'content': 'initial'}]}
    await s.checkpoint(b, previous, [])
    receipt = await s.send(a, 'images', 'accepted request', key='before-growth')
    offered = await s.inbox(b)
    snapshot = {'messages': [{'role': 'user', 'content': 'x' * 2500}]}
    with pytest.raises(CoordinationError) as error:
        await s.checkpoint(b, snapshot, [offered[0]['messageId']])
    assert error.value.code == 'checkpoint_capacity_exceeded'
    actor = s._actors[b.actor_id]
    assert actor.checkpoint == previous and not actor.consumed
    assert s._messages[receipt['receipt']['messageId']].state == 'failed'
    assert s._reserved_replies == 0


@pytest.mark.asyncio
async def test_pending_capacity_is_reserved_at_acceptance_and_not_silently_overfilled():
    s = service(); s._checkpoint_bytes = 1100; a, b = await actors(s)
    await s.send(a, 'images', 'x' * 400, key='one', kind='event', expect_reply=False)
    before = await s.budget.snapshot()
    with pytest.raises(CoordinationError) as error:
        await s.send(a, 'images', 'x' * 400, key='two', kind='event', expect_reply=False)
    assert error.value.code == 'checkpoint_capacity_exceeded'
    assert await s.budget.snapshot() == before
    assert len(await s.inbox(b)) == 1


@pytest.mark.asyncio
async def test_child_service_inherits_omitted_fields_but_honors_explicit_zero_wake_cap():
    parent = service()
    members = {a.config.role: (a.path, a.config) for a in parent._actors.values()}
    for child_limits, expected in [(None, 3), ({'maxInputTokens': 50}, 3), ({'maxWakeupsPerWorkgroup': 0}, 0)]:
        policy = CoordinationPolicy(enabled=True, allowedParticipants=list(members),
            **({'limits': child_limits} if child_limits is not None else {}))
        child = CoordinationService(policy, members, server_limits=parent.limits,
            authorize=lambda: None, budget=parent.budget)
        assert child.limits.max_wakeups_per_workgroup == expected
        assert child.limits.max_wakeups_per_actor == 2
        assert child.deadline <= parent.deadline


@pytest.mark.asyncio
async def test_first_active_service_initializes_defaults_under_server_budget():
    base = service()
    members = {a.config.role: (a.path, a.config) for a in base._actors.values()}
    first = CoordinationService(CoordinationPolicy(enabled=True, allowedParticipants=list(members)),
        members, server_limits=base.limits, authorize=lambda: None, budget=base.budget, inherit_limits=False)
    assert first.limits.max_wakeups_per_actor == 0 and first.limits.max_wakeups_per_workgroup == 0


@pytest.mark.asyncio
async def test_sync_owner_guard_is_read_only_when_request_has_expired():
    now = [100.0]; s = service(clock=lambda: now[0]); a, b = await actors(s)
    sent = await s.send(a, 'images', 'request', key='expire', expires_in=1)
    now[0] = 102
    s.check_owner_open(b)  # called outside condition by provider admission
    assert s._reserved_replies == 1
    await s.inbox(b)  # only a locked safe boundary settles the expiry
    assert s._reserved_replies == 0
    assert s._messages[sent['receipt']['messageId']].state == 'expired'


@pytest.mark.asyncio
async def test_request_expiry_wakes_targeted_wait_with_terminal_outcome_not_wait_timeout():
    s = service(); a, b = await actors(s)
    sent = await s.send(a, 'images', 'request', key='expire', expires_in=.01)
    result = await s.wait_message(a, request_id=sent['receipt']['requestId'], timeout=1)
    assert result['outcome'] == 'request_failed' and result['reason'] == 'message_expired'
    assert s._reserved_replies == 0
