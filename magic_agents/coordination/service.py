"""Attached, scope-local mailbox authority with serialized finish/arrival.

This service is execution-owned, never global observer state. Its state and
receipts promise attached lifetime only. Durable mode uses a transactional
repository and cannot be enabled by supplying authored configuration here.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from magic_agents.models.coordination import CoordinationLimits, CoordinationPolicy, MessagingConfig
from magic_agents.coordination.budget import BudgetError, UsageBound, WorkgroupBudget, inherited_limits
from magic_llm.agent.control import InboxMessage

_INPUT_COUNT = 32
_INPUT_BYTES = 65536
_TERMINAL = frozenset(("failed", "bypassed", "timed_out", "budget_exhausted", "cancelled"))


def native_inbox_message(view: dict) -> InboxMessage:
    """One representation for admission, batching and native-loop delivery."""
    return InboxMessage(message_id=view["messageId"], sender=view["senderRole"],
        content=json.dumps({key: value for key, value in view.items() if key not in ("messageId", "senderRole")},
                           ensure_ascii=False, allow_nan=False, separators=(",", ":")))


class CoordinationError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)

    def as_result(self):
        return {"ok": False, "error": {"code": self.code, "message": str(self)}}


LIFECYCLE_CHECKPOINT_KEY = "coordination_lifecycle"


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _json(value: Any, *, max_bytes: int) -> bytes:
    def check(item, depth=0):
        if depth > 16:
            raise CoordinationError("invalid_payload", "JSON nesting exceeds the limit")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) in (list, dict):
            if len(item) > 2048:
                raise CoordinationError("invalid_payload", "JSON collection exceeds the limit")
            if isinstance(item, dict) and any(type(key) is not str for key in item):
                raise CoordinationError("invalid_payload", "JSON object keys must be strings")
            for child in item.values() if isinstance(item, dict) else item:
                check(child, depth + 1)
            return
        raise CoordinationError("invalid_payload", "Payload must contain finite JSON data")
    check(value)
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (UnicodeError, ValueError) as exc:
        raise CoordinationError("invalid_payload", "Invalid JSON/Unicode payload") from exc
    if len(result) > max_bytes:
        raise CoordinationError("payload_too_large", "Payload exceeds its UTF-8 byte limit")
    return result


@dataclass
class _Message:
    id: str
    sequence: int
    sender: str
    target: str
    text: str
    payload: Any
    kind: str
    request_id: str | None
    expires: float
    roots: tuple[str, ...]
    depth: int
    state: str = "accepted"
    reason: str | None = None
    reply_id: str | None = None
    outcome: str | None = None
    runtime_generated: bool = False


@dataclass
class _Actor:
    id: str
    path: tuple[str, ...]
    config: MessagingConfig
    state: str = "not_started"
    activation_id: str | None = None
    owner_token: object | None = None
    sequence: int = 0
    wakes: int = 0
    inbox: list[str] = field(default_factory=list)
    offered: set[str] = field(default_factory=set)
    consumed: set[str] = field(default_factory=set)
    requests: set[str] = field(default_factory=set)
    reserved_replies: int = 0
    activities: set[str] = field(default_factory=set)
    activity_placeholders: dict[str, _Message] = field(default_factory=dict)
    checkpoint: Any = None
    output: Any = None
    output_revision: int = 0
    activations: dict[str, str] = field(default_factory=dict)
    causal_roots: set[str] = field(default_factory=set)
    causal_depth: int = 0
    failure_reason: str | None = None


@dataclass(frozen=True)
class ActorCaller:
    """Runtime-issued binding; never serialized into a graph/provider schema."""
    service: "CoordinationService" = field(repr=False)
    actor_id: str
    activation_id: str
    _owner_token: object = field(repr=False)


class CoordinationService:
    def __init__(self, policy: CoordinationPolicy, members: dict[str, tuple[tuple[str, ...], MessagingConfig]], *,
                 server_limits: CoordinationLimits, authorize: Callable[[], None],
                 clock: Callable[[], float] = time.time, checkpoint_bytes: int = 4 * 1024 * 1024,
                 budget: WorkgroupBudget | None = None, inherit_limits: bool = True,
                 background_capable: bool = False):
        if not policy.enabled or policy.lifetime != "attached" or (policy.delivery_mode != "safe_boundary" and not
                (policy.delivery_mode == "background_jobs" and background_capable is True)):
            raise CoordinationError("unsupported_capability", "This coordinator requires attached safe-boundary mode")
        if set(members) != set(policy.allowed_participants):
            raise CoordinationError("invalid_config", "Participant registrations differ from the admitted policy")
        self.policy = policy.model_copy(deep=True)
        if type(inherit_limits) is not bool:
            raise CoordinationError("invalid_config", "Limit inheritance must be explicit")
        invocation = bool(budget and budget.invocation)
        requested = inherited_limits(policy.limits, budget._limits, require_complete=not invocation) if budget and inherit_limits else policy.limits
        self.limits = requested.intersect_server_policy(server_limits, require_complete=not invocation)
        self.budget = budget.child(self.limits) if budget else WorkgroupBudget(self.limits, clock=clock)
        self.clock, self.authorize = self.budget.clock, authorize
        self.deadline = self.budget.deadline
        self.workgroup_id = f"group-{self.budget._path[0].id}"
        self.epoch_id, self.scope_id = _id("epoch"), _id("scope")
        self._condition = self.budget.condition
        self._actors = {}
        self._roles = {}
        for role, (path, config) in members.items():
            if not config.enabled or config.role != role or set(config.peers) - set(members) or role in config.peers:
                raise CoordinationError("invalid_config", "Invalid participant registration")
            actor = _Actor(_id("actor"), tuple(path), config.model_copy(deep=True))
            self._actors[actor.id] = actor
            self._roles[role] = actor.id
        self._messages: dict[str, _Message] = {}
        self._requests: dict[str, str] = {}
        self._operations: dict[tuple[str, str], tuple[bytes, dict]] = {}
        self._accepted = self._reserved_replies = self._wakes = 0
        self._state = "open"
        self._activity_manager = None
        self._reserved_activity_notices = 0
        self._publication = None
        self._checkpoint_bytes = checkpoint_bytes
        self.events = None
        self._checkpoint_boundaries = {}
        self._retained_waits = {}
        self._restored_actors = set()
        self._volatile_state_lost = False
        if type(checkpoint_bytes) is not int or checkpoint_bytes < 1:
            raise CoordinationError("invalid_config", "Checkpoint byte ceiling must be positive")

    def _event(self, kind, actor=None, *, terminal=False, message_id=None, request_id=None, **data):
        if self.events is not None:
            self.events.append(self.scope_id, kind, data, actor=actor, terminal=terminal,
                               message_id=message_id, request_id=request_id)

    def _state_change(self, actor, state, reason=None):
        old = actor.state
        actor.state = state
        if old != state:
            self._event('actor_state', actor, oldState=old, newState=state,
                **({'reason': reason} if reason else {}), terminal=state in {'quiescent', 'completed', *_TERMINAL})

    def _activation_end(self, actor, state, reason=None):
        if actor.activation_id and actor.activation_id not in actor.activations:
            actor.activations[actor.activation_id] = state
            kind = 'activation_finished' if state == 'finished' else 'activation_cancelled' if state == 'cancelled' else 'activation_failed'
            self._event(kind, actor, state=state, **({'reason': reason} if reason else {}), terminal=True)

    @contextmanager
    def _waiting(self, actor, state, request_id=None):
        prior, activation = actor.state, actor.activation_id
        self._state_change(actor, state)
        self._event('wait_started', actor, request_id=request_id)
        try:
            yield
        finally:
            if actor.activation_id == activation and actor.state == state:
                self._state_change(actor, prior)
            original = copy.copy(actor)
            original.activation_id = activation
            self._event('wait_finished', original, request_id=request_id, state=actor.state)

    def _access(self, caller: ActorCaller) -> _Actor:
        self.authorize()
        if caller.service is not self or caller.actor_id not in self._actors:
            raise CoordinationError("unknown_ref", "Unknown actor reference")
        return self._actors[caller.actor_id]

    def _owner(self, caller: ActorCaller) -> _Actor:
        actor = self._access(caller)
        if actor.owner_token is not caller._owner_token or actor.activation_id != caller.activation_id:
            raise CoordinationError("stale_activation", "Activation ownership no longer matches")
        return actor

    def _open(self):
        self._expire()
        self._assert_open()

    def _assert_open(self):
        """Read-only admission; expiry settlement belongs to locked methods."""
        try: self.budget._open()
        except BudgetError as exc: raise CoordinationError(exc.code, str(exc)) from exc
        if self._state != "open":
            code = {"cancelled": "cancelled", "timed_out": "deadline_exceeded", "failed": "workgroup_failed",
                    "budget_exhausted": "budget_exhausted"}.get(self._state, "epoch_closed")
            raise CoordinationError(code, "The epoch no longer admits work")
        if self.clock() >= self.deadline:
            raise CoordinationError("deadline_exceeded", "The original workgroup deadline expired")

    def check_owner_open(self, caller: ActorCaller):
        """Synchronous provider guard, safe outside the mailbox transaction."""
        self._access(caller)
        self._assert_open()
        self._owner(caller)

    def _target(self, caller: _Actor, role_or_ref: str) -> _Actor:
        actor = self._actors.get(self._roles.get(role_or_ref, role_or_ref))
        if actor is None or actor.config.role not in caller.config.peers:
            raise CoordinationError("unknown_ref", "Peer is not visible in this scope")
        return actor

    def _pending(self, actor):
        return [self._messages[mid] for mid in actor.inbox if self._messages[mid].state == "accepted"]

    def _reserved_slots(self, actor):
        return actor.reserved_replies + len(actor.activities)

    def admit_activity_locked(self, caller, job_id, *, commit=False):
        self.admit_tool_locked(caller)
        actor = self._actors[caller.actor_id]
        if self._accepted + self._reserved_replies + self._reserved_activity_notices >= self.limits.max_accepted_messages:
            raise CoordinationError('message_budget_exhausted', 'No activity completion allowance remains')
        if len(self._pending(actor)) + self._reserved_slots(actor) >= self.limits.max_pending_messages_per_actor:
            raise CoordinationError('queue_full', 'No activity completion inbox capacity remains')
        self.budget._check_capacity(UsageBound(messages=1), 'quota')
        # Reserve the maximum supported terminal reference/reason after all
        # renderer and checkpoint escaping, not merely a queue record.
        placeholder = _Message('reserved-' + job_id, self.limits.max_accepted_messages, actor.id, actor.id,
            'Owned activity reached a terminal outcome.',
            {'jobId': job_id, 'state': 'completed', 'resultRef': '\x00' * 1024, 'reason': '\x00' * 128},
            'event', None, self.deadline, (), 0, reason='\x00' * 128, outcome='error', runtime_generated=True)
        self._admit_envelope(actor, placeholder)
        if commit:
            self.budget.allocate_messages_locked(accepted=0, reserved_replies=1)
            actor.activities.add(job_id)
            actor.activity_placeholders[job_id] = placeholder
            self._reserved_activity_notices += 1
            self._event('job_state', actor, jobId=job_id, state='queued')

    def activity_finished_locked(self, caller, job_id, *, state, result_ref=None, reason=None):
        actor = self._actors[caller.actor_id]
        if job_id not in actor.activities:
            return
        actor.activity_placeholders.pop(job_id, None)
        notify = self._state == 'open' and actor.state not in _TERMINAL
        notice = _Message(_id('message'), actor.sequence + 1, actor.id, actor.id,
            'Owned activity reached a terminal outcome.',
            {'jobId': job_id, 'state': state, **({'resultRef': result_ref} if result_ref else {}),
             **({'reason': reason} if reason else {})}, 'event', None, self.deadline,
            tuple(sorted(actor.causal_roots or {self.workgroup_id})), actor.causal_depth,
            reason=reason, outcome='success' if state == 'completed' else 'error', runtime_generated=True)
        if notify:
            try: self._admit_envelope(actor, notice)
            except CoordinationError:
                self._terminate_actor(actor, 'failed', 'checkpoint_capacity_exceeded')
                notify = False
        actor.activities.remove(job_id)
        self._reserved_activity_notices -= 1
        self.budget.resolve_reply_locked(accepted=notify)
        if notify:
            actor.sequence += 1
            self._messages[notice.id] = notice
            actor.inbox.append(notice.id)
            self._accepted += 1
        self._event('job_state', actor, jobId=job_id, state=state,
                    **({'reason': reason} if reason else {}), terminal=True)
        self._condition.notify_all()

    def admit_tool_locked(self, caller: ActorCaller):
        """Called in the shared ledger transaction, before charging a new call."""
        if not self._condition.locked():
            raise RuntimeError("Tool admission requires the lineage lock")
        self._open()
        self._owner(caller)

    def _settle_request(self, message, state, reason, *, notify):
        source = self._actors[message.sender]
        if not message.request_id or message.request_id not in source.requests:
            return
        source.requests.remove(message.request_id)
        source.reserved_replies -= 1
        self._reserved_replies -= 1
        message.state, message.reason = state, reason
        self._event('request_resolved', source, message_id=message.id,
                    request_id=message.request_id, state=state, reason=reason, terminal=True)
        notify = notify and source.state not in _TERMINAL and self._state == "open"
        self.budget.resolve_reply_locked(accepted=notify)
        if notify:
            source.sequence += 1
            notice = _Message(_id("message"), source.sequence, message.target, source.id,
                "Request closed without a final reply.", {"reason": reason}, "event", message.request_id,
                self.deadline, message.roots, message.depth, reason=reason, outcome="error", runtime_generated=True)
            self._messages[notice.id] = notice
            source.inbox.append(notice.id)
            self._accepted += 1

    def _close_epoch(self, state, reason):
        if self._state != "open":
            return
        self._state = state
        for actor in self._actors.values():
            if actor.state not in _TERMINAL:
                self._state_change(actor, state, reason)
                actor.failure_reason = reason
            actor.output = None
            actor.owner_token = None
            self._activation_end(actor, actor.state, reason)
            if self._activity_manager is not None:
                self._activity_manager.cancel_actor_locked(actor.id)
        if state in ('budget_exhausted', 'timed_out'):
            self._event('budget_stopped', state=state, reason=reason, terminal=True)
        for message in list(self._messages.values()):
            self._settle_request(message, "cancelled" if state == "cancelled" else "failed", reason, notify=False)
            if message.state in ("accepted", "consumed"):
                message.state, message.reason = "cancelled" if state == "cancelled" else "failed", reason
        self._condition.notify_all()

    def _terminate_actor(self, actor, state, reason):
        if actor.state in _TERMINAL:
            return
        self._state_change(actor, state, reason)
        actor.failure_reason, actor.output, actor.owner_token = reason, None, None
        self._activation_end(actor, state, reason)
        if self.policy.failure_policy == "fail_group":
            self._close_epoch("failed", reason)
        else:
            if self._activity_manager is not None:
                self._activity_manager.cancel_actor_locked(actor.id)
            for message in list(self._messages.values()):
                if message.sender == actor.id or message.target == actor.id:
                    self._settle_request(message, "failed", reason, notify=message.sender != actor.id)
                if message.target == actor.id and message.state in ("accepted", "consumed"):
                    message.state, message.reason = "failed", reason
        self._condition.notify_all()

    def _expire(self):
        now = self.clock()
        changed = False
        if self.budget._shared.cancelled:
            self._close_epoch("cancelled", "workgroup_cancelled")
        elif self.budget._shared.exceeded:
            self._close_epoch("budget_exhausted", "budget_exhausted")
        if now >= self.deadline and self._state == "open":
            self._close_epoch("timed_out", "deadline_exceeded")
        for message in list(self._messages.values()):
            if message.expires > now or message.state not in ("accepted", "consumed"):
                continue
            message.state, message.reason = "expired", "message_expired"
            self._settle_request(message, "expired", "message_expired", notify=True)
            changed = True
        if changed:
            self._condition.notify_all()

    def _project_input(self, actor, snapshot, additional=None):
        projected = copy.deepcopy(snapshot or {})
        if actor.checkpoint and LIFECYCLE_CHECKPOINT_KEY in actor.checkpoint:
            projected.setdefault(LIFECYCLE_CHECKPOINT_KEY,
                                 copy.deepcopy(actor.checkpoint[LIFECYCLE_CHECKPOINT_KEY]))
        messages = projected.setdefault("messages", [])
        consumed = projected.setdefault("consumed_message_ids", [])
        digests = projected.setdefault("message_digests", {})
        for envelope in [*self._pending(actor), *actor.activity_placeholders.values(), *([additional] if additional else [])]:
            item = native_inbox_message(self._view(envelope))
            if item.message_id in consumed:
                continue
            messages.append({"role": "user", "content": item.render()})
            consumed.append(item.message_id)
            digests[item.message_id] = item.digest()
        return projected

    def _admit_envelope(self, actor, envelope):
        try:
            rendered = native_inbox_message(self._view(envelope)).render().encode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise CoordinationError("payload_too_large", "Message cannot fit the native input representation") from exc
        if len(rendered) > _INPUT_BYTES:
            raise CoordinationError("payload_too_large", "Rendered message exceeds the native input byte limit")
        try:
            _json(self._project_input(actor, actor.checkpoint, envelope), max_bytes=self._checkpoint_bytes)
        except CoordinationError as exc:
            if exc.code not in ("payload_too_large", "invalid_payload"):
                raise
            raise CoordinationError("checkpoint_capacity_exceeded", "Pending input cannot fit the actor checkpoint") from exc

    def _activate(self, actor):
        if self.policy.delivery_mode == 'background_jobs' and self._activity_manager is None:
            raise CoordinationError('unsupported_capability', 'Background mode requires an owned activity registry')
        if actor.state not in ("not_started", "queued") or actor.owner_token is not None:
            raise CoordinationError("activation_conflict", "Actor already has an activation owner")
        actor.activation_id, actor.owner_token = _id("activation"), object()
        # Prior output is not a candidate for this activation. Finish replaces
        # it; failures already clear it, and inspection exposes revision only.
        actor.output = None
        self._state_change(actor, 'running')
        self._event('activation_started', actor)
        self._condition.notify_all()
        return ActorCaller(self, actor.id, actor.activation_id, actor.owner_token)

    def _replay(self, caller: _Actor, key: str, args: dict):
        if type(key) is not str or not 1 <= len(key) <= 256:
            raise CoordinationError("invalid_idempotency_key", "A bounded runtime operation key is required")
        digest = hashlib.sha256(_json(args, max_bytes=self.limits.max_inline_message_bytes + 4096)).digest()
        stored = self._operations.get((caller.id, key))
        if stored is None:
            return digest, None
        if stored[0] != digest:
            raise CoordinationError("idempotency_conflict", "Operation key was used with different arguments")
        result = copy.deepcopy(stored[1])
        message = self._messages[result["receipt"]["messageId"]]
        result["currentDisposition"] = {"state": message.state, "reason": message.reason}
        return digest, result

    async def activate(self, role: str) -> ActorCaller:
        async with self._condition:
            self.authorize(); self._expire()
            actor = self._actors[self._roles[role]]
            if actor.id in self._restored_actors:
                self._restored_actors.remove(actor.id)
                if self._state == 'sealed' or actor.state in ({'quiescent'} | _TERMINAL): return None
                self._open()
                if actor.state in {'running', 'awaiting_reply', 'awaiting_tools'} and actor.activation_id:
                    actor.owner_token = object()
                    self._state_change(actor, 'running')
                    return ActorCaller(self, actor.id, actor.activation_id, actor.owner_token)
            self._open()
            return self._activate(actor)

    async def send(self, caller: ActorCaller, target: str, message: str, payload=None, *, key: str,
                   kind: str = "request", expect_reply: bool = True, wake: bool = False,
                   expires_in: float | None = None, expected_epoch: str | None = None,
                   expected_activation: str | None = None):
        args = dict(target=target, message=message, payload=payload, kind=kind, expect_reply=expect_reply,
                    wake=wake, expires_in=expires_in, expected_epoch=expected_epoch, expected_activation=expected_activation)
        async with self._condition:
            source = self._access(caller); self._expire()
            try:
                digest, replay = self._replay(source, key, args)
                if replay is not None: return replay
                self._owner(caller); self._open()
                recipient = self._target(source, target)
                if type(message) is not str or kind not in ("request", "event") or type(expect_reply) is not bool or type(wake) is not bool:
                    raise CoordinationError("invalid_message", "Invalid message kind, text or options")
                if kind == "event" and (expect_reply or wake):
                    raise CoordinationError("invalid_message", "Events cannot request replies or wake completed actors")
                _json({"message": message, "payload": payload}, max_bytes=self.limits.max_inline_message_bytes)
                if expected_epoch is not None and expected_epoch != self.epoch_id:
                    raise CoordinationError("stale_epoch", "Epoch guard does not match")
                if expected_activation is not None and expected_activation != recipient.activation_id:
                    raise CoordinationError("stale_activation", "Target activation guard does not match")
                ttl = self.deadline - self.clock() if expires_in is None else expires_in
                if type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl <= 0:
                    raise CoordinationError("invalid_expiry", "Message expiry must be finite and positive")
                waking = recipient.state == "quiescent"
                if recipient.state not in ("not_started", "queued", "running", "awaiting_reply", "awaiting_tools", "quiescent"):
                    raise CoordinationError("target_terminal", "Target no longer accepts work")
                if waking:
                    if not wake: raise CoordinationError("wake_required", "Quiescent peer requires an explicit wake")
                    if recipient.config.role not in source.config.can_wake_peers or not recipient.config.wake_on_message:
                        raise CoordinationError("wake_denied", "Wake is not permitted by both peers")
                    if recipient.wakes >= self.limits.max_wakeups_per_actor or self._wakes >= self.limits.max_wakeups_per_workgroup:
                        raise CoordinationError("wake_budget_exhausted", "Wake allowance exhausted")
                depth = source.causal_depth + (1 if kind == "request" else 0)
                if depth > self.limits.max_request_depth:
                    raise CoordinationError("request_depth_exceeded", "Causal request depth exceeded")
                reserve_reply = int(kind == "request" and expect_reply)
                if self._accepted + self._reserved_replies + self._reserved_activity_notices + 1 + reserve_reply > self.limits.max_accepted_messages:
                    raise CoordinationError("message_budget_exhausted", "Message and reserved outcome allowance exhausted")
                if len(self._pending(recipient)) + self._reserved_slots(recipient) >= self.limits.max_pending_messages_per_actor:
                    raise CoordinationError("queue_full", "Peer inbox is full")
                if reserve_reply and len(self._pending(source)) + self._reserved_slots(source) >= self.limits.max_pending_messages_per_actor:
                    raise CoordinationError("queue_full", "No reserved reply capacity in caller inbox")
                request_id = _id("request") if reserve_reply else None
                envelope = _Message(_id("message"), recipient.sequence + 1, source.id, recipient.id, message, copy.deepcopy(payload),
                                    kind, request_id, min(self.deadline, self.clock() + ttl),
                                    tuple(sorted(source.causal_roots or {self.workgroup_id})), depth)
                self._admit_envelope(recipient, envelope)
                try: self.budget.allocate_messages_locked(accepted=1, reserved_replies=reserve_reply, wakeups=int(waking))
                except BudgetError as exc: raise CoordinationError(exc.code, str(exc)) from exc
                recipient.sequence += 1
                self._messages[envelope.id] = envelope; recipient.inbox.append(envelope.id)
                self._accepted += 1
                if reserve_reply:
                    self._requests[request_id] = envelope.id; source.requests.add(request_id)
                    source.reserved_replies += 1; self._reserved_replies += 1
                if waking:
                    recipient.wakes += 1; self._wakes += 1
                    self._state_change(recipient, 'queued')
                    self._event('wake_scheduled', recipient, message_id=envelope.id, request_id=request_id)
                elif wake and recipient.state == 'queued':
                    self._event('wake_coalesced', recipient, message_id=envelope.id, request_id=request_id)
                self._event('message_accepted', recipient, message_id=envelope.id, request_id=request_id,
                            queueDepth=len(self._pending(recipient)), wakeScheduled=waking)
                result = {"ok": True, "receipt": {"messageId": envelope.id, "requestId": request_id, "stage": "accepted",
                          "targetState": recipient.state, "targetActivationId": recipient.activation_id,
                          "recipientSequence": envelope.sequence, "epochId": self.epoch_id, "expiresAt": envelope.expires,
                          "wakeScheduled": waking, "lifetime": "attached"}}
                self._operations[(source.id, key)] = (digest, copy.deepcopy(result))
                self._condition.notify_all()
                return result
            except CoordinationError as error:
                self._event('message_rejected', source, reason=error.code)
                if error.code in ('wake_required', 'wake_denied', 'wake_budget_exhausted'):
                    self._event('wake_denied', source, reason=error.code)
                raise

    async def reply(self, caller: ActorCaller, request_id: str, message: str, payload=None, *, key: str, outcome="success", error_code: str | None = None):
        async with self._condition:
            source = self._access(caller); self._expire()
            digest, replay = self._replay(source, key, dict(request_id=request_id, message=message, payload=payload, outcome=outcome, error_code=error_code))
            if replay is not None: return replay
            self._owner(caller); self._open()
            original = self._messages.get(self._requests.get(request_id, ""))
            if original is None or original.target != source.id:
                raise CoordinationError("unknown_request", "Request is not addressed to this actor")
            if original.reply_id is not None:
                raise CoordinationError("request_resolved", "The request already has a final reply")
            if original.state not in ("accepted", "consumed"):
                raise CoordinationError("request_closed", "Request no longer accepts a reply")
            if type(message) is not str or outcome not in ("success", "error"):
                raise CoordinationError("invalid_reply", "Invalid reply message or outcome")
            if error_code is not None and (type(error_code) is not str or not 1 <= len(error_code) <= 128 or outcome != "error"):
                raise CoordinationError("invalid_reply", "Error code requires a bounded error outcome")
            _json({"message": message, "payload": payload}, max_bytes=self.limits.max_inline_message_bytes)
            target = self._actors[original.sender]
            if request_id not in target.requests or target.state not in ("running", "awaiting_reply", "awaiting_tools"):
                raise CoordinationError("request_closed", "Origin has no open reply obligation")
            reply = _Message(_id("message"), target.sequence + 1, source.id, target.id, message, copy.deepcopy(payload),
                             "reply", request_id, original.expires, original.roots, original.depth, outcome=outcome, reason=error_code)
            self._admit_envelope(target, reply)
            target.sequence += 1
            self.budget.resolve_reply_locked(accepted=True)
            self._messages[reply.id] = reply; target.inbox.append(reply.id)
            original.reply_id, original.state = reply.id, "resolved"
            target.requests.remove(request_id); target.reserved_replies -= 1
            self._reserved_replies -= 1; self._accepted += 1
            self._event('message_accepted', target, message_id=reply.id, request_id=request_id,
                        queueDepth=len(self._pending(target)))
            self._event('request_resolved', target, message_id=original.id, request_id=request_id,
                        state='resolved', outcome=outcome, terminal=True)
            result = {"ok": True, "receipt": {"messageId": reply.id, "requestId": request_id, "stage": "accepted", "requestOutcome": outcome}}
            if error_code is not None: result["receipt"]["errorCode"] = error_code
            self._operations[(source.id, key)] = (digest, copy.deepcopy(result))
            self._condition.notify_all()
            return result

    def _view(self, message):
        return {"messageId": message.id, "sequence": message.sequence, "kind": message.kind,
                "senderRole": "runtime" if message.runtime_generated else self._actors[message.sender].config.role, "requestId": message.request_id,
                "outcome": message.outcome, "errorCode": message.reason if message.outcome == "error" else None,
                "message": message.text, "payload": copy.deepcopy(message.payload)}

    async def inbox(self, caller: ActorCaller, *, limit=_INPUT_COUNT, checkpoint: dict | None = None):
        if type(limit) is not int or not 1 <= limit <= _INPUT_COUNT:
            raise CoordinationError("invalid_batch", "Inbox batch size must be between 1 and 32")
        async with self._condition:
            self._open(); actor = self._owner(caller)
            try:
                _json(self._project_input(actor, checkpoint if checkpoint is not None else actor.checkpoint),
                      max_bytes=self._checkpoint_bytes)
            except CoordinationError as exc:
                if exc.code == "payload_too_large":
                    self._terminate_actor(actor, "failed", "checkpoint_capacity_exceeded")
                    raise CoordinationError("checkpoint_capacity_exceeded", "Canonical input can no longer fit the actor checkpoint") from exc
                raise
            batch = []
            total = 0
            for item in self._pending(actor)[:limit]:
                size = len(native_inbox_message(self._view(item)).render().encode("utf-8"))
                if total + size > _INPUT_BYTES: break
                total += size
                batch.append(item)
            actor.offered.update(item.id for item in batch)
            return [self._view(item) for item in batch]

    async def checkpoint(self, caller: ActorCaller, snapshot: dict, consumed_ids: list[str], *, boundary='tool_results'):
        async with self._condition:
            self._open(); actor = self._owner(caller)
            snapshot = copy.deepcopy(snapshot)
            if actor.checkpoint and LIFECYCLE_CHECKPOINT_KEY in actor.checkpoint:
                snapshot.setdefault(LIFECYCLE_CHECKPOINT_KEY,
                                    copy.deepcopy(actor.checkpoint[LIFECYCLE_CHECKPOINT_KEY]))
            if not set(consumed_ids).issubset(actor.offered | actor.consumed):
                raise CoordinationError("invalid_checkpoint", "Checkpoint acknowledges an undelivered message")
            try:
                if LIFECYCLE_CHECKPOINT_KEY in snapshot:
                    canonical = {key: value for key, value in snapshot.items() if key != LIFECYCLE_CHECKPOINT_KEY}
                    snapshot[LIFECYCLE_CHECKPOINT_KEY]['canonicalDigest'] = hashlib.sha256(
                        _json(canonical, max_bytes=self._checkpoint_bytes)).hexdigest()
                encoded = _json(snapshot, max_bytes=self._checkpoint_bytes)
                _json(self._project_input(actor, snapshot), max_bytes=self._checkpoint_bytes)
            except CoordinationError as exc:
                if exc.code == "payload_too_large":
                    self._terminate_actor(actor, "failed", "checkpoint_capacity_exceeded")
                    raise CoordinationError("checkpoint_capacity_exceeded", "Canonical state can no longer fit the actor checkpoint") from exc
                raise
            actor.checkpoint = json.loads(encoded)
            self._checkpoint_boundaries[actor.id] = boundary
            for mid in consumed_ids:
                if self._volatile_state_lost and mid not in self._messages: continue
                item = self._messages[mid]
                if item.state == "accepted": item.state = "consumed" if item.kind == "request" and item.request_id else "resolved"
                if mid not in actor.consumed:
                    self._event('message_consumed', actor, message_id=mid, request_id=item.request_id)
                actor.consumed.add(mid); actor.offered.discard(mid)
                actor.causal_roots.update(item.roots); actor.causal_depth = max(actor.causal_depth, item.depth)
            self._condition.notify_all()

    async def retained_checkpoint(self, caller: ActorCaller):
        async with self._condition:
            actor = self._owner(caller)
            return copy.deepcopy(actor.checkpoint)

    async def checkpoint_lifecycle(self, caller: ActorCaller, *, node_id, output, invocation, child_operations):
        """Commit private effective output/journal beside unchanged native history.

        An input arriving around this commit remains pending and prevents
        finish. Native checkpoints retain the journal on later loop turns.
        """
        async with self._condition:
            self._open(); actor = self._owner(caller)
            if node_id != actor.path[-1] or actor.checkpoint is None:
                raise CoordinationError('invalid_lifecycle_checkpoint', 'Lifecycle checkpoint requires this actor canonical state')
            if (not isinstance(invocation, dict) or invocation.get('node_id') != node_id
                    or not isinstance(invocation.get('outcome'), dict) or invocation['outcome'].get('status') != 'success'
                    or not isinstance(child_operations, list) or len(child_operations) > 100
                    or any(not isinstance(item, dict) or item.get('ownerActorId') != caller.actor_id or item.get('state') != 'completed'
                           for item in child_operations)):
                raise CoordinationError('invalid_lifecycle_checkpoint', 'Lifecycle effects must be complete and owned by this actor')
            snapshot = copy.deepcopy(actor.checkpoint)
            snapshot.pop(LIFECYCLE_CHECKPOINT_KEY, None)
            payload = {'schemaVersion': 1, 'nodeId': node_id, 'activationId': caller.activation_id,
                       'effectiveOutput': copy.deepcopy(output), 'invocation': copy.deepcopy(invocation),
                       'childOperations': copy.deepcopy(child_operations),
                       'canonicalDigest': hashlib.sha256(_json(snapshot, max_bytes=self._checkpoint_bytes)).hexdigest()}
            snapshot[LIFECYCLE_CHECKPOINT_KEY] = payload
            try:
                encoded = _json(snapshot, max_bytes=self._checkpoint_bytes)
                _json(self._project_input(actor, snapshot), max_bytes=self._checkpoint_bytes)
            except CoordinationError as error:
                if error.code == 'payload_too_large':
                    self._terminate_actor(actor, 'failed', 'checkpoint_capacity_exceeded')
                    raise CoordinationError('checkpoint_capacity_exceeded', 'Lifecycle state exceeds canonical checkpoint capacity') from error
                raise
            actor.checkpoint = json.loads(encoded)
            self._checkpoint_boundaries[actor.id] = 'candidate'
            self._condition.notify_all()

    async def finish(self, caller: ActorCaller, output):
        """After lifecycle controls: atomically retain output or continue arrivals."""
        _json(output, max_bytes=self._checkpoint_bytes)
        async with self._condition:
            self._open(); actor = self._owner(caller)
            if self._pending(actor): return "continue"
            if actor.requests: return "awaiting_reply"
            if actor.activities: return "awaiting_tools"
            if actor.checkpoint is None:
                raise CoordinationError("checkpoint_required", "An activation must retain a safe checkpoint before finishing")
            lifecycle = actor.checkpoint.get(LIFECYCLE_CHECKPOINT_KEY)
            if lifecycle is not None:
                canonical = {key: value for key, value in actor.checkpoint.items() if key != LIFECYCLE_CHECKPOINT_KEY}
                digest = hashlib.sha256(_json(canonical, max_bytes=self._checkpoint_bytes)).hexdigest()
                if (lifecycle.get('activationId') != caller.activation_id or lifecycle.get('effectiveOutput') != output
                        or lifecycle.get('canonicalDigest') != digest):
                    raise CoordinationError('invalid_lifecycle_checkpoint', 'Effective output must match the committed final lifecycle state')
            actor.output, actor.output_revision = copy.deepcopy(output), actor.output_revision + 1
            self._state_change(actor, 'quiescent')
            self._activation_end(actor, 'finished')
            actor.owner_token = None
            self._condition.notify_all()
            return "quiescent"

    async def candidate_decision(self, caller: ActorCaller):
        """Wait without a provider permit while retaining the activation owner."""
        async with self._condition:
            self._open(); actor = self._owner(caller)
            if self._pending(actor): return 'continue'
            if not actor.requests and not actor.activities: return 'candidate_ready'
            with self._waiting(actor, 'awaiting_tools' if actor.activities else 'awaiting_reply'):
                while True:
                    self._open(); actor = self._owner(caller)
                    if self._pending(actor): return "continue"
                    if not actor.requests and not actor.activities: return "candidate_ready"
                    deadline = min([self.deadline, *(self._messages[self._requests[r]].expires for r in actor.requests)])
                    try: await asyncio.wait_for(self._condition.wait(), timeout=max(0, deadline - self.clock()))
                    except TimeoutError: continue

    async def guard(self, caller: ActorCaller):
        async with self._condition:
            self._open(); self._owner(caller)

    async def terminate(self, role: str, *, state="failed", reason="participant_failed"):
        """Trusted scheduler settlement, including actors that never activated."""
        if state not in _TERMINAL or type(reason) is not str or not 1 <= len(reason) <= 128:
            raise CoordinationError("invalid_terminal", "Terminal state and bounded reason are required")
        async with self._condition:
            self.authorize(); self._expire()
            actor_id = self._roles.get(role)
            if actor_id is None:
                raise CoordinationError("unknown_ref", "Unknown participant")
            actor = self._actors[actor_id]
            if actor.state in _TERMINAL:
                return
            if self._state == "sealed":
                raise CoordinationError("epoch_closed", "A sealed epoch cannot be changed")
            self._terminate_actor(actor, state, reason)

    def _try_seal(self):
        if self._state == "sealed":
            return copy.deepcopy(self._publication)
        if self._state != "open" or any(a.state not in ({"quiescent"} | _TERMINAL) or self._pending(a) or a.requests or a.activities
                                       for a in self._actors.values()):
            return None
        result = {}
        for actor in self._actors.values():
            value = {"output": copy.deepcopy(actor.output), "revision": actor.output_revision}
            if actor.state in _TERMINAL:
                value["failure"] = {"state": actor.state, "reason": actor.failure_reason}
            else:
                self._state_change(actor, 'completed')
            result[actor.config.role] = value
        self._publication, self._state = result, "sealed"
        self._event('epoch_sealed', participants=[{'actorId': actor.id, 'outputRevision': actor.output_revision,
                    'state': actor.state} for actor in self._actors.values()], terminal=True)
        self._condition.notify_all()
        return copy.deepcopy(result)

    async def wait_for_activation(self, role: str) -> ActorCaller | None:
        """Scheduler-owned waiter: queue/wake and seal share one lock."""
        async with self._condition:
            while True:
                self.authorize(); self._expire()
                if self._state == "sealed":
                    return None
                self._open()
                actor_id = self._roles.get(role)
                if actor_id is None:
                    raise CoordinationError("unknown_ref", "Unknown participant")
                actor = self._actors[actor_id]
                if actor.state == "queued":
                    return self._activate(actor)
                if actor.state not in ({"quiescent"} | _TERMINAL):
                    raise CoordinationError("activation_conflict", "Wait follows a settled activation or terminal participant")
                if self._try_seal() is not None:
                    return None
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=max(0, self.deadline - self.clock()))
                except TimeoutError:
                    continue

    def _wait_deadline(self, caller, name, arguments, end):
        from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL
        call = CURRENT_TOOL_CALL.get()
        identity = getattr(call, 'id', None)
        if identity is None: return end
        key = caller.actor_id + ':' + identity
        digest = hashlib.sha256(_json({'name': name, 'arguments': arguments}, max_bytes=8192)).hexdigest()
        prior = self._retained_waits.get(key)
        if prior is not None:
            if prior['digest'] != digest: raise CoordinationError('idempotency_conflict', 'Wait identity changed arguments')
            return min(end, prior['deadline'])
        self._retained_waits[key] = {'digest': digest, 'deadline': end}
        return end

    async def wait_agent(self, caller: ActorCaller, target: str, *, until: str, timeout: float,
                         activation_id: str | None = None, request_id: str | None = None):
        if until == "request_resolved":
            async with self._condition:
                source = self._owner(caller); recipient = self._target(source, target)
                original = self._messages.get(self._requests.get(request_id, ""))
                if original is None or original.sender != source.id or original.target != recipient.id:
                    raise CoordinationError("unknown_request", "Request does not bind the caller and selected peer")
            result = await self.wait_message(caller, request_id=request_id, timeout=timeout)
            if result["outcome"] == "reply": result["outcome"] = "resolved"
            return result
        if until != "activation_finished" or not activation_id:
            raise CoordinationError("invalid_wait", "Wait requires an exact activation or request guard")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= self.limits.max_single_wait_seconds:
            raise CoordinationError("invalid_timeout", "Wait must have a finite admitted timeout")
        end = min(self.clock() + timeout, self.deadline)
        async with self._condition:
            source = self._owner(caller); recipient = self._target(source, target)
            end = self._wait_deadline(caller, 'wait_agent', [target, until, timeout, activation_id, request_id], end)
            if activation_id != recipient.activation_id and activation_id not in recipient.activations:
                raise CoordinationError("unknown_activation", "Target activation is unknown")
            with self._waiting(source, 'awaiting_reply', request_id):
                while True:
                    self._access(caller); self._expire()
                    if self._state != "open": return {"ok": True, "outcome": "closed"}
                    self._owner(caller)
                    if activation_id in recipient.activations:
                        terminal = recipient.activations[activation_id]
                        return {"ok": True, "outcome": "target_failed" if terminal in _TERMINAL else "finished", "activationId": activation_id, "state": terminal if terminal in _TERMINAL else recipient.state,
                                "outputRevision": recipient.output_revision}
                    if self._pending(source): return {"ok": True, "outcome": "inbox_ready"}
                    remaining = end - self.clock()
                    if remaining <= 0: return {"ok": True, "outcome": "timeout"}
                    try: await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except TimeoutError: continue

    async def seal(self):
        async with self._condition:
            self.authorize(); self._expire()
            if self._state != "sealed": self._open()
            return self._try_seal()

    async def inspect(self, caller: ActorCaller, target: str):
        async with self._condition:
            source = self._access(caller); self._expire(); actor = self._target(source, target)
            state = actor.state
            summary = "not_started" if state in ("not_started", "queued", "blocked_inputs") else "finished" if state in ("completed", "quiescent", *_TERMINAL) else "running"
            remaining = self.budget.remaining_locked()
            permitted_wake = actor.config.role in source.config.can_wake_peers and actor.config.wake_on_message
            has_wake_budget = actor.wakes < self.limits.max_wakeups_per_actor and remaining["remainingWakeups"] > 0
            reason = ("epoch_closed" if self._state != "open" else "wake_denied" if not permitted_wake
                      else "wake_budget_exhausted" if not has_wake_budget else "not_quiescent" if state != "quiescent" else None)
            return {"ok": True, "agentRef": actor.id, "role": actor.config.role, "state": state, "summaryStatus": summary,
                    "activationId": actor.activation_id, "outputRevision": actor.output_revision, "outputPublished": self._state == "sealed",
                    "epochId": self.epoch_id, "epochState": self._state, "pendingMessageCount": len(self._pending(actor)),
                    "unresolvedRequestCount": len(actor.requests), "activeJobCount": len(actor.activities), "liveness": "local_live",
                    "description": actor.config.description, "capabilities": list(actor.config.tools),
                    "canSend": self._state == "open" and state in ("not_started", "queued", "running", "awaiting_reply", "awaiting_tools"),
                    "canRequestWake": permitted_wake and has_wake_budget, "acceptsWake": actor.config.wake_on_message,
                    "canWake": reason is None, "wakeDeniedReason": reason, "remainingBudget": remaining}

    async def list_agents(self, caller: ActorCaller):
        # Each result is a fresh authorized observation; no tool-fingerprint cache.
        source = self._access(caller)
        return {"peers": [await self.inspect(caller, role) for role in source.config.peers]}

    async def wait_message(self, caller: ActorCaller, *, request_id: str | None = None, after_sequence: int = 0, timeout: float):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or timeout > self.limits.max_single_wait_seconds:
            raise CoordinationError("invalid_timeout", "Wait must have a finite admitted timeout")
        if type(after_sequence) is not int or after_sequence < 0:
            raise CoordinationError("invalid_cursor", "Cursor must be nonnegative")
        end = min(self.clock() + timeout, self.deadline)
        async with self._condition:
            actor = self._owner(caller)
            original = self._messages.get(self._requests.get(request_id, "")) if request_id else None
            if request_id and (original is None or original.sender != actor.id):
                raise CoordinationError("unknown_request", "Caller does not own this request")
            if original: end = min(end, original.expires)
            end = self._wait_deadline(caller, 'wait_message', [request_id, after_sequence, timeout], end)
            with self._waiting(actor, 'awaiting_reply', request_id):
                while True:
                    self._access(caller); self._expire()
                    if self._state != "open": return {"ok": True, "outcome": "cancelled" if self._state == "cancelled" else "closed"}
                    self._owner(caller)
                    if original:
                        if original.reply_id:
                            return {"ok": True, "outcome": "reply", "message": self._view(self._messages[original.reply_id]), "requestId": request_id}
                        if original.state in ("expired", "cancelled", "failed"):
                            return {"ok": True, "outcome": "request_failed", "requestId": request_id, "reason": original.reason}
                    pending = [m for m in self._pending(actor) if m.sequence > after_sequence]
                    if pending:
                        return {"ok": True, "outcome": "inbox_ready" if request_id else "message", "message": self._view(pending[0]), "nextSequence": pending[0].sequence}
                    remaining = end - self.clock()
                    if remaining <= 0: return {"ok": True, "outcome": "timeout", "requestId": request_id}
                    try: await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except TimeoutError: continue

    async def cancel(self):
        async with self._condition:
            self.authorize()
            if self._state == "sealed": return
            # Cancellation belongs to the shared root lineage, including children.
            self.budget._shared.cancelled = True
            self._expire()
            self._condition.notify_all()
