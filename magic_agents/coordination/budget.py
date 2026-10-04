"""Shared admission ledger for one root lineage and its narrowed child scopes.

Amounts in flight or with unknown billing remain exposure. Completing a request
releases a concurrency permit, but does not silently refund its unknown charge.
The durable backend must persist equivalent transitions with its owner fence.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable
from uuid import uuid4

from magic_agents.models.coordination import CoordinationLimits


class BudgetError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class UsageBound:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    image_jobs: int = 0
    messages: int = 0
    wakeups: int = 0
    cost: Decimal = Decimal(0)

    def __post_init__(self):
        for name in _RESOURCE_FIELDS:
            value = getattr(self, name)
            if name == "cost":
                if type(value) is not Decimal or not value.is_finite() or value < 0:
                    raise ValueError("Cost must be a finite nonnegative Decimal")
            elif type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


_RESOURCE_FIELDS = ("input_tokens", "output_tokens", "tool_calls", "image_jobs", "messages", "wakeups", "cost")
_CAPS = dict(input_tokens="max_input_tokens", output_tokens="max_output_tokens", tool_calls="max_tool_calls",
             image_jobs="max_image_jobs", messages="max_accepted_messages", wakeups="max_wakeups_per_workgroup", cost="max_cost")


def inherited_limits(requested: CoordinationLimits, parent: CoordinationLimits, *, require_complete=True) -> CoordinationLimits:
    """Only authored child fields narrow the already admitted parent policy."""
    values = parent.model_dump()
    for name in requested.model_fields_set:
        value = getattr(requested, name)
        if value is not None:
            values[name] = value
    return CoordinationLimits.model_validate(values).intersect_server_policy(parent, require_complete=require_complete)


def _zeros():
    return {name: Decimal(0) if name == "cost" else 0 for name in _RESOURCE_FIELDS}


@dataclass
class _Totals:
    spent: dict = field(default_factory=_zeros)
    reserved: dict = field(default_factory=_zeros)
    uncertain: dict = field(default_factory=_zeros)
    model_turns: int = 0
    active_models: int = 0
    active_jobs: int = 0


@dataclass
class _Reservation:
    scope: "WorkgroupBudget"
    estimate: UsageBound
    kind: str
    state: str = "active"
    actual: UsageBound | None = None
    priority: str = 'normal'
    queue_order: int = 0


@dataclass
class _Shared:
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    journal: dict[str, _Reservation] = field(default_factory=dict)
    tool_operations: set[tuple[str, str]] = field(default_factory=set)
    tool_operation_dispatches: set[tuple[str, str]] = field(default_factory=set)
    cancelled: bool = False
    exceeded: bool = False
    queue_order: int = 0
    priority_streak: int = 0


class WorkgroupBudget:
    def __init__(self, limits: CoordinationLimits, *, absolute_deadline: float | None = None,
                 clock: Callable[[], float] = time.time, _parent: "WorkgroupBudget | None" = None, invocation=False):
        # Admission needs finite server-supplied spending ceilings, even when
        # the authored graph omits those optional narrowing fields.
        self.invocation = _parent.invocation if _parent else invocation
        self._limits = limits.intersect_server_policy(limits, require_complete=not self.invocation)
        self.clock = _parent.clock if _parent else clock
        ceiling = self.clock() + self._limits.max_group_lifetime_seconds
        if absolute_deadline is not None:
            if type(absolute_deadline) not in (int, float) or not math.isfinite(absolute_deadline):
                raise ValueError("Absolute deadline must be finite")
            ceiling = min(ceiling, absolute_deadline)
        self.deadline = min(ceiling, _parent.deadline) if _parent else ceiling
        if not math.isfinite(self.deadline): raise ValueError("Absolute deadline must be finite")
        datetime.fromtimestamp(self.deadline, timezone.utc)  # must survive the public/checkpoint UTC format
        self.id = uuid4().hex
        self._shared = _parent._shared if _parent else _Shared()
        self._path = (*_parent._path, self) if _parent else (self,)
        self._totals = _Totals()

    def child(self, requested: CoordinationLimits) -> "WorkgroupBudget":
        return WorkgroupBudget(inherited_limits(requested, self._limits, require_complete=not self.invocation), _parent=self)

    @property
    def condition(self):
        return self._shared.condition

    def _open(self):
        if self._shared.cancelled: raise BudgetError("cancelled", "Root workgroup was cancelled")
        if self._shared.exceeded: raise BudgetError("budget_exhausted", "Observed usage exceeded its admitted bound")
        if self.clock() >= self.deadline: raise BudgetError("deadline_exceeded", "Original workgroup deadline expired")

    def _check_capacity(self, estimate: UsageBound, kind: str):
        for scope in self._path:
            totals = scope._totals
            if kind == "model" and scope._limits.max_model_turns is not None and totals.model_turns >= scope._limits.max_model_turns:
                raise BudgetError("budget_exhausted", "Model attempt allowance exhausted")
            for field, limit_name in _CAPS.items():
                cap = getattr(scope._limits, limit_name)
                if cap is None: continue
                cap = Decimal(cap.amount) if field == "cost" else cap
                exposure = totals.spent[field] + totals.reserved[field] + totals.uncertain[field]
                if exposure + getattr(estimate, field) > cap:
                    raise BudgetError("budget_exhausted", f"Shared {field} allowance exhausted")

    def _has_slot(self, kind):
        return all((scope._totals.active_models < scope._limits.max_concurrent_model_turns if kind == "model"
                    else scope._totals.active_jobs < scope._limits.max_concurrent_jobs) for scope in self._path)

    def allocate_messages_locked(self, *, accepted=1, reserved_replies=0, wakeups=0):
        """Part of the same scope/lineage lock transaction as mailbox acceptance."""
        if not self.condition.locked(): raise RuntimeError("Mailbox allocation requires the lineage lock")
        self._open()
        bound = UsageBound(messages=accepted + reserved_replies, wakeups=wakeups)
        self._check_capacity(bound, "quota")
        for scope in self._path:
            scope._totals.spent["messages"] += accepted
            scope._totals.reserved["messages"] += reserved_replies
            scope._totals.spent["wakeups"] += wakeups

    def resolve_reply_locked(self, *, accepted: bool):
        if not self.condition.locked(): raise RuntimeError("Reply allocation requires the lineage lock")
        if any(scope._totals.reserved["messages"] < 1 for scope in self._path):
            raise BudgetError("reservation_missing", "Reply capacity was not reserved")
        for scope in self._path:
            scope._totals.reserved["messages"] -= 1
            if accepted: scope._totals.spent["messages"] += 1

    def remaining_locked(self):
        if not self.condition.locked(): raise RuntimeError("Budget inspection requires the lineage lock")
        def remaining(field):
            values = []
            for scope in self._path:
                cap = getattr(scope._limits, _CAPS[field])
                if cap is None: continue
                cap = Decimal(cap.amount) if field == "cost" else cap
                used = sum(getattr(scope._totals, category)[field] for category in ("spent", "reserved", "uncertain"))
                values.append(max(0, cap - used))
            return min(values) if values else None
        turns = [scope._limits.max_model_turns - scope._totals.model_turns for scope in self._path
                 if scope._limits.max_model_turns is not None]
        return {"absoluteDeadline": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat().replace("+00:00", "Z"),
                "remainingMessages": remaining("messages"), "remainingWakeups": remaining("wakeups"),
                "remainingModelTurns": min(turns) if turns else None,
                "remainingInputTokens": remaining("input_tokens"), "remainingOutputTokens": remaining("output_tokens"),
                "remainingToolCalls": remaining("tool_calls"), "remainingImageJobs": remaining("image_jobs"),
                "remainingCost": ({"amount": str(remaining("cost")), "currency": self._limits.max_cost.currency}
                                  if self._limits.max_cost is not None else None)}

    async def charge_tool_call(self, operation_id: str, *, guard: Callable[[], None] = lambda: None,
                               new_guard: Callable[[], None] = lambda: None):
        """Read authority precedes replay; new-work authority precedes debit."""
        if type(operation_id) is not str or not 1 <= len(operation_id) <= 512:
            raise ValueError("A stable bounded tool operation identity is required")
        async with self.condition:
            guard()
            identity = (self.id, operation_id)
            if identity in self._shared.tool_operations: return
            new_guard()
            self._open()
            self._check_capacity(UsageBound(tool_calls=1), "quota")
            for scope in self._path: scope._totals.spent["tool_calls"] += 1
            self._shared.tool_operations.add(identity)

    async def reserve(self, operation_id: str, estimate: UsageBound, *, kind="model", guard: Callable[[], None] = lambda: None,
                      charged_tool_operation_id: str | None = None):
        """Reserve before physical dispatch; guard rechecks current owner/auth."""
        if type(operation_id) is not str or not 1 <= len(operation_id) <= 256 or kind not in ("model", "job"):
            raise ValueError("A bounded operation ID and admitted work kind are required")
        if not isinstance(estimate, UsageBound): raise TypeError("UsageBound required")
        credit = (self.id, charged_tool_operation_id) if charged_tool_operation_id is not None else None
        if credit is not None and (kind != 'job' or type(charged_tool_operation_id) is not str
                                   or not 1 <= len(charged_tool_operation_id) <= 512):
            raise ValueError('A charged invocation credit requires a bounded job identity')
        if kind == "job" and estimate.tool_calls + estimate.image_jobs < 1 and credit is None:
            raise ValueError("A job must reserve a bounded tool or image operation")
        async with self.condition:
            while True:
                guard(); self._open()
                if credit is not None and (credit not in self._shared.tool_operations
                                           or credit in self._shared.tool_operation_dispatches):
                    raise BudgetError('invalid_operation_credit', 'Invocation credit is absent, foreign or already dispatched')
                if operation_id in self._shared.journal:
                    raise BudgetError("operation_exists", "An already admitted physical operation cannot dispatch again")
                self._check_capacity(estimate, kind)
                if self._has_slot(kind): break
                try: await asyncio.wait_for(self.condition.wait(), timeout=max(0, self.deadline - self.clock()))
                except TimeoutError: raise BudgetError("deadline_exceeded", "Deadline expired while awaiting compute admission") from None
            self._shared.journal[operation_id] = _Reservation(self, estimate, kind)
            if credit is not None: self._shared.tool_operation_dispatches.add(credit)
            for scope in self._path:
                totals = scope._totals
                if kind == "model": totals.model_turns += 1; totals.active_models += 1
                else: totals.active_jobs += 1
                for name in _RESOURCE_FIELDS: totals.reserved[name] += getattr(estimate, name)

    def queue_job_locked(self, operation_id: str, estimate: UsageBound, *, max_queued: int, priority='normal'):
        """Reserve exposure immediately; queued work holds no compute permit."""
        if not self.condition.locked(): raise RuntimeError("Job acceptance requires the lineage lock")
        self._open()
        if type(operation_id) is not str or not 1 <= len(operation_id) <= 256:
            raise ValueError("A bounded job effect identity is required")
        if not isinstance(estimate, UsageBound) or estimate.tool_calls < 1 or estimate.messages or estimate.wakeups:
            raise ValueError("Job bounds include a tool operation; mailbox reservations are separate")
        if type(max_queued) is not int or max_queued < 1:
            raise ValueError("A positive finite queue bound is required")
        if priority not in ('normal', 'peer_request'):
            raise ValueError('Unknown job priority')
        if operation_id in self._shared.journal:
            raise BudgetError("operation_exists", "Job effect already reserved")
        if sum(entry.state == 'queued' for entry in self._shared.journal.values()) >= max_queued:
            raise BudgetError("job_queue_full", "Shared job queue is full")
        self._check_capacity(estimate, 'quota')
        self._shared.queue_order += 1
        self._shared.journal[operation_id] = _Reservation(self, estimate, 'job', state='queued',
            priority=priority, queue_order=self._shared.queue_order)
        for scope in self._path:
            for name in _RESOURCE_FIELDS: scope._totals.reserved[name] += getattr(estimate, name)
        self.condition.notify_all()

    def _next_queued_job(self):
        """FIFO within priority classes, at most two priority jobs per normal.

        The ordering is shared by sibling scopes. A narrower saturated child
        cannot block unrelated eligible work in the same root queue.
        """
        waiting = sorted(((key, entry) for key, entry in self._shared.journal.items()
                          if entry.state == 'queued' and entry.scope._has_slot('job')),
                         key=lambda pair: pair[1].queue_order)
        normal = next((pair for pair in waiting if pair[1].priority == 'normal'), None)
        priority = next((pair for pair in waiting if pair[1].priority == 'peer_request'), None)
        selected = priority if priority and (normal is None or self._shared.priority_streak < 2) else normal
        return selected[0] if selected else None

    async def start_queued_job(self, operation_id: str, *, guard: Callable[[], None]):
        async with self.condition:
            while True:
                guard(); self._open()
                entry = self._shared.journal.get(operation_id)
                if entry is None or entry.scope is not self or entry.state != 'queued':
                    raise BudgetError("operation_exists", "Job is not an owned queued operation")
                if self._has_slot('job') and self._next_queued_job() == operation_id:
                    entry.state = 'active'
                    self._shared.priority_streak = self._shared.priority_streak + 1 if entry.priority == 'peer_request' else 0
                    for scope in self._path: scope._totals.active_jobs += 1
                    self.condition.notify_all()
                    return
                try: await asyncio.wait_for(self.condition.wait(), timeout=max(0, self.deadline - self.clock()))
                except TimeoutError: continue

    def rollback_unaccepted_job_locked(self, operation_id: str):
        """Undo only a same-transaction queue allocation before its receipt."""
        if not self.condition.locked(): raise RuntimeError('Queue rollback requires the lineage lock')
        entry = self._shared.journal.get(operation_id)
        if entry is None or entry.scope is not self or entry.state != 'queued':
            raise BudgetError('rollback_conflict', 'Only an undispatched unaccepted job can roll back')
        for scope in self._path:
            for name in _RESOURCE_FIELDS: scope._totals.reserved[name] -= getattr(entry.estimate, name)
        del self._shared.journal[operation_id]

    async def settle(self, operation_id: str, actual: UsageBound | None, *, uncertain=False):
        """Settle once even after cancellation; unknown usage retains exposure."""
        if actual is not None and not isinstance(actual, UsageBound): raise TypeError("UsageBound required")
        if not uncertain and actual is None: raise ValueError("Known completion needs actual usage")
        async with self.condition:
            entry = self._shared.journal.get(operation_id)
            if entry is None or entry.scope is not self: raise BudgetError("unknown_operation", "Operation is not owned by this scope")
            state = "uncertain" if uncertain else "settled"
            if entry.state not in ("active", "queued"):
                if entry.state == state and entry.actual == actual: return
                raise BudgetError("settlement_conflict", "Operation has a different recorded settlement")
            bound = entry.estimate
            exceeded = not self.invocation and actual is not None and any(getattr(actual, name) > getattr(bound, name) for name in _RESOURCE_FIELDS)
            for scope in self._path:
                totals = scope._totals
                if entry.kind == "model": totals.active_models -= 1
                elif entry.state == 'active': totals.active_jobs -= 1
                for name in _RESOURCE_FIELDS:
                    totals.reserved[name] -= getattr(bound, name)
                    known = getattr(actual, name) if actual else 0
                    totals.spent[name] += known
                    if uncertain: totals.uncertain[name] += max(0, getattr(bound, name) - known)
            entry.state, entry.actual = state, actual
            if self.invocation:
                try: self._check_capacity(UsageBound(), 'quota')
                except BudgetError: exceeded = True
            self._shared.exceeded |= exceeded
            self.condition.notify_all()
            if exceeded: raise BudgetError("usage_bound_exceeded", "Actual usage exceeded the reserved upper bound; further admission stopped")

    async def reconcile(self, operation_id: str, actual: UsageBound):
        """Replace unknown exposure only with an authoritative final total."""
        if not isinstance(actual, UsageBound): raise TypeError("UsageBound required")
        async with self.condition:
            entry = self._shared.journal.get(operation_id)
            if entry is None or entry.scope is not self: raise BudgetError("unknown_operation", "Operation is not owned by this scope")
            if entry.state == "settled" and entry.actual == actual: return
            if entry.state != "uncertain": raise BudgetError("settlement_conflict", "Only unknown operations can be reconciled")
            previous = entry.actual or UsageBound()
            if any(getattr(actual, name) < getattr(previous, name) for name in _RESOURCE_FIELDS):
                raise BudgetError("settlement_conflict", "Reconciliation cannot erase already observed usage")
            exceeded = any(getattr(actual, name) > getattr(entry.estimate, name) for name in _RESOURCE_FIELDS)
            for scope in self._path:
                for name in _RESOURCE_FIELDS:
                    scope._totals.uncertain[name] -= max(0, getattr(entry.estimate, name) - getattr(previous, name))
                    scope._totals.spent[name] += getattr(actual, name) - getattr(previous, name)
            entry.state, entry.actual = "settled", actual
            self._shared.exceeded |= exceeded
            self.condition.notify_all()
            if exceeded: raise BudgetError("usage_bound_exceeded", "Reconciled usage exceeded the reserved bound")

    async def cancel(self):
        async with self.condition:
            self._shared.cancelled = True
            self.condition.notify_all()

    async def snapshot(self):
        async with self.condition:
            totals = self._totals
            def values(data): return {k: str(v) if isinstance(v, Decimal) else v for k, v in data.items()}
            return {"absoluteDeadline": self.deadline, "modelTurns": totals.model_turns,
                    "activeModels": totals.active_models, "activeJobs": totals.active_jobs,
                    "spent": values(totals.spent), "reserved": values(totals.reserved), "uncertain": values(totals.uncertain),
                    "cancelled": self._shared.cancelled, "exceeded": self._shared.exceeded}
