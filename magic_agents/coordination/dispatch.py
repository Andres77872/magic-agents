"""Trusted physical-operation admission below graph and native-tool wrappers.

Private effective requests reach host policy unchanged. The journal keeps only
bounded metadata and result digests, never credentials or transport objects.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from uuid import uuid4
from weakref import WeakSet

from pydantic import BaseModel

from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.service import CoordinationError, _json


@dataclass(frozen=True)
class ExternalOutcome:
    status: str
    result: object = field(default=None, repr=False)
    error_type: str | None = None
    uncertain: bool = False


@dataclass
class _Credit:
    budget: object
    operation_id: str
    available: bool = True


_CREDIT = ContextVar('coordination_dispatch_invocation_credit', default=None)
_CALLABLES = WeakSet()


def register_dispatch_callable(function):
    """Package-only registration; authored scalar flags cannot bypass admission."""
    _CALLABLES.add(function)
    function._disable_dedup = True
    return function


def dispatches_externally(value):
    from magic_agents.hooks.invocation_control import ControlledTool
    if isinstance(value, ControlledTool):
        return dispatches_externally(value.function)
    try:
        if value in _CALLABLES: return True
    except TypeError:
        pass
    from magic_agents.node_system.NodeFetch import NodeFetch
    from magic_agents.node_system.NodeMemory import NodeMemory
    from magic_agents.node_system.NodeMcp import NodeMcp
    from magic_agents.node_system.NodeConditional import NodeConditional
    return isinstance(value, (NodeFetch, NodeMemory, NodeMcp)) or (
        isinstance(value, NodeConditional) and value.evaluation_mode in ('llm', 'jev'))


@asynccontextmanager
async def invocation_charge(scope, operation_id, guard):
    """Count the wrapper once without holding a slot needed by its subrequest."""
    await scope.budget.charge_tool_call(operation_id, guard=scope.runtime.authorize, new_guard=guard)
    token = _CREDIT.set(_Credit(scope.budget, operation_id))
    try:
        guard()
        yield
        guard()
    finally:
        _CREDIT.reset(token)


def _portable(value):
    if isinstance(value, BaseModel): return value.model_dump(mode='json')
    if isinstance(value, list): return [_portable(item) for item in value]
    if isinstance(value, tuple): return [_portable(item) for item in value]
    if isinstance(value, dict): return {key: _portable(item) for key, item in value.items()}
    return value


def _encoded(value, *, max_bytes=4 * 1024 * 1024):
    # Embedding vectors can exceed mailbox's 2048-item collection ceiling.
    # Keep a separate bounded private wire-data codec, never a public payload.
    def check(item, depth=0):
        if depth > 32: raise ValueError('Private request nesting limit')
        if item is None or type(item) in (str, bool, int, float): return
        if type(item) not in (list, dict) or len(item) > 200000:
            raise ValueError('Private request collection limit')
        if isinstance(item, dict) and any(type(key) is not str for key in item):
            raise ValueError('Private request keys must be strings')
        for child in item.values() if isinstance(item, dict) else item: check(child, depth + 1)
    try:
        check(value)
        encoded=json.dumps(value,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    except (ValueError, TypeError, UnicodeError) as error:
        raise CoordinationError('invalid_external_payload', 'External request or result is not bounded finite JSON') from error
    if len(encoded)>max_bytes: raise CoordinationError('external_payload_limit', 'External request or result exceeds its byte ceiling')
    return encoded


def is_protected(error):
    from magic_agents.hooks.invocation_control import _protected_failure
    return _protected_failure(error)


class DispatchSession:
    """One invocation's immutable owner authority and physical dispatch IDs."""
    def __init__(self, scope, node_id):
        self.scope, self.path = scope, scope.path + (node_id,)
        self.guard = scope.capture_guard(node_id)
        self.guard()
        self.id, self.sequence = uuid4().hex, 0
        runtime = scope.runtime
        self.authorize = getattr(runtime, 'authorize_external', None)
        self.estimate = getattr(runtime, 'external_estimate', None)
        self.usage = getattr(runtime, 'external_usage', None)
        if not all(callable(callback) for callback in (self.authorize, self.estimate, self.usage)):
            raise CoordinationError('external_admission_unavailable', 'External work requires trusted resource and usage adapters')
        if not hasattr(scope, '_external_operations'): scope._external_operations = {}
        self.journal = scope._external_operations

    async def call(self, kind, request, operation, *, operation_id=None, admission_guard=None):
        """A private admission guard raises on denial and returns only None on success."""
        if not isinstance(kind, str) or not 1 <= len(kind) <= 128:
            raise CoordinationError('invalid_external_operation', 'Operation kind must be a bounded name')
        if not callable(operation) or not inspect.iscoroutinefunction(operation):
            raise CoordinationError('unsupported_external_operation', 'External operations require cancellable async dispatch')
        if not isinstance(request, dict):
            raise CoordinationError('invalid_external_operation', 'Effective request must be a private JSON object')
        if admission_guard is not None and not callable(admission_guard):
            raise CoordinationError('invalid_external_adapter', 'Physical admission guard must be a trusted synchronous callable')
        request = json.loads(_encoded(request))
        self.guard()
        self.sequence += 1
        identity = operation_id if operation_id is not None else f'external:{self.id}:{self.sequence}'
        if type(identity) is not str or not 1 <= len(identity) <= 256:
            raise CoordinationError('operation_identity_required', 'Physical dispatch identity must be bounded')
        digest = hashlib.sha256(_encoded({'path': list(self.path), 'kind': kind, 'request': request})).hexdigest()
        if identity in self.journal:
            # A completed physical effect is never automatically dispatched a
            # second time. Logical child replay belongs to the parent journal.
            raise CoordinationError('operation_unresolved', 'Physical operation already exists; reconcile or replay its owning invocation')
        if len(self.journal) >= 10000:
            raise CoordinationError('external_operation_limit', 'Physical operation journal capacity exhausted')
        def authorize():
            self.guard()
            if admission_guard is not None:
                admitted = admission_guard()
                if inspect.isawaitable(admitted):
                    if inspect.iscoroutine(admitted): admitted.close()
                    raise CoordinationError('invalid_external_adapter', 'Physical admission guard must complete synchronously')
                if admitted is not None:
                    raise CoordinationError('invalid_external_adapter', 'Physical admission guard must raise on denial and return None on success')
            result = self.authorize(self.path, kind, copy.deepcopy(request))
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result): result.close()
                raise CoordinationError('invalid_external_adapter', 'Resource authorization must complete synchronously')
        authorize()
        estimate = self.estimate(self.path, kind, copy.deepcopy(request))
        if inspect.iscoroutine(estimate): estimate.close()
        if not isinstance(estimate, UsageBound) or estimate.tool_calls < 1 or estimate.messages or estimate.wakeups:
            raise CoordinationError('usage_bound_unavailable', 'Every external dispatch requires a finite physical-operation bound')
        credit = _CREDIT.get()
        charged = None
        if credit is not None and credit.budget is self.scope.budget and credit.available:
            # No await between testing and consuming the shared object: sibling
            # tasks inherit one credit, not independent copies of an allowance.
            credit.available = False
            charged = credit.operation_id
            estimate = replace(estimate, tool_calls=estimate.tool_calls - 1)
        entry = {'operationId': identity, 'kind': kind, 'nodePath': list(self.path),
                 'requestDigest': digest, 'state': 'admitting'}
        self.journal[identity] = entry
        try:
            await self.scope.budget.reserve(identity, estimate, kind='job', guard=authorize,
                                            charged_tool_operation_id=charged)
        except BaseException:
            entry['state'] = 'not_dispatched'
            raise
        entry['state'] = 'running'
        result, error, actual, started = None, None, None, False
        try:
            authorize()
            async with asyncio.timeout(max(0, self.scope.budget.deadline - self.scope.budget.clock())):
                started = True
                result = await operation()
            _encoded(_portable(result))
            self.guard()
        except BaseException as exc:
            error = exc
        outcome = ExternalOutcome('cancelled' if isinstance(error, asyncio.CancelledError) else 'failed' if error else 'completed',
                                  result, type(error).__name__ if error else None, error is not None and started)
        usage_error = None
        try:
            actual = self.usage(self.path, kind, copy.deepcopy(request), outcome) if started else UsageBound()
            if inspect.iscoroutine(actual): actual.close()
            if actual is not None and (not isinstance(actual, UsageBound) or started and actual.tool_calls < 1):
                raise ValueError('Usage must account for the physical operation')
            if actual is not None and started and charged is not None: actual = replace(actual, tool_calls=actual.tool_calls - 1)
        except Exception as exc:
            usage_error = exc
            actual = None
        uncertain = outcome.uncertain or actual is None
        entry['state'] = 'aborted' if not started else 'unknown' if uncertain else 'completed'
        if not error:
            entry['resultDigest'] = hashlib.sha256(_encoded(_portable(result))).hexdigest()
        settlement = asyncio.create_task(self.scope.budget.settle(identity, actual, uncertain=uncertain))
        interrupted = None
        while not settlement.done():
            try: await asyncio.shield(settlement)
            except asyncio.CancelledError as exc: interrupted = exc
        settlement.result()
        if interrupted is not None: raise interrupted
        if error is not None:
            if is_protected(error): raise error
            raise CoordinationError('external_outcome_unknown', 'External operation may have occurred; reconciliation is required') from error
        if usage_error is not None:
            raise CoordinationError('usage_unavailable', 'External usage reconciliation failed; exposure retained') from usage_error
        return result
