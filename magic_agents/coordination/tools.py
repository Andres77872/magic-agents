"""Native coordination tools bound to one runtime-issued actor capability.

These are ordinary provider tools with complete tool results. Mailbox delivery
and consumption remain owned by the awaited loop control, never these wrappers.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Annotated, Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from magic_agents.coordination.service import ActorCaller, CoordinationError
from magic_agents.coordination.budget import BudgetError
from magic_agents.models.coordination import BUILTIN_NAMES
from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL, ToolExecutor
from magic_llm.engine.tooling import _inline_local_refs

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
Text = Annotated[str, Field(max_length=65536)]
JOB_TOOL_NAMES = ('startToolJob', 'inspectWork', 'waitForWork', 'cancelWork')
RESERVED_TOOL_NAMES = (*BUILTIN_NAMES, *JOB_TOOL_NAMES)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class _Empty(_Arguments):
    pass


class _Target(_Arguments):
    agentRef: Identifier


class _SendOptions(_Arguments):
    kind: Literal['request', 'event'] = 'request'
    expectReply: bool | None = None
    wakeIfCompleted: bool = False
    idempotencyKey: Identifier | None = None
    expiresInSeconds: float | None = Field(default=None, gt=0)
    expectedActivationId: Identifier | None = None
    expectedEpochId: Identifier | None = None


class _Send(_Target):
    message: Text
    payload: JsonValue = None
    options: _SendOptions = Field(default_factory=_SendOptions)


class _ReplyOptions(_Arguments):
    outcome: Literal['success', 'error'] = 'success'
    errorCode: str | None = Field(default=None, min_length=1, max_length=128)
    idempotencyKey: Identifier | None = None

    @model_validator(mode='after')
    def error_outcome(self):
        if self.errorCode is not None and self.outcome != 'error':
            raise ValueError('errorCode requires an error outcome')
        return self


class _Reply(_Arguments):
    requestId: Identifier
    message: Text
    payload: JsonValue = None
    options: _ReplyOptions = Field(default_factory=_ReplyOptions)


class _WaitMessage(_Arguments):
    requestId: Identifier | None = None
    afterSequence: int = Field(default=0, ge=0)
    timeoutSeconds: float = Field(gt=0)


class _WaitAgentOptions(_Arguments):
    activationId: Identifier | None = None
    requestId: Identifier | None = None
    until: Literal['activation_finished', 'request_resolved']
    timeoutSeconds: float = Field(gt=0)

    @model_validator(mode='after')
    def exact_guard(self):
        if self.until == 'activation_finished' and (not self.activationId or self.requestId is not None):
            raise ValueError('Activation wait requires only an activationId')
        if self.until == 'request_resolved' and (not self.requestId or self.activationId is not None):
            raise ValueError('Request wait requires only a requestId')
        return self


class _WaitAgent(_Target):
    options: _WaitAgentOptions


_SPECS = {
    'listAgents': (_Empty, 'Discover only registered peers visible in this scope; this does not start them.'),
    'sendMessageToAgent': (_Send, 'Send bounded peer input and receive an acceptance receipt. Accepted does not mean consumed or completed. Request and wait in separate turns using the returned requestId.'),
    'inspectAgent': (_Target, 'Read fresh peer status and effective limits without accessing private history. Quiescent output remains tentative.'),
    'replyToAgent': (_Reply, 'Resolve one exact incoming request with a final success or error. The runtime derives its destination.'),
    'waitForMessage': (_WaitMessage, 'Wait with a finite timeout for one request or the next inbox sequence. This returns a tool result before canonical mailbox delivery and never acknowledges consumption.'),
    'waitForAgent': (_WaitAgent, 'Wait with a finite timeout for an exact activationId or requestId. A dependency on a new send requires its receipt in a later model turn.'),
}


def _schema(name):
    schema = _SPECS[name][0].model_json_schema()
    if 'payload' in schema.get('properties', {}):
        # JSON arriving from a provider is intrinsically JSON-shaped. Runtime
        # JsonValue/service validation still enforces finite, bounded data.
        # Avoid recursive $refs: the current Gemini mapper expands local refs.
        schema['properties']['payload'] = {'description': 'Optional bounded inline JSON data; prefer owned resource references for large content.'}
        schema.get('$defs', {}).pop('JsonValue', None)
    # Expose nested options as concrete objects to native provider tools. The
    # runtime still validates the original strict models without JSON coercion.
    return _inline_local_refs(schema)


def reserve_builtin_names(existing_names: Iterable[str]) -> None:
    """Reserve the entire namespace, including configured-but-disabled names."""
    collision = set(existing_names).intersection(RESERVED_TOOL_NAMES)
    if collision:
        raise CoordinationError('reserved_tool_name', 'Coordination builtin name collision: ' + ', '.join(sorted(collision)))


def _operation_key(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    current = CURRENT_TOOL_CALL.get()
    identifier = getattr(current, 'id', None)
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 251:
        raise CoordinationError('missing_operation_identity', 'Native coordination requires a bounded runtime tool-call ID')
    return 'tool:' + identifier


def _utc(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z')


class CoordinationTools:
    """Run-local schemas and callables for one immutable actor binding.

    Configure the final run-local executor before invoking any tools. Its
    existing size limits are respected; an oversized observation returns a
    complete error envelope while leaving mailbox consumption untouched.
    """

    def __init__(self, caller: ActorCaller, *, existing_names: Iterable[str] = ()):
        reserve_builtin_names(existing_names)
        self._caller = caller
        actor = caller.service._owner(caller)
        self._names = tuple(actor.config.tools)
        self._output_limits = {name: 50000 for name in self._names}
        self.tool_functions = {name: self._callable(name) for name in self._names}
        self.tools = [{'type': 'function', 'function': {
            'name': name, 'description': _SPECS[name][1],
            'parameters': _schema(name),
        }} for name in self._names]

    def configure_executor(self, executor: ToolExecutor) -> None:
        # Check before merge; permit configuring a second time only for these
        # exact callable objects, never a foreign actor's captured capability.
        existing = {name for name in executor.registered_names()
                    if executor._registry[name] is not self.tool_functions.get(name)
                    and getattr(executor._registry[name], '_coordination_binding', None) != (id(self._caller.service), self._caller.actor_id)}
        reserve_builtin_names(existing)
        limits = {name: executor.content_limit(name) for name in self._names}
        if any(type(limit) is not int or limit < 2048 for limit in limits.values()):
            raise CoordinationError('tool_output_limit', 'Coordination tools require room for a complete 2048-character receipt envelope')
        self._output_limits.update(limits)
        executor.exclude_from_dedup(*self._names)
        executor.require_complete_output(*self._names)
        executor.serialize_tools(*(name for name in ('sendMessageToAgent', 'replyToAgent') if name in self._names))

    def _callable(self, name):
        async def invoke(**arguments):
            try:
                current_key = _operation_key(None)
                await self._caller.service.budget.charge_tool_call(
                    self._caller.actor_id + ':' + current_key,
                    guard=lambda: self._caller.service._access(self._caller),
                    new_guard=lambda: self._caller.service.admit_tool_locked(self._caller))
                parsed = _SPECS[name][0].model_validate(arguments)
                result = await self._dispatch(name, parsed)
            except ValidationError:
                result = {'ok': False, 'error': {'code': 'invalid_arguments', 'message': 'Invalid coordination tool arguments'}}
            except CoordinationError as error:
                result = error.as_result()
            except BudgetError as error:
                result = {'ok': False, 'error': {'code': error.code, 'message': str(error)}}
            return self._bounded(name, result)

        invoke.__name__ = name
        invoke.__doc__ = _SPECS[name][1]
        invoke._disable_dedup = True
        invoke._require_complete_output = True
        invoke._coordination_binding = (id(self._caller.service), self._caller.actor_id)
        return invoke

    def _bounded(self, name, result):
        result = copy.deepcopy(result)
        # Optional transport metadata is omitted, while null inside a caller's
        # JSON payload remains meaningful data and must not be rewritten.
        envelopes = [result, result.get('receipt'), result.get('currentDisposition'), result.get('message')]
        envelopes.extend(result.get('peers', []))
        for envelope in envelopes:
            if isinstance(envelope, dict):
                for key in list(envelope):
                    if key != 'payload' and envelope[key] is None:
                        del envelope[key]
        receipt = result.get('receipt')
        if receipt and type(receipt.get('expiresAt')) in (int, float):
            receipt['expiresAt'] = _utc(receipt['expiresAt'])
        if len(ToolExecutor.serialize_output(result)) <= self._output_limits[name]:
            return result
        # Receipts remain small by service contract. Do not report a successful
        # mutation as rejected merely because some observation became too big.
        if receipt:
            compact = {'ok': result['ok'], 'receipt': receipt}
            if 'currentDisposition' in result:
                compact['currentDisposition'] = result['currentDisposition']
            if len(ToolExecutor.serialize_output(compact)) <= self._output_limits[name]:
                return compact
            raise CoordinationError('tool_output_limit', 'Executor cannot retain the accepted operation receipt')
        compact = {'ok': False, 'error': {'code': 'tool_output_limit',
                   'message': 'Observation exceeds the tool result limit; mailbox input remains available at the next safe boundary'}}
        if result.get('requestId'):
            compact['requestId'] = result['requestId']
        return compact

    async def _dispatch(self, name, args):
        service, caller = self._caller.service, self._caller
        if name == 'listAgents':
            return await service.list_agents(caller)
        if name == 'inspectAgent':
            return await service.inspect(caller, args.agentRef)
        if name == 'sendMessageToAgent':
            options = args.options
            return await service.send(caller, args.agentRef, args.message, args.payload,
                key=_operation_key(options.idempotencyKey), kind=options.kind,
                expect_reply=options.expectReply if options.expectReply is not None else options.kind == 'request',
                wake=options.wakeIfCompleted, expires_in=options.expiresInSeconds,
                expected_epoch=options.expectedEpochId, expected_activation=options.expectedActivationId)
        if name == 'replyToAgent':
            return await service.reply(caller, args.requestId, args.message, args.payload,
                key=_operation_key(args.options.idempotencyKey), outcome=args.options.outcome,
                error_code=args.options.errorCode)
        if name == 'waitForMessage':
            result = await service.wait_message(caller, request_id=args.requestId,
                after_sequence=args.afterSequence, timeout=args.timeoutSeconds)
            result.setdefault('nextSequence', result.get('message', {}).get('sequence', args.afterSequence))
            return result
        if name == 'waitForAgent':
            options = args.options
            result = await service.wait_agent(caller, args.agentRef, until=options.until,
                timeout=options.timeoutSeconds, activation_id=options.activationId, request_id=options.requestId)
            view = await service.inspect(caller, args.agentRef)
            result.setdefault('state', view['state'])
            result.setdefault('outputRevision', view['outputRevision'])
            if result.get('outcome') == 'request_failed':
                result['outcome'], result['requestOutcome'] = 'target_failed', 'error'
            elif result.get('message', {}).get('outcome'):
                result['requestOutcome'] = result['message']['outcome']
            return result
        raise CoordinationError('unsupported_tool', 'Unknown coordination builtin')


def build_coordination_tools(caller: ActorCaller, *, existing_names: Iterable[str] = ()) -> CoordinationTools:
    return CoordinationTools(caller, existing_names=existing_names)
