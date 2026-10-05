"""Awaited native LLM and tool recovery boundaries; no provider reconstruction."""
from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import uuid
from contextlib import aclosing

from anyio import CancelScope

from .recorder import current_node, current_scope
from .storage import (EffectRecord, ExecutionStorageError, ExecutionUsageObservation,
                      ExecutionUsageReceipt, canonical_bytes)


class CoreAttemptControl:
    def __init__(self, scope, node, delegate=None):
        self.scope, self.node, self.delegate = scope, node, delegate
        self.attempts = {}

    async def before_attempt(self, attempt):
        self.node.active_response_id = attempt.provider_attempt_id
        if self.delegate is not None:
            await self.delegate.before_attempt(attempt)
        request = {'provider': attempt.provider, 'model': attempt.model,
                   'messages': list(attempt.messages), 'generation_options': attempt.generation_options,
                   'stream': attempt.stream, 'request_operation_id': attempt.request_operation_id,
                   'parent_attempt_id': getattr(attempt, 'parent_attempt_id', None),
                   'attempt_index': getattr(attempt, 'attempt_index', None),
                   'retry_index': getattr(attempt, 'retry_index', None),
                   'is_fallback': getattr(attempt, 'is_fallback', None),
                   'request_features': list(getattr(attempt, 'request_features', ()))}
        effect = EffectRecord(effect_id=attempt.provider_attempt_id,
            attempt_id=attempt.provider_attempt_id, kind='provider', status='prepared',
            request_digest=hashlib.sha256(canonical_bytes(request)).hexdigest(), cause=self.node.cause)
        await self.scope.record('effect_prepared', {'effect_id': effect.effect_id,
            'execution_id': attempt.provider_attempt_id, 'parent_execution_id': self.node.execution_id,
            'request': request}, node_id=self.node.node_id, effect=effect)
        effect = effect.model_copy(update={'status': 'dispatched'})
        self.attempts[attempt.provider_attempt_id] = effect
        await self.scope.record('llm_start', {'execution_id': attempt.provider_attempt_id,
            'parent_execution_id': self.node.execution_id, 'node_id': self.node.node_id,
            'provider': attempt.provider, 'model': attempt.model, 'request_digest': effect.request_digest},
            node_id=self.node.node_id, effect=effect)

    async def after_attempt(self, attempt, outcome):
        with CancelScope(shield=True):
            delegate_error = None
            if self.delegate is not None:
                try:
                    await self.delegate.after_attempt(attempt, outcome)
                except BaseException as error:
                    delegate_error = error
            effect = self.attempts[attempt.provider_attempt_id]
            state = 'succeeded' if outcome.status == 'completed' else 'unknown'
            effect = effect.model_copy(update={'status': state})
            usage = outcome.usage
            if hasattr(usage, 'model_dump'):
                usage = usage.model_dump(mode='json')
            try:
                await self.scope.record('llm_end', {'execution_id': attempt.provider_attempt_id,
                    'parent_execution_id': self.node.execution_id, 'node_id': self.node.node_id,
                    'provider': attempt.provider, 'model': attempt.model,
                    'status': outcome.status, 'usage': usage, 'usage_uncertain': outcome.usage_uncertain,
                    'semantic_output': outcome.semantic_output, 'error_code': outcome.error_code},
                    node_id=self.node.node_id, effect=effect)
            except BaseException as original_error:
                denied = (getattr(original_error, 'code', None) in {
                    'run_closed', 'version_conflict', 'lease_lost', 'stale_lease'}
                    or isinstance(original_error, asyncio.CancelledError)
                    and self.scope.recorder.state.status == 'cancelled')
                final_usage = (outcome.status == 'completed' and not outcome.usage_uncertain
                    and type(usage) is dict and all(type(usage.get(key)) is int
                        and 0 <= usage[key] <= 2**31 - 1
                        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')))
                if denied and final_usage:
                    try:
                        observation = ExecutionUsageObservation(attempt_id=attempt.provider_attempt_id,
                            request_digest=effect.request_digest, provider=attempt.provider,
                            model=attempt.model, usage=usage)
                        identity = self.scope.recorder.snapshot.identity
                        receipt = await self.scope.recorder.store.record_usage(identity, observation)
                        if (not isinstance(receipt, ExecutionUsageReceipt)
                                or receipt.identity != identity or receipt.attempt_id != observation.attempt_id
                                or receipt.request_digest != observation.request_digest
                                or receipt.usage_digest != observation.usage_digest):
                            raise ExecutionStorageError('usage_receipt_invalid', 'Usage receipt does not match its original attempt')
                    except BaseException as accounting_error:
                        # The original Stop/ownership failure remains primary.
                        # No raw usage, request, credential or driver text leaks.
                        original_error.add_note('Late usage reconciliation failed: ' + type(accounting_error).__name__)
                raise
            if delegate_error is not None:
                raise delegate_error


class CoreLoopControl:
    def __init__(self, scope, node, delegate=None):
        self.scope, self.node, self.delegate = scope, node, delegate
        self.response_capacity_bytes = scope.recorder.store.limits.max_state_bytes
        self.pending_response = (node.saved or {}).get('llm_response')

    async def restore_response(self, checkpoint, *, stream):
        if self.pending_response is None:
            return None
        saved = self.pending_response
        self.pending_response = None
        if canonical_bytes(saved['checkpoint']) != canonical_bytes(checkpoint.model_dump(mode='json')):
            raise ExecutionStorageError('execution_response_conflict', 'Confirmed response no longer matches canonical input')
        self.node.active_response_id = saved.get('physical_attempt_id')
        return saved['response']

    async def response_ready(self, checkpoint, response):
        saved = {'checkpoint': checkpoint.model_dump(mode='json'), 'response': response,
                 'physical_attempt_id': self.node.active_response_id}
        digest = hashlib.sha256(canonical_bytes(saved)).hexdigest()
        await self.scope.record('llm_response', {'execution_id': self.node.execution_id,
            'response_digest': digest}, node_id=self.node.node_id, private_state=saved)

    async def before_turn(self, checkpoint):
        if self.pending_response is not None:
            return []
        return await self.delegate.before_turn(checkpoint) if self.delegate is not None else []

    async def checkpoint(self, checkpoint, boundary):
        if self.delegate is not None:
            await self.delegate.checkpoint(checkpoint, boundary)
        value = checkpoint.model_dump(mode='json')
        await self.scope.record('llm_checkpoint', {'execution_id': self.node.execution_id,
            'boundary': boundary, 'checkpoint_digest': hashlib.sha256(canonical_bytes(value)).hexdigest()},
            node_id=self.node.node_id, private_state={'boundary': boundary, 'checkpoint': value, 'physical_attempt_id': self.node.active_response_id})

    async def finish_candidate(self, checkpoint):
        return await self.delegate.finish_candidate(checkpoint) if self.delegate is not None else 'candidate_ready'



def _retained_tool_ids(saved):
    packet = (saved or {}).get('llm_response', {}).get('response', {})
    frames = packet.get('chunks', []) if packet.get('kind') == 'stream' else [packet.get('response', {})]
    return {call['id'] for frame in frames for choice in frame.get('choices', [])
            for call in (choice.get('delta', choice.get('message', {})).get('tool_calls') or [])
            if isinstance(call, dict) and isinstance(call.get('id'), str)}


def _tool(scope, node, name, function):
    @functools.wraps(function)
    async def call(*args, **kwargs):
        from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL
        canonical = CURRENT_TOOL_CALL.get()
        tool_call_id = getattr(canonical, 'id', None)
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ExecutionStorageError('execution_tool_identity_missing', 'Native tool replay requires the original tool-call ID')
        span = hashlib.sha256(canonical_bytes([scope.instance_id, node.execution_id, tool_call_id])).hexdigest()
        request = {'name': name, 'args': scope.recorder.encode(args),
                   'kwargs': scope.recorder.encode(kwargs), 'tool_call_id': tool_call_id}
        request_digest = hashlib.sha256(canonical_bytes(request)).hexdigest()
        local = bool(getattr(function, '_native_coordination_local', False))
        kind = 'coordinator' if local else 'tool'
        db_local = local and any(item.node_path == scope.path and item.messaging_engine == 'db_persistence' for item in scope.recorder.snapshot.coordination_scopes)
        retained = next((item for item in scope.recorder.state.effects if item.effect_id == span), None)
        from .recovery import volatile_scope_lost
        lost = local and volatile_scope_lost(scope) and (retained is not None or tool_call_id in _retained_tool_ids(node.saved))
        if retained is not None:
            if retained.kind != kind or retained.request_digest != request_digest:
                raise ExecutionStorageError('execution_tool_conflict', 'Original tool identity has different arguments')
            if retained.status == 'succeeded':
                from .recovery import decode_value
                return decode_value(retained.result, scope)
            if retained.status != 'prepared' and not (db_local or lost):
                raise ExecutionStorageError('external_effect_unresolved', 'Original tool effect is not safely repeatable')
            effect = retained
        else:
            effect = EffectRecord(effect_id=span, attempt_id=span, kind=kind, status='prepared',
                request_digest=request_digest, cause=node.cause)
            await scope.record('effect_prepared', {'execution_id': span, 'parent_execution_id': node.execution_id,
                'effect_id': effect.effect_id, 'request': request}, node_id=node.node_id, effect=effect)
        replaying = retained is not None and retained.status in {'dispatched', 'unknown'}
        effect = effect if replaying else effect.model_copy(update={'status': 'dispatched'})
        await scope.record('tool_resume' if replaying else 'tool_start', {'execution_id': span, 'parent_execution_id': node.execution_id,
            'effect_id': effect.effect_id, 'name': name, 'request_digest': effect.request_digest}, node_id=node.node_id, effect=effect)
        try:
            if lost:
                result = {'ok': False, 'error': {'code': 'coordination_state_lost',
                    'message': 'Volatile messaging state was lost during interruption'}}
            elif inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(getattr(function, '__call__', None)):
                result = await function(*args, **kwargs)
            else:
                import asyncio
                result = await asyncio.to_thread(function, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
        except BaseException as error:
            with CancelScope(shield=True):
                effect = effect.model_copy(update={'status': 'unknown'})
                await scope.record('tool_end', {'execution_id': span, 'parent_execution_id': node.execution_id,
                    'name': name, 'status': 'cancelled' if type(error).__name__ in ('CancelledError', 'GeneratorExit') else 'failed',
                    'error_type': type(error).__name__}, node_id=node.node_id, effect=effect)
            raise
        result_value = scope.recorder.encode(result)
        effect = effect.model_copy(update={'status': 'succeeded', 'result': result_value,
            'result_digest': hashlib.sha256(canonical_bytes(result_value)).hexdigest()})
        await scope.record('tool_end', {'execution_id': span, 'parent_execution_id': node.execution_id,
            'name': name, 'status': 'completed', 'result_digest': effect.result_digest},
            node_id=node.node_id, effect=effect)
        return result
    return call


def configure_core_llm(native_options, provider_options, functions, executor, *, agent_loop):
    scope, node = current_scope(), current_node()
    if scope is None or node is None:
        return executor
    existing = native_options.get('provider_attempt_control', provider_options.get('provider_attempt_control'))
    attempts = CoreAttemptControl(scope, node, existing)
    native_options['provider_attempt_control'] = attempts
    provider_options['provider_attempt_control'] = attempts
    if agent_loop:
        native_options['control'] = CoreLoopControl(scope, node, native_options.get('control'))
        saved = node.saved or {}
        checkpoint = saved.get('llm_response') or saved.get('llm_checkpoint')
        if checkpoint is not None:
            native_options['continuation'] = checkpoint['checkpoint']
    for name, function in list(functions.items()):
        functions[name] = _tool(scope, node, name, function)
    if agent_loop:
        if executor is None:
            from magic_llm.agent.tool_executor import ToolExecutor
            executor = ToolExecutor()
        executor.propagate_errors(ExecutionStorageError)
    return executor


def _direct_request(chat, options):
    return {'messages': list(chat.messages), 'max_input_tokens': chat.max_input_tokens,
            'options': {key: value for key, value in options.items() if key != 'provider_attempt_control'}}


async def core_generate(method, chat, **options):
    scope, node = current_scope(), current_node()
    if scope is None or node is None:
        return await method(chat, **options)
    from magic_llm.agent.async_agent_loop import AsyncAgentLoop
    from magic_llm.model.ModelChatResponse import ModelChatResponse
    request = _direct_request(chat, options)
    digest = hashlib.sha256(canonical_bytes(request)).hexdigest()
    retained = (node.saved or {}).get('llm_response')
    if retained is not None:
        if retained.get('direct_request_digest') != digest or retained['response']['kind'] != 'response':
            raise ExecutionStorageError('execution_response_conflict', 'Original direct LLM request changed')
        node.active_response_id = retained['physical_attempt_id']
        return ModelChatResponse.model_validate(retained['response']['response'])
    response = await method(chat, **options)
    saved = {'direct_request_digest': digest, 'physical_attempt_id': node.active_response_id,
             'response': {'kind': 'response', 'response': AsyncAgentLoop._response_json(response)}}
    await scope.record('llm_response', {'execution_id': node.execution_id,
        'response_digest': hashlib.sha256(canonical_bytes(saved)).hexdigest()},
        node_id=node.node_id, private_state=saved)
    return response


async def core_stream(method, chat, **options):
    scope, node = current_scope(), current_node()
    if scope is None or node is None:
        async with aclosing(method(chat, **options)) as source:
            async for chunk in source:
                yield chunk
        return
    from magic_llm.agent.async_agent_loop import AsyncAgentLoop
    from magic_llm.model.ModelChatStream import ChatCompletionModel
    request = _direct_request(chat, options)
    digest = hashlib.sha256(canonical_bytes(request)).hexdigest()
    retained = (node.saved or {}).get('llm_response')
    if retained is not None:
        if retained.get('direct_request_digest') != digest or retained['response']['kind'] != 'stream':
            raise ExecutionStorageError('execution_response_conflict', 'Original direct LLM request changed')
        node.active_response_id = retained['physical_attempt_id']
        for chunk in retained['response']['chunks']:
            yield ChatCompletionModel.model_validate(chunk)
        return
    chunks, size = [], 2
    async with aclosing(method(chat, **options)) as source:
        async for chunk in source:
            value = AsyncAgentLoop._response_json(chunk)
            size += len(canonical_bytes(value)) + 1
            if size > scope.recorder.store.limits.max_state_bytes:
                raise ExecutionStorageError('execution_response_capacity', 'Native stream exceeds storage payload capacity')
            chunks.append(value)
            yield chunk
    saved = {'direct_request_digest': digest, 'physical_attempt_id': node.active_response_id,
             'response': {'kind': 'stream', 'chunks': chunks}}
    await scope.record('llm_response', {'execution_id': node.execution_id,
        'response_digest': hashlib.sha256(canonical_bytes(saved)).hexdigest()},
        node_id=node.node_id, private_state=saved)
