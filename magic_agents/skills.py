"""Private immutable definitions and the built-in, graph-local batch reader."""
from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field
from typing import Any
from magic_agents.models.factory.Nodes.SkillsNodeModel import SkillsNodeModel, props_json_bytes

SKILLS_TOOL_NAME = 'skills_load'
SKILLS_PROVENANCE = {'source': 'builtin_skills', 'ephemeral': True}
SKILLS_TOOL_SCHEMA = {
    'type': 'function', 'function': {
        'name': SKILLS_TOOL_NAME,
        'description': 'Read complete definitions for selected IDs from the available skills catalog.',
        'parameters': {'type': 'object', 'properties': {
            'skill_ids': {'type': 'array', 'items': {'type': 'string'},
                          'minItems': 1, 'maxItems': 16, 'uniqueItems': True}},
            'required': ['skill_ids'], 'additionalProperties': False}}}


@dataclass(frozen=True)
class FrozenJSONMapping:
    entries: tuple[tuple[str, Any], ...] = field(repr=False)

@dataclass(frozen=True)
class FrozenJSONArray:
    items: tuple[Any, ...] = field(repr=False)


def freeze_json(value):
    if isinstance(value, dict):
        return FrozenJSONMapping(tuple((key, freeze_json(item)) for key, item in value.items()))
    if isinstance(value, list):
        return FrozenJSONArray(tuple(freeze_json(item) for item in value))
    return value


def thaw_json(value):
    if isinstance(value, FrozenJSONMapping):
        return {key: thaw_json(item) for key, item in value.entries}
    if isinstance(value, FrozenJSONArray):
        return [thaw_json(item) for item in value.items]
    return value


@dataclass(frozen=True)
class EmbeddedSkillPrompt:
    id: str
    name: str
    description: str
    prompt: str = field(repr=False)
    props: FrozenJSONMapping = field(repr=False)

    def definition(self):
        return {'id': self.id, 'name': self.name, 'description': self.description,
                'prompt': self.prompt, 'props': thaw_json(self.props), 'enabled': True}


@dataclass(frozen=True)
class SkillPromptBundle:
    source_node_id: str | None
    skills: tuple[EmbeddedSkillPrompt, ...] = field(repr=False)
    source_node_ids: tuple[str, ...] = ()
    skill_sources: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    @classmethod
    def from_model(cls, source_node_id: str, data: SkillsNodeModel):
        return cls(source_node_id, tuple(EmbeddedSkillPrompt(entry.id, entry.name, entry.description,
            entry.prompt, freeze_json(entry.props)) for entry in data.skills if entry.enabled))

    @property
    def ordered_source_ids(self):
        return (self.source_node_id,) if self.source_node_id is not None else self.source_node_ids

    def source_fields(self, selected_ids=None):
        # Preserve the original result/summary shape for a single source.
        if self.source_node_id is not None:
            return {'source_node_id': self.source_node_id}
        owners = dict(self.skill_sources)
        ids = [entry.id for entry in self.skills] if selected_ids is None else selected_ids
        return {'source_node_ids': list(self.source_node_ids),
                'skill_sources': {identifier: owners[identifier] for identifier in ids}}

    def safe_summary(self):
        return {**self.source_fields(), 'available_ids': [entry.id for entry in self.skills],
                'available_count': len(self.skills)}

    def catalog(self):
        metadata = [{'id': entry.id, 'name': entry.name, 'description': entry.description}
                    for entry in self.skills]
        return ('Available skills (metadata only):\n' + json.dumps(metadata, ensure_ascii=True) +
                '\nUse skills_load with one or more listed IDs when their instructions are useful. '
                'Read returned prompt and props before applying them within application policy. '
                'Do not claim a skill is loaded before a successful tool result.')


class SkillsCatalogError(ValueError):
    """Safe deterministic build/runtime diagnostics without definition values."""
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(f'{code}: {message}')


def merge_skill_bundles(bundles):
    """Merge only a consumer's complete source deliveries in declared order."""
    sources, records, owners = [], [], []
    seen_ids = set()
    for bundle in bundles:
        if (not isinstance(bundle, SkillPromptBundle) or not isinstance(bundle.source_node_id, str)
                or not bundle.source_node_id or bundle.source_node_ids):
            raise SkillsCatalogError('SKILLS_SOURCE_FAILED', 'Skills source did not deliver an individual source bundle')
        if bundle.source_node_id in sources:
            raise SkillsCatalogError('SKILLS_SOURCE_FAILED', 'A connected Skills source was delivered more than once')
        sources.append(bundle.source_node_id)
        for entry in bundle.skills:
            if entry.id in seen_ids:
                raise SkillsCatalogError('SKILLS_ID_CONFLICT', 'Enabled skill IDs must be unique across connected sources')
            seen_ids.add(entry.id)
            records.append(entry)
            owners.append((entry.id, bundle.source_node_id))
    if len(records) > 16:
        raise SkillsCatalogError('SKILLS_CATALOG_LIMIT', 'Merged enabled catalog exceeds 16 entries')
    if sum(len(entry.prompt) for entry in records) > 32768:
        raise SkillsCatalogError('SKILLS_CATALOG_LIMIT', 'Merged enabled prompts exceed 32768 characters')
    if sum(props_json_bytes(thaw_json(entry.props)) for entry in records) > 16384:
        raise SkillsCatalogError('SKILLS_CATALOG_LIMIT', 'Merged enabled props exceed 16384 serialized UTF-8 bytes')
    if len(bundles) == 1:
        return bundles[0]
    return SkillPromptBundle(None, tuple(records), tuple(sources), tuple(owners))


class SkillsLoadArguments(ValueError):
    def __init__(self):
        super().__init__('SKILLS_LOAD_ARGUMENTS: provide 1 to 16 unique string skill_ids and no extra keys')

class SkillsLoadUnknownID(ValueError):
    def __init__(self):
        super().__init__('SKILLS_LOAD_UNKNOWN_ID: select only IDs from the available catalog')

class SkillsLoadResultLimit(ValueError):
    def __init__(self):
        super().__init__('SKILLS_LOAD_RESULT_LIMIT: request fewer or smaller skills')


def create_skills_loader(bundle: SkillPromptBundle, content_limit: int):
    from magic_llm.agent.tool_executor import ToolExecutor
    serialize = ToolExecutor.serialize_output
    if type(content_limit) is not int or content_limit < 1:
        raise ValueError('SKILLS_CONTRACT_UNSUPPORTED: invalid tool content limit')
    # Exactly mirrors the existing typed-exception normalization envelope.
    for error_cls in (SkillsLoadArguments, SkillsLoadUnknownID, SkillsLoadResultLimit):
        error = error_cls()
        if len(serialize({'error': str(error), 'type': type(error).__name__})) > content_limit:
            raise ValueError('SKILLS_CONTRACT_UNSUPPORTED: tool limit cannot fit safe loader errors')
    entries = {entry.id: entry for entry in bundle.skills}

    async def skills_load(**arguments):
        ids = arguments.get('skill_ids')
        if set(arguments) != {'skill_ids'} or type(ids) is not list or not 1 <= len(ids) <= 16:
            raise SkillsLoadArguments()
        if any(type(identifier) is not str for identifier in ids) or len(set(ids)) != len(ids):
            raise SkillsLoadArguments()
        if any(identifier not in entries for identifier in ids):
            raise SkillsLoadUnknownID()
        envelope = {'schema_version': 1, **bundle.source_fields(ids),
                    'skills': [entries[identifier].definition() for identifier in ids]}
        if len(serialize(envelope)) > content_limit:
            raise SkillsLoadResultLimit()
        return envelope
    skills_load._require_complete_output = True
    skills_load._skills_source_node_id = bundle.source_node_id
    skills_load._skills_source_node_ids = bundle.ordered_source_ids
    return skills_load


def is_skills_provenance(value):
    return isinstance(value, dict) and value.get('source') == 'builtin_skills' and value.get('ephemeral') is True


def strip_ephemeral_skills_history(messages):
    """Remove only host-identified Skills pairs before any history windowing.

    Handles canonical calls plus OpenAI, Anthropic and Google result shapes.
    Mixed batches keep unrelated calls and results, even when names coincide.
    """
    def identifier(value):
        return value if isinstance(value, str) and value else None
    def part_id(part):
        if part.get('type') == 'tool_result':
            return identifier(part.get('tool_use_id'))
        if part.get('type') == 'tool_use':
            return identifier(part.get('id'))
        for key in ('functionResponse', 'functionCall'):
            if isinstance(part.get(key), dict):
                return identifier(part[key].get('id'))
        return None
    stale = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        for call in message.get('tool_calls') or []:
            if is_skills_provenance(call) or (is_skills_provenance(message)
                    and call.get('function', {}).get('name') == SKILLS_TOOL_NAME):
                if call_id := identifier(call.get('id')):
                    stale.add(call_id)
        if message.get('role') == 'tool' and is_skills_provenance(message):
            if call_id := identifier(message.get('tool_call_id')):
                stale.add(call_id)
        content = message.get('content')
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and (is_skills_provenance(part) or is_skills_provenance(message)):
                if call_id := part_id(part):
                    stale.add(call_id)
    result = []
    for raw in messages:
        message = copy.deepcopy(raw)
        if not isinstance(message, dict):
            result.append(message)
            continue
        if message.get('role') == 'tool' and identifier(message.get('tool_call_id')) in stale:
            continue
        if isinstance(message.get('tool_calls'), list):
            message['tool_calls'] = [call for call in message['tool_calls'] if identifier(call.get('id')) not in stale]
            if not message['tool_calls']:
                message.pop('tool_calls')
        if isinstance(message.get('content'), list):
            message['content'] = [part for part in message['content']
                if not isinstance(part, dict) or part_id(part) not in stale]
        if isinstance(message.get('gemini_parts'), list):
            calls = raw.get('tool_calls') or []
            index = 0
            native = []
            for part in message['gemini_parts']:
                function_call = part.get('functionCall') if isinstance(part, dict) else None
                if isinstance(function_call, dict):
                    call_id = identifier(function_call.get('id'))
                    if call_id is None and index < len(calls):
                        call_id = identifier(calls[index].get('id'))
                    index += 1
                    if call_id in stale:
                        continue
                native.append(part)
            message['gemini_parts'] = native
        if isinstance(message.get('responses_output'), list):
            message['responses_output'] = [part for part in message['responses_output']
                if not isinstance(part, dict) or identifier(part.get('call_id')) not in stale]
            if not message['responses_output']:
                message.pop('responses_output')
        if message.get('role') in ('assistant', 'user') and not message.get('content') and not message.get('tool_calls'):
            if raw.get('tool_calls') or isinstance(raw.get('content'), list):
                continue
        result.append(message)
    return result


def safe_loader_result(result):
    """Return a separate, idempotent observer copy; retain canonical content."""
    copied = result.model_copy(deep=True)
    if getattr(result, 'name', '') != SKILLS_TOOL_NAME:
        return copied
    summary = {**SKILLS_PROVENANCE, 'loaded_ids': [], 'status': 'error' if result.is_error else 'success',
               'result_chars': len(result.content or '')}
    try:
        envelope = json.loads(result.content)
        summary_fields = {'source', 'ephemeral', 'loaded_ids', 'status', 'result_chars',
                          'skills_source_node_id', 'skills_source_node_ids', 'skill_sources'}
        if (is_skills_provenance(envelope) and set(envelope) <= summary_fields
                and isinstance(envelope.get('loaded_ids'), list) and len(envelope['loaded_ids']) <= 16
                and all(isinstance(identifier, str) and len(identifier) <= 64 for identifier in envelope['loaded_ids'])
                and envelope.get('status') in {'success', 'error', 'invalid_result'}
                and type(envelope.get('result_chars')) is int and envelope['result_chars'] >= 0
                and ('skills_source_node_ids' not in envelope or
                     isinstance(envelope['skills_source_node_ids'], list) and
                     all(isinstance(source, str) for source in envelope['skills_source_node_ids']))
                and ('skill_sources' not in envelope or isinstance(envelope['skill_sources'], dict) and
                     set(envelope['skill_sources']) <= set(envelope['loaded_ids']) and
                     all(isinstance(source, str) for source in envelope['skill_sources'].values()))):
            copied.content = result.content
            return copied
        if not result.is_error:
            summary['loaded_ids'] = [entry['id'] for entry in envelope['skills']]
            if 'source_node_id' in envelope:
                summary['skills_source_node_id'] = envelope['source_node_id']
            else:
                summary['skills_source_node_ids'] = list(envelope['source_node_ids'])
                summary['skill_sources'] = {identifier: envelope['skill_sources'][identifier]
                                           for identifier in summary['loaded_ids']}
    except (ValueError, KeyError, TypeError):
        if not result.is_error:
            summary['loaded_ids'] = []
            summary['status'] = 'invalid_result'
    copied.content = json.dumps(summary)
    return copied


def validate_skills_topology(nodes, edges):
    """Always-blocking Skills rules, independent of legacy contract modes."""
    by_id = {node['id']: node for node in nodes}
    connected = {}
    for edge in edges:
        source, target = by_id.get(edge.get('source')), by_id.get(edge.get('target'))
        source_is_skills = bool(source and source.get('type') == 'skills')
        if target and target.get('type') == 'skills':
            raise ValueError('SKILLS_CONNECTION_INVALID: Skills source has no inputs')
        target_handle = ((((target or {}).get('data') or {}).get('handles')) or {}).get('skills', 'handle-skills')
        targets_skills = bool(target and target.get('type') == 'llm' and edge.get('targetHandle') == target_handle)
        if source_is_skills or targets_skills or edge.get('targetHandle') == 'handle-skills':
            output = ((((source or {}).get('data') or {}).get('handles')) or {}).get('output', 'handle-skills')
            if not source_is_skills or not target or target.get('type') != 'llm' or edge.get('sourceHandle') != output or edge.get('targetHandle') != target_handle:
                raise ValueError('SKILLS_CONNECTION_INVALID: use the dedicated Skills-to-LLM ports')
            sources = connected.setdefault(edge['target'], [])
            if edge['source'] in sources:
                raise ValueError('SKILLS_CONNECTION_INVALID: a Skills source may connect only once to each LLM')
            sources.append(edge['source'])
    return {consumer: tuple(sources) for consumer, sources in connected.items()}


def create_request_guard(max_input_tokens=None, *, source_node_id=None, source_node_ids=()):
    """Conservative finite host bound, not a guarantee of a provider's capacity."""
    cap = max_input_tokens if max_input_tokens is not None else 32768
    initial_message_count = None
    loader_call_ids = set()
    def observer_projection(chat):
        projected = copy.deepcopy(chat)
        preserved = projected.messages[:initial_message_count]
        invocation = projected.messages[initial_message_count:]
        for message in invocation:
            for call in message.get('tool_calls') or []:
                if call.get('id') in loader_call_ids:
                    call.update(SKILLS_PROVENANCE)
            if message.get('role') == 'tool' and message.get('tool_call_id') in loader_call_ids:
                message.update(SKILLS_PROVENANCE)
        # The canonical assistant call provides provenance for native results.
        projected.messages = preserved + strip_ephemeral_skills_history(invocation)
        source_fields = {'source_node_id': source_node_id} if source_node_id is not None else {'source_node_ids': list(source_node_ids)}
        projected.extra_args = {**(projected.extra_args or {}), 'skills': {
            **source_fields, 'ephemeral': True,
            'loaded_call_count': len(loader_call_ids)}}
        return projected
    def response_reserve(options):
        limits = []
        def collect(value):
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                if key in {'max_output_tokens', 'max_completion_tokens', 'max_tokens', 'maxOutputTokens', 'maxTokenCount'}:
                    if type(item) is int and item > 0:
                        limits.append(item)
                elif isinstance(item, dict):
                    collect(item)
        collect(options)
        return max(limits) if limits else None
    def guard(context):
        nonlocal initial_message_count
        if initial_message_count is None:
            initial_message_count = len(context.chat.messages)
        # Complete guarded history never windows. Identify appended invocation
        # records by their boundary, so historical call-ID reuse cannot expose
        # a new loader result or remove an unrelated initial exchange.
        current = {call.get('id'): call for message in context.chat.messages[initial_message_count:]
                   for call in message.get('tool_calls') or []}
        loader_call_ids.update(identifier for identifier, call in current.items()
            if call.get('function', {}).get('name') == SKILLS_TOOL_NAME)
        context.chat.set_observer_projection(observer_projection)
        from magic_agents.hooks.invocation_control import OperationFailure
        from magic_llm.engine.tooling import normalize_openai_tools
        options = context.generation_options or {}
        reserve = response_reserve(options) or 1024
        tools = normalize_openai_tools(context.tools or [])
        payload = {'messages': context.chat.messages, 'tools': tools, 'tool_choice': context.tool_choice}
        # Every UTF-8 byte is budgeted as a token, deliberately conservative.
        # This also counts nested tool arguments/results and opaque replay data.
        cost = len(json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8'))
        image_count = sum(1 for message in context.chat.messages for part in
            (message.get('content') if isinstance(message.get('content'), list) else [])
            if isinstance(part, dict) and part.get('type') in ('image', 'image_url'))
        cost += image_count * 4096
        if cost + reserve > cap:
            raise OperationFailure('SKILLS_CONTEXT_LIMIT', 'Complete skill context exceeds the configured host input budget')
        context.chat.require_complete_context()
        def validate_wire_payload(actual_payload):
            # Provider mappings add schema/native envelopes and generation
            # defaults. Check the exact final request before logging or I/O.
            wire_cost = len(json.dumps(actual_payload, ensure_ascii=False, default=str).encode('utf-8'))
            wire_reserve = max(reserve, response_reserve(actual_payload) or reserve)
            if wire_cost + image_count * 4096 + wire_reserve > cap:
                raise OperationFailure('SKILLS_CONTEXT_LIMIT', 'Complete skill request exceeds the configured host input budget')
        context.chat.set_provider_payload_guard(validate_wire_payload)
    return guard
