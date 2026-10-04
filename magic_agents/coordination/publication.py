"""Immutable effective-author output handed to a host publication adapter.

This is not a public journal or consumer acknowledgement. The host validates
its admitted action schema and commits its terminal record independently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from magic_agents.coordination.service import CoordinationError, _json


MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_ACTION_BYTES = 256 * 1024
MAX_ACTIONS = 512


@dataclass(frozen=True)
class PublicationWarning:
    code: str
    message: str


@dataclass(frozen=True)
class PublicationDisposition:
    kind: str
    warnings: tuple[PublicationWarning, ...] = ()


@dataclass(frozen=True)
class PublicationCandidate:
    source_node_path: tuple[str, ...]
    scope_instance_id: str
    workgroup_id: str
    epoch_id: str
    output_revision: int
    text: str
    action_json: tuple[str, ...]
    disposition: PublicationDisposition


def _invalid(message):
    raise CoordinationError('invalid_publication', message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: _invalid('Action arguments contain duplicate object keys')
        result[key] = value
    return result


def freeze_author_output(node) -> tuple[str, tuple[str, ...]]:
    """Read only effective scheduler outputs, never a provider or debug stream."""
    generated = node.outputs.get(node.OUTPUT_HANDLE_GENERATED)
    if not isinstance(generated, dict) or 'content' not in generated:
        _invalid('Author has no completed effective output')
    text = generated['content']
    if not isinstance(text, str):
        text = _json(text, max_bytes=MAX_OUTPUT_BYTES).decode('utf-8')
    try:
        total = len(text.encode('utf-8', errors='strict'))
    except UnicodeError:
        _invalid('Author output must contain valid Unicode scalars')
    if total > MAX_OUTPUT_BYTES: _invalid('Author output exceeds its byte ceiling')

    carrier = node.outputs.get(node.OUTPUT_HANDLE_TOOL_CALLS)
    if carrier is None: return text, ()
    envelope = carrier.get('content') if isinstance(carrier, dict) else None
    if (not isinstance(envelope, dict) or envelope.get('execution') != 'client'
            or envelope.get('source') != 'schema_only' or envelope.get('node_id') != node.node_id):
        _invalid('Only the admitted author schema-only client actions may publish')
    calls = envelope.get('tool_calls')
    if not isinstance(calls, list) or len(calls) > MAX_ACTIONS:
        _invalid('Author action count exceeds the bounded contract')
    actions = []
    for raw in calls:
        call = raw.model_dump(exclude_none=True) if hasattr(raw, 'model_dump') else raw
        if not isinstance(call, dict) or call.get('type') != 'function':
            _invalid('Author action must be a complete function call')
        if call.get('execution', 'client') != 'client' or call.get('source', 'schema_only') != 'schema_only':
            _invalid('Callable or untrusted action provenance cannot publish')
        function = call.get('function')
        if not isinstance(function, dict): _invalid('Author action has no function')
        name, arguments = function.get('name'), function.get('arguments')
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name):
            _invalid('Author action name is invalid')
        if not isinstance(arguments, str): _invalid('Action arguments must be complete JSON text')
        try:
            if len(arguments.encode('utf-8', errors='strict')) > MAX_ACTION_BYTES:
                _invalid('Action arguments exceed their byte ceiling')
            parsed = json.loads(arguments, object_pairs_hook=_object,
                                parse_constant=lambda _: _invalid('Non-finite action argument'))
        except (ValueError, UnicodeError, RecursionError):
            _invalid('Action arguments must be valid finite JSON')
        if not isinstance(parsed, dict): _invalid('Action arguments must be an object')
        _json(parsed, max_bytes=MAX_ACTION_BYTES)
        encoded = _json({'type': 'function', 'function': {'name': name, 'arguments': arguments},
                         'execution': 'client', 'source': 'schema_only'}, max_bytes=MAX_ACTION_BYTES).decode('utf-8')
        total += len(encoded.encode('utf-8'))
        if total > MAX_OUTPUT_BYTES: _invalid('Author output exceeds its aggregate byte ceiling')
        actions.append(encoded)
    return text, tuple(actions)
