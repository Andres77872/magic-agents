"""Pure selection of values from a frozen, workflow-owned lookup table."""
from magic_agents.util.strict_json import strict_json


def _text(value, limit):
    return type(value) is str and 0 < len(value) <= limit and value == value.strip()


def _lookup(envelope, table, name, argument, content, empty_text, max_calls):
    if (not _text(name, 64) or not _text(argument, 64) or not _text(empty_text, 256)
            or type(max_calls) is not int or not 1 <= max_calls <= 32):
        raise ValueError()
    if type(envelope) is str:
        envelope = strict_json(envelope)
    if type(table) is str:
        table = strict_json(table)
    if (type(table) is not dict or len(table) > 1000
            or any(not _text(key, 256) or type(value) is not str or not value
                   for key, value in table.items())
            or sum(len(key.encode('utf-8')) + len(value.encode('utf-8'))
                   for key, value in table.items()) > 1_048_576):
        raise ValueError()
    if (type(envelope) is not dict
            or set(envelope) != {'execution', 'source', 'node_id', 'tool_calls'}
            or envelope['execution'] != 'client' or envelope['source'] != 'schema_only'
            or not _text(envelope['node_id'], 256)
            or type(envelope['tool_calls']) is not list
            or len(envelope['tool_calls']) > max_calls):
        raise ValueError()
    calls = envelope['tool_calls']
    if not calls:
        if content != empty_text or type(content) is not str:
            raise ValueError()
        return {'references': [], 'content': ''}
    if content is not None and (type(content) is not str or content != ''):
        raise ValueError()
    references, call_ids = [], set()
    for index, call in enumerate(calls):
        if (type(call) is not dict or not {'id', 'type', 'function'} <= set(call)
                or set(call) - {'id', 'type', 'function', 'index'}
                or call['type'] != 'function' or not _text(call['id'], 256)
                or call['id'] in call_ids
                or ('index' in call and (type(call['index']) is not int or call['index'] != index))):
            raise ValueError()
        function = call['function']
        if (type(function) is not dict or set(function) != {'name', 'arguments'}
                or function['name'] != name or type(function['arguments']) is not str
                or len(function['arguments'].encode('utf-8')) > 2048):
            raise ValueError()
        arguments = strict_json(function['arguments'])
        if type(arguments) is not dict or set(arguments) != {argument}:
            raise ValueError()
        reference = arguments[argument]
        if not _text(reference, 256) or reference not in table or reference in references:
            raise ValueError()
        references.append(reference)
        call_ids.add(call['id'])
    return {'references': references, 'content': '\n\n---\n\n'.join(table[key] for key in references)}


def tool_lookup(envelope, table, name, argument, content, empty_text, max_calls):
    """Resolve schema-only calls without I/O, retries or provider requests.

    The table and selection protocol belong to the authored workflow. Invalid
    selections fail as a whole and never skip, truncate or repair a call.
    """
    result = None
    try:
        result = _lookup(envelope, table, name, argument, content, empty_text, max_calls)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        pass
    if result is None:
        raise ValueError('Invalid tool lookup selection')
    return result
