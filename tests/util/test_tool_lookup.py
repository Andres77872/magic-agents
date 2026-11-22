import copy
import json

import pytest

from magic_agents.util.tool_lookup import tool_lookup


TABLE = {'alpha': 'Private alpha details with {{ literal_template }}.',
         'beta': 'Private beta details.', 'gamma': 'Private gamma details.'}


def selection(*ids):
    return {'execution': 'client', 'source': 'schema_only', 'node_id': 'planner',
            'tool_calls': [{'id': f'call-{index}', 'type': 'function',
                            'function': {'name': 'read_item', 'arguments': json.dumps({'item_id': reference})}}
                           for index, reference in enumerate(ids)]}


def lookup(envelope, content='', table=None):
    return tool_lookup(envelope, TABLE if table is None else table,
                       'read_item', 'item_id', content, 'NO_LOOKUP', 3)


@pytest.mark.parametrize('references', [(), ('alpha',), ('beta', 'alpha'), ('gamma', 'beta', 'alpha')])
def test_lookup_preserves_the_authored_snapshot_and_selection_order(references):
    envelope, table = selection(*references), copy.deepcopy(TABLE)
    before = copy.deepcopy((envelope, table))
    result = lookup(envelope, 'NO_LOOKUP' if not references else '', table)
    assert result['references'] == list(references)
    assert result['content'] == '\n\n---\n\n'.join(TABLE[key] for key in references)
    assert (envelope, table) == before


@pytest.mark.parametrize('arguments', [
    '{"item_id":"alpha","item_id":"beta"}', '{"item_id":"alpha","other":1}',
    '{"item_id":null}', '{"item_id":true}', '{"item_id":1}', '{"item_id":" alpha"}',
    '{"item_id":"unknown"}', '["alpha"]', '```json\n{"item_id":"alpha"}\n```',
    '{"item_id":"alpha","unused":NaN}', '{"item_id":"alpha","unused":Infinity}',
    '{"item_id":"alpha","unused":1e999}', '{"item_id":"alpha","unused":-0}',
    '{"item_id":"alpha","unused":-0.0}', '{"item_id":"alpha"} trailing',
    '{"item_id":' + '[' * 1100 + '0' + ']' * 1100 + '}',
])
def test_invalid_arguments_never_return_partial_details_or_keep_parser_errors(arguments):
    envelope = selection('beta', 'alpha')
    envelope['tool_calls'][1]['function']['arguments'] = arguments
    with pytest.raises(ValueError, match='^Invalid tool lookup selection$') as error:
        lookup(envelope)
    assert error.value.__context__ is None
    assert arguments not in str(error.value)


@pytest.mark.parametrize('content', [None, '', 'NO_LOOKUP ', 'private explanation', {}, []])
def test_empty_selection_requires_the_exact_authored_sentinel(content):
    with pytest.raises(ValueError, match='^Invalid tool lookup selection$'):
        lookup(selection(), content)


@pytest.mark.parametrize('content', ['NO_LOOKUP', 'private explanation', ' ', {}, 1])
def test_calls_cannot_be_combined_with_generated_prose(content):
    with pytest.raises(ValueError, match='^Invalid tool lookup selection$'):
        lookup(selection('alpha'), content)


@pytest.mark.parametrize('variant', ['unknown', 'duplicate', 'too_many', 'duplicate_call_id',
                                    'wrong_tool', 'extra_argument', 'extra_call', 'extra_envelope',
                                    'wrong_source', 'wrong_execution', 'boolean_index', 'wrong_index',
                                    'missing_call_id', 'missing_arguments', 'parsed_arguments'])
def test_invalid_selections_fail_as_a_whole(variant):
    envelope = selection('alpha', 'beta')
    calls = envelope['tool_calls']
    if variant == 'unknown': calls[1]['function']['arguments'] = '{"item_id":"unknown"}'
    if variant == 'duplicate': calls[1]['function']['arguments'] = '{"item_id":"alpha"}'
    if variant == 'too_many': envelope = selection('alpha', 'beta', 'gamma', 'alpha')
    if variant == 'duplicate_call_id': calls[1]['id'] = calls[0]['id']
    if variant == 'wrong_tool': calls[1]['function']['name'] = 'other_tool'
    if variant == 'extra_argument': calls[1]['function']['arguments'] = '{"item_id":"beta","extra":"x"}'
    if variant == 'extra_call': calls[1]['extra'] = 'x'
    if variant == 'extra_envelope': envelope['extra'] = 'x'
    if variant == 'wrong_source': envelope['source'] = 'runtime'
    if variant == 'wrong_execution': envelope['execution'] = 'server'
    if variant == 'boolean_index': calls[0]['index'] = False
    if variant == 'wrong_index': calls[1]['index'] = 0
    if variant == 'missing_call_id': del calls[1]['id']
    if variant == 'missing_arguments': del calls[1]['function']['arguments']
    if variant == 'parsed_arguments': calls[1]['function']['arguments'] = {'item_id': 'beta'}
    with pytest.raises(ValueError, match='^Invalid tool lookup selection$') as error:
        lookup(envelope)
    assert error.value.__context__ is None


@pytest.mark.parametrize('field, value', [('table', []), ('table', {'alpha': None}),
                                       ('table', {'alpha': 'x' * 1_048_577}),
                                       ('envelope', '{"tool_calls":[],"tool_calls":[]}')])
def test_lookup_rejects_invalid_frozen_tables_and_raw_envelopes(field, value):
    kwargs = {field: value}
    if field == 'table': kwargs['envelope'] = selection('alpha')
    with pytest.raises(ValueError, match='^Invalid tool lookup selection$'):
        lookup(**kwargs)
