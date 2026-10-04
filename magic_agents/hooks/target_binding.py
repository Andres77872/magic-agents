"""Canonical same-scope Hook targets, with backward scalar reads."""


def hook_target_ids(data):
    plural, scalar = data.get('target_node_ids'), data.get('target_node_id')
    if scalar is not None and (type(scalar) is not str or not scalar.strip() or len(scalar) > 256):
        raise ValueError('Hook target_node_id must be a non-empty node identity or null')
    if 'target_node_ids' not in data:
        return (scalar,) if scalar is not None else ()
    if (type(plural) is not list or len(plural) > 128
            or any(type(value) is not str or not value.strip() or len(value) > 256 for value in plural)
            or len(set(plural)) != len(plural)):
        raise ValueError('Hook target_node_ids must contain unique non-empty node identities')
    if scalar is not None:
        raise ValueError('Conflicting scalar and plural Hook targets')
    return tuple(plural)
