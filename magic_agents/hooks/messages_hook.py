"""Messages Hook association and static effective configuration.

The association is target_node_ids, never a graph/data edge. The copied view is
for admission/validation only; build retains and binds the actual Hook objects.
"""
from copy import deepcopy
from magic_agents.models.factory.Nodes.HookNodeModel import HookNodeModel


class MessagesHookBindingError(ValueError):
    pass


def messages_hook_bindings(nodes, edges):
    hooks = [node for node in nodes if node.get('type') == 'hook'
             and (node.get('data') or {}).get('hook_mode') == 'messages']
    if not hooks:
        return {}
    by_id = {node.get('id'): node for node in nodes}
    if len(by_id) != len(nodes):
        raise MessagesHookBindingError('Messages Hooks require unique node identities')
    ids = {node.get('id') for node in hooks}
    if any(edge.get('source') in ids or edge.get('target') in ids
           or (edge.get('hooks') or {}).get('hook_node_id') in ids for edge in edges):
        raise MessagesHookBindingError('Messages Hooks bind by target identities, not graph edges or edge hooks')
    bindings = {}
    for hook in hooks:
        try:
            model = HookNodeModel.model_validate(hook.get('data'), strict=True)
        except (ValueError, TypeError):
            raise MessagesHookBindingError('Invalid Messages Hook configuration') from None
        if not model.target_ids:
            raise MessagesHookBindingError('Messages Hook requires at least one bound LLM')
        for target_id in model.target_ids:
            target = by_id.get(target_id)
            if target is None or target.get('type') != 'llm':
                raise MessagesHookBindingError('Messages Hook target must be an LLM in the same graph scope')
            if target_id in bindings or 'messaging' in (target.get('data') or {}):
                raise MessagesHookBindingError('An LLM accepts one Messages Hook and cannot also declare inline messaging')
            bindings[target_id] = (hook['id'], model.messaging_by_target[target_id])
    return bindings


def effective_messages_nodes(nodes, edges):
    """Private compiler view; never persist this view as authored graph data."""
    bindings = messages_hook_bindings(nodes, edges)
    if not bindings:
        return nodes
    result = []
    for node in nodes:
        if node.get('type') == 'hook' and (node.get('data') or {}).get('hook_mode') == 'messages':
            continue
        item = deepcopy(node)
        if node['id'] in bindings:
            item.setdefault('data', {})['messaging'] = bindings[node['id']][1].model_dump(mode='json', by_alias=True)
        result.append(item)
    return result


def bind_messages_hooks(nodes):
    """Bind real Hook instances before participant discovery and node dispatch."""
    from magic_agents.node_system.NodeHook import NodeHook
    targets = set()
    for hook in nodes.values():
        if isinstance(hook, NodeHook) and hook.hook_mode == 'messages':
            if not hook.target_node_ids:
                raise MessagesHookBindingError('Messages Hook requires at least one bound LLM')
            for target_id in hook.target_node_ids:
                if target_id in targets:
                    raise MessagesHookBindingError('An LLM accepts only one Messages Hook')
                targets.add(target_id)
                hook.bind_target(nodes.get(target_id))
