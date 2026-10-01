"""Callable adapters for fixed Python code and isolated child graphs."""
from __future__ import annotations

import asyncio
import copy
import json
import uuid

from magic_agents.models.tool_schema import function_tool_schema, validate_tool_arguments


class FixedPythonTool:
    def __init__(self, *, code, name, description, parameters, node_id, safety_mode, timeout, max_output_chars):
        self.tool_schema = function_tool_schema(name, description, parameters)
        self.__name__ = name
        self._source_node_id = node_id
        self._code = code
        self._config = dict(safety_mode=safety_mode, timeout=timeout, max_output_chars=max_output_chars)

    @property
    def tool_callable(self):
        return self

    async def __call__(self, **arguments):
        from magic_agents.node_system.python_code_runner import CodeRunner
        validate_tool_arguments(self.tool_schema['function']['parameters'], arguments)
        # Each invocation owns its runner and arguments, including parallel calls.
        result = await CodeRunner(**self._config).execute(self._code, copy.deepcopy(arguments))
        if 'error' in result:
            raise RuntimeError(result['error'])
        content = json.dumps(result['result'], ensure_ascii=False, allow_nan=False)
        limit = self._config['max_output_chars']
        if len(content) > limit:
            return content[:limit] + f"\n... [truncated {len(content) - limit} chars]"
        return content


class GraphTool:
    def __init__(self, node, chat_log):
        data = node._tool_data
        self.tool_schema = function_tool_schema(data.tool_name, data.tool_description, data.tool_parameters)
        self.__name__ = data.tool_name
        self._source_node_id = node.node_id
        self._node = node
        self._chat_log = chat_log
        self._extras = copy.deepcopy(node._prepare_child_extras(
            node.inputs.get(node.HANDLER_CLIENT_EXTRAS, {}),
            getattr(chat_log, 'flow_state', None) or {},
        ))
        self._timeout = data.tool_timeout
        self._events = []

    @property
    def tool_callable(self):
        return self

    def drain_events(self):
        events, self._events = self._events, []
        return events

    def _capture_usage(self, event):
        if event.get('type') == 'debug':
            content = event.get('content', {})
            if content.get('event_type') == 'TOOL_USAGE':
                nested = copy.deepcopy(content['data'])
                nested['source_node_path'] = [self._source_node_id, *nested.get('source_node_path', [])]
                self._events.append(nested)
            return
        if event.get('type') != 'content':
            return
        chunk = event.get('content')
        usage = getattr(chunk, 'usage', None)
        if usage is None:
            return
        usage = usage.model_dump(exclude_none=True) if hasattr(usage, 'model_dump') else usage
        if not isinstance(usage, dict) or not any(usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            return
        self._events.append({
            'type': 'tool_usage',
            'data': {'id': getattr(chunk, 'id', None), 'usage': usage,
                     'provider': getattr(chunk, 'provider', None), 'model': getattr(chunk, 'model', None)},
            'source_node': event.get('source_node'),
            'source_node_path': [self._source_node_id, *event.get('source_node_path', [event.get('source_node')])],
        })
        # Newer magic-llm executors expose call identity without adding hidden
        # arguments to the author's JSON schema.
        try:
            from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL
            call = CURRENT_TOOL_CALL.get()
            if call is not None:
                self._events[-1]['tool_call_id'] = call.id
        except ImportError:
            pass

    async def __call__(self, **arguments):
        validate_tool_arguments(self.tool_schema['function']['parameters'], arguments)
        if self._node._tool_graph_factory is None:
            raise ValueError('Inner tool graph factory is unavailable; build the parent graph before execution')
        async with asyncio.timeout(self._timeout):
            return await self._run(arguments)

    async def _run(self, arguments):
        from contextlib import aclosing
        from magic_agents.execution.reactive_executor import execute_graph_reactive
        from magic_agents.hooks.hook_registry import HookRegistry
        from magic_agents.node_system.NodeInner import SubFlowOutcome
        graph = self._node._tool_graph_factory(copy.deepcopy(arguments), copy.deepcopy(self._extras))
        child_run_id = f'run-{uuid.uuid4().hex}'
        parent_hooks = self._node._hooks
        child_hooks = parent_hooks.fork_for_child(child_run_id=child_run_id, parent_node_id=self._source_node_id) if parent_hooks else HookRegistry()
        if graph.hooks is not None:
            child_hooks.register_graph(graph.hooks)
        # Same rule as step mode: the call fails when a child node ends in
        # ERROR, not when a (possibly recovered) diagnostic was emitted.
        outcome = SubFlowOutcome()
        execution = execute_graph_reactive(
            graph, id_chat=getattr(self._chat_log, 'id_chat', None),
            id_thread=getattr(self._chat_log, 'id_thread', None),
            id_user=getattr(self._chat_log, 'id_user', None), extras=copy.deepcopy(self._extras),
            flow_state=None, run_id=child_run_id,
            parent_run_id=getattr(parent_hooks, 'run_id', None) or getattr(self._chat_log, 'run_id', None),
            hooks=child_hooks, result=outcome.result,
        )
        async with aclosing(execution):
            async for event in execution:
                self._capture_usage(event)
                if event.get('type') == 'debug':
                    content = event.get('content', {})
                    path = event.get('source_node_path') or [content.get('node_id') or event.get('source_node')]
                    outcome.observe(content, path)
        values = []
        for node_id in sorted(graph.nodes):
            node = graph.nodes[node_id]
            if getattr(node, 'node_type', None) == 'end':
                values.extend(node.inputs.values())
        if outcome.failed:
            partial = '\n'.join(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
                                for value in values if value is not None)
            raise outcome.failure(partial_content=partial, child_run_id=child_run_id, tool=True)
        if not values:
            raise RuntimeError('Inner tool completed without a value reaching End')
        result = values[0] if len(values) == 1 else values
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
