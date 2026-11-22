from __future__ import annotations

import asyncio
import logging
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from magic_agents.hooks.debug_sse import DebugSSEHook
from magic_agents.hooks.context_factory import HookContextFactory
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.persistence import AssistantMessageContext, GraphPersistenceHook
from magic_agents.hooks.runtime_config import RuntimeConfig


@pytest.mark.asyncio
@pytest.mark.parametrize('result_kind', ['valid', 'malformed', 'provider_error'])
async def test_conditional_llm_persists_provider_usage_even_when_judgment_is_invalid(result_kind):
    from magic_agents.node_system.NodeConditional import NodeConditional
    from magic_agents.hooks.hook_registry import HookRegistry

    sink = FakeExecutionPersistencePort()
    persistence = GraphPersistenceHook(sink=sink, id_chat='chat', id_thread='thread', id_user='user')
    await persistence.on_graph_start(context())
    hooks = HookRegistry(global_hooks=[persistence])
    node = NodeConditional(node_id='judge', node_type='conditional', evaluation_mode='llm',
        questions={'ok': {'type': 'noul', 'instructions': 'Is this approved?'}},
        condition="{{ 'yes' if answers.ok.noul >= 0.5 else 'no' }}", output_handles=['yes', 'no'])
    generate = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({'answers': {'ok': .9}}) if result_kind == 'valid' else 'invalid JSON',
        id='provider-123', model='judge-model', usage={'prompt_tokens': 21, 'completion_tokens': 4, 'total_tokens': 25,
            'prompt_tokens_details': {'cached_tokens': 5}, 'completion_tokens_details': {'reasoning_tokens': 2}},
    ))
    if result_kind == 'provider_error':
        generate.side_effect = RuntimeError('provider unavailable')
    node.inputs = {'handle_input': 'Yes', 'handle-client-provider': SimpleNamespace(
        llm=SimpleNamespace(async_generate=generate, model='judge-model', engine_name='openai'))}
    events = [event async for event in node(None, hooks=hooks)]
    requests = [(config, identifier) for config, identifier in sink.execution_results if config.get('execution_kind') == 'llm_request']
    assert len(requests) == 1
    request_config, identifier = requests[0]
    assert request_config['execution_meta'] == {'model': 'judge-model', 'provider': 'openai'}
    completions = [data for name, data in sink.calls if name == 'complete_execution' and data['id_execution'] == identifier]
    assert completions == [{'id_execution': identifier, 'status': 'failed' if result_kind == 'provider_error' else 'completed'}]
    if result_kind != 'provider_error':
        facts = [data for name, data in sink.calls if name == 'record_usage_fact']
        assert len(facts) == 1
        assert facts[0]['provider_request_id'] == 'provider-123'
        assert facts[0]['provider'] == 'openai'
        assert facts[0]['usage_data']['total_tokens'] == 25
        assert facts[0]['usage_data']['cached_tokens_read'] == 5
        assert facts[0]['usage_data']['reasoning_tokens'] == 2
    if result_kind != 'valid':
        assert any(event['type'] == '__bypass_all__' for event in events)
        assert node.outputs == {}


class FakeExecutionPersistencePort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._counter = 0
        self._run_counter = 0
        self.execution_results: list[tuple[dict, str]] = []

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    async def begin_run(self, **kwargs):
        self.calls.append(("begin_run", kwargs))
        self._run_counter += 1
        return f"run-{self._run_counter}"

    async def begin_execution(self, **kwargs):
        self.calls.append(("begin_execution", kwargs))
        execution_id = self._id("exec")
        self.execution_results.append((kwargs, execution_id))
        return execution_id

    async def record_event(self, **kwargs):
        self.calls.append(("record_event", kwargs))
        return self._id("event")

    async def record_usage_fact(self, **kwargs):
        self.calls.append(("record_usage_fact", kwargs))
        return ("usage-1", 0.0)

    async def complete_execution(self, **kwargs):
        self.calls.append(("complete_execution", kwargs))

    async def complete_run(self, **kwargs):
        self.calls.append(("complete_run", kwargs))


def context(**overrides) -> HookContext:
    metadata = overrides.pop("metadata", {})
    ctx = HookContextFactory.build_node_context(
        execution_id=overrides.pop("execution_id", "graph-exec"),
        run_id=overrides.pop("run_id", "runtime-run"),
        node_id=overrides.pop("node_id", None),
        node_type=overrides.pop("node_type", None),
        node_class=overrides.pop("node_class", None),
        inputs=overrides.pop("inputs", {}),
        outputs=overrides.pop("outputs", {}),
    )
    ctx.metadata = metadata
    for name, value in overrides.items():
        setattr(ctx, name, value)
    return ctx


def call_names(port: FakeExecutionPersistencePort) -> list[str]:
    return [name for name, _ in port.calls]


@pytest.mark.asyncio
async def test_rejects_blank_business_identity_before_sink_call():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="   ",
        id_user="user-1",
        id_agent="agent-1",
    )

    with pytest.raises(ValueError, match="id_thread"):
        await hook.on_graph_start(context())

    assert port.calls == []


@pytest.mark.asyncio
async def test_successful_graph_lifecycle_completes_run_and_root_execution():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
        id_agent="agent-1",
    )

    await hook.on_graph_start(context(inputs={"graph_type": "chat"}))
    await hook.on_graph_end(context())

    assert call_names(port) == [
        "begin_run",
        "begin_execution",
        "record_event",
        "record_event",
        "complete_execution",
        "complete_run",
    ]
    assert port.calls[0][1]["id_conversation"] == "chat-1"
    assert port.calls[-2][1] == {"id_execution": "exec-1", "status": "completed"}
    assert port.calls[-1][1] == {"id_run": "run-1", "status": "completed"}


@pytest.mark.asyncio
async def test_child_fork_keeps_parent_persistence_state_isolated_and_correlated():
    port = FakeExecutionPersistencePort()
    parent = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
    )
    await parent.on_graph_start(context())
    await parent.on_node_start(context(node_id="inner", node_type="inner"))

    parent_run_id = parent.run_id
    parent_root_id = parent.root_execution_id
    parent_inner_id = next(
        execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "node"
        and kwargs.get("execution_name") == "inner"
    )
    child = parent.fork_for_child(
        child_run_id="runtime-child-run",
        parent_node_id="inner",
    )
    await child.on_graph_start(context())

    assert child is not parent
    assert parent.run_id == parent_run_id == "run-1"
    assert parent.root_execution_id == parent_root_id
    assert child.run_id == "run-2"
    child_root = next(
        kwargs
        for kwargs, _ in port.execution_results
        if kwargs.get("execution_kind") == "run_root"
        and kwargs.get("id_run") == "run-2"
    )
    assert child_root["parent_execution_id"] == parent_inner_id
    assert child_root["parent_run_id"] == "run-1"
    assert child_root["nested_depth"] == 1
    assert child_root["nested_request_id"] == "runtime-child-run"


@pytest.mark.asyncio
async def test_real_inner_graph_persists_distinct_parent_and_child_runs():
    from magic_agents.agt_flow import build, execute_graph

    inner_spec = {
        "type": "graph",
        "debug": False,
        "nodes": [
            {"id": "child_input", "type": "user_input"},
            {"id": "child_text", "type": "text", "data": {"text": "child"}},
            {"id": "child_end", "type": "end"},
        ],
        "edges": [
            {
                "id": "child-input-text",
                "source": "child_input",
                "sourceHandle": "handle_user_message",
                "target": "child_text",
                "targetHandle": "handle_flow_input",
            },
            {
                "id": "child-text-end",
                "source": "child_text",
                "sourceHandle": "handle_text_output",
                "target": "child_end",
                "targetHandle": "handle_flow_input",
            },
        ],
    }
    outer_spec = {
        "type": "graph",
        "debug": False,
        "nodes": [
            {"id": "outer_input", "type": "user_input"},
            {"id": "inner", "type": "inner", "data": {"magic_flow": inner_spec}},
            {"id": "outer_end", "type": "end"},
        ],
        "edges": [
            {
                "id": "outer-input-inner",
                "source": "outer_input",
                "sourceHandle": "handle_user_message",
                "target": "inner",
                "targetHandle": "handle_user_message",
            },
            {
                "id": "outer-inner-end",
                "source": "inner",
                "sourceHandle": "handle_execution_content",
                "target": "outer_end",
                "targetHandle": "handle_flow_input",
            },
        ],
    }
    port = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    config.enable_persistence(port)
    graph = build(outer_spec, message="hello")

    _ = [
        event
        async for event in execute_graph(
            graph,
            id_chat="chat-1",
            id_thread="thread-1",
            id_user="user-1",
            hooks=config,
        )
    ]

    completed_runs = [
        kwargs["id_run"]
        for name, kwargs in port.calls
        if name == "complete_run"
    ]
    assert completed_runs == ["run-2", "run-1"]
    roots = [
        (kwargs, execution_id)
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "run_root"
    ]
    assert [kwargs["id_run"] for kwargs, _ in roots] == ["run-1", "run-2"]
    outer_inner_execution = next(
        execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("id_run") == "run-1"
        and kwargs.get("execution_kind") == "node"
        and kwargs.get("execution_name") == "inner"
    )
    child_root = roots[1][0]
    assert child_root["parent_execution_id"] == outer_inner_execution
    assert child_root["parent_run_id"] == "run-1"
    assert child_root["nested_depth"] == 1
    assert child_root["nested_request_id"].startswith("run-")

    completed_executions = [
        kwargs["id_execution"]
        for name, kwargs in port.calls
        if name == "complete_execution"
    ]
    assert completed_executions.count(outer_inner_execution) == 1


@pytest.mark.asyncio
async def test_recursive_inner_graph_persists_depth_and_parent_links():
    """Each recursive child owns a run rooted under its active Inner node."""
    from magic_agents.agt_flow import build, execute_graph

    leaf_spec = {
        "type": "graph",
        "debug": False,
        "nodes": [
            {"id": "leaf_input", "type": "user_input"},
            {"id": "leaf_text", "type": "text", "data": {"text": "leaf"}},
            {"id": "leaf_end", "type": "end"},
        ],
        "edges": [
            {
                "id": "leaf-input-text",
                "source": "leaf_input",
                "sourceHandle": "handle_user_message",
                "target": "leaf_text",
                "targetHandle": "handle_flow_input",
            },
            {
                "id": "leaf-text-end",
                "source": "leaf_text",
                "sourceHandle": "handle_text_output",
                "target": "leaf_end",
                "targetHandle": "handle_flow_input",
            },
        ],
    }
    middle_spec = {
        "type": "graph",
        "debug": False,
        "nodes": [
            {"id": "middle_input", "type": "user_input"},
            {"id": "middle_inner", "type": "inner", "data": {"magic_flow": leaf_spec}},
            {"id": "middle_end", "type": "end"},
        ],
        "edges": [
            {
                "id": "middle-input-inner",
                "source": "middle_input",
                "sourceHandle": "handle_user_message",
                "target": "middle_inner",
                "targetHandle": "handle_user_message",
            },
            {
                "id": "middle-inner-end",
                "source": "middle_inner",
                "sourceHandle": "handle_execution_content",
                "target": "middle_end",
                "targetHandle": "handle_flow_input",
            },
        ],
    }
    outer_spec = {
        "type": "graph",
        "debug": False,
        "nodes": [
            {"id": "outer_input", "type": "user_input"},
            {"id": "outer_inner", "type": "inner", "data": {"magic_flow": middle_spec}},
            {"id": "outer_end", "type": "end"},
        ],
        "edges": [
            {
                "id": "outer-input-inner",
                "source": "outer_input",
                "sourceHandle": "handle_user_message",
                "target": "outer_inner",
                "targetHandle": "handle_user_message",
            },
            {
                "id": "outer-inner-end",
                "source": "outer_inner",
                "sourceHandle": "handle_execution_content",
                "target": "outer_end",
                "targetHandle": "handle_flow_input",
            },
        ],
    }
    port = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    config.enable_persistence(port)

    _ = [
        event
        async for event in execute_graph(
            build(outer_spec, message="hello"),
            id_chat="chat-1",
            id_thread="thread-1",
            id_user="user-1",
            run_id="runtime-outer",
            hooks=config,
        )
    ]

    roots = {
        kwargs["id_run"]: (kwargs, execution_id)
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "run_root"
    }
    assert list(roots) == ["run-1", "run-2", "run-3"]

    outer_inner_execution = next(
        execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("id_run") == "run-1"
        and kwargs.get("execution_kind") == "node"
        and kwargs.get("execution_name") == "outer_inner"
    )
    middle_inner_execution = next(
        execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("id_run") == "run-2"
        and kwargs.get("execution_kind") == "node"
        and kwargs.get("execution_name") == "middle_inner"
    )
    outer_root, _ = roots["run-1"]
    middle_root, _ = roots["run-2"]
    leaf_root, _ = roots["run-3"]

    assert outer_root["nested_depth"] == 0
    assert outer_root["parent_run_id"] is None
    assert outer_root["parent_execution_id"] is None
    assert middle_root["nested_depth"] == 1
    assert middle_root["parent_run_id"] == "run-1"
    assert middle_root["parent_execution_id"] == outer_inner_execution
    assert leaf_root["nested_depth"] == 2
    assert leaf_root["parent_run_id"] == "run-2"
    assert leaf_root["parent_execution_id"] == middle_inner_execution


@pytest.mark.asyncio
async def test_interleaved_llm_nodes_keep_usage_on_their_own_execution():
    """A-first/B-second starts followed by A-first ends must not cross spans."""
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
    )
    await hook.on_graph_start(context())
    await hook.on_node_start(context(node_id="llm-a", node_type="llm"))
    await hook.on_node_start(context(node_id="llm-b", node_type="llm"))

    await hook.on_llm_start(
        context(
            node_id="llm-a",
            inputs={"model": "model-a", "provider": "test", "iteration": 0},
        )
    )
    await hook.on_llm_start(
        context(
            node_id="llm-b",
            inputs={"model": "model-b", "provider": "test", "iteration": 0},
        )
    )
    llm_execution_by_model = {
        kwargs["execution_name"]: execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "llm_request"
    }

    await hook.on_llm_end(
        context(
            node_id="llm-a",
            outputs={
                "model": "model-a",
                "provider": "test",
                "iteration": 0,
                "total_tokens": 11,
            },
        )
    )
    await hook.on_llm_end(
        context(
            node_id="llm-b",
            outputs={
                "model": "model-b",
                "provider": "test",
                "iteration": 0,
                "total_tokens": 22,
            },
        )
    )

    usage_by_model = {
        kwargs["model_alias"]: (
            kwargs["id_execution"],
            kwargs["usage_data"]["total_tokens"],
        )
        for name, kwargs in port.calls
        if name == "record_usage_fact"
    }
    assert usage_by_model == {
        "model-a": (llm_execution_by_model["model-a"], 11),
        "model-b": (llm_execution_by_model["model-b"], 22),
    }
    completed = [
        kwargs["id_execution"]
        for name, kwargs in port.calls
        if name == "complete_execution"
    ]
    assert completed[-2:] == [
        llm_execution_by_model["model-a"],
        llm_execution_by_model["model-b"],
    ]


@pytest.mark.asyncio
async def test_llm_end_waits_for_same_node_start_persistence():
    """A scheduled end cannot race ahead of a slow begin-execution write."""

    class PausingLLMStartPort(FakeExecutionPersistencePort):
        def __init__(self) -> None:
            super().__init__()
            self.llm_start_entered = asyncio.Event()
            self.release_llm_start = asyncio.Event()

        async def begin_execution(self, **kwargs):
            if kwargs.get("execution_kind") == "llm_request":
                self.llm_start_entered.set()
                await self.release_llm_start.wait()
            return await super().begin_execution(**kwargs)

    port = PausingLLMStartPort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
    )
    await hook.on_graph_start(context())
    start_context = context(
        node_id="llm-a",
        inputs={"model": "slow-model", "provider": "test", "iteration": 0},
    )
    end_context = context(
        node_id="llm-a",
        outputs={
            "model": "slow-model",
            "provider": "test",
            "iteration": 0,
            "total_tokens": 13,
        },
    )

    start_task = asyncio.create_task(hook.on_llm_start(start_context))
    await port.llm_start_entered.wait()
    end_task = asyncio.create_task(hook.on_llm_end(end_context))
    await asyncio.sleep(0)
    assert not end_task.done()

    port.release_llm_start.set()
    await asyncio.gather(start_task, end_task)

    usage_call = next(
        kwargs
        for name, kwargs in port.calls
        if name == "record_usage_fact"
    )
    llm_execution = next(
        execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "llm_request"
    )
    assert usage_call["id_execution"] == llm_execution
    assert usage_call["usage_data"]["total_tokens"] == 13


@pytest.mark.asyncio
async def test_repeated_llm_calls_on_one_node_preserve_each_iteration_span():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
    )
    await hook.on_graph_start(context())

    for iteration, tokens in ((0, 7), (1, 9)):
        await hook.on_llm_start(
            context(
                node_id="agent-loop",
                inputs={
                    "model": "loop-model",
                    "provider": "test",
                    "iteration": iteration,
                },
            )
        )
        await hook.on_llm_end(
            context(
                node_id="agent-loop",
                outputs={
                    "model": "loop-model",
                    "provider": "test",
                    "iteration": iteration,
                    "total_tokens": tokens,
                },
            )
        )

    llm_executions = [
        execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "llm_request"
    ]
    usages = [
        (kwargs["id_execution"], kwargs["usage_data"]["total_tokens"])
        for name, kwargs in port.calls
        if name == "record_usage_fact"
    ]
    assert usages == list(zip(llm_executions, (7, 9), strict=True))


@pytest.mark.asyncio
async def test_interleaved_tool_ends_resolve_exact_call_ids_after_mapping_consumption():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
    )
    await hook.on_graph_start(context())

    tool_a = context(
        node_id="llm-a",
        inputs={"tool_name": "tool-a", "tool_call_id": "call-a", "iteration": 0},
    )
    tool_b = context(
        node_id="llm-b",
        inputs={"tool_name": "tool-b", "tool_call_id": "call-b", "iteration": 0},
    )
    await hook.on_tool_start(tool_a)
    await hook.on_tool_start(tool_b)
    tool_execution_by_name = {
        kwargs["execution_name"]: execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "tool_call"
    }

    # Public mapping remains one-shot, but consuming it must not destroy the
    # hook's private active-span correlation used by later end callbacks.
    assert hook.consume_tool_execution_id_mapping() == {
        "call-a": tool_execution_by_name["tool-a"],
        "call-b": tool_execution_by_name["tool-b"],
    }
    assert hook.consume_tool_execution_id_mapping() == {}

    await hook.on_tool_end(
        context(
            node_id="llm-a",
            inputs={"tool_name": "tool-a", "tool_call_id": "call-a", "iteration": 0},
            outputs={"tool_call_id": "call-a", "result": "a"},
        )
    )
    await hook.on_tool_end(
        context(
            node_id="llm-b",
            inputs={"tool_name": "tool-b", "tool_call_id": "call-b", "iteration": 0},
            outputs={"tool_call_id": "call-b", "result": "b"},
        )
    )

    completed = [
        kwargs["id_execution"]
        for name, kwargs in port.calls
        if name == "complete_execution"
    ]
    assert completed[-2:] == [
        tool_execution_by_name["tool-a"],
        tool_execution_by_name["tool-b"],
    ]


@pytest.mark.asyncio
async def test_tool_end_without_call_id_uses_matching_context_not_lifo():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
    )
    await hook.on_graph_start(context())

    for node_id, tool_name in (("llm-a", "tool-a"), ("llm-b", "tool-b")):
        await hook.on_tool_start(
            context(
                node_id=node_id,
                inputs={"tool_name": tool_name, "iteration": 0},
            )
        )
    tool_execution_by_name = {
        kwargs["execution_name"]: execution_id
        for kwargs, execution_id in port.execution_results
        if kwargs.get("execution_kind") == "tool_call"
    }

    for node_id, tool_name in (("llm-a", "tool-a"), ("llm-b", "tool-b")):
        await hook.on_tool_end(
            context(
                node_id=node_id,
                inputs={"tool_name": tool_name, "iteration": 0},
                outputs={"result": node_id},
            )
        )

    completed = [
        kwargs["id_execution"]
        for name, kwargs in port.calls
        if name == "complete_execution"
    ]
    assert completed[-2:] == [
        tool_execution_by_name["tool-a"],
        tool_execution_by_name["tool-b"],
    ]


@pytest.mark.asyncio
async def test_assistant_message_id_is_forwarded_to_usage_fact():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
        id_agent="agent-1",
        assistant_message=AssistantMessageContext(
            id_message="msg-123",
            id_conversation="chat-1",
            id_user_message="msg-user",
            id_agent="agent-1",
        ),
    )

    await hook.on_graph_start(context())
    await hook.on_llm_start(context(node_id="llm-1", inputs={"model": "gpt-4", "provider": "openai"}))
    await hook.on_llm_end(
        context(
            node_id="llm-1",
            outputs={
                "model": "gpt-4",
                "provider": "openai",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "provider_request_id": "req-1",
            },
        )
    )

    usage_call = next(kwargs for name, kwargs in port.calls if name == "record_usage_fact")
    assert usage_call["id_message"] == "msg-123"
    assert usage_call["usage_data"]["total_tokens"] == 15
    assert usage_call["provider_request_id"] == "req-1"


@pytest.mark.asyncio
async def test_failed_graph_lifecycle_marks_run_and_root_failed():
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
        id_agent="agent-1",
    )

    await hook.on_graph_start(context())
    await hook.on_graph_error(context(), RuntimeError("boom"))

    assert port.calls[-3][0] == "record_event"
    assert port.calls[-3][1]["event_type"] == "error"
    assert port.calls[-2][1] == {"id_execution": "exec-1", "status": "failed"}
    assert port.calls[-1][1] == {"id_run": "run-1", "status": "failed"}


@pytest.mark.asyncio
async def test_on_llm_end_consumes_detail_fields():
    """P1.7 (b): GPH.on_llm_end() consumes forwarded detail fields.

    When response_data contains detail fields (cached_tokens_read, etc.),
    they MUST be forwarded to record_usage_fact() in the usage_data dict.
    """
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
        id_agent="agent-1",
    )

    await hook.on_graph_start(context())
    await hook.on_llm_start(context(node_id="llm-1", inputs={"model": "gpt-4", "provider": "openai"}))
    await hook.on_llm_end(
        context(
            node_id="llm-1",
            outputs={
                "model": "gpt-4",
                "provider": "openai",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cached_tokens_read": 3,
                "cached_tokens_write": 1,
                "reasoning_tokens": 7,
                "audio_tokens": 0,
                "raw_usage_json": {"cache_read": 3},
                "provider_request_id": "req-1",
            },
        )
    )

    usage_call = next(kwargs for name, kwargs in port.calls if name == "record_usage_fact")
    usage_data = usage_call["usage_data"]
    assert usage_data["prompt_tokens"] == 10
    assert usage_data["completion_tokens"] == 5
    assert usage_data["total_tokens"] == 15
    assert usage_data["cached_tokens_read"] == 3
    assert usage_data["cached_tokens_write"] == 1
    assert usage_data["reasoning_tokens"] == 7
    assert usage_data["audio_tokens"] == 0
    assert usage_data["raw_usage_json"] == {"cache_read": 3}
    # call_source defaults to "hook" (not passed in constructor)
    assert usage_call["call_source"] == "hook"


@pytest.mark.asyncio
async def test_on_llm_end_missing_detail_fields_default_safely():
    """P1.7 (b): Missing detail fields default to 0/None without KeyError."""
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
        id_agent="agent-1",
    )

    await hook.on_graph_start(context())
    await hook.on_llm_start(context(node_id="llm-1", inputs={"model": "gpt-4", "provider": "openai"}))
    await hook.on_llm_end(
        context(
            node_id="llm-1",
            outputs={
                "model": "gpt-4",
                "provider": "openai",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                # Note: NO detail fields — simulates pre-fix HookRelay
            },
        )
    )

    usage_call = next(kwargs for name, kwargs in port.calls if name == "record_usage_fact")
    usage_data = usage_call["usage_data"]
    # Basic fields present
    assert usage_data["prompt_tokens"] == 10
    # Detail fields default to 0/None (no KeyError)
    assert usage_data["cached_tokens_read"] == 0
    assert usage_data["cached_tokens_write"] == 0
    assert usage_data["reasoning_tokens"] == 0
    assert usage_data["audio_tokens"] == 0
    assert usage_data["raw_usage_json"] is None


@pytest.mark.asyncio
async def test_on_llm_end_configurable_call_source():
    """P1.7 (b): call_source passed to constructor is forwarded to record_usage_fact."""
    port = FakeExecutionPersistencePort()
    hook = GraphPersistenceHook(
        sink=port,
        id_chat="chat-1",
        id_thread="thread-1",
        id_user="user-1",
        id_agent="agent-1",
        call_source="agent",
    )

    await hook.on_graph_start(context())
    await hook.on_llm_start(context(node_id="llm-1", inputs={"model": "gpt-4", "provider": "openai"}))
    await hook.on_llm_end(
        context(
            node_id="llm-1",
            outputs={
                "model": "gpt-4",
                "provider": "openai",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        )
    )

    usage_call = next(kwargs for name, kwargs in port.calls if name == "record_usage_fact")
    assert usage_call["call_source"] == "agent"


# ── P1-3: Persistence Wiring Tests (RuntimeConfig integration) ───────────────


def test_build_persistence_hook_defaults():
    """P1-3 Scenario 1: Default RuntimeConfig (no sink) returns None."""
    config = RuntimeConfig()
    hook = config.build_persistence_hook(
        id_chat="chat-1", id_thread="thread-1", id_user="user-1",
    )
    assert hook is None, "Default config with no sink must return None"


def test_build_persistence_hook_enabled_with_sink():
    """P1-3 Scenario 2: persistence_enabled=True + sink → GraphPersistenceHook."""
    sink = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    config.enable_persistence(sink)

    hook = config.build_persistence_hook(
        id_chat="chat-1", id_thread="thread-1", id_user="user-1",
        id_agent="agent-1",
    )
    assert hook is not None, "Enabled with sink must return a hook"
    assert isinstance(hook, GraphPersistenceHook)
    assert hook._sink is sink
    assert hook._id_chat == "chat-1"
    assert hook._id_thread == "thread-1"
    assert hook._id_user == "user-1"
    assert hook._id_agent == "agent-1"


def test_persistence_enabled_without_sink_logs_warning(caplog):
    """P1-3 Scenario 6: persistence_enabled=True without sink logs WARNING."""
    caplog.set_level(logging.WARNING)
    config = RuntimeConfig()
    # Default: persistence_enabled=True, but _persistence_sink is None

    with caplog.at_level(logging.WARNING):
        hook = config.build_persistence_hook(
            id_chat="chat-1", id_thread="thread-1", id_user="user-1",
        )

    assert hook is None, "No sink → must return None"
    assert any(
        "persistence_enabled=True" in rec.message and "no persistence_sink" in rec.message
        for rec in caplog.records
    ), "Warning must mention enabled=True and missing sink"


def test_build_persistence_hook_respects_disabled_flag():
    """P1-3: persistence_enabled=False returns None even with sink."""
    sink = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    # Manually disable after enabling (simulates overriding)
    config._persistence_enabled = False
    config._persistence_sink = sink

    hook = config.build_persistence_hook(
        id_chat="chat-1", id_thread="thread-1", id_user="user-1",
    )
    assert hook is None, "Disabled flag must return None even with sink"


def test_is_persistence_enabled_default_true():
    """P1-3 Design: persistence_enabled defaults to True."""
    config = RuntimeConfig()
    assert config.is_persistence_enabled() is True


# ── P1-4: DebugSSE Auto-Wiring Tests (RuntimeConfig integration) ────────────


def test_build_debug_sse_hook_enabled_no_sink():
    """P1-4: debug_sse_enabled=True but no sink → None."""
    config = RuntimeConfig()
    config._debug_sse_enabled = True  # Enabled but no sink configured

    hook = config.build_debug_sse_hook(id_chat="chat-1")
    assert hook is None, "Enabled without sink must return None"


def test_build_debug_sse_hook_disabled_with_sink():
    """P1-4: debug_sse_enabled=False even with sink → None."""
    config = RuntimeConfig()
    config._debug_sse_sink = asyncio.Queue()  # Sink present but disabled

    hook = config.build_debug_sse_hook(id_chat="chat-1")
    assert hook is None, "Disabled even with sink must return None"


def test_is_debug_sse_enabled_after_enable():
    """P1-4: enable_debug_sse sets enabled=True."""
    config = RuntimeConfig()
    config.enable_debug_sse(asyncio.Queue())
    assert config.is_debug_sse_enabled() is True


# ── P1-3: AgentFlowModel Graph Override Integration ──────────────────────────


class FakeGraph:
    """Minimal AgentFlowModel-like object for testing graph override behavior.

    Duck-types the single attribute _wire_hooks_to_registry cares about:
    `persistence_enabled`.
    """
    def __init__(self, persistence_enabled=None):
        self.persistence_enabled = persistence_enabled


class FakeHookRegistry:
    """Minimal HookRegistry stub that tracks register_global calls."""
    def __init__(self):
        self.global_hooks: list = []

    def register_global(self, hook) -> None:
        self.global_hooks.append(hook)

    def is_empty(self) -> bool:
        return True


def _count_gph_hooks(registry: FakeHookRegistry) -> int:
    """Count GraphPersistenceHook instances in global hooks."""
    return sum(1 for h in registry.global_hooks if isinstance(h, GraphPersistenceHook))


def _count_dsse_hooks(registry: FakeHookRegistry) -> int:
    """Count DebugSSEHook instances in global hooks."""
    return sum(1 for h in registry.global_hooks if isinstance(h, DebugSSEHook))


def test_graph_override_false_wins_over_runtime_config():
    """P1-3 Scenario 3: graph.persistence_enabled=False overrides RuntimeConfig True.

    Even when RuntimeConfig has persistence enabled + sink, the graph-level
    False must prevent hook registration.
    """
    sink = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    config.enable_persistence(sink)

    graph = FakeGraph(persistence_enabled=False)
    registry = FakeHookRegistry()

    # Simulate _wire_hooks_to_registry logic
    from magic_agents.execution.reactive_executor import _wire_hooks_to_registry
    _wire_hooks_to_registry(registry, config, graph, "c1", "t1", "u1")

    assert _count_gph_hooks(registry) == 0, "Graph override False must suppress hook"


def test_graph_defer_to_runtime_config_when_none():
    """P1-3 Scenario 4: graph.persistence_enabled=None defers to RuntimeConfig True."""
    sink = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    config.enable_persistence(sink)

    graph = FakeGraph(persistence_enabled=None)  # Default — defer
    registry = FakeHookRegistry()

    from magic_agents.execution.reactive_executor import _wire_hooks_to_registry
    _wire_hooks_to_registry(registry, config, graph, "c1", "t1", "u1")

    assert _count_gph_hooks(registry) == 1, "None must defer to RuntimeConfig True"


def test_graph_defer_skips_when_runtimeconfig_disabled():
    """P1-3: graph=None defers to RuntimeConfig but RuntimeConfig disabled → no hook."""
    config = RuntimeConfig()
    config._persistence_enabled = False  # Disabled globally
    config._persistence_sink = FakeExecutionPersistencePort()

    graph = FakeGraph(persistence_enabled=None)
    registry = FakeHookRegistry()

    from magic_agents.execution.reactive_executor import _wire_hooks_to_registry
    _wire_hooks_to_registry(registry, config, graph, "c1", "t1", "u1")

    assert _count_gph_hooks(registry) == 0, "None → defer but RuntimeConfig disabled → no hook"


def test_debug_sse_auto_wired_alongside_persistence():
    """P1-4 Scenario 9: Both hooks are wired via register_global on same registry."""
    p_sink = FakeExecutionPersistencePort()
    d_sink: asyncio.Queue = asyncio.Queue()

    config = RuntimeConfig()
    config.enable_persistence(p_sink)
    config.enable_debug_sse(d_sink)

    graph = FakeGraph(persistence_enabled=None)
    registry = FakeHookRegistry()

    from magic_agents.execution.reactive_executor import _wire_hooks_to_registry
    _wire_hooks_to_registry(registry, config, graph, "c1", "t1", "u1")

    assert _count_gph_hooks(registry) == 1, "Persistence hook must be registered"
    assert _count_dsse_hooks(registry) == 1, "DebugSSE hook must be registered"
    assert len(registry.global_hooks) == 2, "Both hooks on same registry"


def test_hook_wiring_noop_when_runtime_config_none():
    """P1-3: _wire_hooks_to_registry is no-op when runtime_config is None."""
    registry = FakeHookRegistry()
    graph = FakeGraph()

    from magic_agents.execution.reactive_executor import _wire_hooks_to_registry
    _wire_hooks_to_registry(registry, None, graph, "c1", "t1", "u1")

    assert len(registry.global_hooks) == 0, "No hooks registered with None RuntimeConfig"


def test_hook_wiring_debug_sse_default_not_wired():
    """P1-4: Default RuntimeConfig (debug_sse disabled) does not wire DebugSSEHook."""
    p_sink = FakeExecutionPersistencePort()
    config = RuntimeConfig()
    config.enable_persistence(p_sink)
    # debug_sse is NOT enabled (default False)

    graph = FakeGraph()
    registry = FakeHookRegistry()

    from magic_agents.execution.reactive_executor import _wire_hooks_to_registry
    _wire_hooks_to_registry(registry, config, graph, "c1", "t1", "u1")

    assert _count_gph_hooks(registry) == 1, "Persistence wired normally"
    assert _count_dsse_hooks(registry) == 0, "DebugSSE not wired (default disabled)"
