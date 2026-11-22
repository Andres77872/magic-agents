"""
Integration Tests: RuntimeConfig.is_empty() scope (Task 4.7 / M6).

Proves that RuntimeConfig.is_empty() checks only instance-level _graph_hooks,
not class-level _global_hooks. Verifies the has_global_hooks() classmethod.

Scenarios:
1. Instance with graph hooks → is_empty() returns False
2. Instance without hooks but global hooks exist → is_empty() returns True,
   has_global_hooks() returns True
3. No hooks anywhere → both return False
"""
import asyncio

import pytest

from magic_agents.agt_flow import build, execute_graph_loop
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.hooks.flow_hooks import HookContext


class InMemoryPersistenceSink:
    """Concrete persistence port used to verify executor auto-wiring."""

    def __init__(self):
        self.calls = []
        self._next_id = 0

    def _id(self, prefix):
        self._next_id += 1
        return f"{prefix}-{self._next_id}"

    async def begin_run(self, **kwargs):
        self.calls.append(("begin_run", kwargs))
        return self._id("run")

    async def begin_execution(self, **kwargs):
        self.calls.append(("begin_execution", kwargs))
        return self._id("execution")

    async def record_event(self, **kwargs):
        self.calls.append(("record_event", kwargs))
        return self._id("event")

    async def record_usage_fact(self, **kwargs):
        self.calls.append(("record_usage_fact", kwargs))
        return self._id("usage"), 0.0

    async def complete_execution(self, **kwargs):
        self.calls.append(("complete_execution", kwargs))

    async def complete_run(self, **kwargs):
        self.calls.append(("complete_run", kwargs))


class TestIsEmptyScoping:
    """RuntimeConfig.is_empty() must check instance scope only."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    def test_instance_with_graph_hooks_is_not_empty(self):
        """is_empty() returns False when graph hooks are registered."""
        class TestHook:
            async def on_graph_start(self, context: HookContext) -> None:
                pass

        config = RuntimeConfig(graph_hooks=[TestHook()])
        assert not config.is_empty(), (
            "is_empty() should return False when graph hooks are registered"
        )
        assert not RuntimeConfig.has_global_hooks(), (
            "has_global_hooks() should return False when no global hooks"
        )

    def test_instance_without_hooks_but_global_hooks_exist(self):
        """is_empty() returns True with only global hooks; has_global_hooks() returns True."""
        class TestHook:
            async def on_graph_start(self, context: HookContext) -> None:
                pass

        # Register a global hook
        RuntimeConfig.register_global_hook(TestHook())

        config = RuntimeConfig()  # No graph hooks
        assert config.is_empty(), (
            "is_empty() should return True when only global hooks exist"
        )
        assert RuntimeConfig.has_global_hooks(), (
            "has_global_hooks() should return True when global hooks registered"
        )

    def test_no_hooks_anywhere(self):
        """Both is_empty() and has_global_hooks() return False with no hooks."""
        config = RuntimeConfig()
        assert config.is_empty(), (
            "is_empty() should return True with no hooks"
        )
        assert not RuntimeConfig.has_global_hooks(), (
            "has_global_hooks() should return False with no global hooks"
        )

    def test_clear_global_hooks_affects_has_global_hooks(self):
        """Clearing global hooks changes has_global_hooks() to False."""
        class TestHook:
            async def on_graph_start(self, context: HookContext) -> None:
                pass

        RuntimeConfig.register_global_hook(TestHook())
        assert RuntimeConfig.has_global_hooks()

        RuntimeConfig.clear_global_hooks()
        assert not RuntimeConfig.has_global_hooks()

    def test_multiple_graph_hooks_still_not_empty(self):
        """Multiple graph hooks still make is_empty() return False."""
        class HookA:
            async def on_node_start(self, context): pass

        class HookB:
            async def on_node_end(self, context): pass

        config = RuntimeConfig(graph_hooks=[HookA(), HookB()])
        assert not config.is_empty()

    def test_create_registry_with_global_hooks_and_no_graph_hooks(self):
        """create_registry works when is_empty() returns True due to global hooks."""
        class TestHook:
            async def on_graph_start(self, context: HookContext) -> None:
                pass

        RuntimeConfig.register_global_hook(TestHook())
        config = RuntimeConfig()

        # This should work fine — empty instance but global hooks exist
        registry = config.create_registry()
        assert not registry.is_empty(), (
            "Registry should have the global hook even though config is empty"
        )

    def test_has_configured_hooks_includes_global_and_auto_wired_hooks(self):
        class TestHook:
            async def on_graph_start(self, context: HookContext) -> None:
                pass

        config = RuntimeConfig()
        assert config.has_configured_hooks() is False

        RuntimeConfig.register_global_hook(TestHook())
        assert config.has_configured_hooks() is True

        RuntimeConfig.clear_global_hooks()
        config.enable_debug_sse(__import__("asyncio").Queue())
        assert config.has_configured_hooks() is True

    @pytest.mark.asyncio
    async def test_direct_loop_wrapper_auto_wires_debug_sse_exactly_once(self):
        """The public loop wrapper must not omit or duplicate auto-wired hooks."""
        graph = build({
            "type": "graph",
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "items", "type": "text", "data": {"text": '["one"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {"text": "done"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "items",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "items", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "end",
                 "sourceHandle": "handle_end", "targetHandle": "handle_flow_input"},
            ],
        }, message="run")
        sink = asyncio.Queue()
        persistence_sink = InMemoryPersistenceSink()
        config = RuntimeConfig()
        config.enable_debug_sse(sink)
        config.enable_persistence(persistence_sink)

        async for _item in execute_graph_loop(
            graph,
            id_chat="chat",
            id_thread="thread",
            id_user="user",
            hooks=config,
        ):
            pass

        events = []
        while not sink.empty():
            events.append(sink.get_nowait())
        event_types = [event["event_type"] for event in events]

        assert event_types.count("graph_start") == 1
        assert event_types.count("graph_end") == 1
        assert event_types.count("node_start") == len(graph.nodes)
        assert event_types.count("node_end") == len(graph.nodes)
        assert {
            event["content"]["node_id"]
            for event in events
            if event["event_type"] == "node_start"
        } == set(graph.nodes)
        assert {
            event["content"]["node_id"]
            for event in events
            if event["event_type"] == "node_end"
        } == set(graph.nodes)

        persisted_events = [
            kwargs["event_type"]
            for name, kwargs in persistence_sink.calls
            if name == "record_event"
        ]
        assert persisted_events.count("graph_start") == 1
        assert persisted_events.count("graph_end") == 1
        assert persisted_events.count("node_start") == len(graph.nodes)
        assert persisted_events.count("node_end") == len(graph.nodes)
        assert sum(name == "begin_run" for name, _ in persistence_sink.calls) == 1
        assert sum(name == "complete_run" for name, _ in persistence_sink.calls) == 1
