"""
P0 Regression Tests for NodeInner — guards against re-introduction of the two
P0 issues identified in `docs/inner/NODE_INNER_REVIEW.md` §11 and analyzed in
`docs/inner/P0_REGRESSION_ANALYSIS.md`.

Suite scope:
- P0-1 canaries (3 tests): NodeInner MUST NOT mutate chat_log.run_id /
  parent_run_id. The fix deleted the in-place assignments at the (former)
  NodeInner.py:122-123 site. These tests guard the invariant.
- P0-2 hook propagation (5 tests): NodeInner MUST forward parent hooks into
  the recursive child graph execution and MUST merge inner_graph.hooks. The
  fix switched the recursive call to `execute_graph_reactive` and now passes
  `hooks=self._hooks` (with inner_graph.hooks merged).

These tests are designed to PASS today (post-fix). If any test goes RED on a
green main branch, the fix has regressed and the offending behavior must be
restored before merging unrelated work.

Mirrors style and fixtures from `test/test_inner_node_integration.py`.
"""
from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import NodeInner

# Helpers copied verbatim from test/test_inner_node_integration.py — the test
# directory is not a Python package so cross-file imports aren't available.
def extract_streamed_content(item):
    """Extract streamed content from send_message output."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "content":
        return ""
    content = item.get("content")
    if content is None:
        return ""
    if hasattr(content, "choices") and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
    return ""


def get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    executed = set()
    if not debug_summary:
        return executed
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            executed.add(node.get("node_id"))
    return executed


# ─── Recording hook (mirrors patterns from tests/integration/test_3_tier_hooks.py) ──

class _RecordingHook:
    """FlowHooks-protocol implementation that captures every invocation.

    Intentionally not a pytest test class. Stateful: holds a per-instance list
    of (hook_name, node_id, run_id, parent_run_id) tuples.
    """

    __test__ = False

    def __init__(self, label: str = "rec"):
        self.label = label
        self.calls: List[dict] = []

    async def on_graph_start(self, context: HookContext) -> None:
        self.calls.append(self._snap("on_graph_start", context))

    async def on_graph_end(self, context: HookContext) -> None:
        self.calls.append(self._snap("on_graph_end", context))

    async def on_node_start(self, context: HookContext) -> None:
        self.calls.append(self._snap("on_node_start", context))

    async def on_node_end(self, context: HookContext) -> None:
        self.calls.append(self._snap("on_node_end", context))

    @staticmethod
    def _snap(name: str, ctx: HookContext) -> dict:
        return {
            "hook_name": name,
            "node_id": ctx.node_id,
            "run_id": ctx.run_id,
            "parent_run_id": ctx.parent_run_id,
            "execution_id": ctx.execution_id,
        }


# ─── Graph builders (kept tiny — we only care about lifecycle wiring) ─────────

def _simple_inner_graph() -> dict:
    """Trivial inner graph: user_input → text → end."""
    return {
        "type": "graph",
        "debug": False,
        "nodes": [
            {"id": "inner_input", "type": "user_input"},
            {"id": "inner_text", "type": "text", "data": {"text": "INNER"}},
            {"id": "inner_end", "type": "end"},
        ],
        "edges": [
            {"id": "ie1", "source": "inner_input", "target": "inner_text",
             "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
            {"id": "ie2", "source": "inner_text", "target": "inner_end",
             "sourceHandle": "handle_text_output", "targetHandle": "h1"},
        ],
    }


def _outer_graph_with_inner(inner_graph: dict, inner_id: str = "inner") -> dict:
    """Outer graph: user_input → NodeInner → send → end."""
    return {
        "type": "graph",
        "debug": True,
        "nodes": [
            {"id": "input", "type": "user_input"},
            {"id": inner_id, "type": "inner", "data": {"magic_flow": inner_graph}},
            {"id": "send", "type": "send_message",
             "data": {"message": "", "json_extras": "OUTER_OK"}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": inner_id,
             "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
            {"id": "e2", "source": inner_id, "target": "send",
             "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
            {"id": "e3", "source": "send", "target": "end",
             "sourceHandle": "handle_message_output", "targetHandle": "h1"},
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# P0-1 — chat_log MUST NOT be mutated by NodeInner
# ──────────────────────────────────────────────────────────────────────────────

class TestP0_1_ChatLogIsolation:
    """Canaries for the deleted in-place mutation (former NodeInner.py:122-123)."""

    @pytest.mark.asyncio
    async def test_chat_log_run_id_unchanged_after_inner_execution(self):
        """Outer chat_log object identity, run_id, and parent_run_id must be
        preserved across NodeInner.process() invocation.

        We invoke NodeInner.process() directly with a chat_log we control so we
        can compare object identity (id()) before/after.
        """
        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")
        inner_node: NodeInner = graph.nodes["inner"]
        # Bootstrap the inputs that the executor would normally fan in.
        inner_node.inputs[inner_node.INPUT_HANDLE] = "hi"
        inner_node._hooks = None

        chat_log = ModelAgentRunLog(
            id_chat="c1", id_thread="t1", id_user="u1",
            run_id="parent-run-XYZ",
            parent_run_id="grandparent-run-ABC",
        )
        before_id = id(chat_log)
        before_run_id = chat_log.run_id
        before_parent_run_id = chat_log.parent_run_id

        async for _ in inner_node.process(chat_log):
            pass

        assert id(chat_log) == before_id, "chat_log object identity changed"
        assert chat_log.run_id == before_run_id, "chat_log.run_id was mutated"
        assert chat_log.parent_run_id == before_parent_run_id, (
            "chat_log.parent_run_id was mutated"
        )

    @pytest.mark.asyncio
    async def test_chat_log_not_mutated_when_run_id_already_set(self):
        """Pre-set sentinel run_id must survive NodeInner execution untouched."""
        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")
        inner_node: NodeInner = graph.nodes["inner"]
        inner_node.inputs[inner_node.INPUT_HANDLE] = "hi"
        inner_node._hooks = None

        SENTINEL = "SENTINEL-RUN-ID-do-not-touch"
        chat_log = ModelAgentRunLog(run_id=SENTINEL, parent_run_id=None)

        async for _ in inner_node.process(chat_log):
            pass

        assert chat_log.run_id == SENTINEL
        assert chat_log.parent_run_id is None

    @pytest.mark.asyncio
    async def test_sibling_node_inner_chat_log_isolation(self):
        """Two NodeInner instances run sequentially against the same chat_log.

        Neither must mutate the parent chat_log's run_id / parent_run_id. This
        is the forward-looking canary for the parallel-fan-out scenario called
        out in P0_REGRESSION_ANALYSIS.md §2.6.
        """
        graph_a = build(_outer_graph_with_inner(_simple_inner_graph(), inner_id="inner_a"),
                        message="A")
        graph_b = build(_outer_graph_with_inner(_simple_inner_graph(), inner_id="inner_b"),
                        message="B")
        node_a: NodeInner = graph_a.nodes["inner_a"]
        node_b: NodeInner = graph_b.nodes["inner_b"]
        node_a.inputs[node_a.INPUT_HANDLE] = "A"
        node_b.inputs[node_b.INPUT_HANDLE] = "B"
        node_a._hooks = None
        node_b._hooks = None

        chat_log = ModelAgentRunLog(run_id="parent-shared", parent_run_id="root")
        snap_run_id = chat_log.run_id
        snap_parent = chat_log.parent_run_id

        async for _ in node_a.process(chat_log):
            pass
        # Mid-point invariant: still untouched after first sibling
        assert chat_log.run_id == snap_run_id
        assert chat_log.parent_run_id == snap_parent

        async for _ in node_b.process(chat_log):
            pass
        assert chat_log.run_id == snap_run_id
        assert chat_log.parent_run_id == snap_parent


# ──────────────────────────────────────────────────────────────────────────────
# P0-2 — hooks propagate into the child graph
# ──────────────────────────────────────────────────────────────────────────────

class TestP0_2_HookPropagation:
    """Hooks registered on the parent must fire for child-graph nodes too."""

    @pytest.fixture(autouse=True)
    def _clear_global_hooks(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_hooks_propagate_to_child_graph(self):
        """A parent-registered hook must observe child-graph node events."""
        rec = _RecordingHook(label="parent")
        RuntimeConfig.register_global_hook(rec)
        config = RuntimeConfig()

        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")

        async for _ in run_agent(graph, hooks=config):
            pass

        node_starts = [c for c in rec.calls if c["hook_name"] == "on_node_start"]
        observed_ids = {c["node_id"] for c in node_starts}

        # Outer nodes
        assert "input" in observed_ids
        assert "inner" in observed_ids
        # Inner-graph nodes — these prove propagation
        assert "inner_input" in observed_ids, (
            f"child node 'inner_input' did not fire on_node_start; observed={observed_ids}"
        )
        assert "inner_text" in observed_ids, (
            f"child node 'inner_text' did not fire on_node_start; observed={observed_ids}"
        )

    @pytest.mark.asyncio
    async def test_inner_graph_own_hooks_merged_with_parent(self):
        """Both parent hooks and inner_graph.hooks must observe child events."""
        parent_hook = _RecordingHook(label="parent")
        inner_hook = _RecordingHook(label="inner_graph")

        RuntimeConfig.register_global_hook(parent_hook)
        config = RuntimeConfig()

        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")
        # Attach a graph-level hook directly to the inner AgentFlowModel
        inner_node: NodeInner = graph.nodes["inner"]
        inner_node.inner_graph.hooks = inner_hook

        async for _ in run_agent(graph, hooks=config):
            pass

        parent_child_obs = {
            c["node_id"] for c in parent_hook.calls
            if c["hook_name"] == "on_node_start"
        }
        inner_obs = {
            c["node_id"] for c in inner_hook.calls
            if c["hook_name"] == "on_node_start"
        }

        # Parent hook sees outer + inner nodes
        assert "input" in parent_child_obs
        assert "inner_input" in parent_child_obs
        # inner_graph.hooks fires for child nodes
        assert "inner_input" in inner_obs, (
            f"inner_graph hook did not fire for child nodes; observed={inner_obs}"
        )
        assert "inner_text" in inner_obs

    @pytest.mark.asyncio
    async def test_parent_run_id_correlation_in_child_hook_context(self):
        """Child hook contexts must carry parent_run_id matching the outer run_id."""
        rec = _RecordingHook(label="parent")
        RuntimeConfig.register_global_hook(rec)
        config = RuntimeConfig()

        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")

        # Drive execution through execute_graph_reactive directly so we can
        # set a deterministic outer run_id for correlation assertions.
        OUTER_RUN_ID = "parent-run-for-correlation-check"
        registry = config.create_registry()
        if graph.hooks is not None:
            registry.register_graph(graph.hooks)

        async for _ in execute_graph_reactive(
            graph,
            extras=None,
            flow_state=None,
            run_id=OUTER_RUN_ID,
            parent_run_id=None,
            hooks=registry,
        ):
            pass

        # Find a child-node start event (e.g. inner_text) — must NOT have
        # OUTER_RUN_ID as its run_id (it has the child's run_id), but its
        # parent_run_id must equal OUTER_RUN_ID.
        child_starts = [
            c for c in rec.calls
            if c["hook_name"] == "on_node_start" and c["node_id"] in ("inner_input", "inner_text")
        ]
        assert child_starts, "no child-node hook events were captured"

        for c in child_starts:
            assert c["parent_run_id"] == OUTER_RUN_ID, (
                f"child hook for {c['node_id']} has parent_run_id={c['parent_run_id']!r}, "
                f"expected {OUTER_RUN_ID!r}"
            )
            assert c["run_id"] != OUTER_RUN_ID, (
                f"child hook for {c['node_id']} run_id collides with outer run_id "
                f"({c['run_id']!r}); child must have its own run_id"
            )

    @pytest.mark.asyncio
    async def test_debug_callback_not_double_fired(self):
        """Each debug event must reach the parent's debug_callback exactly once.

        Validates Option γ from P0_REGRESSION_ANALYSIS.md §3.5: NodeInner does
        NOT propagate debug_callback to the child executor, relying on the
        manual `yield evt` forwarding instead. If we ever start propagating
        debug_callback AND keep the manual yield, child debug events would
        double-fire — this test guards against that regression.
        """
        seen: List[Any] = []

        async def cb(event):
            seen.append(event)

        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")

        async for _ in run_agent(graph, debug_callback=cb):
            pass

        # Identity-based dedup check: each event object must appear at most once.
        ids = [id(e) for e in seen]
        assert len(ids) == len(set(ids)), (
            f"debug_callback received the same event object multiple times: "
            f"total={len(ids)}, unique={len(set(ids))}"
        )

    @pytest.mark.asyncio
    async def test_no_hooks_no_crash(self):
        """Backward-compat canary: NodeInner must run cleanly with no hooks
        registered anywhere (parent has self._hooks=None, no inner_graph.hooks).

        Guards against AttributeError / NoneType bugs in the propagation path.
        """
        graph = build(_outer_graph_with_inner(_simple_inner_graph()), message="hi")

        content = []
        debug_summary = None
        async for item in run_agent(graph):  # NO hooks kwarg, NO debug_callback
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content.append(text)

        assert "OUTER_OK" in "".join(content)
        executed = get_executed_nodes(debug_summary)
        assert "inner" in executed
        assert "send" in executed
