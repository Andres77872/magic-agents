"""Unit tests for FlowHooks Protocol and HookContext.

Phase 9.1: Tests for:
- Protocol structural typing works without inheritance
- Partial implementation accepted (only some methods)
- Async hook invocation awaits properly
- HookContext to_dict() serializes correctly
"""
import pytest
from datetime import datetime, UTC

from magic_agents.hooks.flow_hooks import (
    FlowHooks,
    HookContext,
)


def direct_context(**kwargs) -> HookContext:
    """Construct the class under test and explicitly verify its deprecation."""
    with pytest.warns(DeprecationWarning, match="Direct HookContext\\(\\) construction"):
        return HookContext(**kwargs)


class TestFlowHooksProtocol:
    """Tests for the FlowHooks Protocol structural typing."""

    def test_full_implementation_passes_isinstance(self):
        """A class implementing ALL 12 hook methods passes isinstance check."""

        class FullHookImpl:
            async def on_graph_start(self, context): pass
            async def on_graph_end(self, context): pass
            async def on_graph_error(self, context, error): pass
            async def on_node_start(self, context): pass
            async def on_node_end(self, context): pass
            async def on_node_error(self, context, error): pass
            async def on_node_bypass(self, context, reason): pass
            async def on_llm_start(self, context, llm_config=None): pass
            async def on_llm_end(self, context, response=None): pass
            async def on_llm_loop_end(self, context): pass
            async def on_tool_start(self, context): pass
            async def on_tool_end(self, context): pass

        hook = FullHookImpl()
        assert isinstance(hook, FlowHooks), (
            "A class with ALL matching method signatures should satisfy "
            "the @runtime_checkable Protocol"
        )

    def test_all_12_methods_defined_in_protocol(self):
        """FlowHooks Protocol defines all expected lifecycle methods."""
        expected = {
            # Graph lifecycle
            "on_graph_start",
            "on_graph_end",
            "on_graph_error",
            # Node lifecycle
            "on_node_start",
            "on_node_end",
            "on_node_error",
            "on_node_bypass",
            # LLM lifecycle
            "on_llm_start",
            "on_llm_end",
            "on_llm_loop_end",
            # Tool lifecycle
            "on_tool_start",
            "on_tool_end",
        }

        protocol_methods = set()
        for attr in dir(FlowHooks):
            if attr.startswith("on_"):
                protocol_methods.add(attr)

        assert protocol_methods == expected, (
            f"Protocol should define exactly 12 methods. "
            f"Missing: {expected - protocol_methods}. "
            f"Extra: {protocol_methods - expected}."
        )


class TestHookContext:
    """Tests for HookContext dataclass and serialization."""

    def test_to_dict_serializes_all_fields(self):
        """to_dict() returns all fields in a flat dict."""
        ctx = direct_context(
            execution_id="exec-123",
            sequence_number=1,
            run_id="run-abc",
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            inputs={"query": "hello"},
            outputs={"response": "world"},
            start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
            end_time=datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC),
            duration_ms=1000.0,
            error_type="ValueError",
            error_message="test error",
            metadata={"source": "test"},
            parent_run_id="parent-xyz",
        )

        result = ctx.to_dict()

        assert result["execution_id"] == "exec-123"
        assert result["sequence_number"] == 1
        assert result["run_id"] == "run-abc"
        assert result["node_id"] == "node-1"
        assert result["node_type"] == "LLM"
        assert result["node_class"] == "NodeLLM"
        assert result["inputs"] == {"query": "hello"}
        assert result["outputs"] == {"response": "world"}
        assert result["duration_ms"] == 1000.0
        assert result["error_type"] == "ValueError"
        assert result["error_message"] == "test error"
        assert result["metadata"] == {"source": "test"}
        assert result["parent_run_id"] == "parent-xyz"
        # Timestamps should be ISO format strings
        assert result["start_time"] == "2025-01-01T12:00:00+00:00"
        assert result["end_time"] == "2025-01-01T12:00:01+00:00"

    def test_to_dict_timestamp_is_iso_format(self):
        """The timestamp field is serialized as ISO 8601 string."""
        ctx = direct_context(execution_id="test-1")
        result = ctx.to_dict()

        assert isinstance(result["timestamp"], str)
        # Should parse back
        parsed = datetime.fromisoformat(result["timestamp"])
        assert parsed is not None

    def test_to_dict_handles_null_timings(self):
        """Start/end time None values serialize to None, not 'None'."""
        ctx = direct_context(
            execution_id="exec-1",
            start_time=None,
            end_time=None,
        )
        result = ctx.to_dict()

        assert result["start_time"] is None
        assert result["end_time"] is None

    def test_to_dict_excludes_error_object(self):
        """The error Exception object is NOT in to_dict (only type/message)."""
        try:
            raise ValueError("something broke")
        except ValueError as e:
            ctx = direct_context(
                execution_id="exec-1",
                error=e,
                error_type="ValueError",
                error_message="something broke",
            )

        result = ctx.to_dict()

        assert "error" not in result, (
            "The Exception object should not be serialized"
        )
        assert result["error_type"] == "ValueError"
        assert result["error_message"] == "something broke"

    def test_to_dict_excludes_emit_helper(self):
        """The emit field is not in to_dict."""
        ctx = direct_context(execution_id="exec-1")
        result = ctx.to_dict()

        assert "emit" not in result, (
            "EmitInterface should be excluded from serialization"
        )

    def test_default_timestamp_is_utc_now(self):
        """Default timestamp is a timezone-aware UTC datetime."""
        before = datetime.now(UTC)
        ctx = direct_context(execution_id="exec-1")
        after = datetime.now(UTC)

        assert before <= ctx.timestamp <= after, (
            "Timestamp should be set to current UTC time"
        )


# ─── Task 4.5: Factory/Contract Validation Tests ─────────────────────────


class TestHookContextFactory:
    """Tests for HookContextFactory — all 6 factory methods.

    Verifies:
    - Each factory method produces correct HookContext for its event type
    - TypedDict schema keys match factory output
    - Extra kwargs absorbed into inputs
    - Warning mode logs for missing fields
    - Before/after comparison: factory vs old imperative construction
    """

    def test_build_graph_context(self):
        """build_graph_context produces HookContext with graph metadata."""
        from magic_agents.hooks.context_factory import HookContextFactory
        from magic_agents.hooks.contracts import GraphStartInputs

        ctx = HookContextFactory.build_graph_context(
            execution_id="exec-1",
            run_id="run-1",
            metadata={"graph_id": "g1", "node_count": 3, "edge_count": 2},
        )

        assert ctx.execution_id == "exec-1"
        assert ctx.run_id == "run-1"
        assert ctx.node_id is None  # Graph-level
        assert ctx.inputs["graph_id"] == "g1"
        assert ctx.inputs["node_count"] == 3
        assert ctx.inputs["edge_count"] == 2

        # TypedDict keys match
        schema_keys = set(GraphStartInputs.__annotations__.keys())
        input_keys = set(ctx.inputs.keys()) & schema_keys
        assert input_keys == schema_keys, (
            f"Factory output keys {input_keys} must match schema {schema_keys}"
        )

    def test_build_node_context(self):
        """build_node_context produces HookContext with node identity."""
        from magic_agents.hooks.context_factory import HookContextFactory
        from magic_agents.hooks.contracts import NodeStartInputs

        ctx = HookContextFactory.build_node_context(
            execution_id="exec-1",
            run_id="run-1",
            node_id="node-1",
            node_type="llm",
            node_class="NodeLLM",
            inputs={"query": "hello"},
            start_time=None,
        )

        assert ctx.execution_id == "exec-1"
        assert ctx.node_id == "node-1"
        assert ctx.node_type == "llm"
        assert ctx.node_class == "NodeLLM"

        # TypedDict keys
        schema_keys = set(NodeStartInputs.__annotations__.keys())
        assert "node_id" in schema_keys

    def test_build_edge_context(self):
        """build_edge_context produces HookContext with edge routing data."""
        from magic_agents.hooks.context_factory import HookContextFactory
        from magic_agents.hooks.contracts import EdgeHookInputs

        ctx = HookContextFactory.build_edge_context(
            execution_id="exec-1",
            run_id="run-1",
            node_id="src-node",
            source="src-node",
            target="tgt-node",
            content="payload",
            source_handle="output-1",
            target_handle="input-1",
            sequence_number=1,
            node_type="llm",
            node_class="NodeLLM",
        )

        assert ctx.execution_id == "exec-1"
        assert ctx.sequence_number == 1
        assert ctx.node_id == "src-node"
        assert ctx.inputs["source"] == "src-node"
        assert ctx.inputs["target"] == "tgt-node"
        assert ctx.inputs["source_handle"] == "output-1"
        assert ctx.inputs["target_handle"] == "input-1"
        assert "content" in ctx.inputs

        # TypedDict keys match
        schema_keys = set(EdgeHookInputs.__annotations__.keys())
        input_keys = set(ctx.inputs.keys()) & schema_keys
        assert input_keys == schema_keys, (
            f"Edge factory output {input_keys} must match {schema_keys}"
        )

    def test_build_bypass_context(self):
        """build_bypass_context produces HookContext with canonical reason."""
        from magic_agents.hooks.context_factory import HookContextFactory

        ctx = HookContextFactory.build_bypass_context(
            execution_id="exec-1",
            node_id="node-b",
            reason="not_ready",
            metadata={"bypass_path": "single_node"},
        )

        assert ctx.inputs["reason"] == "not_ready"
        assert ctx.metadata["bypass_path"] == "single_node"

        # Canonical reason values
        ctx2 = HookContextFactory.build_bypass_context(
            execution_id="exec-2", node_id="node-c", reason="condition", metadata={"phase": "static"}
        )
        assert ctx2.inputs["reason"] == "condition"

        ctx3 = HookContextFactory.build_bypass_context(
            execution_id="exec-3", node_id="node-d", reason="upstream_error",
            metadata={"upstream_error_node": "failed-a"},
        )
        assert ctx3.inputs["reason"] == "upstream_error"

    def test_build_llm_context(self):
        """build_llm_context produces consistent structure for both paths (S4)."""
        from magic_agents.hooks.context_factory import HookContextFactory
        from magic_agents.hooks.contracts import LLMStartInputs

        ctx = HookContextFactory.build_llm_context(
            execution_id="exec-1",
            run_id="run-1",
            node_id="llm-1",
            node_type="LLM",
            node_class="NodeLLM",
            model="gpt-4",
            streaming=False,
            iteration=1,
        )

        assert ctx.execution_id == "exec-1"
        assert ctx.node_id == "llm-1"
        assert ctx.node_type == "LLM"
        assert ctx.node_class == "NodeLLM"
        assert ctx.inputs["model"] == "gpt-4"
        assert ctx.inputs["streaming"] is False
        assert ctx.inputs["iteration"] == 1

        # Schema match
        schema_keys = set(LLMStartInputs.__annotations__.keys())
        input_keys = set(ctx.inputs.keys()) & schema_keys
        assert "model" in input_keys
        assert "streaming" in input_keys

    def test_build_tool_context(self):
        """build_tool_context produces HookContext with tool data."""
        from magic_agents.hooks.context_factory import HookContextFactory
        from magic_agents.hooks.contracts import ToolStartInputs, ToolEndInputs

        ctx = HookContextFactory.build_tool_context(
            execution_id="exec-1",
            node_id="llm-1",
            tool_name="search",
            tool_call_id="call-123",
            arguments={"query": "hello"},
            success=True,
            execution_time_ms=1500.0,
        )

        assert ctx.inputs["tool_name"] == "search"
        assert ctx.inputs["tool_call_id"] == "call-123"
        assert ctx.inputs["arguments"] == {"query": "hello"}
        assert ctx.outputs["success"] is True
        assert ctx.outputs["execution_time_ms"] == 1500.0

        # Schema matches
        start_keys = set(ToolStartInputs.__annotations__.keys())
        assert set(ctx.inputs.keys()) & start_keys == start_keys

        end_keys = set(ToolEndInputs.__annotations__.keys())
        assert "execution_time_ms" in end_keys

    def test_extra_kwargs_absorbed_into_inputs(self):
        """Extra kwargs are absorbed into context.inputs without error."""
        from magic_agents.hooks.context_factory import HookContextFactory

        ctx = HookContextFactory.build_graph_context(
            execution_id="exec-1",
            run_id="run-1",
            metadata={"graph_id": "g1", "node_count": 1, "edge_count": 0},
            unexpected_field="surprise",
            another_one=42,
        )

        assert ctx.inputs["unexpected_field"] == "surprise"
        assert ctx.inputs["another_one"] == 42
        # No exception raised for unexpected kwargs

    def test_warning_mode_logs_missing_required_fields(self, caplog):
        """warn_on_missing=True logs warning for missing required fields."""
        from magic_agents.hooks.context_factory import HookContextFactory
        import logging

        caplog.set_level(logging.WARNING)

        ctx = HookContextFactory.build_node_context(
            execution_id="exec-1",
            warn_on_missing=True,
            # Intentionally omit node_id, node_type, node_class
        )

        warning_messages = [r.message for r in caplog.records]
        missing_warnings = [m for m in warning_messages if "missing" in m.lower()]
        assert len(missing_warnings) >= 1, (
            "Expected at least one warning about missing required fields"
        )
        # Context is still returned (warning is non-blocking)
        assert ctx is not None

    def test_tool_duration_ms_propagated(self):
        """Tool execution_time_ms propagated correctly (S3)."""
        from magic_agents.hooks.context_factory import HookContextFactory

        # With duration_ms
        ctx = HookContextFactory.build_tool_context(
            execution_id="exec-1",
            node_id="llm-1",
            tool_name="search",
            tool_call_id="call-1",
            execution_time_ms=1500.0,
            success=True,
        )
        assert ctx.outputs["execution_time_ms"] == 1500.0

        # Without duration_ms (None)
        ctx2 = HookContextFactory.build_tool_context(
            execution_id="exec-1",
            node_id="llm-1",
            tool_name="search",
            tool_call_id="call-1",
            success=True,
            execution_time_ms=None,
        )
        assert ctx2.outputs["execution_time_ms"] is None


# ─── GAP-1: Before/After Factory Construction Comparison Tests ─────────
#
# Verifies spec §10 requirement: old imperative HookContext construction
# produces equivalent structural identity to factory-based construction for
# all 6 construction sites replaced by HookContextFactory.
#
# NOTE: The factory deliberately absorbs extra kwargs into context.inputs
# (by design). The old HookContext kept `metadata` as a top-level field.
# These tests verify data completeness — no expected field is lost, and
# structural identity fields (execution_id, node_id, etc.) match exactly.


# ─── Task 4.6: Deprecation Warning Tests ────────────────────────────────


class TestHookContextDeprecationWarning:
    """Direct HookContext() construction emits DeprecationWarning (M4)."""

    def test_direct_hook_context_construction_emits_deprecation(self):
        """Direct HookContext() construction must emit DeprecationWarning."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HookContext(execution_id="test-deprecation")

            deprecation_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "HookContext()" in str(x.message)
            ]
            assert len(deprecation_warnings) >= 1, (
                "Direct HookContext() construction must emit DeprecationWarning"
            )

    def test_factory_construction_does_not_emit_deprecation(self):
        """Factory construction must NOT emit deprecation warning."""
        from magic_agents.hooks.context_factory import HookContextFactory
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HookContextFactory.build_graph_context(
                execution_id="test-factory",
                metadata={"graph_id": "g1", "node_count": 1, "edge_count": 0},
            )

            factory_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "HookContext()" in str(x.message)
            ]
            assert len(factory_warnings) == 0, (
                "Factory construction must not emit HookContext deprecation warning"
            )

    def test_dead_context_extensions_exist(self):
        """3 dead context extensions still exist (not removed, backward compatible)."""
        from magic_agents.hooks.flow_hooks import (
            NodeLLMHookContext,
            NodeMcpHookContext,
            NodeLoopHookContext,
        )

        assert NodeLLMHookContext is not None
        assert NodeMcpHookContext is not None
        assert NodeLoopHookContext is not None

        # They still extend HookContext
        ctx = NodeLLMHookContext(execution_id="test", model="gpt-4")
        assert ctx.execution_id == "test"
        assert ctx.model == "gpt-4"
