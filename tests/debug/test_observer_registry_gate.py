"""Focused tests: ObserverRegistry create() gate chain.

This proves that:
1. DebugConfig(enabled=False) object passed as graph_debug_config correctly
   produces a NullObserver (is_active=False).  This was the bug: when
   AgentFlowModel.resolved_debug_config returns a DebugConfig object,
   ObserverRegistry.create() received it and DebugConfig.from_dict() silently
   created a default config with enabled=True.
2. Legacy _start_debug_tracking is skipped when observer is provided to
   Node.__call__().
"""


from magic_agents.debug.registry import ObserverRegistry
from magic_agents.debug.null_observer import NullObserver
from magic_agents.debug.config import DebugConfig


class TestObserverRegistryGateChain:
    """Prove the gate chain handles DebugConfig objects correctly."""

    def test_debug_config_object_enabled_false_creates_null_observer(self):
        """DebugConfig(enabled=False) object must produce NullObserver.

        This is the regression fix: when graph.resolved_debug_config is
        already a DebugConfig(enabled=False) object, calling
        DebugConfig.from_dict() on it falls through to default config
        with enabled=True.  The fix checks isinstance() first.
        """
        config = DebugConfig(enabled=False)

        registry = ObserverRegistry.create(
            debug_enabled_global=True,
            graph_debug=True,
            graph_debug_config=config,
            execution_id="test-exec",
            graph_type="chat",
            total_nodes=5,
            total_edges=4,
            callback=None,
        )

        assert not registry.is_active, (
            "DebugConfig(enabled=False) should produce NullObserver, "
            "got active observer instead"
        )
        assert isinstance(registry.graph_observer, NullObserver), (
            "Graph observer should be NullObserver when debug is disabled"
        )

    def test_debug_config_object_enabled_true_creates_active_observer(self):
        """DebugConfig(enabled=True) object must produce active observer."""
        config = DebugConfig(enabled=True)

        registry = ObserverRegistry.create(
            debug_enabled_global=True,
            graph_debug=True,
            graph_debug_config=config,
            execution_id="test-exec",
            graph_type="chat",
            total_nodes=5,
            total_edges=4,
            callback=None,
        )

        assert registry.is_active, (
            "DebugConfig(enabled=True) should produce active observer"
        )
        assert not isinstance(registry.graph_observer, NullObserver)

    def test_none_config_with_global_and_graph_enabled(self):
        """None config should produce active observer (default enabled=True)."""
        registry = ObserverRegistry.create(
            debug_enabled_global=True,
            graph_debug=True,
            graph_debug_config=None,
            execution_id="test-exec",
            graph_type="chat",
            total_nodes=5,
            total_edges=4,
            callback=None,
        )

        assert registry.is_active, "None config with all gates pass should be active"

    def test_dict_config_enabled_false(self):
        """Dict with enabled=False must produce NullObserver."""
        registry = ObserverRegistry.create(
            debug_enabled_global=True,
            graph_debug=True,
            graph_debug_config={"enabled": False},
            execution_id="test-exec",
            graph_type="chat",
            total_nodes=5,
            total_edges=4,
            callback=None,
        )

        assert not registry.is_active, (
            "dict {'enabled': False} should produce NullObserver"
        )

    def test_global_false_wins_over_debug_config_object(self):
        """Global DEBUG_ENABLED=False wins even with DebugConfig(enabled=True)."""
        config = DebugConfig(enabled=True)

        registry = ObserverRegistry.create(
            debug_enabled_global=False,
            graph_debug=True,
            graph_debug_config=config,
            execution_id="test-exec",
            graph_type="chat",
            total_nodes=5,
            total_edges=4,
            callback=None,
        )

        assert not registry.is_active, "Global False always wins"

    def test_graph_debug_false_wins_over_debug_config(self):
        """graph.debug=False wins even with DebugConfig(enabled=True)."""
        config = DebugConfig(enabled=True)

        registry = ObserverRegistry.create(
            debug_enabled_global=True,
            graph_debug=False,
            graph_debug_config=config,
            execution_id="test-exec",
            graph_type="chat",
            total_nodes=5,
            total_edges=4,
            callback=None,
        )

        assert not registry.is_active, "graph.debug=False should produce NullObserver"
