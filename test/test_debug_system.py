"""
Tests for the Debug System.

Comprehensive tests for all debug system components:
- Events
- Capture
- Transform
- Emitter
- Collector
- Context
- Config
"""

import asyncio
import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock, AsyncMock

# Import all debug components
from magic_agents.debug.events import (
    DebugEvent,
    DebugEventType,
    DebugEventSeverity,
    node_start_event,
    node_end_event,
    node_error_event,
    node_bypass_event,
    graph_start_event,
    graph_end_event,
)
from magic_agents.debug.capture import (
    DefaultDebugCapture,
    DebugCaptureHook,
)
from magic_agents.debug.transform import (
    TransformPipeline,
    RedactTransformer,
    FilterTransformer,
    TruncateTransformer,
    EnrichTransformer,
    TagFilterTransformer,
    SamplingTransformer,
    create_default_pipeline,
)
from magic_agents.debug.emitter import (
    EmitterRegistry,
    QueueEmitter,
    LogEmitter,
    CallbackEmitter,
    BufferedEmitter,
    NullEmitter,
)
from magic_agents.debug.collector import (
    DebugCollector,
    GraphExecutionSummary,
    NodeExecutionSummary,
)
from magic_agents.debug.context import (
    DebugContext,
    debug_context,
    NoopDebugContext,
    create_debug_context,
)
from magic_agents.debug.config import (
    DebugConfig,
    default_config,
    minimal_config,
    verbose_config,
    production_config,
    errors_only_config,
    get_preset,
)


# ============================================================================
# Event Tests
# ============================================================================

class TestDebugEvent:
    """Tests for DebugEvent dataclass."""
    
    def test_create_event(self):
        """Test creating a basic event."""
        event = DebugEvent(
            event_type=DebugEventType.NODE_START,
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
        )
        
        assert event.event_type == DebugEventType.NODE_START
        assert event.node_id == "node-1"
        assert event.event_id  # Auto-generated
        assert event.timestamp  # Auto-generated
    
    def test_to_dict(self):
        """Test serialization to dict."""
        event = DebugEvent(
            event_type=DebugEventType.NODE_END,
            node_id="node-1",
            payload={"duration_ms": 100.5}
        )
        
        d = event.to_dict()
        
        assert d["event_type"] == "node_end"
        assert d["node_id"] == "node-1"
        assert d["payload"]["duration_ms"] == 100.5
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "event_type": "node_start",
            "severity": "info",
            "node_id": "node-1",
            "payload": {"inputs": {"x": 1}}
        }
        
        event = DebugEvent.from_dict(data)
        
        assert event.event_type == DebugEventType.NODE_START
        assert event.node_id == "node-1"
        assert event.payload["inputs"]["x"] == 1
    
    def test_to_legacy_format_node_end(self):
        """Test legacy format conversion for NODE_END."""
        event = DebugEvent(
            event_type=DebugEventType.NODE_END,
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            payload={
                "start_time": "2025-01-01T00:00:00",
                "end_time": "2025-01-01T00:00:01",
                "duration_ms": 1000,
                "inputs": {"prompt": "hello"},
                "outputs": {"response": "world"},
                "internal_state": {"tokens": 10},
            }
        )
        
        legacy = event.to_legacy_format()
        
        assert legacy["node_id"] == "node-1"
        assert legacy["was_executed"] is True
        assert legacy["was_bypassed"] is False
        assert legacy["execution_duration_ms"] == 1000
        assert legacy["inputs"]["prompt"] == "hello"
    
    def test_to_legacy_format_node_bypass(self):
        """Test legacy format conversion for NODE_BYPASS."""
        event = DebugEvent(
            event_type=DebugEventType.NODE_BYPASS,
            node_id="node-2",
            payload={
                "reason": "conditional",
                "bypass_source": "node-1",
                "inputs_at_bypass": {"x": 1},
            }
        )
        
        legacy = event.to_legacy_format()
        
        assert legacy["was_executed"] is False
        assert legacy["was_bypassed"] is True
        assert legacy["bypass_reason"] == "conditional"
    
    def test_is_error(self):
        """Test error detection."""
        error_event = DebugEvent(event_type=DebugEventType.NODE_ERROR)
        normal_event = DebugEvent(event_type=DebugEventType.NODE_END)
        
        assert error_event.is_error() is True
        assert normal_event.is_error() is False
    
    def test_with_payload(self):
        """Test payload modification."""
        event = DebugEvent(payload={"a": 1})
        new_event = event.with_payload(b=2)
        
        assert "a" in new_event.payload
        assert new_event.payload["b"] == 2
        assert "b" not in event.payload  # Original unchanged
    
    def test_with_tags(self):
        """Test tag addition."""
        event = DebugEvent(tags=["existing"])
        new_event = event.with_tags("new", "another")
        
        assert "existing" in new_event.tags
        assert "new" in new_event.tags
        assert "another" in new_event.tags


class TestEventFactories:
    """Tests for event factory functions."""
    
    def test_node_start_event(self):
        """Test node_start_event factory."""
        event = node_start_event(
            execution_id="exec-1",
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            inputs={"prompt": "hello"},
            sequence=1
        )
        
        assert event.event_type == DebugEventType.NODE_START
        assert event.execution_id == "exec-1"
        assert event.sequence_number == 1
        assert "inputs" in event.payload
    
    def test_node_end_event(self):
        """Test node_end_event factory."""
        start = datetime.now(UTC)
        event = node_end_event(
            execution_id="exec-1",
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            outputs={"response": "world"},
            internal_state={"tokens": 10},
            duration_ms=100,
            start_time=start
        )
        
        assert event.event_type == DebugEventType.NODE_END
        assert event.payload["duration_ms"] == 100
    
    def test_node_error_event(self):
        """Test node_error_event factory."""
        event = node_error_event(
            execution_id="exec-1",
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            error=ValueError("test error"),
            context={"attempt": 1}
        )
        
        assert event.event_type == DebugEventType.NODE_ERROR
        assert event.severity == DebugEventSeverity.ERROR
        assert event.payload["error_type"] == "ValueError"
        assert event.payload["error_message"] == "test error"


# ============================================================================
# Capture Tests
# ============================================================================

class TestDefaultDebugCapture:
    """Tests for DefaultDebugCapture."""
    
    def test_enabled_property(self):
        """Test enabled property."""
        capture = DefaultDebugCapture("exec-1", enabled=True)
        assert capture.enabled is True
        
        capture.enabled = False
        assert capture.enabled is False
    
    def test_on_node_start(self):
        """Test capturing node start."""
        capture = DefaultDebugCapture("exec-1")
        event = capture.on_node_start(
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            inputs={"x": 1}
        )
        
        assert event.event_type == DebugEventType.NODE_START
        assert event.execution_id == "exec-1"
        assert event.node_id == "node-1"
        assert event.sequence_number == 1
    
    def test_sequence_numbers(self):
        """Test sequence number incrementing."""
        capture = DefaultDebugCapture("exec-1")
        
        e1 = capture.on_node_start("n1", "LLM", "NodeLLM", {})
        e2 = capture.on_node_start("n2", "TEXT", "NodeText", {})
        e3 = capture.on_node_end("n1", "LLM", "NodeLLM", {}, {}, 100, datetime.now(UTC))
        
        assert e1.sequence_number == 1
        assert e2.sequence_number == 2
        assert e3.sequence_number == 3
    
    def test_safe_copy(self):
        """Test safe data copying."""
        capture = DefaultDebugCapture("exec-1")
        
        # Test with nested dict
        data = {"a": 1, "b": {"c": [1, 2, 3]}}
        copied = capture._safe_copy(data)
        assert copied == data
        
        # Test with non-serializable object
        class CustomObj:
            pass
        
        data = {"obj": CustomObj()}
        copied = capture._safe_copy(data)
        assert "<CustomObj>" in copied["obj"]


# ============================================================================
# Transform Tests
# ============================================================================

class TestTransformPipeline:
    """Tests for TransformPipeline."""
    
    def test_add_transformer(self):
        """Test adding transformers."""
        pipeline = TransformPipeline()
        pipeline.add(RedactTransformer())
        pipeline.add(FilterTransformer())
        
        assert len(pipeline.transformers) == 2
    
    def test_transformer_ordering(self):
        """Test transformers are ordered by priority."""
        pipeline = TransformPipeline()
        pipeline.add(TruncateTransformer())  # order 30
        pipeline.add(RedactTransformer())    # order 10
        pipeline.add(FilterTransformer())    # order 20
        
        orders = [t.order for t in pipeline.transformers]
        assert orders == [10, 20, 30]
    
    def test_remove_transformer(self):
        """Test removing transformers."""
        pipeline = TransformPipeline()
        pipeline.add(RedactTransformer())
        pipeline.add(FilterTransformer())
        
        pipeline.remove("redact")
        
        assert len(pipeline.transformers) == 1
        assert pipeline.transformers[0].name == "filter"
    
    def test_process_event(self):
        """Test processing an event."""
        pipeline = TransformPipeline()
        pipeline.add(RedactTransformer())
        
        event = DebugEvent(payload={"api_key": "secret123", "data": "normal"})
        processed = pipeline.process(event)
        
        assert processed.payload["api_key"] == "***REDACTED***"
        assert processed.payload["data"] == "normal"
    
    def test_process_filters_out(self):
        """Test filtering out events."""
        pipeline = TransformPipeline()
        pipeline.add(FilterTransformer(min_severity=DebugEventSeverity.ERROR))
        
        event = DebugEvent(severity=DebugEventSeverity.INFO)
        processed = pipeline.process(event)
        
        assert processed is None


class TestRedactTransformer:
    """Tests for RedactTransformer."""
    
    def test_redact_sensitive_keys(self):
        """Test redacting sensitive keys."""
        transformer = RedactTransformer()
        event = DebugEvent(payload={
            "api_key": "secret",
            "password": "secret",
            "data": "normal",
            "nested": {"token": "secret", "value": 123}
        })
        
        result = transformer.transform(event)
        
        assert result.payload["api_key"] == "***REDACTED***"
        assert result.payload["password"] == "***REDACTED***"
        assert result.payload["data"] == "normal"
        assert result.payload["nested"]["token"] == "***REDACTED***"
        assert result.payload["nested"]["value"] == 123
    
    def test_custom_redact_keys(self):
        """Test adding custom redact keys."""
        transformer = RedactTransformer(additional_keys={"custom_secret"})
        event = DebugEvent(payload={"custom_secret": "value"})

        result = transformer.transform(event)

        assert result.payload["custom_secret"] == "***REDACTED***"

    def test_non_string_keys_do_not_crash(self):
        """Non-string keys in payload mappings must pass through without crashing."""
        transformer = RedactTransformer()
        event = DebugEvent(payload={
            "inputs": {
                None: "keep",
                123: {"token": "secret"},
                "api_key": "secret",
            }
        })

        result = transformer.transform(event)

        assert result.payload["inputs"][None] == "keep"
        assert result.payload["inputs"][123]["token"] == "***REDACTED***"
        assert result.payload["inputs"]["api_key"] == "***REDACTED***"


class TestFilterTransformer:
    """Tests for FilterTransformer."""
    
    def test_filter_by_severity(self):
        """Test filtering by severity."""
        transformer = FilterTransformer(min_severity=DebugEventSeverity.WARN)
        
        debug_event = DebugEvent(severity=DebugEventSeverity.DEBUG)
        warn_event = DebugEvent(severity=DebugEventSeverity.WARN)
        error_event = DebugEvent(severity=DebugEventSeverity.ERROR)
        
        assert transformer.transform(debug_event) is None
        assert transformer.transform(warn_event) is not None
        assert transformer.transform(error_event) is not None
    
    def test_filter_by_event_type(self):
        """Test filtering by event type."""
        transformer = FilterTransformer(
            include_types={DebugEventType.NODE_START, DebugEventType.NODE_END}
        )
        
        start = DebugEvent(event_type=DebugEventType.NODE_START)
        bypass = DebugEvent(event_type=DebugEventType.NODE_BYPASS)
        
        assert transformer.transform(start) is not None
        assert transformer.transform(bypass) is None
    
    def test_filter_by_node(self):
        """Test filtering by node ID."""
        transformer = FilterTransformer(
            include_nodes={"node-1", "node-2"}
        )
        
        node1 = DebugEvent(node_id="node-1")
        node3 = DebugEvent(node_id="node-3")
        
        assert transformer.transform(node1) is not None
        assert transformer.transform(node3) is None


class TestTruncateTransformer:
    """Tests for TruncateTransformer."""
    
    def test_truncate_strings(self):
        """Test truncating long strings."""
        transformer = TruncateTransformer(max_length=10)
        event = DebugEvent(payload={"long": "a" * 100, "short": "abc"})
        
        result = transformer.transform(event)
        
        assert len(result.payload["long"]) < 100
        assert result.payload["long"].endswith("...[truncated]")
        assert result.payload["short"] == "abc"
    
    def test_truncate_lists(self):
        """Test truncating long lists."""
        transformer = TruncateTransformer(max_list_items=3)
        event = DebugEvent(payload={"list": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        
        result = transformer.transform(event)
        
        assert len(result.payload["list"]) == 4  # 3 items + truncation message
        assert "more items" in result.payload["list"][-1]


# ============================================================================
# Emitter Tests
# ============================================================================

class TestEmitterRegistry:
    """Tests for EmitterRegistry."""
    
    @pytest.mark.asyncio
    async def test_register_emitter(self):
        """Test registering emitters."""
        registry = EmitterRegistry()
        queue = asyncio.Queue()
        
        registry.register(QueueEmitter(queue))
        
        assert len(registry.emitters) == 1
    
    @pytest.mark.asyncio
    async def test_emit_to_all(self):
        """Test emitting to all emitters."""
        registry = EmitterRegistry()
        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()
        
        registry.register(QueueEmitter(queue1, use_legacy_format=False))
        
        # Create a second queue emitter with a different name
        emitter2 = QueueEmitter(queue2, use_legacy_format=False)
        emitter2.name = "queue2"
        registry.register(emitter2)
        
        event = DebugEvent(event_type=DebugEventType.NODE_START)
        await registry.emit(event)
        
        item1 = await queue1.get()
        item2 = await queue2.get()
        
        assert item1["type"] == "debug"
        assert item2["type"] == "debug"


class TestQueueEmitter:
    """Tests for QueueEmitter."""
    
    @pytest.mark.asyncio
    async def test_emit_legacy_format(self):
        """Test emitting in legacy format."""
        queue = asyncio.Queue()
        emitter = QueueEmitter(queue, use_legacy_format=True)
        
        event = DebugEvent(
            event_type=DebugEventType.NODE_END,
            node_id="node-1",
            payload={"duration_ms": 100}
        )
        await emitter.emit(event)
        
        item = await queue.get()
        
        assert item["type"] == "debug"
        assert "node_id" in item["content"]
    
    @pytest.mark.asyncio
    async def test_emit_new_format(self):
        """Test emitting in new format."""
        queue = asyncio.Queue()
        emitter = QueueEmitter(queue, use_legacy_format=False)
        
        event = DebugEvent(event_type=DebugEventType.NODE_START)
        await emitter.emit(event)
        
        item = await queue.get()
        
        assert item["type"] == "debug"
        assert "event_id" in item["content"]


class TestCallbackEmitter:
    """Tests for CallbackEmitter."""
    
    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test async callbacks."""
        emitter = CallbackEmitter()
        received = []
        
        async def handler(event):
            received.append(event)
        
        emitter.add_callback(handler)
        
        event = DebugEvent()
        await emitter.emit(event)
        
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_sync_callback(self):
        """Test sync callbacks."""
        emitter = CallbackEmitter()
        received = []
        
        def handler(event):
            received.append(event)
        
        emitter.add_sync_callback(handler)
        
        event = DebugEvent()
        await emitter.emit(event)
        
        assert len(received) == 1


# ============================================================================
# Collector Tests
# ============================================================================

class TestDebugCollector:
    """Tests for DebugCollector."""
    
    def test_collect_node_events(self):
        """Test collecting node events."""
        collector = DebugCollector("exec-1", "agent", total_nodes=3)
        
        collector.collect(DebugEvent(
            event_type=DebugEventType.NODE_START,
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            payload={"inputs": {"x": 1}}
        ))
        
        collector.collect(DebugEvent(
            event_type=DebugEventType.NODE_END,
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
            payload={"outputs": {"y": 2}, "duration_ms": 100}
        ))
        
        summary = collector.get_summary()
        
        assert len(summary.nodes) == 1
        assert summary.nodes["node-1"].was_executed is True
        assert summary.nodes["node-1"].inputs["x"] == 1
        assert summary.nodes["node-1"].outputs["y"] == 2
    
    def test_collect_bypass_event(self):
        """Test collecting bypass events."""
        collector = DebugCollector("exec-1", "agent")
        
        collector.collect(DebugEvent(
            event_type=DebugEventType.NODE_BYPASS,
            node_id="node-2",
            node_type="TEXT",
            node_class="NodeText",
            payload={"reason": "conditional"}
        ))
        
        summary = collector.get_summary()
        
        assert summary.nodes["node-2"].was_bypassed is True
        assert summary.bypassed_nodes == 1
    
    def test_get_errors(self):
        """Test getting error events."""
        collector = DebugCollector("exec-1", "agent")
        
        collector.collect(DebugEvent(event_type=DebugEventType.NODE_START))
        collector.collect(DebugEvent(
            event_type=DebugEventType.NODE_ERROR,
            payload={"error_message": "test error"}
        ))
        collector.collect(DebugEvent(event_type=DebugEventType.NODE_END))
        
        errors = collector.get_errors()
        
        assert len(errors) == 1
        assert errors[0].event_type == DebugEventType.NODE_ERROR
    
    def test_to_legacy_format(self):
        """Test converting summary to legacy format."""
        collector = DebugCollector("exec-1", "agent", total_nodes=1)
        
        collector.collect(DebugEvent(
            event_type=DebugEventType.GRAPH_START,
            payload={"graph_type": "agent"}
        ))
        collector.collect(DebugEvent(
            event_type=DebugEventType.NODE_END,
            node_id="node-1",
            node_type="LLM",
            node_class="NodeLLM",
        ))
        
        summary = collector.finalize()
        legacy = summary.to_legacy_format()
        
        assert legacy["execution_id"] == "exec-1"
        assert legacy["graph_type"] == "agent"
        assert len(legacy["nodes"]) == 1


# ============================================================================
# Context Tests
# ============================================================================

class TestDebugContext:
    """Tests for DebugContext."""
    
    @pytest.mark.asyncio
    async def test_start_and_finish(self):
        """Test starting and finishing context."""
        ctx = DebugContext(
            execution_id="exec-1",
            graph_type="agent",
            total_nodes=2
        )
        
        queue = asyncio.Queue()
        ctx.add_queue_emitter(queue)
        
        await ctx.start()
        
        # Should have emitted graph start
        item = await queue.get()
        assert item["type"] == "debug"
        
        summary = await ctx.finish()
        
        assert summary.execution_id == "exec-1"
        assert summary.graph_type == "agent"
    
    @pytest.mark.asyncio
    async def test_node_events(self):
        """Test emitting node events via context."""
        ctx = DebugContext(execution_id="exec-1", graph_type="agent")
        
        queue = asyncio.Queue()
        ctx.add_queue_emitter(queue)
        
        await ctx.start()
        await queue.get()  # Clear graph start
        
        await ctx.node_start("node-1", "LLM", "NodeLLM", {"x": 1})
        
        item = await queue.get()
        assert item["content"]["node_id"] == "node-1"
    
    @pytest.mark.asyncio
    async def test_disabled_context(self):
        """Test that disabled context doesn't emit."""
        ctx = DebugContext(enabled=False)
        
        queue = asyncio.Queue()
        ctx.add_queue_emitter(queue)
        
        event = DebugEvent()
        await ctx.emit(event)
        
        assert queue.empty()


class TestDebugContextManager:
    """Tests for debug_context context manager."""
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using as context manager."""
        queue = asyncio.Queue()
        
        async with debug_context(
            graph_type="agent",
            output_queue=queue
        ) as ctx:
            await ctx.node_start("n1", "LLM", "NodeLLM", {})
        
        # Should have graph_start, node_start, graph_end
        assert not queue.empty()


# ============================================================================
# Config Tests
# ============================================================================

class TestDebugConfig:
    """Tests for DebugConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = default_config()
        
        assert config.enabled is True
        assert config.redact_sensitive is True
        assert config.use_legacy_format is True
    
    def test_minimal_config(self):
        """Test minimal configuration."""
        config = minimal_config()
        
        assert config.min_severity == DebugEventSeverity.WARN
        assert config.capture_inputs is False
    
    def test_verbose_config(self):
        """Test verbose configuration."""
        config = verbose_config()
        
        assert config.min_severity == DebugEventSeverity.TRACE
        assert config.max_payload_length == 5000
    
    def test_get_preset(self):
        """Test getting config by preset name."""
        config = get_preset("production")
        
        assert config.sample_rate == 0.1
        
        with pytest.raises(ValueError):
            get_preset("unknown")
    
    def test_with_severity(self):
        """Test creating config with different severity."""
        config = default_config()
        new_config = config.with_severity(DebugEventSeverity.ERROR)
        
        assert new_config.min_severity == DebugEventSeverity.ERROR
        assert config.min_severity == DebugEventSeverity.DEBUG  # Unchanged


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the debug system."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete capture-transform-emit pipeline."""
        queue = asyncio.Queue()
        
        async with debug_context(
            graph_type="test",
            output_queue=queue,
            total_nodes=2
        ) as ctx:
            # Simulate node execution
            start_time = datetime.now(UTC)
            
            await ctx.node_start(
                "node-1", "LLM", "NodeLLM",
                {"prompt": "hello", "api_key": "secret"}
            )
            
            await ctx.node_end(
                "node-1", "LLM", "NodeLLM",
                outputs={"response": "world"},
                internal_state={},
                duration_ms=100,
                start_time=start_time
            )
            
            await ctx.node_bypass(
                "node-2", "TEXT", "NodeText",
                reason="conditional",
                bypass_source="node-1"
            )
        
        # Collect all events
        events = []
        while not queue.empty():
            events.append(await queue.get())
        
        # Should have: graph_start, node_start, node_end, node_bypass, graph_end
        assert len(events) >= 5
        
        # Check redaction worked
        node_start = next(
            e for e in events 
            if e.get("event_type") == "node_start"
        )
        assert "api_key" not in str(node_start["content"]) or \
               "REDACTED" in str(node_start["content"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
