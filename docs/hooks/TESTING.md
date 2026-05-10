# Hooks Testing

## Purpose

Describes current test coverage, fixture patterns, and where to find specific test categories.

## Test Suite Overview

Total: **234+ tests** (verified passing).

| Category | File | Lines | What It Tests |
|----------|------|-------|---------------|
| **Fixtures** | `tests/hooks/conftest.py` | 256 | RecordingHook, AsyncRecordingHook shared fixtures |
| **Unit — Registry** | `tests/hooks/test_hook_registry.py` | 453 | 3-tier dispatch, error isolation, sync/async, parallel |
| **Unit — Factory** | `tests/hooks/test_flow_hooks.py` | — | HookContextFactory validation + context construction |
| **Unit — Relay** | `tests/hooks/test_hook_relay.py` | — | HookRelay unit tests |
| **Unit — Emit** | `tests/hooks/test_emit_interface.py` | — | EmitInterface tests |
| **Integration — E2E** | `tests/e2e/test_full_graph_hooks.py` | 940 | Full lifecycle, error paths, bypass, build() integration |
| **Integration — Edge** | `tests/integration/test_edge_hook_execution.py` | 674 | Edge-level hook dispatch + NodeHook execution |
| **Integration — 3-Tier** | `tests/integration/test_3_tier_hooks.py` | 301 | 3-tier registration integration |
| **Integration — Context** | `tests/integration/test_hook_context_runtime_payload.py` | 554 | Production payload completeness |
| **Integration — LLM** | `tests/integration/test_llm_hook_relay.py` | 466 | LLM hook relay + tool collection |
| **Performance** | `tests/performance/test_hook_overhead.py` | 538 | <5ms overhead verification |

## Key Fixture Patterns

**File**: `tests/hooks/conftest.py`

### RecordingHook (sync)

```python
class RecordingHook:
    """Records all hook invocations for assertions."""
    calls: List[Dict[str, Any]]
    
    async def on_node_start(self, ctx): self.calls.append(...)
```

### AsyncRecordingHook (async-only)

```python
class AsyncRecordingHook:
    """Same as RecordingHook but all methods are async."""
```

### RuntimeConfig Cleanup

Hook tests use manual `RuntimeConfig.clear_global_hooks()` calls in setup/teardown paths where global hook state is exercised, preventing class-level state pollution without relying on an autouse fixture.

## E2E Test Patterns

**File**: `tests/e2e/test_full_graph_hooks.py` (940 lines)

- Full lifecycle: graph_start → node_start → node_end → graph_end
- Error paths: graph_error, node_error
- Runtime vs model hooks (RuntimeConfig vs AgentFlowModel.hooks)
- Bypass: on_node_bypass with reason propagation
- build() integration: JSON dict hooks warning
- RuntimeConfig cleanup isolation

## Edge Hook Tests

**File**: `tests/integration/test_edge_hook_execution.py` (674 lines)

- EdgeHookConfig dispatch correctness
- NodeHook execution via edge traversal
- `hook_type` deprecation warning
- Edge context payload validation

## 3-Tier Integration

**File**: `tests/integration/test_3_tier_hooks.py` (301 lines)

- All three tiers fire in correct order
- Node-only hooks don't fire for other nodes
- Graph hooks fire for all nodes

## Performance Benchmarks

**File**: `tests/performance/test_hook_overhead.py` (538 lines)

- 1000-round hook overhead benchmarks
- Verifies added latency is below 5ms
- Covers empty registry, single hook, multi-hook scenarios

## Known Gaps

| Gap | Reason |
|-----|--------|
| NodeInner hook propagation tests | Planned in P0_REGRESSION_ANALYSIS.md:308-311 but may not exist |
| `on_node_bypass` with `reason="not_ready"` | Timing-dependent, not e2e tested (`test_full_graph_hooks.py:934-940`) |
| `emit.debug()` / `emit.feedback()` runtime integration | No runtime to test against |
| Contracts TypedDict schemas | No validation tests — doc-only |
