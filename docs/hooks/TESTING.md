# Hooks Testing

## Purpose

Describes the hook tests and where to find each category. All files are under
`tests/`; run them with the command in
[../wiki/DEVELOPMENT.md](../wiki/DEVELOPMENT.md#setup-and-tests).

## Test files

| Category | Location | What it tests |
|----------|----------|---------------|
| Edge hooks | `tests/hooks/edge/test_edge_hook_ordering.py`, `test_edge_hook_execution.py` | Edge Hooks finish and emit before the original input reaches its target, including cancellation of a running Hook; bypassed and unselected edges |
| Nested edge hooks | `tests/hooks/edge/test_nested_edge_hook_ordering.py` | Nested edge Hooks gate delivery; skipped Hook routes preserve convergence |
| Lifecycle hooks | `tests/hooks/lifecycle/test_fallback.py` | `onStart`/`onError`/`onFinish`/`onCancel`/`onDeliver` through the real graph, Fetch HTTP and the LLM agent loop, including the Studio provider-fallback catalog graph (`tests/fixtures/graphs/browsing_provider_fallback.json`) |
| Lifecycle boundaries | `tests/hooks/lifecycle/test_boundaries.py` | Reliability boundaries on the real executor and node classes |
| Lifecycle composition | `tests/hooks/lifecycle/test_composition.py` | Hook child operations and isolated terminal edge deliveries (a suppressed delivery releases downstream nodes immediately) |
| Build validation | `tests/hooks/lifecycle/test_validation.py` | Build-time lifecycle checks, without executing templates |
| Shared bindings | `tests/hooks/lifecycle/test_shared_bindings.py` | Shared Python and built-in `messages` Hook bindings |
| LLM relay | `tests/hooks/relay/` | `HookRelay` translation, correlated tool call/result collection, pending-hook flush and error isolation |
| Registry and tiers | `tests/hooks/test_hook_registry.py`, `test_hook_tiers.py`, `test_no_hook_parity.py` | Node → graph → global order, concurrency within a tier, identical events with and without hooks |
| Persistence and SSE | `tests/hooks/test_graph_persistence_hook.py`, `test_debug_sse_hook.py`, `test_debug_sse_frames.py` | Persistence port calls and debug SSE envelopes |
| Inner flows | `tests/nodes/inner/test_inner_p0_regression.py` | Hook propagation into `inner` sub-flows and run/execution id separation |

## Key fixture patterns

`tests.support.RecordingHook` is an async `FlowHooks` implementation that
records every invocation:

```python
from tests.support import RecordingHook

rec = RecordingHook()
# ... run a graph with run_agent(graph, hooks=RuntimeConfig(graph_hooks=[rec])) ...
rec.assert_method_called("on_node_start", times=3)
starts = rec.get_invocations("on_node_start")
```

`tests.support.AsyncRecordingHook` adds a delay before recording, for
async-bridge and flush tests.

### RuntimeConfig cleanup

Hook tests call `RuntimeConfig.clear_global_hooks()` in setup/teardown where
global hook state is exercised, preventing class-level state pollution without
relying on an autouse fixture.

## Known gaps

| Gap | Reason |
|-----|--------|
| `on_node_bypass` with `reason="not_ready"` | Timing-dependent, not end-to-end tested |
| `emit.debug()` / `emit.feedback()` runtime integration | No runtime consumer to test against |
| Contracts TypedDict schemas | No validation tests — doc-only |
| Loop executor `on_graph_error` on cancellation | Not fired today (see [ARCHITECTURE.md](ARCHITECTURE.md)) |
