# Hooks Architecture

## Purpose

Describes the layered architecture of the hooks system: protocol layer, registry layer, runtime layer, and the node/edge integration layer.

## Layered Architecture

```
User Code (FlowHooks impl)
        |
RuntimeConfig           → class-level global hooks, instance-level graph hooks
        |
HookRegistry            → 3-tier dispatch: Node → Graph → Global, async parallel
        |
reactive_executor.py    → drives on_graph_*, on_node_*, on_node_bypass
Node.__call__()         → drives on_node_start / end / error
NodeLLM                 → on_llm_*, creates HookRelay for magic-llm bridge
event_dispatcher.py     → edge-level: EdgeHookConfig → NodeHook dispatch
NodeHook.process()      → exec() template execution
```

### Layer 1: Protocol (`magic_agents/hooks/flow_hooks.py`)

Defines `FlowHooks` — a `@runtime_checkable` Protocol with 11 async methods (`flow_hooks.py:22-45`). All methods are optional. Hooks MUST NOT modify execution state.

### Layer 2: Registry (`magic_agents/hooks/hook_registry.py`)

`HookRegistry` is an execution-scoped registry (`hook_registry.py:19-31`). Dies with execution. No module-level global state.

### Layer 3: Runtime Config (`magic_agents/hooks/runtime_config.py`)

`RuntimeConfig` provides application-scoped global hook registration (`runtime_config.py:19-42`). Class-level `_global_hooks` list persists across executions.

### Layer 4: Context Factory (`magic_agents/hooks/context_factory.py`)

`HookContextFactory` provides 6 static factory methods for validated `HookContext` construction (`context_factory.py:52-58`). Direct `HookContext()` construction is deprecated (`flow_hooks.py:146-162`).

## Runtime Flow

1. `agt_flow.py:368` (`execute_graph()`) or `agt_flow.py:422` (`execute_graph_loop()`) → merges `RuntimeConfig` + `AgentFlowModel.hooks` → creates `HookRegistry` (`agt_flow.py:396-405`).

2. `reactive_executor.py:370-391` → sets `hooks.execution_id` and `hooks.run_id` → `HookContextFactory.build_graph_context()` → `hooks.invoke("on_graph_start", ctx)`.

3. For each node: `Node.__call__()` (`Node.py:220-237`) → stores `self._hooks` → `HookContextFactory.build_node_context()` → `hooks.invoke("on_node_start", ctx)` → `process()` → on success: `hooks.invoke("on_node_end", ctx)` → on error: `hooks.invoke("on_node_error", ctx, error=e)`.

4. `NodeLLM.process()` → `_create_hook_relay()` at `NodeLLM.py:292-321` → passes `HookRelay` as `hooks=` to magic-llm's `run_agent_async()` / `run_agent_stream_async()`.

5. On error/bypass: `reactive_executor.py:586-628` → `_propagate_error_bypass_with_hooks()` → `dispatcher.propagate_error_bypass()` → `HookContextFactory.build_bypass_context(reason="upstream_error")` → `hooks.invoke("on_node_bypass", ctx, reason="upstream_error")`.

## 3-Tier Dispatch Order

**Execution order**: Node → Graph → Global (innermost-first).

| Tier | Scope | Registration | Fire Condition |
|------|-------|-------------|----------------|
| 3 (Node) | Per-node | `registry.register_node(node_id, hook)` | Only when that specific node executes |
| 2 (Graph) | Per-graph | `registry.register_graph(hook)` or `AgentFlowModel.hooks` | Every node in the graph |
| 1 (Global) | Application | `RuntimeConfig.register_global_hook(hook)` | ALL executions in the process |

All hooks execute in parallel via `asyncio.gather(*tasks, return_exceptions=True)` (`hook_registry.py:135-167`). Errors are logged but never propagated.

## Graph Integration

- **AgentFlowModel.hooks** (`AgentFlowModel.py:155-159`): graph-level `FlowHooks` instance, merged with `RuntimeConfig` at execution time.
- **NodeInner hook propagation** (`NodeInner.py:143-171`): creates a child `HookRegistry`, copies `execution_id`/`run_id`, registers `inner_graph.hooks`, passes to `execute_graph_reactive()`.
- **EdgeHookConfig** (`EdgeNodeModel.py:24-52`): attaches a `NodeHook` to an edge; dispatcher invokes on traversal.

## LLM Integration

`NodeLLM` creates a `HookRelay` adapter (`NodeLLM.py:292-321`) bridging magic-llm's sync `AgentHooks` protocol to magic-agents async `FlowHooks`. HookRelay implements `on_iteration_start`, `on_llm_response`, `on_tool_start`, `on_tool_complete`, `on_loop_complete`, `on_budget_exceeded` (`hook_relay.py:139-283`).

## NodeHook Integration

`NodeHook.process()` (`NodeHook.py:101-192`) receives a `HookContext` via its input handle (`INPUT_HANDLE_HOOK_CONTEXT`), compiles the Python function template via `exec()`, and executes it with timeout enforcement.
