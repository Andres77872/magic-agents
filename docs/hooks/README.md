# Hooks System — Entry Point

## Purpose

The magic-agents hook system provides **observer-only lifecycle hooks** for graph execution, LLM calls, and tool invocations. It has two subsystems:

1. **FlowHooks Protocol** — async observer interface (12 lifecycle methods) for programmatic consumers (logging, metrics, tracing).
2. **NodeHook** — a `type: "hook"` node that executes user-defined Python function templates via `exec()`.
3. **CallbackEmitter** — a module-level compatibility/debug callback registry for selected executor debug events.

Hooks are **observers**: they see events but MUST NOT modify execution state or alter control flow.

## Quick Start

### Programmatic: implement FlowHooks

```python
from magic_agents.hooks.flow_hooks import FlowHooks, HookContext
from magic_agents.hooks.runtime_config import RuntimeConfig

class MyHook:
    async def on_node_start(self, ctx: HookContext) -> None:
        print(f"Node started: {ctx.node_id}")

RuntimeConfig.register_global_hook(MyHook())
config = RuntimeConfig()
registry = config.create_registry()
await execute_graph_reactive(graph, hooks=registry)
```

### Declarative (JSON): use hook nodes + EdgeHookConfig

```json
{
  "type": "hook",
  "function_template": "def my_hook(ctx, log): emit.user('hook fired!')"
}
```

Add `EdgeHookConfig` to an edge: `"hooks": { "hook_node_id": "my-hook-node", "enabled": true }`.

## Status Summary

| Subsystem | Status | Tests |
|-----------|--------|-------|
| FlowHooks Protocol (12 methods) | ✅ Implemented | ~30 unit |
| HookContext + HookContextFactory | ✅ Implemented | ~50 tests |
| HookRegistry (3-tier dispatch) | ✅ Implemented | 453 lines |
| RuntimeConfig (global/graph) | ✅ Implemented | e2e + integration |
| HookRelay (magic-llm bridge) | ✅ Implemented | ~200 unit + ~400 integration |
| CallbackEmitter (module-level debug bridge) | ✅ Implemented | graph boundary events only |
| EdgeHookConfig | ✅ Implemented | 674 lines integration |
| NodeHook (Python template) | ✅ Implemented (Phase 1 only) | via edge + e2e |
| EmitInterface | ✅ Implemented | ~100 unit |
| Contracts (TypedDict schemas) | ✅ Implemented | doc-only, NOT runtime-enforced |
| 5 NodeInner-specific hooks | ❌ Not implemented | None |
| `emit.debug()` runtime integration | ❌ Structure-only | TBD |
| `emit.feedback()` extras transport | ❌ Placeholder | TBD |
| NodeHook sandboxing | ❌ Phase 2/3 | None |
| 3 deprecated HookContext subclasses | ⚠️ Deprecated | 3 deprecation-warning tests |

## Doc Map

| Doc | Audience | Content |
|-----|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Maintainers | Layered design, 3-tier dispatch, runtime flow |
| [PROTOCOL.md](PROTOCOL.md) | Hook implementers | FlowHooks 12 methods, signatures, semantics |
| [../JSON_CONTRACT.md](../JSON_CONTRACT.md) | Graph authors | Serialized JSON hook fields and edge hook contract details |
| [CONTRACTS.md](CONTRACTS.md) | Hook implementers | HookContext, factory, TypedDict schemas, bypass reasons |
| [RUNTIME_CONFIG.md](RUNTIME_CONFIG.md) | Application devs | RuntimeConfig, HookRegistry, registration API |
| [HOOK_RELAY.md](HOOK_RELAY.md) | Integration devs | magic-llm bridge, async bridge, flushing |
| [CALLBACK_EMITTER.md](CALLBACK_EMITTER.md) | Integration devs | Module-level debug callback registry and persistence tradeoffs |
| [NODE_HOOK.md](NODE_HOOK.md) | NodeHook authors | exec template, emit API, timeout, safety |
| [EDGE_HOOKS.md](EDGE_HOOKS.md) | Graph builders | EdgeHookConfig, hook dispatch on traversal |
| [BUILD_INTEGRATION.md](BUILD_INTEGRATION.md) | Platform devs | build()/JSON behavior, hooks stripping |
| [TESTING.md](TESTING.md) | QA / contributors | Test coverage, fixtures, e2e patterns |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | All | Deprecations, migration paths |
