# Magic Agents documentation

This is the canonical documentation entry point for the repository.

Use this page as the navigation hub. The wiki explains concepts and execution behavior, `nodes/` and `hooks/` are reference sections, `JSON_CONTRACT.md` documents the serialized graph contract, and `issues/` tracks known limitations or documentation drift.

## Start here

| Goal | Read |
| --- | --- |
| Understand what the project is | [wiki/PROJECT_OVERVIEW.md](wiki/PROJECT_OVERVIEW.md) |
| Understand build/runtime internals | [wiki/ARCHITECTURE.md](wiki/ARCHITECTURE.md) and [wiki/EXECUTION_MODEL.md](wiki/EXECUTION_MODEL.md) |
| Author graph JSON correctly | [wiki/GRAPH_FORMAT.md](wiki/GRAPH_FORMAT.md) |
| Integrate MCP tools | [wiki/MCP_INTEGRATION.md](wiki/MCP_INTEGRATION.md) |
| Learn routing and handles | [wiki/HANDLES_AND_ROUTING.md](wiki/HANDLES_AND_ROUTING.md) |
| Debug or validate flows | [wiki/DEBUG_SYSTEM.md](wiki/DEBUG_SYSTEM.md) and [wiki/VALIDATION.md](wiki/VALIDATION.md) |
| Look up a specific node | [nodes/README.md](nodes/README.md) or [wiki/NODE_REFERENCE.md](wiki/NODE_REFERENCE.md) |
| Work with lifecycle hooks | [hooks/README.md](hooks/README.md) |
| Check serialized graph shape | [JSON_CONTRACT.md](JSON_CONTRACT.md) |
| Understand task subagents | [wiki/TASK_SUBAGENTS.md](wiki/TASK_SUBAGENTS.md) |
| Review known issues and limitations | [issues/README.md](issues/README.md) |

## Documentation map

- [wiki/README.md](wiki/README.md) — conceptual guides, architecture, execution model, validation, routing, and contributor notes
- [nodes/README.md](nodes/README.md) — one page per current built-in node type
- [wiki/NODE_REFERENCE.md](wiki/NODE_REFERENCE.md) — concise index of all built-in node types
- [hooks/README.md](hooks/README.md) — flow hook protocol, `hook` nodes, edge hook behavior, and hook lifecycle docs
- [JSON_CONTRACT.md](JSON_CONTRACT.md) — JSON graph contract, top-level fields, node/edge shape, and compatibility notes
- [issues/README.md](issues/README.md) — documented repo issues, limitations, and known documentation drift

## Recommended reading order

1. [wiki/PROJECT_OVERVIEW.md](wiki/PROJECT_OVERVIEW.md) — start with the mental model and repository scope.
2. [wiki/GRAPH_FORMAT.md](wiki/GRAPH_FORMAT.md) — learn the graph JSON structure.
3. [JSON_CONTRACT.md](JSON_CONTRACT.md) — use this when you need a stricter serialized contract reference.
4. [wiki/HANDLES_AND_ROUTING.md](wiki/HANDLES_AND_ROUTING.md) — understand how edges connect outputs to inputs.
5. [nodes/README.md](nodes/README.md) — look up the node types used by a graph.
6. [wiki/EXECUTION_MODEL.md](wiki/EXECUTION_MODEL.md) — then read how the runtime executes the graph.
7. [hooks/README.md](hooks/README.md) — read after the execution model if you use lifecycle hooks or `hook` nodes.
8. [issues/README.md](issues/README.md) — check known limitations before relying on edge cases.

## Example material outside `docs/`

- [examples/json/INDEX.md](../examples/json/INDEX.md) — complete graph specs and their local runner
- [examples/conditional/INDEX.md](../examples/conditional/INDEX.md) — branching examples
- [examples/loop/INDEX.md](../examples/loop/INDEX.md) — loop examples

## Source of truth policy

- Runtime behavior is documented from the current codebase, not from stale prose.
- If code and older docs disagree, the new docs here win until the runtime changes.
- Unresolved mismatches are tracked under [docs/issues/](issues/README.md).
