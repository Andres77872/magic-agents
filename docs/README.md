# Magic Agents documentation

This is the canonical documentation entry point for the repository.

## Start here

| Goal | Read |
| --- | --- |
| Understand what the project is | [wiki/PROJECT_OVERVIEW.md](wiki/PROJECT_OVERVIEW.md) |
| Understand build/runtime internals | [wiki/ARCHITECTURE.md](wiki/ARCHITECTURE.md) and [wiki/EXECUTION_MODEL.md](wiki/EXECUTION_MODEL.md) |
| Author graph JSON correctly | [wiki/GRAPH_FORMAT.md](wiki/GRAPH_FORMAT.md) |
| Integrate MCP tools | [wiki/MCP_INTEGRATION.md](wiki/MCP_INTEGRATION.md) |
| Learn routing and handles | [wiki/HANDLES_AND_ROUTING.md](wiki/HANDLES_AND_ROUTING.md) |
| Debug or validate flows | [wiki/DEBUG_SYSTEM.md](wiki/DEBUG_SYSTEM.md) and [wiki/VALIDATION.md](wiki/VALIDATION.md) |
| Look up a specific node | [nodes/README.md](nodes/README.md) |
| Understand task subagents | [wiki/TASK_SUBAGENTS.md](wiki/TASK_SUBAGENTS.md) |
| Review known issues and limitations | [issues/README.md](issues/README.md) |

## Documentation map

- [wiki/README.md](wiki/README.md) — conceptual guides and architecture
- [nodes/README.md](nodes/README.md) — one page per built-in node type
- [issues/README.md](issues/README.md) — documented repo issues and limitations

## Recommended reading order

1. [wiki/PROJECT_OVERVIEW.md](wiki/PROJECT_OVERVIEW.md)
2. [wiki/GRAPH_FORMAT.md](wiki/GRAPH_FORMAT.md)
3. [wiki/HANDLES_AND_ROUTING.md](wiki/HANDLES_AND_ROUTING.md)
4. [nodes/README.md](nodes/README.md)
5. [wiki/EXECUTION_MODEL.md](wiki/EXECUTION_MODEL.md)

## Example material outside `docs/`

- [examples/json/](../examples/json/) — complete graph specs
- [examples/conditional/INDEX.md](../examples/conditional/INDEX.md) — branching examples
- [examples/loop/INDEX.md](../examples/loop/INDEX.md) — loop examples

## Source of truth policy

- Runtime behavior is documented from the current codebase, not from stale prose.
- If code and older docs disagree, the new docs here win until the runtime changes.
- Unresolved mismatches are tracked under [docs/issues/](issues/README.md).
