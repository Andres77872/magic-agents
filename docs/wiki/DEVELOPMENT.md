# Development notes

This page is for contributors touching docs or runtime internals.

## Repo paths that matter for documentation

| Path | Why it matters |
| --- | --- |
| `docs/` | Canonical documentation |
| `examples/` | Real graph examples referenced by docs |
| `magic_agents/agt_flow.py` | Build entry point and validation behavior |
| `magic_agents/execution/` | Runtime and loop semantics |
| `magic_agents/node_system/` | Node behavior |
| `magic_agents/models/factory/Nodes/` | Node config fields and aliases |
| `magic_agents/mcp/` | MCP integration |
| `.issue/` | Issue investigation source material |

## Documentation maintenance rules

- Prefer `docs/README.md` as the human entry point.
- Put conceptual material under `docs/wiki/`.
- Put per-node material under `docs/nodes/`.
- Put unresolved problems under `docs/issues/`.
- If runtime code and older prose disagree, update the docs from code and log unresolved gaps as issues.

## When editing docs

Check these sources first:

- node implementation class
- node model class
- executor/build path
- real example JSON under `examples/`

## Related indexes

- [../README.md](../README.md)
- [NODE_REFERENCE.md](NODE_REFERENCE.md)
- [../issues/README.md](../issues/README.md)
