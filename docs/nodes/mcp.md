# `mcp`

## Purpose

Connect to an MCP server, discover tools, and expose them to downstream `llm` nodes as an `MCPToolBundle`.

## Runtime class

- `NodeMcp`
- model: `McpNodeModel`

## Default output

- `handle-tool-definition`

## Supported transports

- `stdio`
- `http` (streamable HTTP)

## Current runtime contract

- the config model allows `servers: [...]`
- the runtime currently requires **exactly one server per node in v1**
- session lifecycle is per run
- cleanup happens in `finally`

## Server config fields

- `transport`
- `command`, `args`, `env`, `cwd` for `stdio`
- `url`, `headers` for `http`
- `init_timeout`, `tool_timeout`, `discovery_timeout`
- `prefix`
- `tool_allowlist`
- `tool_denylist`

## Important behavior

1. connect and initialize session
2. discover tools
3. filter and prefix names
4. build `MCPToolBundle`
5. yield bundle to an `llm` node
6. cleanup the session

## Failure model

Emits a debug error and then re-raises for:

- configuration errors
- protocol errors
- transport errors
- unexpected runtime failures

Because the error is re-raised, the node fails: its downstream edges are
bypassed and `on_node_error` fires. Server identifiers in these errors and in
debug state are shown as `stdio:<command>` or `http:<scheme://host[:port]/path>`;
URL user info, query and fragment (where `{{env.*}}` tokens usually live) are
dropped.

## Example

```json
{
  "id": "filesystem-tools",
  "type": "mcp",
  "data": {
    "servers": [
      {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-filesystem", "/tmp"],
        "prefix": "fs"
      }
    ]
  }
}
```
