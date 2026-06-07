# `node_tool`

## Purpose

`node_tool` provides a schema-only tool definition to an `llm` node. It accepts one raw OpenAI-compatible function tool object and emits it on `handle-tool-definition`.

The backend **does not execute** this function. When the provider returns a tool call, callers may display it or route it to their own trusted client-side execution layer. `magic-ui` v1 intentionally **does not execute** browser tools and only displays schema-only calls.

## Runtime class

- `NodeTool`
- model: `ToolNodeModel`

## Data contract

```json
{
  "tool": {
    "type": "function",
    "function": {
      "name": "client_lookup_order",
      "description": "Ask the client application to look up an order by ID.",
      "parameters": {"type": "object", "properties": {}}
    }
  }
}
```

Validation is strict at model/graph build time:

- `tool.type` must be `"function"`
- `tool.function` must be an object
- `tool.function.name` must match `^[a-zA-Z0-9_-]{1,64}$`
- `tool.function.description` is required and non-empty
- `tool.function.parameters` must be an object JSON Schema with `type: "object"`

## Handles

- inputs: none in v1
- outputs: `handle-tool-definition`

## NodeLLM behavior

Connect `node_tool.handle-tool-definition` to an `llm` tool input such as `handle-tool-definition-0`. NodeLLM passes the raw schema through provider `tools=` using the direct `magic-llm` generation path.

Schema-only NodeTool must not be mixed with callable server tools (`fetch` tool mode, `python_exec`, `mcp`, task subagents) on the same LLM node in v1. Split the graph into separate LLM nodes if you need both.

## Tool-call envelope

When a schema-only tool call is returned, `handle-tool-calls` emits:

```json
{
  "execution": "client",
  "source": "schema_only",
  "tool_calls": [
    {"id": "call_123", "type": "function", "function": {"name": "client_lookup_order", "arguments": "{}"}}
  ]
}
```

The objects inside `tool_calls` are raw provider payloads and are not mutated. The backend emits client-executable visibility as `TOOL_CALL` debug data only; it does not emit server `TOOL_RESULT` for NodeTool.

## Frontend rollout dependency

Frontend/UI support lives in a separate repository. Public UI rollout requires that repo to add `node_tool` node type/data handling, edge validation, save/load support, and display-only handling for schema-only tool calls. Real client execution and client result submission are out of scope for `magic-ui` v1.
