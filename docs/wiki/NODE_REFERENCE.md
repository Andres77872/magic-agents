# Node reference

This wiki page is the conceptual node index used while reading the guide pages. Detailed per-node contracts, handles, and examples live in [../nodes/README.md](../nodes/README.md).

## Built-in node types

| Type | Purpose | Detail page |
| --- | --- | --- |
| `user_input` | Inject user text/files/images and create chat/thread IDs | [../nodes/user_input.md](../nodes/user_input.md) |
| `text` | Emit static text | [../nodes/text.md](../nodes/text.md) |
| `constant` | Emit a typed primitive constant value | [../nodes/constant.md](../nodes/constant.md) |
| `parser` | Render a Jinja2 template from inputs | [../nodes/parser.md](../nodes/parser.md) |
| `fetch` | HTTP request node, optionally a tool provider | [../nodes/fetch.md](../nodes/fetch.md) |
| `client` | Construct a MagicLLM client | [../nodes/client.md](../nodes/client.md) |
| `llm` | Generate text/JSON/tool calls | [../nodes/llm.md](../nodes/llm.md) |
| `chat` | Build or reuse a chat transcript | [../nodes/chat.md](../nodes/chat.md) |
| `send_message` | Emit user-facing message/extras payloads | [../nodes/send_message.md](../nodes/send_message.md) |
| `loop` | Iterate over a list and aggregate results | [../nodes/loop.md](../nodes/loop.md) |
| `conditional` | Route to one branch handle | [../nodes/conditional.md](../nodes/conditional.md) |
| `inner` | Execute a nested graph | [../nodes/inner.md](../nodes/inner.md) |
| `end` | Terminal completion node | [../nodes/end.md](../nodes/end.md) |
| `void` | Internal sink node | [../nodes/void.md](../nodes/void.md) |
| `python_exec` | Expose a Python execution tool | [../nodes/python_exec.md](../nodes/python_exec.md) |
| `mcp` | Discover MCP tools and expose an `MCPToolBundle` | [../nodes/mcp.md](../nodes/mcp.md) |
| `hook` | Execute a Python hook template with `HookContext` input/output handles | [../nodes/hook.md](../nodes/hook.md) |

## Cross-cutting reads

- [GRAPH_FORMAT.md](GRAPH_FORMAT.md)
- [HANDLES_AND_ROUTING.md](HANDLES_AND_ROUTING.md)
- [EXECUTION_MODEL.md](EXECUTION_MODEL.md)
