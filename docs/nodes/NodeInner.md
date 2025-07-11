# NodeInner

Executes a **nested agent flow** (`magic_flow`) and streams its outputs to the outer graph.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeInner` |
| **Type key** | `inner` |
| **Input handle** | `handle_user_message` |
| **Output handles** | `handle_execution_content`, `handle_execution_extras` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `magic_flow` | `dict` | A full agent-flow specification (same format used by `build`). |

## Example

```json
{
  "id": "summarize_each",
  "type": "inner",
  "magic_flow": {
    "type": "chat",
    "nodes": [
      { "id": "inner_user", "type": "user_input" },
      { "id": "summarizer", "type": "llm", "data": { "stream": false } },
      { "id": "inner_end", "type": "end" }
    ],
    "edges": [
      { "source": "inner_user", "target": "summarizer" },
      { "source": "summarizer", "target": "inner_end" }
    ],
    "master": "inner_user"
  }
}
```

## Runtime Overview

1. Receives a user message on `handle_user_message`.
2. Before execution, **patches** the inner flow’s `user_input` and/or `chat` nodes to use this message.
3. Calls `execute_graph` to run the inner graph.
4. Streams resulting `ChatCompletionModel` chunks:
   • Aggregates text into `content` string.  
   • Collects any `extras` payloads.
5. Yields two outputs:
   • `handle_execution_content` – the concatenated text.  
   • `handle_execution_extras` – list of extras (optional).

## Simplified Logic

```python
input_message = self.inputs['handle_user_message']
for n in inner_graph.nodes.values():
    if isinstance_user_or_chat(n):
        n.message_or_text = input_message
async for evt in execute_graph(inner_graph):
    ...
yield self.yield_static(content)
if extras:
    yield self.yield_static(extras)
```

- Enables **modular composition**: build a complex flow once, reuse it as a sub-routine.
- Supports recursion when an inner flow also contains another `NodeInner`.
