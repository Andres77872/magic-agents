[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-agents)

# magic_agents

A graph-based orchestration library for LLM workflows.

## Documentation

Start with:

- [docs/README.md](docs/README.md)
- [docs/wiki/ARCHITECTURE.md](docs/wiki/ARCHITECTURE.md)
- [docs/wiki/GRAPH_FORMAT.md](docs/wiki/GRAPH_FORMAT.md)
- [docs/nodes/README.md](docs/nodes/README.md)

## Installation

```bash
git clone https://github.com/your/repo/magic_agents.git
cd magic_agents
pip install -e .
```

## Quickstart

```python
import asyncio
from magic_agents import run_agent
from magic_agents.agt_flow import build

spec = {
    "type": "chat",
    "debug": True,
    "nodes": [
        {"id": "user_input", "type": "user_input"},
        {
            "id": "llm_client",
            "type": "client",
            "data": {
                "engine": "openai",
                "model": "gpt-4o-mini",
                "api_info": {"api_key": "{{env.OPENAI_API_KEY}}"}
            }
        },
        {"id": "answer", "type": "llm", "data": {"stream": True}},
        {"id": "finish", "type": "end"}
    ],
    "edges": [
        {"source": "user_input", "sourceHandle": "handle_user_message", "target": "answer", "targetHandle": "handle_user_message"},
        {"source": "llm_client", "sourceHandle": "handle-client-provider", "target": "answer", "targetHandle": "handle-client-provider"},
        {"source": "answer", "sourceHandle": "handle_generated_content", "target": "finish", "targetHandle": "handle_flow_input"}
    ]
}

graph = build(spec, message="Hi")

async def main():
    async for event in run_agent(graph):
        if hasattr(event, 'choices') and event.choices:
            print(event.choices[0].delta.content or "", end="")

asyncio.run(main())
```

## Built-in nodes

The current runtime supports 15 built-in node types:

`user_input`, `text`, `parser`, `fetch`, `client`, `llm`, `chat`, `send_message`, `loop`, `conditional`, `inner`, `end`, `void`, `python_exec`, `mcp`

Use the docs for current per-node behavior and routing details.

## Examples

- [examples/json/](examples/json/)
- [examples/conditional/INDEX.md](examples/conditional/INDEX.md)
- [examples/loop/INDEX.md](examples/loop/INDEX.md)
