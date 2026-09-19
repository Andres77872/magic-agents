[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-agents)

# magic_agents

A graph-based orchestration library for LLM workflows.

## Documentation

Start with:

- [docs/README.md](docs/README.md)
- [docs/wiki/ARCHITECTURE.md](docs/wiki/ARCHITECTURE.md)
- [docs/wiki/GRAPH_FORMAT.md](docs/wiki/GRAPH_FORMAT.md)
- [docs/nodes/README.md](docs/nodes/README.md)
- [docs/hooks/README.md](docs/hooks/README.md)
- [docs/JSON_CONTRACT.md](docs/JSON_CONTRACT.md)
- [docs/issues/README.md](docs/issues/README.md)

## Installation

```bash
git clone https://github.com/Andres77872/magic-agents.git
cd magic-agents
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
        content = event.get("content")
        if hasattr(content, "choices") and content.choices:
            print(content.choices[0].delta.content or "", end="")

asyncio.run(main())
```

This example uses the OpenAI client configuration, so set `OPENAI_API_KEY` before running it.

## Built-in nodes

The current runtime supports 20 built-in node types:

`user_input`, `text`, `constant`, `parser`, `fetch`, `client`, `llm`, `chat`, `send_message`, `loop`, `conditional`, `inner`, `end`, `void`, `hook`, `python_exec`, `mcp`, `memory`, `codex`, `node_tool`

Use the docs for current per-node behavior and routing details: [docs/nodes/README.md](docs/nodes/README.md) for per-node pages and [docs/wiki/NODE_REFERENCE.md](docs/wiki/NODE_REFERENCE.md) for the wiki-level node index.

## Examples

- [examples/json/INDEX.md](examples/json/INDEX.md)
- [examples/conditional/INDEX.md](examples/conditional/INDEX.md)
- [examples/loop/INDEX.md](examples/loop/INDEX.md)

## Testing

Install the test tools and run the deterministic node/integration suite without
provider credentials:

```bash
pip install -e ".[test]"
make test-credential-free
make coverage-credential-free
```

The coverage target exercises both complete test suites with branch coverage,
a 70% aggregate gate, and a 60% minimum for every `Node*.py` module.
Live-provider tests are marked `needs_api`/`credential_gated` and excluded from
this target.
