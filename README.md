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

## Installation

Requires Python 3.11 or newer. For a checkout of this repo:

```bash
git clone https://github.com/Andres77872/magic-agents.git
cd magic-agents
python -m pip install -e .
```

This is enough for applications that only use the JSON graph API. `magic-agents` installs the official `magic-llm` from GitHub as a dependency; application code doesn't need to import it. Standalone LLM applications can install [magic-llm](https://github.com/Andres77872/magic-llm) without `magic-agents`.

If an application declares both packages from Git, use these exact references so pip resolves a single copy of `magic-llm`:

```text
magic_agents @ git+https://github.com/Andres77872/magic-agents.git@master
magic_llm @ git+https://github.com/Andres77872/magic-llm.git@master
```

To work on **both local checkouts**, put `magic-llm` beside this repo, activate one virtualenv, and install from the `magic-agents` root:

```sh
printf '%s\n' '-e ../magic-llm[test]' | uv pip install --overrides - -r requirements.txt
```

`uv` resolves all dependencies while keeping both checkouts editable. Don't install either package by its bare name from public PyPI; those names belong to unrelated projects.

Optional extras: `pdf` parses PDF chat attachments, `redis` caches those parsed PDFs, and `qdrant` stores `memory` node vectors. For example, `python -m pip install -e ".[pdf,redis]"` adds PDF parsing and Redis caching to an Agents checkout. Text and Markdown attachments work without them; see [docs/chat-file-attachments.md](docs/chat-file-attachments.md).

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

The runtime supports these built-in node types:

`user_input`, `text`, `constant`, `parser`, `fetch`, `client`, `llm`, `chat`, `send_message`, `loop`, `conditional`, `inner`, `end`, `void`, `hook`, `python_exec`, `mcp`, `memory`, `codex`, `node_tool`, `skills`

Use the docs for current per-node behavior and routing details: [docs/nodes/README.md](docs/nodes/README.md) for per-node pages and [docs/wiki/NODE_REFERENCE.md](docs/wiki/NODE_REFERENCE.md) for the wiki-level node index.

## Testing

Install the test tools with `python -m pip install -r requirements.txt`. If working on both repos, use the local override above instead. For a focused installation check without provider calls:

```sh
env OPENAI_API_KEY= OPENROUTER_API_KEY= SERPER_API_KEY= JINA_API_KEY= \
  python -m pytest --strict-config --strict-markers --timeout=30 \
    tests/graph/test_package_dependencies.py tests/coordination/test_mcp_http.py \
    tests/coordination/test_mcp_discovery.py tests/nodes/chat/test_chat_file_attachments.py
```

Install the `pdf` extra to include the real PDF parser case. The whole suite (`tests/`) runs without credentials; `setup.cfg` deselects the live-provider tests by default:

```sh
env OPENAI_API_KEY= OPENROUTER_API_KEY= SERPER_API_KEY= JINA_API_KEY= \
  python -m pytest --strict-config --strict-markers tests/ --timeout=30 \
    -m 'not needs_api and not credential_gated'
```

The opt-in live smoke tests call real providers and skip without real keys: `python -m pytest tests/live -m needs_api`.

CI runs that command on Python 3.11, 3.12 and 3.13 (`.github/workflows/ci.yml`). Contributor notes are in [docs/wiki/DEVELOPMENT.md](docs/wiki/DEVELOPMENT.md). Problems whose root cause is in `magic-llm` or another dependency are listed in [docs/external/](docs/external/README.md).
