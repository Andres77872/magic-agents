# Development notes

This page is for contributors touching docs or runtime internals.

## Repo paths that matter

| Path | Why it matters |
| --- | --- |
| `docs/` | Canonical documentation |
| `docs/external/` | Problems whose root cause is in another repository (for example `magic-llm`) |
| `magic_agents/agt_flow.py` | Build entry point and validation behavior |
| `magic_agents/execution/` | Runtime and loop semantics |
| `magic_agents/node_system/` | Node behavior |
| `magic_agents/models/factory/Nodes/` | Node config fields and aliases |
| `magic_agents/mcp/` | MCP integration |
| `magic_agents/util/aio.py` | `wait_for`, the timeout helper to use instead of `asyncio.wait_for` |
| `tests/` | The test suite, organized by area (`graph`, `execution`, `hooks`, `nodes/<node>`, `flows`, `live`, ...) |
| `tests/fixtures/graphs/` | Complete graph JSON used by tests and referenced by docs |
| `tests/support/` | Helpers tests import (`RecordingHook`, `load_fixture_graph`, `debug_errors`, credential helpers) |

`examples/`, `Makefile` and `scripts/check_node_coverage.py` are local
development material excluded by `.gitignore`; a fresh clone does not have
them, so tests and docs must not depend on them.

## Setup and tests

Install from the repository root with Python 3.11 or newer:

```sh
python -m pip install -r requirements.txt   # installs -e .[test]
```

To work on `magic-llm` at the same time, follow the editable-install recipe in
the top-level [README.md](../../README.md#installation).

Run the suite without provider calls (empty values override `.env.test`;
`setup.cfg` also deselects the live tests by default):

```sh
env OPENAI_API_KEY= OPENROUTER_API_KEY= SERPER_API_KEY= JINA_API_KEY= \
  python -m pytest --strict-config --strict-markers tests/ --timeout=30 \
    -m 'not needs_api and not credential_gated'
```

Live provider smoke tests are opt-in and skip without real keys:
`python -m pytest tests/live -m needs_api`.

The `CI` GitHub Actions workflow (`.github/workflows/ci.yml`) runs the same
command on Python 3.11, 3.12 and 3.13.

## Code rules worth knowing

- Use `magic_agents.util.aio.wait_for`, not `asyncio.wait_for`. Before Python
  3.12, `asyncio.wait_for` awaits through another task; that swallows a
  cancellation that races with completion and breaks AnyIO cancel scopes
  (the MCP SDK).
- Relay nested async generators with `async with aclosing(...)`. `async for`
  does not close its iterator on early exit.
- Render authored Jinja templates with `jinja2.sandbox.SandboxedEnvironment`
  (parser, conditional and fetch already do).
- Never put resolved `{{env.*}}` values in prompts, debug events or error
  messages; strip URL credentials and query strings for display.

## Documentation maintenance rules

- Prefer `docs/README.md` as the human entry point.
- Put conceptual material under `docs/wiki/` and per-node material under `docs/nodes/`.
- Record problems that belong to another repository under `docs/external/`.
- If runtime code and older prose disagree, update the docs from the code.
- Link only to tracked files; use `tests/fixtures/graphs/` for complete example graphs.

## When editing docs

Check these sources first:

- node implementation class
- node model class
- executor/build path
- the graphs under `tests/fixtures/graphs/`

## Related indexes

- [../README.md](../README.md)
- [NODE_REFERENCE.md](NODE_REFERENCE.md)
- [../external/README.md](../external/README.md)
