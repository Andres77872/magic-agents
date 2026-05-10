# Project overview

Magic Agents is a graph-based orchestration library for LLM workflows.

## Core mental model

A flow is a graph definition made of:

- **nodes** — units of work such as `user_input`, `parser`, `llm`, `fetch`, `loop`, or `mcp`
- **edges** — routing rules from one node output handle to another node input handle
- **runtime state** — node inputs/outputs, chat identifiers, optional extras, per-flow state, and optional runtime hooks/debug config

The runtime is **reactive**, not a simple linear runner:

- nodes execute when their expected inputs are ready
- independent paths can run in parallel
- conditionals can bypass downstream branches
- loops switch to a specialized three-phase executor

## What build does

`magic_agents.agt_flow.build()` is the main graph preparation entry point.

At build time the library:

1. normalizes graph input
2. resolves `{{env.NAME}}` placeholders
3. validates basic graph shape
4. auto-assigns tool handles for tool-capable nodes targeting LLM nodes
5. sorts nodes/edges for stable processing
6. injects an internal `void` sink node
7. instantiates node classes from JSON
8. builds nested `inner` graphs recursively
9. attaches structural/conditional validation results and a contract-validation report to the final `AgentFlowModel`

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed pipeline.

## What execution does

`run_agent()` delegates to the reactive executor.

- normal graphs use `execute_graph_reactive()`
- graphs containing a `loop` node use `execute_graph_loop_reactive()`
- graph/node hooks can observe execution lifecycle events
- debug events are emitted as structured dict events
- streaming content is emitted as `type: "content"`

See [EXECUTION_MODEL.md](EXECUTION_MODEL.md).

## Built-in node surface

The current runtime accepts 17 built-in node types:

`user_input`, `text`, `constant`, `parser`, `fetch`, `client`, `llm`, `chat`, `send_message`, `loop`, `conditional`, `inner`, `end`, `void`, `python_exec`, `mcp`, `hook`

See [NODE_REFERENCE.md](NODE_REFERENCE.md) and [../nodes/README.md](../nodes/README.md).

## Important current constraints

- exactly one `user_input` node is required by `validate_graph()`
- `loop` has dedicated executor semantics; it is not a generic cycle mechanism
- `mcp` is v1-limited to **exactly one server per node at runtime**
- `master` may appear in older specs, but the current build/runtime path does **not** read it
- a hidden `void` sink node is added during build for unrouted outputs
- graph-level `timeout` defaults to 60 seconds for input waiting
- contract validation defaults to `warn`; stricter runtime enforcement is only partially implemented today

Documented open gaps live in [../issues/README.md](../issues/README.md).
