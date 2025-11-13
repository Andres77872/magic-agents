# Magic Agents Architecture

> **Version:** <!-- 2025-07-11 -->
>
> This document provides a high-level architectural overview of **`magic_agents`** including the main building blocks, how nodes are defined, and how the **compile** (`build`) and **execute** (`run_agent` / `execute_graph`) pipeline works.

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         magic_agents Package                       │
├─────────────────────────────────────────────────────────────────────┤
│  agt_flow.py        ──┐                                            │
│  node_system/          │  • Build & run directed graphs            │
│  models/               │  • Node implementations (LLM, Fetch, …)   │
│  util/                 │  • Pydantic models                        │
└────────────────────────┴───────────────────────────────────────────┘
```

1. **`agt_flow.py`** – Orchestrates everything. It exposes:
   • `build(...)` – compiles a JSON spec into an `AgentFlowModel` (in-memory graph).  
   • `run_agent(...)` – async generator that streams `ChatCompletionModel` chunks.  
   • Internal helpers: `execute_graph`, `execute_graph_loop` (for Loop nodes).
2. **`node_system/`** – One Python file per node type (e.g. `NodeLLM.py`, `NodeFetch.py`). All inherit from the common `Node` base class defined in `node_system/Node.py`.
3. **`models/`** – Pydantic models that describe:
   • Graph (`AgentFlowModel`, `EdgeNodeModel`).  
   • Node payloads (e.g. `LlmNodeModel`).
4. **`util/const.py`** – Small constants such as the special `HANDLE_VOID` output handle.

The runtime executes nodes **topologically**; each node yields 0-N `ChatCompletionModel` chunks that are passed downstream via *handles* (named edges).

---

## 2. Node Definitions

| Node Type (`type`) | Python Class            | Purpose |
|--------------------|-------------------------|---------|
| `user_input`       | `NodeUserInput`         | Seeds the graph with the user's message & images, assigns chat/thread IDs. |
| `text`             | `NodeText`              | Emits a static text string. |
| `parser`           | `NodeParser`            | Renders a Jinja2 template based on upstream inputs. |
| `fetch`            | `NodeFetch`             | Executes an HTTP request (sync/async) and returns parsed JSON / text. |
| `client`           | `NodeClientLLM`         | Provides a configured MagicLLM client instance to child nodes. |
| `llm`              | `NodeLLM`               | Streams or batches LLM completions; can enforce JSON output. |
| `chat`             | `NodeChat`              | Augments prompts with memory & conversation context. |
| `send_message`     | `NodeSendMessage`       | Sends extra payloads (references, UI hints) via `ChatCompletionModel.extras`. |
| `loop`             | `NodeLoop`              | Iterates over a list; children run once per item & aggregate results. |
| `conditional`      | `NodeConditional`       | Branches execution based on a Jinja2 condition; bypasses non-selected paths. |
| `inner`            | `NodeInner`             | Runs a *nested* agent flow (`magic_flow`) as a sub-graph. |
| `end` / `void`     | `NodeEND`               | Terminator; swallows un-connected outputs. |

Each node exposes:

```python
class Node:
    id: str               # uuid / author-chosen
    type: str             # same value as spec
    inputs: dict[str, Any]

    async def process(self, chat_log) -> AsyncGenerator[ChatCompletionModel, None]:
        ...               # yield 0-N streamed chunks
```

All inputs arrive through **handles** (`edge.sourceHandle ➜ edge.targetHandle`). They are stored in `self.inputs` and available to templates (`handle_parser_input`, etc.).

---

## 3. Graph Compilation (`build`)

`agt_flow.build(spec, message, images)` performs the following steps:

1. **Sort Nodes / Edges** – Uses `sort_nodes` utility to ensure master node (usually `user_input`) is first.
2. **Add *void* Node** – A hidden sink node is appended so every edge has a valid target.
3. **Patch Edges** – Any edge with an unspecified `targetHandle` is wired to `HANDLE_VOID` on the *void* node.
4. **Inject Message / Images** – For `user_input` & `chat` nodes the user’s message is inserted into `node.data`.
5. **Instantiate Runtime Nodes** – Maps the JSON dicts to real Python node objects via `create_node`.
6. **Inner Graphs** – For every `NodeInner` it recursively calls `build` on its `magic_flow` sub-spec.
7. **Return** an **`AgentFlowModel`** that contains:
   • `nodes: dict[str, Node]`  
   • `edges: list[EdgeNodeModel]`  
   • metadata (`debug`, `master`, ...)

![Compile sequence](https://mermaid.live/svg?pako:eNqVjjEOgzAMRX8l5DFTjcLJoh46i3VAuBA9iVjo4wkyJw5GtCFO3Hk7e--v7saQIwRzBAtEFGo07kuZXAzFK6NiOJw9eSmhhVw7p5mLZEau7_cuDi6HedPl36fXmhzA2G2Ph1CXQcLl_qhVmXfVq_4HVHOxPiNLcJqWqN83LttP8wV4CsAcwhqtDcKR0ZUum7JmLqaCY0Q0XFYDaUtc_r3LYyg_)  <!-- tiny mermaid diagram -->

---

## 4. Graph Execution

Execution is handled by **`execute_graph`** (or **`execute_graph_loop`** when a `NodeLoop` is present). The algorithm is roughly:

```text
while edges remain:
    pick an edge whose *source node* has all dependencies satisfied
    await source.process(chat_log)               # may yield streamed chunks
    forward chunk ➜ target.inputs[targetHandle]
```

Key details:

- **Streaming** – Nodes can `yield` multiple `ChatCompletionModel` chunks; these are *immediately* propagated downstream so the caller receives partial output in real-time.
- **Dependency Check** – A node may require multiple handles to be filled before it runs (fan-in). `are_dependencies_satisfied` verifies this.
- **Loop Node** – When encountered, `execute_graph_loop` dynamically expands edges per item and collects aggregated results through `NodeLoop.OUTPUT_HANDLE_END`.
- **Error Handling** – Any exception aborts execution unless user wraps `process` with try/except.

---

## 5. Putting It All Together

```python
from magic_agents.agt_flow import build, run_agent

spec = ...                      # JSON-compatible flow spec (see README Quickstart)

agent_graph = build(spec, message="Hello!")
async for chunk in run_agent(agent_graph):
    print(chunk.choices[0].delta.content, end="")
```

`build` gives you a validated, runtime-ready graph; `run_agent` executes it and yields streamed LLM chunks + extras.

---

## 6. Additional Resources

* [Root README](../README.md) – full feature list & API reference.
* `magic_agents/node_system/*.py` – deep-dive into each node implementation.
* `magic_agents/agt_flow.py` – graph compiler & scheduler.
