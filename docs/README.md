# Magic Agents Documentation

Welcome to the **magic_agents** documentation site.

## Contents

| Doc | Description |
|------|-------------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | High-level overview of the internal design, node system, and compile/execute pipeline. |
| **Node Reference ›** | Detailed docs per node type (see table below). |
| Root [`README.md`](../README.md) | Installation guide, quick-start, advanced examples. |

## Node Reference

| Type | Doc |
|------|-----|
| `user_input` | [NodeUserInput](nodes/NodeUserInput.md) |
| `text` | [NodeText](nodes/NodeText.md) |
| `parser` | [NodeParser](nodes/NodeParser.md) |
| `fetch` | [NodeFetch](nodes/NodeFetch.md) |
| `client` | [NodeClientLLM](nodes/NodeClientLLM.md) |
| `llm` | [NodeLLM](nodes/NodeLLM.md) |
| `chat` | [NodeChat](nodes/NodeChat.md) |
| `send_message` | [NodeSendMessage](nodes/NodeSendMessage.md) |
| `loop` | [NodeLoop](nodes/NodeLoop.md) |
| `inner` | [NodeInner](nodes/NodeInner.md) |
| `end` / `void` | [NodeEND](nodes/NodeEND.md) |

---

## Getting Started

1. 📦 **Install** the package (see root README).
2. 🏗️ **Understand the architecture** – start with `ARCHITECTURE.md`.
3. 🤖 **Build your first agent** – follow the Quick-start snippet in the root README.

---

Need more info? Check the source code – every node resides in `magic_agents/node_system/`, and `magic_agents/agt_flow.py` is the main orchestrator.
