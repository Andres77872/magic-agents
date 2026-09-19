# `chat`

## Purpose

Build or reuse a `ModelChat` transcript for downstream LLM generation. NodeChat composes messages from five merge slots (see [5-Slot Merge Order](#5-slot-merge-order)), applies optional STM windowing, and yields the assembled chat to downstream nodes (typically an `llm` node).

**Explicit boundary**: Conversation STM (short-term memory) windowing belongs to NodeChat. NodeMemory handles insights and punctual memory data only — it does **not** manage conversation messages or interactions. If you need per-turn message count limits or token budget enforcement, configure STM fields on the chat node (see [STM Windowing](#stm-windowing)).

## Runtime class

- `NodeChat`
- model: `ChatNodeModel`

## Default handles

| Direction | Handle | Description |
|-----------|--------|-------------|
| Input | `handle-system-context` | System context message — inserted at index 0 (Slot 4). |
| Input | `handle_user_message` | Current-turn user message — appended last (Slot 5). |
| Input | `handle_messages` | Runtime messages list — appended or replaces history (Slot 2). |
| Input | `handle_user_files` | Extracted file content, optionally paired with a page/image reference. |
| Input | `handle_user_images` | User image content for multimodal message attachment. |
| Output | `handle_chat_output` | Assembled `ModelChat` with windowed messages. Canonical output. |

### Handle aliases

Handle names are configurable via the `handles` dict on `ChatNodeModel`. The following aliases are supported:

| Handle Key | Aliases (in precedence order) | Default |
|------------|------------------------------|---------|
| `system_context` | `system` | `handle-system-context` |
| `user_message` | `message` | `handle_user_message` |
| `messages` | — | `handle_messages` |
| `user_files` | `files` | `handle_user_files` |
| `user_images` | `images` | `handle_user_images` |
| `output` | `chat` | `handle_chat_output` |

Example override:
```json
{
  "id": "chat",
  "type": "chat",
  "data": {
    "handles": {
      "user_message": "custom-user-msg",
      "system_context": "custom-system"
    }
  }
}
```

## Model fields

### `ChatNodeModel` — all fields

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `session_id` | `string` or `null` | No | `null` | — | Thread/conversation ID for persistence. Backend-authoritative — injected by backend, not from JSON config. |
| `session_required` | `boolean` | No | `false` | — | If `true`, enforce session presence (auto-create fallback). |
| `history_messages` | `array` or `null` | No | `null` | — | Backend-injected persisted + runtime messages (Slot 1). Populated by backend via `build()` call, NOT from JSON config. |
| `custom_messages` | `array` or `null` | No | `null` | — | Pre-user context injection (Slot 3). Array of message dicts with `role`/`content`. |
| `messages_append_mode` | `boolean` | No | `false` | — | When `false` (legacy REPLACE mode), runtime messages in Slot 2 overwrite history. When `true` (APPEND mode), runtime messages extend history. |
| `message` | `string` or `null` | No | `null` | — | Legacy field kept for backward compatibility. Not the primary input — use `handle_user_message` instead. |
| `memory` | `object` or `null` | No | `null` | — | Legacy memory dict. **DEPRECATED** for `max_input_tokens`. Supports fallback: if `memory.max_input_tokens` is set and the new first-class `max_input_tokens` is not, the legacy value is used (with a deprecation warning). **The legacy `memory.stm` key is deprecated and ignored. Use `max_messages` instead.** Other keys (`ltm`, etc.) are out of scope and not part of this cleanup. |
| `handles` | `object` or `null` | No | `null` | — | Handle name overrides (see [Handle aliases](#handle-aliases)). |
| `max_messages` | `int` or `null` | No | `null` | `>= 1` | Maximum non-system conversation messages to keep. System messages are always preserved and NOT counted toward this limit. Applied after the 5-slot merge, before yield. |
| `max_input_tokens` | `int` or `null` | No | `null` | `>= 1` | Maximum estimated tokens for the assembled chat. Applied AFTER `max_messages`. When `null` and legacy `memory.max_input_tokens` is set, falls back to the legacy value with a deprecation warning. |
| `truncation_strategy` | `string` | No | `"tail"` | `"tail"` or `"token_budget"` | Windowing strategy. `"tail"` = keep the newest N non-system messages. `"token_budget"` = defer to token budget (skip the message-count step and go directly to token-level enforcement). |
| `model` | `string` or `null` | No | `null` | — | Usage/logging only. Does **not** affect STM windowing token estimation, which uses a hardcoded tokenizer. Optional — purely informational. |

### Backward compatibility

All new STM fields are **additive with defaults**. Existing graphs that do not include `max_messages`, `max_input_tokens`, or `truncation_strategy` in their JSON config continue to work identically — no STM windowing is applied.

## 5-Slot Merge Order

NodeChat assembles the final message list by merging messages from five ordered slots. Each slot is processed sequentially:

```
Slot 1: history_messages (backend-injected)   ← base message list
  ↓
Slot 2: runtime handle_messages input          ← APPEND or REPLACE (per messages_append_mode)
  ↓
Slot 3: custom_messages from config            ← APPEND
  ↓
Slot 4: system context (INSERT at index 0)     ← from handle-system-context input
  ↓
Slot 5: user message (APPEND)                  ← from handle_user_message input
  ↓
[NEW] STM Windowing (optional)                 ← apply_windowing() if limits configured
  ↓
yield self.yield_static(self.chat, ...)        ← windowed output
```

### Slot details

| Slot | Source | Operation | Description |
|------|--------|-----------|-------------|
| **1** | `data.history_messages` | Copy to `base_messages` | Backend-authoritative persisted + runtime history. Copy avoids mutation of original. |
| **2** | `handle_messages` input | `extend()` (APPEND) or replace (REPLACE) | Runtime messages from upstream node. Mode controlled by `messages_append_mode` field. |
| **3** | `data.custom_messages` | `extend()` (always APPEND) | Config-level messages appended after history + runtime. |
| **4** | `handle-system-context` input | `set_system()` (INSERT at index 0) | System context message inserted at the beginning of the message list. |
| **5** | `handle_user_message` input | `add_user_message()` (APPEND) | Current-turn user input — always the final slot. Supports multimodal content via `handle_user_images` and extracted attachments via `handle_user_files`. |

### Image and file inputs

`handle_user_images` accepts image references (URL/data URL/bytes), including a
JSON-encoded list. All image references are attached to the current-turn user
message.

`handle_user_files` accepts provider-neutral descriptors because `ModelChat`
does not expose a raw-file primitive. Each descriptor is either a
`[text, image]` pair or a mapping with `text`/`content` and an optional
`image`/`url`/`image_url`. Extracted file messages are inserted immediately
before the current-turn user message. For backward compatibility, legacy
`[text, image]` descriptors routed through `handle_user_images` remain valid.

The image and file handles are mutually exclusive for a single Chat node
execution. Invalid or mixed attachment shapes produce a `ValidationError`
debug event and no chat output.

## STM Windowing

STM windowing controls how many messages (or how many tokens) reach the LLM. It is applied **after** the 5-slot merge and **before** the yield, making it the final gate before the downstream LLM node receives the chat.

### When windowing runs

Windowing is evaluated **every execution/turn** over the accumulated messages array. Each non-system message array element counts individually toward `max_messages`. System messages are **always preserved** and **excluded** from the `max_messages` count.

Windowing is **skipped entirely** when both `max_messages` and `max_input_tokens` are `null` (default). This means zero overhead for existing graphs that don't configure STM fields.

### Fields

| Field | Effect |
|-------|--------|
| `max_messages` | Message-count limit. Keeps the newest N non-system messages (excluding system messages and the last user message). Older messages are dropped. |
| `max_input_tokens` | Token budget limit. Estimated tokens via `tiktoken`; oldest messages dropped until within budget. Applied after `max_messages`. |
| `truncation_strategy` | `"tail"` = apply message-count then token-budget. `"token_budget"` = skip message-count, go directly to token enforcement. |

**Token estimation uses a hardcoded GPT-5 tokenizer (not the `model` field).** The `model` field on `ChatNodeModel` has no effect on token counting or windowing behavior. If GPT-5 is unknown to tiktoken, the system silently falls back to `cl100k_base` encoding — this is by design and requires no configuration.

### `max_input_tokens` precedence

When resolving `max_input_tokens` at runtime, NodeChat uses the following precedence:

1. **New first-class `max_input_tokens` field** on `ChatNodeModel` (if not `null`)
2. **Legacy `memory.max_input_tokens`** from the `memory` dict (if the dict exists and key is present) — a deprecation warning is logged
3. **`null`** (no token budget enforcement) if neither is set

### Semantics

- **System messages** are always preserved and excluded from `max_messages`.
- **Last user message** (the current turn) is always preserved — it is the primary prompt for the LLM.
- **Tool-call chains** are atomic (see [Tool-Call Atomicity](#tool-call-atomicity)).
- Windowing is **stateless** — computed fresh every `process()` call.
- The input message list is **never mutated** — a new list is returned.

### Interaction with NodeLLM (no-CHAT fallback)

When a graph connects `user_input → llm` directly (no `chat` node), NodeLLM's no-CHAT fallback path applies its own STM windowing on `history_messages`. In this path, if `max_messages` is not configured on the LLM node, a **default of 30** (`NodeLLM.DEFAULT_MAX_MESSAGES`) is applied as the effective limit. This protects inline graphs from unbounded context.

When the CHAT path IS active (NodeChat yields to NodeLLM), NodeLLM does NOT apply additional windowing — the chat node is the owner of STM windowing. The no-CHAT default of 30 does NOT apply to the CHAT path.

### Examples

**Message-count limit**:
```json
{
  "id": "chat",
  "type": "chat",
  "data": {
    "max_messages": 10
  }
}
```
Keeps the newest 10 non-system conversation messages each turn. System messages and the current-turn user message are always preserved (so actual message count may exceed 10).

**Token budget**:
```json
{
  "id": "chat",
  "type": "chat",
  "data": {
    "max_input_tokens": 32000
  }
}
```
Enforces a 32k token budget (using hardcoded GPT-5 tokenizer). Messages beyond the budget are dropped oldest-first. The `model` field, if set, has no effect on token estimation.

**Full STM configuration**:
```json
{
  "id": "chat",
  "type": "chat",
  "data": {
    "max_messages": 20,
    "max_input_tokens": 64000,
    "truncation_strategy": "token_budget"
  }
}
```
Uses token-budget strategy (skips message-count step), applies a 64k token budget with hardcoded GPT-5 tokenizer estimates. Falls back to `cl100k_base` if GPT-5 is unknown to `tiktoken`. The `model` field (if present in the config) is purely informational and does not affect token counting.

## Tool-Call Atomicity

Tool-call chains (an `assistant` message with `tool_calls` + consecutive `tool` role messages) are treated as **atomic units**. A chain is either kept in full or dropped entirely — chains are never split.

### Chain definition

1. An `assistant` message with a non-empty `tool_calls` array
2. Zero or more consecutive `tool` role messages immediately following (tool results)

Chains are detected by a pre-scan that tags each chain message with a shared `_chain_group_id` (UUID). Standalone messages (not part of a chain) get `_chain_group_id = None`.

### Truncation behavior

When truncating, `find_safe_tail_cut()` scans from newest to oldest and adjusts the cut boundary to chain boundaries:

- If a chain **straddles** the naive cut point (part in the kept region, part in the dropped region), the cut moves to the **start of the chain** so the entire chain is kept.
- If a chain is entirely in the **dropped region** (all messages older than the cut), the entire chain is dropped.
- Standalone `tool` messages without a preceding `assistant` (orphan tools) are NOT chain-tagged and can be dropped individually.

### Example

Given messages `[user(1), assistant(tool_calls), tool, tool, user(2)]` with `max_messages=3`:
- Naive cut would keep the newest 3: `[tool, tool, user(2)]` — **splitting** the chain
- Safe cut adjusts to: `[assistant(tool_calls), tool, tool, user(2)]` — **chain fully kept** (4 messages)

Given `[assistant(tool_calls), tool, user(1), user(2)]` with `max_messages=2`:
- The entire chain is dropped: `[user(1), user(2)]`
- Chain was atomically dropped, not split.

## Two-Layer Token Budget Defense

| Layer | Location | Mechanism | Responsibility |
|-------|----------|-----------|----------------|
| **Layer 1 (PRIMARY)** | `NodeChat.process()` (post-merge) | `apply_windowing()` with `max_messages` + `max_input_tokens` | NodeChat owns STM windowing. Applied before yield. |
| **Layer 2 (SAFETY NET)** | `ModelChat.__init__()` (in magic-llm) | `max_input_tokens` → `get_messages()` token truncation | Provider-level fallback. Retained from existing behavior. |

Layer 1 is the primary control. Layer 2 is a safety net that operates independently — if `max_input_tokens` is resolved (from either the new field or legacy `memory` fallback), it is passed to `ModelChat.__init__()` as well.

In standard flow (NodeChat → NodeLLM), the LLM provider reads `chat.messages` directly, so Layer 2's `get_messages()` is NOT called. Layer 1 is the sole active enforcement.

## Diagnostics / Internal State

The following fields are exposed via `NodeChat._capture_internal_state()`:

| Field | Type | Description |
|-------|------|-------------|
| `max_messages` | `int` or `null` | Configured max non-system messages. |
| `max_input_tokens` | `int` or `null` | Configured (or legacy-resolved) max input tokens. |
| `truncation_strategy` | `str` | Current truncation strategy (`"tail"` or `"token_budget"`). |
| `messages_before_windowing` | `int` | Message count before `apply_windowing()` was called this turn. |
| `messages_after_windowing` | `int` | Message count after windowing. |
| `messages_discarded` | `int` | `messages_before_windowing − messages_after_windowing`. |
| `total_tokens_estimated` | `int` or `null` | Estimated token count after windowing (or `null` if not estimated). |
| `windowing_applied` | `bool` | `true` when any STM limit was active this turn. |

## Graph Wiring Examples

### Basic chat (with Codex)

```json
{
  "nodes": [
    {"id": "input", "type": "user_input", "data": {}},
    {"id": "codex", "type": "codex", "data": {"codex_entries": []}},
    {"id": "client", "type": "client", "data": {"engine": "openai"}},
    {"id": "chat", "type": "chat", "data": {}},
    {"id": "answer", "type": "llm", "data": {}},
    {"id": "end", "type": "end"}
  ],
  "edges": [
    {"source": "input", "target": "codex", "sourceHandle": "handle_user_message", "targetHandle": "handle_codex_input"},
    {"source": "codex", "target": "chat", "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
    {"source": "client", "target": "answer", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
    {"source": "chat", "target": "answer", "sourceHandle": "handle_chat_output", "targetHandle": "handle-chat"},
    {"source": "answer", "target": "end", "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input"}
  ]
}
```

### Chat without Codex

```json
{
  "nodes": [
    {"id": "input", "type": "user_input", "data": {}},
    {"id": "client", "type": "client", "data": {"engine": "openai"}},
    {"id": "chat", "type": "chat", "data": {}},
    {"id": "answer", "type": "llm", "data": {}},
    {"id": "end", "type": "end"}
  ],
  "edges": [
    {"source": "input", "target": "chat", "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
    {"source": "client", "target": "answer", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
    {"source": "chat", "target": "answer", "sourceHandle": "handle_chat_output", "targetHandle": "handle-chat"},
    {"source": "answer", "target": "end", "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input"}
  ]
}
```

### Chat with NodeMemory (memory → chat wiring)

```json
{
  "nodes": [
    {"id": "input", "type": "user_input", "data": {}},
    {"id": "memory-1", "type": "memory", "data": {"memory_entries": []}},
    {"id": "client", "type": "client", "data": {"engine": "openai"}},
    {"id": "chat", "type": "chat", "data": {}},
    {"id": "answer", "type": "llm", "data": {}},
    {"id": "end", "type": "end"}
  ],
  "edges": [
    {"source": "input", "target": "memory-1", "sourceHandle": "handle_user_message", "targetHandle": "handle_memory_input"},
    {"source": "memory-1", "target": "chat", "sourceHandle": "handle_memory_output", "targetHandle": "handle_user_message"},
    {"source": "client", "target": "answer", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
    {"source": "client", "target": "memory-1", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
    {"source": "chat", "target": "answer", "sourceHandle": "handle_chat_output", "targetHandle": "handle-chat"},
    {"source": "answer", "target": "end", "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input"}
  ]
}
```

### Chat with STM windowing

```json
{
  "nodes": [
    {"id": "input", "type": "user_input", "data": {}},
    {"id": "client", "type": "client", "data": {"engine": "openai"}},
    {"id": "chat", "type": "chat", "data": {
      "max_messages": 10,
      "max_input_tokens": 32000,
      "truncation_strategy": "tail"
    }},
    {"id": "answer", "type": "llm", "data": {}},
    {"id": "end", "type": "end"}
  ],
  "edges": [
    {"source": "input", "target": "chat", "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
    {"source": "client", "target": "answer", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
    {"source": "chat", "target": "answer", "sourceHandle": "handle_chat_output", "targetHandle": "handle-chat"},
    {"source": "answer", "target": "end", "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input"}
  ]
}
```

## Out of Scope (Not Part of This Change)

The following are explicitly **not** part of this STM windowing change:

- `session_required` enforcement — unrelated to STM windowing; P2 priority.
- `middle` truncation strategy — adds complexity with limited use cases; deferred.
- Role preservation flags — system and last-user preservation are always-on and non-configurable in MVP.
- NodeMemory interaction protocol — NodeMemory handles insights/memory, not conversation messages.
- `max_image_tokens` / multimodal budget — P3 priority; existing `ModelChat.get_messages()` serves as fallback.
- `memory` dict field removal — deprecation only; removal deferred to the next major version.
