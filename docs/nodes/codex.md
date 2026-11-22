# `codex`

## Purpose

Prepend trigger-matched context to a user message. Matched entries are wrapped as `<codex_content>...</codex_content>` prompt text and emitted as user-message-compatible content.

## Runtime class

- `NodeCodex`
- model: `CodexNodeModel`

## Handles

| Direction | Handle | Purpose |
| --- | --- | --- |
| Input | `handle_codex_input` | User message/context input. |
| Output | `handle_user_message` | Codex context emitted as user-message-compatible content. |

Handle overrides are supported through `data.handles`:

- `input`
- `output`

## Model fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `codex_entries` | `array` | No | `[]` | List of strict `CodexEntry` objects. |

### `CodexEntry`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `string` | No | Auto-generated UUID4 hex | If provided, must be valid UUID4 in hyphenated or non-hyphenated form. |
| `triggers` | `array[string]` | Yes | - | Trigger keywords. Minimum length: 1. |
| `content` | `string` | Yes | - | Content to prepend when a trigger matches. Minimum length: 1. |

Unknown fields are rejected.

## Important behavior

- matching is case-insensitive
- each entry prepends at most once per invocation
- a trigger matches as a whole token: it may not be preceded or followed by a letter, digit or underscore
- an empty or missing input yields an empty passthrough
- if no entries match, the original message is emitted unchanged
- `<codex_content>` is prompt annotation text, not parsed or escaped XML

## Gotchas

Do not keep a direct `user_input.handle_user_message -> chat.handle_user_message` edge alongside the Codex path. That creates competing writes to the downstream `handle_user_message` input; route through Codex instead:

```json
[
  {"source": "input", "target": "codex", "sourceHandle": "handle_user_message", "targetHandle": "handle_codex_input"},
  {"source": "codex", "target": "chat", "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"}
]
```

Triggers that start or end with a non-word character, such as `c++`, `.net` or `node.js`, match too: `c++` matches `use c++ now` but not `c++x`.

## Example

See the Memory + Codex graph in [memory.md](memory.md#memory--codex-coexistence); it builds without provider credentials.
