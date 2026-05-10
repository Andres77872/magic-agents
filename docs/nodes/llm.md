# `llm`

## Purpose

Run LLM generation in batch or streaming mode, optionally with tools or JSON parsing.

## Runtime class

- `NodeLLM`
- model: `LlmNodeModel`

## Default inputs

- `handle-client-provider`
- `handle-chat`
- `handle-system-context`
- `handle_user_message`
- dynamic tool handles with prefix `handle-tool-`

## Runtime-overridable inputs

The runtime also supports input handles that can override selected generation settings at execution time:

- `handle-llm-temperature`
- `handle-llm-top_p`
- `handle-llm-max_tokens`
- `handle-llm-stream`
- `handle-llm-iterate`
- `handle-llm-json_output`

## Default outputs

- `handle_streaming_content` — streaming chunks during generation
- `handle_generated_content` — complete response after generation
- `handle-tool-calls` — tool call requests (when tools are present)

## Canonical output

`handle_generated_content` is the canonical routed output for downstream nodes. Use this handle for edge connections.

## Important behavior

- supports streaming and non-streaming execution
- supports `json_output` with code-block extraction before JSON parsing
- supports `iterate: true` so the node re-runs on each loop iteration
- collects tools from `fetch`, `python_exec`, `mcp`, and task-subagent bundles
- warns for engines known to have weak/no tool support

## Gotchas

- if debug is enabled, consumers must handle non-content debug events too
- `handle-tool-calls` is only emitted when tools are present

## Example

```json
{
  "id": "answer",
  "type": "llm",
  "data": {
    "stream": true,
    "temperature": 0.2,
    "max_tokens": 512
  }
}
```
