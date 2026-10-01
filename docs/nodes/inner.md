# `inner`

## Purpose

Execute a nested graph from inside a parent graph.

## Runtime class

- `NodeInner`
- model: `InnerNodeModel`

## Default input

- `handle_user_message`
- optional `handle_client_extras`

## Default outputs

- `handle_content_stream` — emitted during execution (real-time streaming chunks)
- `handle_execution_content` — emitted after execution completes (aggregated content)
- `handle_execution_extras` — emitted when inner graph produces extras (final extras)

## Important behavior

- accepts embedded graph config via `magic_flow`, `flow`, `graph`, or `subgraph`
- builds the child graph recursively during parent build
- forwards child streaming chunks in real time
- merges client extras with parent state exposure
- exposes full parent state as `parent_state` by default, or mapped keys via `parent_state_mapping`

## Failures

The sub-flow fails when one of its nodes ends in `ERROR`: it raised, it timed out waiting for inputs, or it emitted `__bypass_all__`. This is decided from child node state (the executor's result sink), not from diagnostic frames: a child that emits a non-fatal diagnostic and still completes, for example a `JSONParseError` that an `onFinish` Hook recovers, leaves the sub-flow successful and its Result is delivered.

On failure the Inner node:

- forwards the child frames as usual, then emits `SUBGRAPH_END` with `status: "error"`
- raises `OperationFailure` with code `INNER_FLOW_FAILED`; its executor frame has `error_type: "OperationFailure"` and `error_code: "INNER_FLOW_FAILED"`
- delivers nothing: no `handle_execution_content`, no extras; downstream is bypassed (edge-scoped, see [EXECUTION_MODEL.md](../wiki/EXECUTION_MODEL.md))

The error message names the root cause: `Sub-flow step '<path>' failed: <error_code or error_type>: <message>` (a nested path is joined with `/`). The cause is the first child error frame, after the child's `GRAPH_START`, from a step that is in `failed_nodes`; a diagnostic from a step that recovered and completed (for example a `JSONParseError` an `onFinish` Hook repaired) is never named, even when it came first. Input-wait `TimeoutError`s are effects: they are named only when no other step failed, and then the first frame of the run (typically the diagnostic of the producer that stayed silent) is the cause. For a nested Inner Flow, the path continues into the step the nested Inner Flow itself named. The same message is the parent run's error (`on_node_error`).

| `error` field | Value |
|---|---|
| `code` | `INNER_FLOW_FAILED` |
| `retryable` | taken from the root-cause child's typed outcome when it has one (a Fetch HTTP 429/5xx or network error, a Hook-controlled node, a nested Inner Flow); otherwise `false` |
| `details.failed_nodes` | child node ids in `ERROR` |
| `details.cause_path` | path (relative to the sub-flow) of the step the message names, or `null` when no frame names one |
| `details.child_errors` | up to 20 `{node_path, error_type, error_message (at most 500 chars), error_code?}` in frame order, including diagnostics of steps that recovered; frame `context` dicts are never copied |
| `details.partial_content` | the text the sub-flow produced before failing (what reached End) |
| `details.child_run_id` | the child run id |

An `onError` lifecycle Hook targeting the Inner node receives this outcome and can recover it (`{"action": "outcome", "outcome": {"status": "success", "content": {"handle_execution_content": ...}}}`); downstream then receives the recovered value and the invocation record has `recovered: true`. `onFinish` sees `status: "error"` when it is not recovered. Under the node's own lifecycle Hooks the child trace (`SUBGRAPH_START`, child error frames, `SUBGRAPH_END`) stays in the stream in both cases; an Inner Flow reached through a Hook child call or redirect reports only through its invocation record. A redirect into a backup Inner Flow that fails adopts the backup's `INNER_FLOW_FAILED` error, so the redirecting node fails too (no empty success). In a Loop, a failed iteration aggregates `null`, or the recovered value.

### Tolerated failures inside the sub-flow

Any child node in `ERROR` fails the Inner Flow, including a failure the sub-flow itself absorbs: a Loop item whose body step fails (its slot is `null`), or a fan-in step that renders with the inputs that did arrive. This is the same rule as a top-level run, which also ends with `on_graph_error` in those cases. What reached End is not delivered downstream; it is kept in `error.details.partial_content`. To keep such a degraded Result, add an `onError` Hook on the Inner node that adopts it:

```python
def adopt(context, chat_log):
    partial = context.outcome["error"]["details"]["partial_content"]
    return {"action": "outcome", "outcome": {"status": "success",
            "content": {"handle_execution_content": partial}}}
```

Tool mode follows the same rule: the tool call fails with `OperationFailure("INNER_FLOW_FAILED")` whose message starts with `Inner tool failed:`; the model receives `{"error": "Inner tool failed: sub-flow step '...' failed: ...", "type": "OperationFailure"}`.

## Gotchas

- malformed `magic_flow` does not crash build; execution emits a `ConfigurationError` diagnostic and `__bypass_all__` (an `InputError` when `handle_user_message` is missing), so the node fails and downstream is bypassed. Under a lifecycle Hook this is the typed outcome `NODE_ERROR`
- child `flow_state` is intentionally isolated
- the child graph uses its own `timeout` (default 60 s), not the parent's
- when a Hook-controlled Inner node is cancelled by its invocation budget (or interrupted by an unexpected exception), the kept trace still closes every open sub-flow with a `SUBGRAPH_END` (`status: "error"`, `reason: "cancelled"` or `"interrupted"`)

## Example

```json
{
  "id": "inner-step",
  "type": "inner",
  "data": {
    "magic_flow": {
      "type": "chat",
      "nodes": [{"id": "u", "type": "user_input"}, {"id": "e", "type": "end"}],
      "edges": []
    }
  }
}
```
