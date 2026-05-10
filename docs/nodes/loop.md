# `loop`

## Purpose

Iterate over a list and aggregate per-iteration feedback.

## Runtime class

- `NodeLoop`
- model: `LoopNodeModel` (currently just a stub)

## Default inputs

- `handle_list`
- `handle_loop`

## Default outputs

- `handle_item`
- `handle_end`

## Important behavior

- accepts a JSON string or Python list
- emits each item during iteration
- aggregates values fed back into `handle_loop`
- triggers the specialized loop executor for the entire graph

## Critical runtime nuance

Loop behavior does **not** come from `NodeLoop.process()` alone. The real semantics live mostly in `execute_graph_loop_reactive()`.

That executor adds:

- static phase
- per-iteration topological execution
- branch bypass inside iterations
- aggregation and post-loop execution
- `loop_progress` events

## Gotchas

- generic cycles are not the same thing as loop support
- `llm` nodes must set `iterate: true` if they should re-run for each item

## Example

See [../../examples/loop/loop_with_llm_processing.json](../../examples/loop/loop_with_llm_processing.json).
