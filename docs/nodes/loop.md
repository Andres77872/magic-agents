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

## Iteration inputs

Each iteration starts clean. Inputs that iteration nodes receive from the loop item or from other iteration nodes are cleared before the next item; inputs from nodes that ran before the loop are kept.

- A body step runs with whatever arrived **for this item**, and is skipped when no value from the loop item or from another body step arrived (its producer failed, was bypassed, or completed without emitting that handle). Pre-loop inputs such as a Client or a fixed Text never make it run on their own, so an `llm` whose message producer failed is skipped rather than called without a message.
- A body step that still received some per-item value (for example the item itself) runs without the missing one, and its output reaches the slot.
- The slot is `null` when the step feeding `handle_loop` failed or was skipped; it never repeats the previous item's value.
- A skip caused by a failure is reported with `on_node_bypass` reason `upstream_error`; one caused by a silent producer with `not_ready`.
- A failure in any iteration still fails the run. Inside an Inner Flow this fails the Inner Flow too (see [inner.md](inner.md#tolerated-failures-inside-the-sub-flow)).

## Gotchas

- generic cycles are not the same thing as loop support
- `llm` nodes must set `iterate: true` if they should re-run for each item

## Example

See [../../examples/loop/loop_with_llm_processing.json](../../examples/loop/loop_with_llm_processing.json).
