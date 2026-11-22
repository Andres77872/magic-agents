# `void`

## Purpose

Internal sink node for outputs that should be dropped.

## Runtime class

- implemented with the dedicated silent `NodeVoid` sink

## Why it is special

- build injects it automatically with a generated UUID
- edges using `targetHandle: "handle-void"` are rewritten to target this node
- an edge with no `targetHandle` (after tool-handle backfilling) defaults to `handle-void`, so it also ends here and delivers nothing
- users usually do not need to declare it manually, but the runtime accepts the type literal

## Practical meaning

`void` is how the build/runtime gives a safe landing place to unrouted outputs without requiring every edge to terminate in a user-visible node.
