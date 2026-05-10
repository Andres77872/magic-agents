# `master` field is ignored by runtime execution

## Status

Documented limitation.

## What the current docs say

[../wiki/EXECUTION_MODEL.md](../wiki/EXECUTION_MODEL.md) describes runtime execution as reactive:

- tasks are created for nodes up front
- readiness comes from incoming edges
- nodes with no incoming edges can be ready immediately
- routing is based on handles and edges

The same page explicitly notes that there is no single imperative "walk the graph from master node" loop.

## Practical meaning

Graph data may include a legacy `master` field, and existing JSON can still be accepted by the surrounding graph model/build surface. However, current runtime execution should not be understood as starting from or being controlled by `master`.

Rely on graph connectivity, handles, and node readiness instead. For authoring details, use:

- [../wiki/GRAPH_FORMAT.md](../wiki/GRAPH_FORMAT.md)
- [../wiki/HANDLES_AND_ROUTING.md](../wiki/HANDLES_AND_ROUTING.md)
- [../wiki/EXECUTION_MODEL.md](../wiki/EXECUTION_MODEL.md)

## Caveat

This page documents the current runtime-facing docs. It does not claim that every importer, editor, or historical JSON producer treats `master` the same way.
