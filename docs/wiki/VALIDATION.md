# Validation

This page documents what the current code validates and what it does not.

## Build-time validation

`magic_agents.agt_flow.validate_graph()` currently checks:

1. exactly one `user_input` node
2. duplicate edges by `(source, target, sourceHandle, targetHandle)`
3. missing source/target nodes
4. self-loop edges

Invalid source/target edges and self-loops are also filtered before sorting/build continues.

That function is only the first layer.

## Broader contract validation chain

After the graph is instantiated as an `AgentFlowModel`, `run_all_validations()` runs a wider validation chain:

1. edge connectivity
2. source handle validation (`validate_edge_handles`) including legacy-handle rejection
3. target-handle validation (`validate_edge_target_handles`)
4. fan-in/cardinality validation (`validate_edge_fan_in_compatibility`)
5. conditional-specific validation

The resulting diagnostics are attached as a `GraphContractReport` on the built graph model.

### Target-handle validation

The code validates `edge.targetHandle` against the target node's declared input contract.

- unknown target handles usually surface as warnings in `shadow` / `warn` modes
- dynamic `handle-tool-definition-*` targets are allowed for tool wiring into `llm`
- `void` targets are treated specially
- stricter rejection behavior is declared for `strict` mode, but parts of that path are still deferred

### Fan-in / cardinality validation

The runtime now explicitly models multi-edge fan-in instead of treating it as universally invalid.

- multi-edge fan-in to the same `targetHandle` is supported in principle
- exclusive ports (`cardinality="one"`, `exclusive=true`) with multiple incoming edges surface diagnostics
- ambiguous ports produce advisory warnings
- explicitly multi-compatible ports (`cardinality="many"`, `multi_compatible=true`) are accepted

This is why current docs should talk about fan-in compatibility, not just duplicate-edge rejection.

## Conditional validation

After node instantiation, `ConditionalEdgeValidator.validate()` checks conditional nodes for:

- missing edges for declared `output_handles`
- missing edge for `default_handle`
- undeclared outputs as warnings

These results are attached to the built graph and emitted by the executor.

## Validation modes

`AgentFlowModel.contract_config` supports:

- `off` — skip contract validation entirely
- `shadow` — compute diagnostics and attach the report, but do not surface them
- `warn` — compute diagnostics and surface them as warnings/log output (default)
- `strict` — declared on the model surface, but strict enforcement is still partially deferred in follow-up work

`strict_runtime` also exists on the config surface, but runtime delivery enforcement is currently deferred.

## Runtime handling of validation errors

The executors emit `_validation_errors` as `debug` events.

Only blocking structural errors with `error_type == "GraphValidationError"` abort execution immediately.

Conditional routing validation issues do **not** automatically abort execution. They can still surface as runtime bypass/routing behavior.

Contract diagnostics are attached to `graph.contract_report`; they are not all treated as hard runtime blockers today.

## Actual cycle behavior

This repo contains two different cycle-related ideas:

- `detect_cycles()` in `magic_agents/node_system/__init__.py` raises `ValueError`
- the build path actually calls `sort_nodes()`, which uses `perform_topological_sort()` and falls back to insertion order on `NetworkXUnfeasible`

So the important truth is:

> the current build path does **not** use `detect_cycles()` to hard-fail graphs.

That means older claims like "all cycles abort execution" are stale.

### What you should rely on instead

- do not author arbitrary cycles
- use `loop` for supported iteration semantics
- treat non-loop cycle handling as a documented limitation, not a supported feature

See [../issues/cycle-behavior-and-doc-drift.md](../issues/cycle-behavior-and-doc-drift.md).
