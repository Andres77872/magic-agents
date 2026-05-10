# `conditional`

## Purpose

Choose one output handle by evaluating a Jinja2 expression.

## Runtime class

- `NodeConditional`
- model: `ConditionalNodeModel`

## Default input

- `handle_input`

## Outputs

- dynamic user-defined handle such as `adult`, `minor`, `approved`, `rejected`
- internal bookkeeping `end`
- system signals like `__bypass_all__` on error paths

## Important behavior

- condition template must render the **name of the output handle**
- supports `merge_strategy: flat | namespaced`
- exposes convenience alias `value` for the primary input
- stores `selected_handle` for executor bypass propagation
- uses `default_handle` only when the rendered result is empty
- `__bypass_all__` is emitted on configuration/template errors, causing executor to skip all downstream branches

## Recommended fields

- `condition` (required)
- `merge_strategy`
- `handles` — custom input handle name mappings (e.g., `{'input': 'my_input'}`)
- `output_handles`
- `default_handle`

## Example

See [../../examples/conditional/conditional_simple_if_else.json](../../examples/conditional/conditional_simple_if_else.json).
