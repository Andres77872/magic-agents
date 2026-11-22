# `python_exec`

## Purpose

Expose a Python execution tool to an `llm` node, or execute configured Python code as a normal graph node.

## Runtime class

- `NodePythonExec`
- model: `PythonExecNodeModel`

## Modes

| Mode | Trigger | Output |
|------|---------|--------|
| Tool mode (model-written code) | `data.code` is absent or empty, `tool_mode` not `false` | `handle-tool-definition` |
| Fixed tool | `data.code` is a non-empty string and `tool_mode: true` | `handle-tool-definition` (a `FixedPythonTool` whose arguments match `tool_parameters`) |
| Node mode | `data.code` is a non-empty string and `tool_mode` is not `true` | `handle-python_exec-result` |

`tool_mode: false` without `code` is rejected at build ("Python node mode requires code defining run(handler)").

## Config fields

- `safety_mode`
- `timeout`
- `max_output_chars`
- `code` — optional Python source for direct node-mode execution via the `run(handler)` contract
- `tool_name` — optional tool-mode name; defaults to `execute_python`
- `tool_mode` — `null` (default) infers the mode from `code`; `true` with `code` makes a fixed tool; `false` forces node mode
- `tool_description` — fixed-tool description (default "Run the configured Python function with structured arguments.")
- `tool_parameters` — fixed-tool JSON Schema of the arguments
- `handles` — supports `safety_mode`, `timeout`, `max_output_chars`, and `output`

## Important behavior

- wraps MagicLLM's `PythonExecutor`
- warns when `safety_mode == "in_process"` because arbitrary code execution is then enabled
- commonly acts as a tool provider for an `llm` node
- with `data.code`, all non-config input handles become entries in the `handler` dict passed to `run(handler)`
- build skips automatic LLM tool-handle assignment for node-mode `python_exec`

## Example

```json
{
  "id": "pytool",
  "type": "python_exec",
  "data": {
    "safety_mode": "subprocess",
    "timeout": 20
  }
}
```

Node-mode example:

```json
{
  "id": "py-node",
  "type": "python_exec",
  "data": {
    "code": "def run(handler):\n    return handler.get('value', 0) + 1"
  }
}
```
