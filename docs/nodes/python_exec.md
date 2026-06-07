# `python_exec`

## Purpose

Expose a Python execution tool to an `llm` node, or execute configured Python code as a normal graph node.

## Runtime class

- `NodePythonExec`
- model: `PythonExecNodeModel`

## Modes

| Mode | Trigger | Output |
|------|---------|--------|
| Tool mode | `data.code` is absent or empty | `handle-tool-definition` |
| Node mode | `data.code` is a non-empty string | `handle-python_exec-result` |

## Config fields

- `safety_mode`
- `timeout`
- `max_output_chars`
- `code` — optional Python source for direct node-mode execution via the `run(handler)` contract
- `tool_name` — optional tool-mode name; defaults to `execute_python`
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
