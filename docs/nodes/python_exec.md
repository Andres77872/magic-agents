# `python_exec`

## Purpose

Expose a Python execution tool to an `llm` node.

## Runtime class

- `NodePythonExec`
- model: `PythonExecNodeModel`

## Default output

- `handle-tool-definition`

## Config fields

- `safety_mode`
- `timeout`
- `max_output_chars`
- `code` — optional Python source for direct node-mode execution when the runtime supports the `run(handler)` contract

## Important behavior

- wraps MagicLLM's `PythonExecutor`
- warns when `safety_mode == "in_process"` because arbitrary code execution is then enabled
- commonly acts as a tool provider for an `llm` node
- the model also exposes `code` for a direct node-mode contract based on a `run(handler)` function when supported by the runtime

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
