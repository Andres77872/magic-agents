# Node (Base Class)

All node implementations inherit from this **abstract base class** located at `magic_agents.node_system.Node`.

## Responsibilities

1. Store `inputs` dictionary collected from upstream edges.
2. Provide `yield_static()` helper to wrap values in the standard event envelope.
3. Maintain common attributes: `node_id`, `node_type`, `debug` flag.
4. Cache results so downstream edges can reuse already-executed outputs.

## Key API

```python
class Node:
    def __init__(
        self,
        cost: float = 0.0,
        node_id: str | None = None,
        node_type: str | None = None,
        debug: bool = False,
        **kwargs,
    ): ...

    async def process(self, chat_log):
        """Subclasses override: yield 0-N events"""

    async def __call__(self, chat_log):
        # Executes once and caches (_response)
        ...

    def add_parent(self, parent_outputs: dict, source_handle: str, target_handle: str): ...
    def get_input(self, key: str, default=None, required: bool = False): ...
    def yield_static(self, value, content_type: str = 'end'): ...
```

- `process()` is **async generator**; use `yield` to emit events.
- `__call__()` handles caching and forwards to `process()`.

## Debug Methods

```python
def yield_debug_error(self, error_type: str, error_message: str, context: dict = None) -> dict:
    """Yield a debug error message without raising an exception."""

def get_debug_info(self) -> Optional[NodeDebugInfo]:
    """Get the debug information for this node."""

def mark_bypassed(self):
    """Mark this node as bypassed (e.g., in conditional flow)."""

def _capture_internal_state(self) -> Dict[str, Any]:
    """Capture internal state for debugging. Subclasses override to add specific variables."""
```

## Debug Information Structure

When `debug=True`, nodes track execution details via `NodeDebugInfo`:

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Node identifier |
| `node_type` | `str` | Node type key |
| `node_class` | `str` | Python class name |
| `start_time` | `str` | ISO format execution start |
| `end_time` | `str` | ISO format execution end |
| `execution_duration_ms` | `float` | Execution time in milliseconds |
| `inputs` | `dict` | Captured input values |
| `outputs` | `dict` | Captured output values |
| `internal_variables` | `dict` | Node-specific internal state |
| `was_executed` | `bool` | Whether process() was called |
| `was_bypassed` | `bool` | Whether node was bypassed (conditional) |
| `error` | `str | None` | Error message if failed |

Understanding this base class helps when authoring **custom nodes**.
