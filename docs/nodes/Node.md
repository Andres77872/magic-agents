# Node (Base Class)

All node implementations inherit from this **abstract base class** located at `magic_agents.node_system.Node`.

## Responsibilities

1. Store `inputs` dictionary collected from upstream edges.
2. Provide `yield_static()` helper to wrap values in the standard event envelope.
3. Maintain common attributes: `node_id`, `node_type`, `debug` flag.
4. Cache results (unless overridden) so downstream nodes can access already-executed results.

## Key API

```python
class Node:
    def __init__(self, debug: bool = False, node_id: str = '', **kwargs): ...

    async def process(self, chat_log):
        """Subclasses override: yield 0-N events"""

    async def __call__(self, chat_log):
        # Executes once (unless iterate flag) and caches
        ...

    def get_input(self, handle: str, required: bool = False): ...
    def yield_static(self, value, content_type: str = 'content'): ...
```

- `process()` is **coroutine**; use `yield` for streaming.
- `__call__()` handles caching and forwards to `process()`.

Understanding this base class helps when authoring **custom nodes**.
