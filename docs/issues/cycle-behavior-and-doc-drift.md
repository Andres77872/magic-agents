# Cycle behavior and documentation drift

## Status

Documented limitation and documentation-drift note.

## What the current docs say

[../wiki/VALIDATION.md](../wiki/VALIDATION.md) identifies two cycle-related behaviors in the repository:

- `detect_cycles()` raises `ValueError`
- the current build path calls `sort_nodes()`, which uses `perform_topological_sort()` and falls back to insertion order when NetworkX cannot topologically sort the graph

[../wiki/ARCHITECTURE.md](../wiki/ARCHITECTURE.md) repeats the same nuance in the build-pipeline section.

## Practical meaning

Older documentation that claims all cycles hard-abort execution is too broad for the current documented build path.

Do not treat arbitrary non-loop cycles as supported graph semantics. Use the `loop` node for supported iteration behavior, and treat non-loop cycles as a limitation unless the runtime code and validation docs are updated together.

## Related docs

- [../wiki/VALIDATION.md](../wiki/VALIDATION.md)
- [../wiki/ARCHITECTURE.md](../wiki/ARCHITECTURE.md)
- [../wiki/EXECUTION_MODEL.md](../wiki/EXECUTION_MODEL.md)
- [../nodes/loop.md](../nodes/loop.md)
