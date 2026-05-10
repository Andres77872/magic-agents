# Task-subagent documentation drift

## Status

Documentation drift.

## What the current docs say

[../wiki/TASK_SUBAGENTS.md](../wiki/TASK_SUBAGENTS.md) describes the current ownership boundary:

- MagicLLM owns `load_subagents()` and the runtime subagent bundle
- Magic Agents keeps the application-side toggle and code-registry wiring in `agt_flow.py`
- `NodeLLM.process()` loads subagent manifests through the client when task subagents are enabled

That page also warns that docs importing from `magic_agents.subagents` are stale for the current codebase.

## Practical meaning

When using or documenting task subagents, prefer the current API surface described in [../wiki/TASK_SUBAGENTS.md](../wiki/TASK_SUBAGENTS.md). Do not rely on older import paths unless they are revalidated against the current repository.

The currently documented bundled example is `research.web`.

## Related docs

- [../wiki/TASK_SUBAGENTS.md](../wiki/TASK_SUBAGENTS.md)
- [../nodes/llm.md](../nodes/llm.md)
