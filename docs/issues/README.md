# Known issues and limitations

This catalog tracks documented behavior gaps, limitations, and documentation drift that are referenced from the wiki guides.

These pages are intentionally conservative: they describe what the current docs say and what the runtime-facing docs identify as unresolved or nuanced. If behavior is uncertain, treat the issue page as a limitation rather than a feature promise.

## Catalog

| Issue | Status | Summary |
| --- | --- | --- |
| [master-field-is-ignored.md](master-field-is-ignored.md) | Documented limitation | The graph-level `master` field can appear in graph data, but the current execution model is reactive and does not execute by walking from that field. |
| [cycle-behavior-and-doc-drift.md](cycle-behavior-and-doc-drift.md) | Documented limitation / doc drift | Validation docs identify a mismatch between older "cycles abort" wording and the current build path's sorting fallback. |
| [task-subagent-docs-drift.md](task-subagent-docs-drift.md) | Documentation drift | Task-subagent ownership currently sits across Magic Agents wiring and MagicLLM loading; older import-path docs may be stale. |

## Related docs

- [../wiki/VALIDATION.md](../wiki/VALIDATION.md)
- [../wiki/EXECUTION_MODEL.md](../wiki/EXECUTION_MODEL.md)
- [../wiki/TASK_SUBAGENTS.md](../wiki/TASK_SUBAGENTS.md)
- [../wiki/KNOWN_ISSUES_AND_INCOMPLETE_WORK.md](../wiki/KNOWN_ISSUES_AND_INCOMPLETE_WORK.md) — backward-compatible redirect stub
