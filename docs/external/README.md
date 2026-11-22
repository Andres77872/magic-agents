# Issues in external repositories

These pages record problems observed from `magic-agents` whose root cause lives
in another repository. They are out of scope for this repository: fix them
upstream. Each entry says how `magic-agents` is affected and what, if
anything, this repository does to work around it.

| Repository | Page |
| --- | --- |
| [magic-llm](https://github.com/Andres77872/magic-llm) (`magic_llm`) | [magic-llm.md](magic-llm.md) |

## Conventions

- Upstream locations are given as `magic_llm/<path>:<line>` for the
  published `magic_llm` 0.1.54 that `setup.cfg` resolves from
  `git+https://github.com/Andres77872/magic-llm.git@master`. Line numbers
  move; search for the quoted symbol if they no longer match.
- Every entry was checked against that published package, not against a local
  checkout with uncommitted changes.
- When an issue is fixed upstream, delete its entry and any workaround that
  the entry lists here.
