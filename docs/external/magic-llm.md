# magic-llm

Issues found while auditing `magic-agents` against the published `magic_llm`
0.1.54. See [README.md](README.md) for conventions.

## ML-1: Attempt-control callbacks can replace a cancellation

**Where:** `magic_llm/engine/attempt_control.py:343-358` (the streaming
attempt wrapper).

The wrapper catches `BaseException` (so also `CancelledError` and
`GeneratorExit`) into `error`, then awaits `control.after_attempt(...)` in
`finally` and `_observe(...)` before `_terminal_error(error, ...)` re-raises.
If either callback raises, its exception propagates instead of the
cancellation. On the `aclose()` path the caller then sees an unrelated error,
and a cancelled graph can look like a provider failure.

**Effect on magic-agents:** `magic_agents/coordination/control.py` and
`magic_agents/execution/llm_storage.py` implement those callbacks. If one of
them raises while a run is being cancelled (for example a storage or budget
failure), that error is reported instead of the cancellation.

**Suggested fix:** re-raise non-`Exception` errors (`CancelledError`,
`GeneratorExit`, `KeyboardInterrupt`) before awaiting the callbacks, or run the
callbacks so that their failures are logged and chained, never substituted.

## ML-2: `ModelChat.set_system()` inserts instead of setting

**Where:** `magic_llm/model/ModelChat.py:82-83`:
`self.messages.insert(index, {"role": "system", "content": system})`.

The name says "set", but every call adds another system message. Calling it on
a chat that already has one, or calling it repeatedly on a reused chat, sends
several system prompts.

**Effect on magic-agents:** a Chat node shares one `ModelChat` with every LLM
it feeds and with every Loop iteration. `NodeLLM` now works on a deep copy
before calling `set_system()`/`add_user_message()`
(`magic_agents/node_system/NodeLLM.py`, Chat branch of `process()`), so
consumers no longer leak prompts into each other. An LLM node that gets both a
Chat input with a system message and its own `handle-system-context` still
sends two system messages; that is documented in
[../nodes/llm.md](../nodes/llm.md).

**Suggested fix:** make `set_system()` replace the system message at `index`
when one exists, and add an explicit `add_system_message()` for the current
behaviour (it already exists, so `set_system()` can simply replace).

## ML-3: `asyncio.wait_for` on Python 3.10/3.11

**Where:** `magic_llm/agent/tool_executor.py:488`,
`magic_llm/agent/task_executor.py:453`,
`magic_llm/util/python_executor.py:119`. The package declares
`Requires-Python: >=3.10`.

Before Python 3.12, `asyncio.wait_for` awaits through a separate task. A
cancellation that arrives in the same loop step as completion is swallowed and
the result returned (CPython gh-86296). So a cancelled agent run can keep
running a tool or task loop. Code that enters an AnyIO cancel scope inside the
awaited coroutine also exits it from the wrong task.

**Effect on magic-agents:** this repository hit the same problem in its own
code and replaced every `asyncio.wait_for` with
`magic_agents.util.aio.wait_for`, a backport of the 3.12 implementation. The
upstream calls above still have the 3.11 behaviour when `magic-agents` runs on
3.11.

**Suggested fix:** use `async with asyncio.timeout(...)` (3.11+) or a
backport like `magic_agents/util/aio.py`. On 3.10, `asyncio.timeout` is
unavailable, so a backport is needed, or the minimum should become 3.11.

## ML-4: magic-agents depends on private upstream names

**Where:**

| Name | Upstream | Used by magic-agents |
| --- | --- | --- |
| `CURRENT_TOOL_CALL` | `magic_llm/agent/tool_executor.py:37` | `coordination/context.py`, `coordination/service.py`, `coordination/tools.py` |
| `_inline_local_refs` | `magic_llm/engine/tooling.py:70` | `coordination/tools.py` (coordination tool schemas) |
| `AsyncAgentLoop._response_json` | `magic_llm/agent/async_agent_loop.py:323` | `execution/llm_storage.py` (durable LLM replay) |

None of these has a compatibility contract, so an upstream refactor can break
coordination or durable replay without any public API change.

**Suggested fix:** export supported equivalents (for example
`magic_llm.agent.current_tool_call()`, `magic_llm.engine.tooling.inline_refs()`,
and a public response serializer), then switch these imports.

## ML-5: `AgentHooks` is synchronous

**Where:** `magic_llm/agent/hooks.py:18` (`class AgentHooks(Protocol)`, sync
`on_*` methods).

Async consumers can't await their own hooks from the loop's callbacks. They have
to schedule tasks and flush them later, so there's no backpressure and ordering
depends on the flush. When the loop runs in a worker thread (the sync
`run_agent()`/`run_agent_stream()` APIs) there is no running event loop to
schedule on.

**Effect on magic-agents:** `magic_agents/hooks/hook_relay.py` schedules async
`FlowHooks` with `asyncio.create_task()` and flushes them on exit. In
`NodeLLM`'s legacy sync fallbacks (used only when the client has no
`run_agent_async()`/`run_agent_stream_async()`), async hooks called from the
worker thread are dropped with a warning. The streaming fallback therefore
iterates the sync stream on the event-loop thread, which blocks the loop. The
pinned `magic_llm` provides the async APIs, so these fallbacks are not used in
practice.

**Suggested fix:** add an async hook protocol (`async def on_*`) that
`AsyncAgentLoop` awaits, or accept a loop/executor to which sync callbacks are
dispatched.

## ML-6: `load_subagents()` silently returns nothing when disabled

**Where:** `magic_llm/__init__.py:123-133` returns an empty `SubagentBundle()`
when `magic_llm.agent.config.ENABLE_SUBAGENTS` (default `False`,
`magic_llm/agent/config.py:17`) is off.

`magic-agents` has its own opt-in (`magic_agents.agt_flow.enable_task_subagents()`).
Enabling only that one loads no subagents and gives no warning, so a
configuration mistake looks like an empty `subagents/` directory.

**Effect on magic-agents:** documented in
[../wiki/TASK_SUBAGENTS.md](../wiki/TASK_SUBAGENTS.md): both flags must be
enabled.

**Suggested fix:** log a warning, or raise when a manifest directory is passed
but the feature flag is off.
