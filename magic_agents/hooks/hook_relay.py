"""
HookRelay - Adapter bridging magic-agents FlowHooks to magic-llm AgentHooks.

HookRelay implements magic-llm's AgentHooks Protocol and translates events
from magic-llm's agent loop (on_iteration_start, on_tool_start, etc.) to
magic-agents FlowHooks invocations (on_llm_start, on_tool_start, etc.).

This enables graph-level visibility of tool execution inside magic-llm
without modifying magic-llm's Protocol contract.

Contracts:
- AgentHooks is sync → HookRelay methods are sync (implements the Protocol)
- FlowHooks is async → HookRelay._safe_invoke_sync handles async/sync mismatch
- NodeID, GraphID, RunID are injected into every translated HookContext
- Errors are isolated (logged, not propagated) per spec requirement
"""
import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from magic_llm.agent.hooks import AgentHooks
from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.model import ModelChatResponse

from magic_agents.hooks.flow_hooks import FlowHooks, HookContext

if TYPE_CHECKING:
    from magic_agents.hooks.emit_context import EmitInterface
    from magic_agents.hooks.hook_registry import HookRegistry

logger = logging.getLogger(__name__)


class HookRelay(AgentHooks):
    """Adapter bridging magic-agents FlowHooks to magic-llm AgentHooks.

    Implements magic-llm's AgentHooks Protocol, translating each event into
    a FlowHooks invocation with graph-level context injection.

    Created per NodeLLM execution with node_id, graph_id, and run_id for
    traceability. Passed as hooks= parameter to run_agent_async() and
    run_agent_stream_async().

    Supports two invocation modes:
    1. Via FlowHooks instance: calls method directly on a single FlowHooks.
    2. Via HookRegistry: delegates to registry.invoke() for multi-hook dispatch.

    Attributes:
        _flow_hooks: Optional FlowHooks implementation (single hook).
        _registry: Optional HookRegistry (multi-hook dispatch, preferred).
        _node_id: Source NodeLLM node ID (injected into all contexts).
        _graph_id: Current execution graph ID (injected into all contexts).
        _run_id: Current execution run ID (injected into all contexts).
        _emit: Optional EmitInterface for hook function outputs.
        _sequence: Auto-incrementing sequence number for HookContext.
    """

    def __init__(
        self,
        flow_hooks: Optional[FlowHooks] = None,
        node_id: str = "",
        graph_id: str = "",
        run_id: str = "",
        emit: Optional[Any] = None,
        registry: Optional['HookRegistry'] = None,
        timeout: float = 5.0,
        llm_config: Optional[Dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
        nested_depth: int = 0,
        nested_request_id: Optional[str] = None,
    ):
        """Initialize HookRelay with flow hooks and execution context.

        Args:
            flow_hooks: Optional FlowHooks implementation to relay events to.
                Used when a single FlowHooks instance is available.
            node_id: Source NodeLLM node ID for context injection.
            graph_id: Current execution graph ID for traceability.
            run_id: Current execution run ID for traceability.
            emit: Optional EmitInterface for hook function outputs.
            registry: Optional HookRegistry for multi-hook dispatch.
                Preferred over flow_hooks when both are provided.
            timeout: Default timeout in seconds for flush_pending_hooks().
            llm_config: Optional dict with LLM configuration data (model,
                provider, streaming, tools, tool_choice, deduplicate, etc.)
                populated best-effort from NodeLLM context.
            parent_run_id: Optional run_id of the parent loop for nested
                LLM hook correlation.
            nested_depth: Nesting depth for nested LLM hook events.
                0 = root loop, 1 = first child, etc.
            nested_request_id: Unique UUID hex for nested invocation
                correlation. Auto-generated if not provided.
        """
        self._flow_hooks = flow_hooks
        self._registry = registry
        self._node_id = node_id
        self._graph_id = graph_id
        self._run_id = run_id
        self._emit = emit
        self._sequence: int = 0

        self._pending_futures: Set[asyncio.Task] = set()
        self._flush_timeout: float = timeout

        self._current_provider_request_id: Optional[str] = None
        self._llm_config: Dict[str, Any] = llm_config or {}
        self._parent_run_id: Optional[str] = parent_run_id
        self._nested_depth: int = nested_depth
        self._nested_request_id: str = nested_request_id or uuid.uuid4().hex

        self._collected_tool_calls: List[Dict[str, Any]] = []
        self._collected_tool_results: List[Dict[str, Any]] = []

    @staticmethod
    def _iteration_from_state(state: AgentState, fallback: int = 0) -> int:
        """Return canonical loop iteration metadata from AgentState.

        Newer magic-llm exposes the current 0-indexed loop counter as
        ``AgentState.step``. Older states used ``iteration``. Keep the fallback
        so existing consumers with legacy state objects remain compatible.
        """
        value = getattr(state, "step", None)
        if value is None:
            value = getattr(state, "iteration", fallback)
        return value

    def _next_sequence(self) -> int:
        """Get the next sequence number for HookContext."""
        self._sequence += 1
        return self._sequence

    def _build_context(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> HookContext:
        """Build a HookContext with execution identity injected.

        Uses HookContextFactory.build_llm_context() for core identity fields,
        then merges outputs and error_message for backward compatibility.

        Injects nested correlation metadata (nested_depth, parent_run_id,
        nested_request_id) into the context for nested LLM loop tracking.
        Uses the runtime DEPTH ContextVar from magic-llm (when available)
        for accurate nesting depth, falling back to construction-time
        _nested_depth.

        Args:
            inputs: Optional input data for the context.
            outputs: Optional output data for the context.
            error_message: Optional error message for error contexts.

        Returns:
            HookContext with node_id, graph_id, run_id populated,
            plus nested correlation metadata when applicable.
        """
        from magic_agents.hooks.context_factory import HookContextFactory
        ctx = HookContextFactory.build_llm_context(
            execution_id=self._graph_id,
            run_id=self._run_id,
            node_id=self._node_id,
            node_type="LLM",
            node_class="NodeLLM",
            sequence_number=self._next_sequence(),
            **(inputs or {}),
        )
        if outputs:
            ctx.outputs = outputs
        if error_message:
            ctx.error_message = error_message
        ctx.emit = self._emit

        # Inject nested correlation metadata for ALL events.
        # Uses runtime DEPTH ContextVar when available (more accurate
        # than construction-time _nested_depth for shared HookRelay
        # across parent/child loops).
        try:
            from magic_llm.agent import DEPTH as _runtime_depth
            ctx.metadata["nested_depth"] = _runtime_depth.get()
        except (ImportError, AttributeError):
            ctx.metadata["nested_depth"] = self._nested_depth

        ctx.metadata["nested_request_id"] = self._nested_request_id
        if self._parent_run_id is not None:
            ctx.metadata["parent_run_id"] = self._parent_run_id
            ctx.parent_run_id = self._parent_run_id

        return ctx

    # === AgentHooks Protocol Implementation ===

    def on_iteration_start(self, iteration: int, state: AgentState) -> None:
        """Called before each LLM call in the agent loop.

        Translates to: FlowHooks.on_llm_start(context, llm_config=...)

        Populates context.inputs with best-effort LLM config data (model,
        provider, streaming, tools, tool_choice, deduplicate) from the
        HookRelay's _llm_config dict. Delivers the full llm_config dict
        as an extra kwarg to on_llm_start.

        Note: Nested correlation metadata (nested_depth, parent_run_id,
        nested_request_id) is injected at _build_context() level for
        ALL events, not here.

        Args:
            iteration: The current 0-indexed iteration number.
            state: The current agent state (read-only).
        """
        state_iteration = self._iteration_from_state(state, iteration)
        context = self._build_context(
            inputs={
                "iteration": iteration,
                "llm_call_count": state_iteration,
                "model": self._llm_config.get("model"),
                "provider": self._llm_config.get("provider"),
                "streaming": self._llm_config.get("streaming"),
                "tools": self._llm_config.get("tools"),
                "tool_choice": self._llm_config.get("tool_choice"),
                "deduplicate": self._llm_config.get("deduplicate"),
            }
        )

        # Build llm_config dict for extra kwarg (None when empty to distinguish
        # "no data" from "all fields are None")
        extra = None
        if self._llm_config:
            extra = {"llm_config": dict(self._llm_config)}
        self._safe_invoke_sync("on_llm_start", context, extra=extra)

    def on_llm_response(
        self, response: ModelChatResponse, state: AgentState
    ) -> None:
        """Called after each LLM response, before tool extraction.

        Translates to: FlowHooks.on_llm_end(context)

        Injects provider_request_id from response.id (cached for tool event
        correlation), token usage from response.usage, and iteration metadata
        from state.

        Args:
            response: The raw LLM response.
            state: The current agent state (read-only).
        """
        self._current_provider_request_id = getattr(response, 'id', None)

        usage = getattr(response, 'usage', None) or {}
        prompt_tokens = usage.get('prompt_tokens') if isinstance(usage, dict) else getattr(usage, 'prompt_tokens', None)
        completion_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)
        total_tokens = usage.get('total_tokens') if isinstance(usage, dict) else getattr(usage, 'total_tokens', None)

        context = self._build_context(
            outputs={
                "model": getattr(response, 'model', 'unknown'),
                "content": getattr(response, 'content', ''),
                "finish_reason": getattr(response, 'finish_reason', None),
                "provider_request_id": self._current_provider_request_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "iteration": self._iteration_from_state(state),
            }
        )
        self._safe_invoke_sync("on_llm_end", context)

    def on_tool_start(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Dict[str, Any],
        state: AgentState,
    ) -> None:
        """Called before each tool execution.

        Translates to: FlowHooks.on_tool_start(context)

        Injects provider_request_id from the most recent LLM response
        (cached in _current_provider_request_id) and iteration metadata
        into both the HookContext and the collected tool call entry.

        Args:
            tool_name: The name of the tool about to be executed.
            tool_call_id: The provider-specific tool call identifier.
            arguments: The parsed tool arguments.
            state: The current agent state (read-only).
        """
        context = self._build_context(
            inputs={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "provider_request_id": self._current_provider_request_id,
                "iteration": self._iteration_from_state(state),
            }
        )
        self._safe_invoke_sync("on_tool_start", context)

        self._collected_tool_calls.append({
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
            },
            "provider_request_id": self._current_provider_request_id,
            "iteration": self._iteration_from_state(state),
        })

    def on_tool_complete(self, result: ToolResult, state: AgentState) -> None:
        """Called after each tool execution (success or error).

        Translates to: FlowHooks.on_tool_end(context)

        Injects provider_request_id from the most recent LLM response
        (cached in _current_provider_request_id) and iteration metadata
        into both the HookContext and the collected tool result entry.

        Args:
            result: The structured tool execution result.
            state: The current agent state (read-only).
        """
        context = self._build_context(
            outputs={
                "tool_name": getattr(result, 'name', 'unknown'),
                        "result": getattr(result, 'content', ''),
                "success": getattr(result, 'error', None) is None,
                "execution_time_ms": getattr(result, 'duration_ms', None),
                "provider_request_id": self._current_provider_request_id,
                "iteration": self._iteration_from_state(state),
            },
            error_message=getattr(result, 'error', None),
        )
        self._safe_invoke_sync("on_tool_end", context)

        status = "error" if result.is_error else "completed"
        entry: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": result.tool_call_id or "",
            "content": result.content,
            "status": status,
            "execution_time_ms": result.duration_ms,
            "provider_request_id": self._current_provider_request_id,
            "iteration": self._iteration_from_state(state),
        }
        if result.is_error and result.error:
            entry["tool_error"] = result.error
        self._collected_tool_results.append(entry)

    def on_loop_complete(
        self, final_response: ModelChatResponse, state: AgentState
    ) -> None:
        """Called after the loop exits (NORMAL exit only, NOT budget-exceeded).

        Translates to: FlowHooks.on_llm_loop_end(context) for aggregated
        loop completion. Fires ONCE per loop, after all per-iteration
        on_llm_end events.

        Carries accumulated content from ALL iterations, iteration metadata
        (0-indexed iteration, 1-indexed total_iterations), provider_request_id
        from final response, and token usage data.

        Clears _current_provider_request_id after firing to prevent stale
        values from leaking between loops.

        NOTE: This fires for normal completion only. Budget-exceeded exits
        are handled by on_budget_exceeded() and never reach this method.

        Args:
            final_response: The final LLM response.
            state: The final agent state (read-only).
        """
        iteration = self._iteration_from_state(state)
        usage = getattr(final_response, 'usage', None) or {}
        prompt_tokens = usage.get('prompt_tokens') if isinstance(usage, dict) else getattr(usage, 'prompt_tokens', None)
        completion_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else getattr(usage, 'completion_tokens', None)
        total_tokens = usage.get('total_tokens') if isinstance(usage, dict) else getattr(usage, 'total_tokens', None)

        context = self._build_context(
            outputs={
                "model": getattr(final_response, 'model', 'unknown'),
                "content": getattr(final_response, 'content', ''),
                "content_preview": (getattr(final_response, 'content', '') or '')[:200],
                "finish_reason": getattr(final_response, 'finish_reason', None),
                "iteration": iteration,
                "total_iterations": iteration + 1,
                "provider_request_id": getattr(final_response, 'id', None),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        )
        self._safe_invoke_sync("on_llm_loop_end", context)

        # Clear cached provider_request_id to prevent stale values between loops
        self._current_provider_request_id = None

    def on_budget_exceeded(self, budget_type: str, details: str) -> None:
        """Called when a budget constraint is violated.

        Translates to: on_node_error via FlowHooks.

        Note: Budget exceeded is treated as an error context since the
        loop terminated due to constraint violation.

        Args:
            budget_type: Which budget limit was exceeded.
            details: Human-readable details about the breach.
        """
        context = self._build_context(
            error_message=f"Budget exceeded ({budget_type}): {details}",
            inputs={"budget_type": budget_type, "details": details},
        )
        self._safe_invoke_sync("on_node_error", context, {"error": None})

    # === Async Bridge: Pending Futures ===

    async def flush_pending_hooks(self) -> None:
        """Wait for all pending async hook tasks to complete.

        Snapshots _pending_futures to prevent list mutation race during
        asyncio.wait(). Cancels any timed-out tasks and clears the set.
        Safe to call multiple times (clears after each call).

        Task exceptions are handled via return_exceptions=True and are NOT
        propagated to the caller — errors are isolated per spec requirement.
        """
        if not self._pending_futures:
            return

        # Snapshot to prevent list mutation race during asyncio.wait
        futures = list(self._pending_futures)
        done, pending = await asyncio.wait(
            futures,
            timeout=self._flush_timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        if pending:
            logger.warning(
                "%d pending hook(s) timed out after %.1fs. Cancelling.",
                len(pending), self._flush_timeout,
            )
            for task in pending:
                task.cancel()

        # Clear ALL tasks — completed tasks are done, pending are cancelled
        self._pending_futures.clear()

    # === Internal Helpers ===

    def _safe_invoke_sync(
        self,
        hook_name: str,
        context: HookContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Invoke a FlowHooks method safely from a sync context.

        Since AgentHooks is sync but FlowHooks is async, this method:
        1. Uses registry.invoke() when a HookRegistry is available.
        2. Falls back to getattr on a single FlowHooks instance.
        3. If async and we have a running event loop, schedules it.
        4. If async but no event loop, logs a warning.
        5. If sync, calls directly.
        6. Catches all exceptions and logs them.

        Args:
            hook_name: The FlowHooks method name (e.g., 'on_llm_start').
            context: The HookContext to pass to the hook.
            extra: Optional extra keyword arguments for the hook method.
        """
        try:
            # Prefer registry mode (multi-hook dispatch)
            if self._registry is not None:
                self._invoke_via_registry(hook_name, context, extra)
                return

            # Fallback to single FlowHooks instance
            if self._flow_hooks is None:
                return  # No hooks configured

            method = getattr(self._flow_hooks, hook_name, None)
            if method is None:
                return  # Hook method not implemented (partial implementation)

            # FlowHooks methods may be async or sync
            if asyncio.iscoroutinefunction(method):
                self._invoke_async_method(method, hook_name, context, extra)
            else:
                # Sync hook — call directly
                if extra is not None:
                    method(context, **extra)
                else:
                    method(context)

        except Exception as e:
            logger.warning(
                "HookRelay.%s raised exception: %s",
                hook_name,
                e,
                exc_info=True,
            )

    def _invoke_via_registry(
        self,
        hook_name: str,
        context: HookContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Invoke hook via HookRegistry (async, with error isolation).

        Uses asyncio.run_coroutine_threadsafe or ensure_future to bridge
        from sync AgentHooks context to async HookRegistry.invoke().

        Args:
            hook_name: The hook method name.
            context: The HookContext to pass.
            extra: Optional extra kwargs for the hook.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Schedule the async registry invoke in the event loop
                if extra is not None:
                    task = asyncio.create_task(
                        self._registry.invoke(hook_name, context, **extra)
                    )
                else:
                    task = asyncio.create_task(
                        self._registry.invoke(hook_name, context)
                    )
                self._pending_futures.add(task)
            else:
                logger.warning(
                    "HookRelay cannot invoke '%s' via registry: "
                    "event loop is not running.",
                    hook_name,
                )
        except RuntimeError:
            logger.warning(
                "HookRelay cannot invoke '%s' via registry: "
                "no event loop available (sync-only context).",
                hook_name,
            )

    def _invoke_async_method(
        self,
        method,
        hook_name: str,
        context: HookContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Invoke an async FlowHooks method from sync context.

        Schedules the async call in the running event loop.

        Args:
            method: The async method to invoke.
            hook_name: Method name for logging.
            context: The HookContext to pass.
            extra: Optional extra kwargs.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                if extra is not None:
                    task = asyncio.create_task(method(context, **extra))
                else:
                    task = asyncio.create_task(method(context))
                self._pending_futures.add(task)
            else:
                logger.warning(
                    "FlowHooks.%s is async but event loop is not running. "
                    "Hook will be skipped in sync context.",
                    hook_name,
                )
        except RuntimeError:
            logger.warning(
                "FlowHooks.%s is async but no event loop is available. "
                "Hook will be skipped in sync context.",
                hook_name,
            )

    def get_collected_tool_data_for_yield(self, clear: bool = True) -> List[Dict[str, Any]]:
        """Return tool call/result data as yieldable events.

        Each event dict has ``type`` ('tool_call' or 'tool_result') and ``data``
        containing the structured payload for insert_tool_messages().

        When clear=True (default), internal lists are emptied after read to
        prevent duplicate accumulation across iterations (F18 fix).

        Args:
            clear: If True, clear internal collections after reading.
                Use clear=False for inspection/debugging purposes.

        Returns:
            list of event dicts with type and data keys.
        """
        events: List[Dict[str, Any]] = []

        # Snapshot tool calls
        tool_calls_copy = list(self._collected_tool_calls)
        for tc in tool_calls_copy:
            events.append({
                "type": "tool_call",
                "data": {
                    "role": "assistant",
                    "tool_calls": [tc],
                },
            })

        # Snapshot tool results
        tool_results_copy = list(self._collected_tool_results)
        for tr in tool_results_copy:
            events.append({
                "type": "tool_result",
                "data": tr,
            })

        # Clear if requested (prevents double-accumulation, F18)
        if clear:
            self._collected_tool_calls.clear()
            self._collected_tool_results.clear()

        return events

    @property
    def collected_tool_calls(self) -> List[Dict[str, Any]]:
        """Raw collected tool call dicts (OpenAI format)."""
        return list(self._collected_tool_calls)
