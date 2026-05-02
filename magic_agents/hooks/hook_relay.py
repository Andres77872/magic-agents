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
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

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
        """
        self._flow_hooks = flow_hooks
        self._registry = registry
        self._node_id = node_id
        self._graph_id = graph_id
        self._run_id = run_id
        self._emit = emit
        self._sequence: int = 0

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

        Args:
            inputs: Optional input data for the context.
            outputs: Optional output data for the context.
            error_message: Optional error message for error contexts.

        Returns:
            HookContext with node_id, graph_id, run_id populated.
        """
        return HookContext(
            execution_id=self._graph_id,
            sequence_number=self._next_sequence(),
            run_id=self._run_id,
            node_id=self._node_id,
            node_type="LLM",
            node_class="NodeLLM",
            inputs=inputs or {},
            outputs=outputs or {},
            error_message=error_message,
            emit=self._emit,
        )

    # === AgentHooks Protocol Implementation ===

    def on_iteration_start(self, iteration: int, state: AgentState) -> None:
        """Called before each LLM call in the agent loop.

        Translates to: FlowHooks.on_llm_start(context)

        Args:
            iteration: The current 0-indexed iteration number.
            state: The current agent state (read-only).
        """
        context = self._build_context(
            inputs={
                "iteration": iteration,
                "llm_call_count": getattr(state, 'iteration', iteration),
            }
        )
        self._safe_invoke_sync("on_llm_start", context)

    def on_llm_response(
        self, response: ModelChatResponse, state: AgentState
    ) -> None:
        """Called after each LLM response, before tool extraction.

        Translates to: FlowHooks.on_llm_end(context)

        Args:
            response: The raw LLM response.
            state: The current agent state (read-only).
        """
        context = self._build_context(
            outputs={
                "model": getattr(response, 'model', 'unknown'),
                "content": getattr(response, 'content', ''),
                "finish_reason": getattr(response, 'finish_reason', None),
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
            }
        )
        self._safe_invoke_sync("on_tool_start", context)

    def on_tool_complete(self, result: ToolResult, state: AgentState) -> None:
        """Called after each tool execution (success or error).

        Translates to: FlowHooks.on_tool_end(context)

        Args:
            result: The structured tool execution result.
            state: The current agent state (read-only).
        """
        context = self._build_context(
            outputs={
                "tool_name": getattr(result, 'name', 'unknown'),
                        "result": getattr(result, 'content', ''),
                "success": getattr(result, 'error', None) is None,
            },
            error_message=getattr(result, 'error', None),
        )
        self._safe_invoke_sync("on_tool_end", context)

    def on_loop_complete(
        self, final_response: ModelChatResponse, state: AgentState
    ) -> None:
        """Called after the loop exits (normal completion or budget-exceeded).

        Translates to: FlowHooks.on_llm_end(context) for final aggregation.

        Args:
            final_response: The final LLM response.
            state: The final agent state (read-only).
        """
        context = self._build_context(
            outputs={
                "model": getattr(final_response, 'model', 'unknown'),
                "content": getattr(final_response, 'content', ''),
                "iterations": getattr(state, 'iteration', 0),
                "loop_complete": True,
            }
        )
        self._safe_invoke_sync("on_llm_end", context)

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
                    asyncio.ensure_future(
                        self._registry.invoke(hook_name, context, **extra)
                    )
                else:
                    asyncio.ensure_future(
                        self._registry.invoke(hook_name, context)
                    )
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
                    asyncio.ensure_future(method(context, **extra))
                else:
                    asyncio.ensure_future(method(context))
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
