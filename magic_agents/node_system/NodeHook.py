"""
NodeHook - Python function template node for hooks.

NodeHook is a dedicated node type for executing Python hook function templates
with timeout enforcement and error isolation (Phase 1 safety).

Receives HookContext at runtime via input handle and can emit via
emit.user/debug/feedback through dedicated output handles.

Phase 6.2: Node class implementation.
Phase 6.3: Hook function template compilation (Phase 1 safety: exec with
constrained namespace, no sandboxing yet).
"""
import asyncio
import inspect
import logging
import ast
from datetime import datetime, UTC
from typing import Optional, AsyncGenerator, Dict, Any, Callable

from magic_agents.hooks.emit_context import EmitInterface
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.models.factory.Nodes.HookNodeModel import (
    HookNodeModel,
    DEFAULT_INPUT_HOOK_CONTEXT,
    DEFAULT_OUTPUT_USER,
    DEFAULT_OUTPUT_DEBUG,
    DEFAULT_OUTPUT_FEEDBACK,
)
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeHook(Node):
    """Python function template for hooks.

    A dedicated node type that executes user-defined Python function templates
    at hook lifecycle points. The hook function receives a HookContext with
    emit helpers for producing outputs.

    Phase 1 safety:
      - Timeout enforced via asyncio.wait_for (global default 30s)
      - Error isolation: exceptions logged, execution continues
      - Constrained exec namespace (emit, logger, datetime, UTC)
    
    Phase 2/3 (future):
      - Restricted globals
      - Full sandboxing (subprocess isolation or RestrictedPython)

    Attributes:
        DEFAULT_TIMEOUT_SECONDS: Global default timeout (30s).
        DEFAULT_INPUT_HOOK_CONTEXT: Default input handle name.
        DEFAULT_OUTPUT_USER: Default output handle for emit.user.
        DEFAULT_OUTPUT_DEBUG: Default output handle for emit.debug.
        DEFAULT_OUTPUT_FEEDBACK: Default output handle for emit.feedback.
    """

    DEFAULT_TIMEOUT_SECONDS = 30

    # Default handle names
    DEFAULT_INPUT_HOOK_CONTEXT = DEFAULT_INPUT_HOOK_CONTEXT
    DEFAULT_OUTPUT_USER = DEFAULT_OUTPUT_USER
    DEFAULT_OUTPUT_DEBUG = DEFAULT_OUTPUT_DEBUG
    DEFAULT_OUTPUT_FEEDBACK = DEFAULT_OUTPUT_FEEDBACK

    def __init__(self, data: HookNodeModel, **kwargs):
        """Initialize NodeHook from HookNodeModel.

        Args:
            data: HookNodeModel with function_template, timeout_override, hook_type.
            **kwargs: Additional kwargs passed to Node base class.
                'handles' dict is extracted from kwargs for handle resolution.
        """
        # Extract handles from kwargs (passed as a separate arg by node factory)
        handles = kwargs.pop('handles', {}) or {}
        node_id = kwargs.pop('node_id', None)
        debug = kwargs.pop('debug', False)

        super().__init__(node_id=node_id, debug=debug, **kwargs)
        self._function_template = data.function_template
        self.hook_mode = data.hook_mode
        self._messaging_by_target = {key: value.model_copy(deep=True)
                                    for key, value in (data.messaging_by_target or {}).items()}
        self._timeout_seconds = (
            data.timeout_override if data.timeout_override is not None
            else self.DEFAULT_TIMEOUT_SECONDS
        )
        self.lifecycle_event = data.lifecycle_event
        self.target_node_id = data.target_node_id
        self.target_node_ids = data.target_ids
        self.failure_policy = data.failure_policy
        self.OUTPUT_HANDLE_CALL = handles.get('child_call', 'handle-child-call')
        self._hook_type = data.hook_type  # 'pre', 'post', 'error', 'custom'

        # Resolve handle names from JSON override (handles dict)
        self.INPUT_HANDLE_HOOK_CONTEXT = handles.get(
            'hook_context', self.DEFAULT_INPUT_HOOK_CONTEXT
        )
        self.OUTPUT_HANDLE_USER = handles.get(
            'user_output', self.DEFAULT_OUTPUT_USER
        )
        self.OUTPUT_HANDLE_DEBUG = handles.get(
            'debug_output', self.DEFAULT_OUTPUT_DEBUG
        )
        self.OUTPUT_HANDLE_FEEDBACK = handles.get(
            'feedback_output', self.DEFAULT_OUTPUT_FEEDBACK
        )

    def bind_target(self, target):
        """Install this built-in Hook before the target joins its actor scope."""
        from magic_agents.hooks.messages_hook import MessagesHookBindingError
        from magic_agents.node_system.NodeLLM import NodeLLM
        if (self.hook_mode != 'messages' or not isinstance(target, NodeLLM)
                or target.node_id not in self.target_node_ids):
            raise MessagesHookBindingError('Messages Hook target must be an LLM in the same graph scope')
        prior = getattr(target, '_messages_hook_id', None)
        if prior is not None and prior != self.node_id or prior is None and target.messaging is not None:
            raise MessagesHookBindingError('An LLM accepts one Messages Hook and cannot also declare inline messaging')
        target.messaging = self._messaging_by_target[target.node_id].model_copy(deep=True)
        target._messages_hook_id = self.node_id

    async def process(
        self, chat_log
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute the hook function template with safety measures.

        Phase 1 safety: timeout + error isolation.
        Phase 2/3 (future): restricted globals + sandboxing.

        Args:
            chat_log: The agent run log (not used directly but required by Node interface).

        Yields:
            Dict with type/content for emit outputs or debug error events.
        """
        if self.hook_mode == 'messages':
            raise RuntimeError('Messages Hooks are bound by the Hook controller, not ordinary node dispatch')
        # Get HookContext from input handle
        hook_context = self.get_input(self.INPUT_HANDLE_HOOK_CONTEXT)

        if not hook_context:
            logger.warning(
                "NodeHook:%s received no hook context on handle '%s'",
                self.node_id,
                self.INPUT_HANDLE_HOOK_CONTEXT,
            )
            return

        # Ensure hook_context is a proper HookContext
        if not isinstance(hook_context, HookContext):
            logger.warning(
                "NodeHook:%s input is not a HookContext (got %s) — skipping",
                self.node_id,
                type(hook_context).__name__,
            )
            return

        # Create emit interface and inject into context
        emit = EmitInterface(node=self, node_id=self.node_id)
        hook_context.emit = emit

        # Compile the function template
        hook_function = self._compile_hook_function(self._function_template)
        if hook_function is None:
            logger.error(
                "NodeHook:%s failed to compile function template — skipping",
                self.node_id,
            )
            return

        # Templates are compiled per invocation, so their namespace is isolated.
        # Support both documented global emit.user(...) and context.emit.user(...).
        hook_function.__globals__['emit'] = emit

        # An edge override belongs to this invocation, not the shared node's
        # configuration. Zero is an explicit immediate timeout, not a fallback.
        timeout_seconds = hook_context.metadata.get('timeout_override')
        if timeout_seconds is None:
            timeout_seconds = self._timeout_seconds

        # Execute with timeout (Phase 1 safety)
        try:
            # The compiled hook function may be async or sync
            result = await asyncio.wait_for(
                self._execute_function(hook_function, hook_context, chat_log),
                timeout=timeout_seconds,
            )

            # If the function returned a result dict, yield it
            if result is not None:
                yield result

        except asyncio.TimeoutError:
            logger.warning(
                "NodeHook:%s timed out after %ss (template truncated: %s...)",
                self.node_id,
                timeout_seconds,
                self._function_template[:80],
            )
            yield self.yield_debug_error(
                error_type="HookTimeout",
                error_message=(
                    f"Hook exceeded {timeout_seconds}s timeout"
                ),
                context={
                    "function_template": self._function_template[:200],
                    "timeout": timeout_seconds,
                },
            )

        except Exception as e:
            logger.warning(
                "NodeHook:%s failed: %s (template truncated: %s...)",
                self.node_id,
                e,
                self._function_template[:80],
            )
            yield self.yield_debug_error(
                error_type="HookError",
                error_message=str(e),
                context={
                    "function_template": self._function_template[:200],
                },
            )

    async def invoke_control(self, context, chat_log):
        """Run this real Hook template without shared graph-node input state."""
        if self.hook_mode == 'messages':
            raise RuntimeError('Messages mode does not execute a Python lifecycle callback')
        function = self._compile_hook_function(self._function_template)
        if function is None:
            raise ValueError("Hook function template could not be compiled")
        if getattr(chat_log, "coordination", None) is not None and not inspect.iscoroutinefunction(function):
            from magic_agents.coordination.service import CoordinationError
            raise CoordinationError("unsupported_coordination_capability",
                                    "Coordinated lifecycle callbacks require an async function; synchronous threads cannot be fenced after cancellation")
        function.__globals__['emit'] = context.emit
        return await self._execute_function(function, context, chat_log)

    async def _execute_function(
        self,
        func: Callable,
        hook_context: HookContext,
        chat_log,
    ) -> Optional[Dict[str, Any]]:
        """Execute the compiled hook function (handles sync/async).

        Args:
            func: The compiled hook function.
            hook_context: HookContext with emit helpers injected.
            chat_log: Agent run log.

        Returns:
            Optional result dict from the function, or None.
        """
        if inspect.iscoroutinefunction(func):
            return await func(hook_context, chat_log)
        else:
            return await asyncio.to_thread(func, hook_context, chat_log)

    def _compile_hook_function(self, template: str) -> Optional[Callable]:
        """Compile Python function template for execution.

        Phase 1 safety: simple exec with constrained namespace.
        Phase 2 (future): restricted globals.
        Phase 3 (future): full sandboxing.

        The constrained namespace includes:
          - emit: Injected at runtime with EmitInterface
          - logger: Standard Python logger
          - datetime, UTC: For timestamping

        Args:
            template: Python code string with function definition.

        Returns:
            The compiled callable, or None if compilation failed.
        """
        if not template or not template.strip():
            logger.warning(
                "NodeHook:%s empty function template", self.node_id
            )
            return None

        # Create execution namespace with Phase 1 safe globals
        namespace = {
            'emit': None,      # Injected at runtime via hook_context
            'logger': logger,
            'datetime': datetime,
            'UTC': UTC,
        }

        try:
            exec(template, namespace)
        except Exception as e:
            logger.error(
                "NodeHook:%s template compilation failed: %s",
                self.node_id,
                e,
            )
            return None

        # Extract the function from the namespace
        func_name = self._extract_function_name(template)
        func = namespace.get(func_name)

        if func is None or not callable(func):
            logger.error(
                "NodeHook:%s could not find callable '%s' in compiled template",
                self.node_id,
                func_name or '(unknown)',
            )
            return None

        return func

    def _extract_function_name(self, template: str) -> Optional[str]:
        """Extract function name from template for invocation.

        Parses the template to find the first 'async def' or 'def' declaration
        and returns the function name.

        Args:
            template: Python code string.

        Returns:
            Function name string, or None if not found.
        """
        try:
            statements = ast.parse(template).body
        except SyntaxError:
            return None
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return statement.name

        # Keep the existing lambda fallback, limited to real top-level syntax.
        for statement in statements:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Lambda):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        return target.id
        return None

    def _capture_internal_state(self):
        """Capture NodeHook-specific internal state for debugging."""
        state = super()._capture_internal_state()
        state['hook_type'] = self._hook_type
        state['timeout'] = self._timeout_seconds
        state['template_length'] = len(self._function_template)
        state['template_preview'] = self._function_template[:100]
        return state
