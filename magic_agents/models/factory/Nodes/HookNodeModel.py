"""
HookNodeModel - Pydantic model for NodeHook function template nodes.

NodeHook nodes execute Python function templates at defined lifecycle points
within graph execution. They receive HookContext via input handle and can
emit messages via emit.user/debug/feedback through dedicated output handles.

Phase 6.1: Pydantic model with function_template, timeout_override, hook_type.
"""
from typing import Optional

from pydantic import Field

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


# Default handle names for NodeHook
DEFAULT_INPUT_HOOK_CONTEXT = 'handle-hook-context'
DEFAULT_OUTPUT_USER = 'handle-user-output'
DEFAULT_OUTPUT_DEBUG = 'handle-debug-output'
DEFAULT_OUTPUT_FEEDBACK = 'handle-feedback-output'


class HookNodeModel(BaseNodeModel):
    """Pydantic model for NodeHook configuration.

    NodeHook nodes execute Python function templates with timeout enforcement
    and error isolation (Phase 1 safety). The function template receives a
    HookContext with emit helpers at runtime.

    Attributes:
        function_template: Python code string defining the hook function.
            Must contain an 'async def' or 'def' declaration as the entry point.
        timeout_override: Optional per-hook timeout in seconds.
            Defaults to global default (30s) when None.
        hook_type: Lifecycle point for this hook: 'pre', 'post', 'error', 'custom'.
        handles: Dict mapping handle names to customized values.
            Supports: hook_context, user_output, debug_output, feedback_output.
    """
    function_template: str = Field(
        default="",
        description="Python function template for hook execution"
    )
    timeout_override: Optional[int] = Field(
        default=None,
        description="Per-hook timeout override in seconds (default: 30s global)"
    )
    hook_type: str = Field(
        default="custom",
        description="Hook lifecycle type: pre, post, error, custom"
    )
