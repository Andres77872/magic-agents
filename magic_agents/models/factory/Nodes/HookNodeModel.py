"""
HookNodeModel - Pydantic model for NodeHook function template nodes.

NodeHook nodes execute Python function templates at defined lifecycle points
within graph execution. They receive HookContext via input handle and can
emit messages via emit.user/debug/feedback through dedicated output handles.

Phase 6.1: Pydantic model with function_template, timeout_override, hook_type.
"""
from typing import Optional, Literal

from pydantic import Field, model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel
from magic_agents.models.coordination import MessagingConfig


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
    hook_mode: Literal['python', 'messages'] = 'python'
    messaging_by_target: Optional[dict[str, MessagingConfig]] = None

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

    lifecycle_event: Optional[Literal["onStart", "onError", "onFinish", "onCancel", "onDeliver"]] = None
    target_node_id: Optional[str] = None
    target_node_ids: Optional[list[str]] = None
    failure_policy: Literal["preserve", "fail"] = "preserve"

    @model_validator(mode='after')
    def validate_mode(self):
        targets = self.target_ids
        if self.hook_mode == 'messages':
            if (self.messaging_by_target is None or set(self.messaging_by_target) != set(targets)
                    or self.function_template.strip() or self.lifecycle_event is not None):
                raise ValueError('Messages mode requires messaging and cannot also run Python or a lifecycle callback')
        elif self.messaging_by_target is not None:
            raise ValueError('Messaging configuration requires hook_mode messages')
        return self

    @property
    def target_ids(self):
        from magic_agents.hooks.target_binding import hook_target_ids
        data = {'target_node_id': self.target_node_id}
        if 'target_node_ids' in self.model_fields_set:
            data['target_node_ids'] = self.target_node_ids
        return hook_target_ids(data)
