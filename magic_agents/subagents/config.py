"""Task subagents configuration.

Feature flags and settings for subagent functionality.
"""

# Feature flag: Enable task subagents injection
# When False, TaskToolBundle injection is skipped in NodeLLM._collect_tools()
# Default: False for backward compatibility
ENABLE_TASK_SUBAGENTS: bool = False

# Maximum summary length for TaskResult
# Prevents token blowup from long child outputs
MAX_SUMMARY_LENGTH: int = 5000

# Default values for SubagentManifest fields
DEFAULT_TIMEOUT_SECONDS: int = 30
DEFAULT_MAX_CONCURRENCY: int = 5
DEFAULT_MAX_DEPTH: int = 3


def is_task_subagents_enabled() -> bool:
    """Check if task subagents feature is enabled.
    
    Returns:
        True if ENABLE_TASK_SUBAGENTS is True and registry is initialized
    """
    if not ENABLE_TASK_SUBAGENTS:
        return False
    
    try:
        from .registry import get_registry
        registry = get_registry()
        return registry.is_initialized()
    except RuntimeError:
        # Registry not initialized
        return False


def enable_task_subagents() -> None:
    """Enable task subagents feature globally."""
    global ENABLE_TASK_SUBAGENTS
    ENABLE_TASK_SUBAGENTS = True


def disable_task_subagents() -> None:
    """Disable task subagents feature globally."""
    global ENABLE_TASK_SUBAGENTS
    ENABLE_TASK_SUBAGENTS = False