"""Subagent error taxonomy — wrapper-specific errors.

NOTE: Runtime execution errors (depth limit exceeded, timeout during execution)
are now handled by magic-llm's TaskExecutor and represented via TaskError:
    from magic_llm.agent import TaskError
    TaskError.DEPTH_LIMIT  # Depth exceeded
    TaskError.TIMEOUT      # Execution timeout

This file contains wrapper-specific errors for:
- Registration/validation issues (before execution)
- Unknown subagent lookup errors
- Backward compatibility with existing error handling code
"""
from typing import Optional


class DuplicateSubagentError(Exception):
    """Duplicate registration error.
    
    Raised when attempting to register a subagent with an ID that
    already exists in the registry. This is a hard error per spec.
    """
    
    def __init__(
        self,
        agent_id: str,
        existing_source: Optional[str] = None,
        new_source: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.existing_source = existing_source
        self.new_source = new_source
        message = f"Duplicate registration: agent_id '{agent_id}' already registered"
        if existing_source:
            message += f" (existing: {existing_source})"
        if new_source:
            message += f" (new: {new_source})"
        message += ". Duplicate registration is not allowed."
        super().__init__(message)


class SubagentValidationError(Exception):
    """Validation error for subagent input or binding.
    
    Raised when:
    - Input validation fails against input_schema
    - Callable signature mismatches manifest schema
    - Required fields are missing
    """
    
    def __init__(
        self,
        agent_id: str,
        message: str,
        validation_type: str = "input"
    ):
        self.agent_id = agent_id
        self.message = message
        self.validation_type = validation_type
        full_message = f"Validation error for subagent '{agent_id}' ({validation_type}): {message}"
        super().__init__(full_message)


class DepthLimitError(Exception):
    """Depth limit exceeded error — wrapper-specific (pre-execution validation).
    
    NOTE: Runtime depth tracking is now handled by magic-llm's TaskExecutor.
    During execution, depth exceeded returns TaskError.DEPTH_LIMIT:
        TaskResult(status="cancelled", error=TaskError(error_type=TaskError.DEPTH_LIMIT, ...))
    
    This exception is kept for backward compatibility with wrapper-level
    error handling code (e.g., pre-execution validation checks).
    
    Non-retryable (deterministic failure).
    """
    
    def __init__(
        self,
        agent_id: str,
        current_depth: int,
        max_depth: int
    ):
        self.agent_id = agent_id
        self.current_depth = current_depth
        self.max_depth = max_depth
        message = (
            f"Depth limit exceeded for subagent '{agent_id}': "
            f"current_depth={current_depth}, max_depth={max_depth}. "
            f"Task cancelled."
        )
        super().__init__(message)


class UnknownSubagentError(Exception):
    """Unknown subagent ID error.
    
    Raised when attempting to resolve or invoke a subagent
    that is not registered. Includes registered types for debugging.
    """
    
    def __init__(
        self,
        agent_id: str,
        registered_ids: list[str] = []
    ):
        self.agent_id = agent_id
        self.registered_ids = registered_ids
        message = f"Unknown subagent_id '{agent_id}'."
        if registered_ids:
            message += f" Registered: {', '.join(registered_ids)}"
        else:
            message += " No subagents registered."
        super().__init__(message)


class SubagentTimeoutError(Exception):
    """Timeout error for subagent execution — wrapper-specific.
    
    NOTE: Runtime timeout enforcement is now handled by magic-llm's TaskExecutor.
    During execution, timeout returns TaskError.TIMEOUT:
        TaskResult(status="timeout", error=TaskError(error_type=TaskError.TIMEOUT, ...))
    
    This exception is kept for backward compatibility with wrapper-level
    error handling code (e.g., pre-execution timeout configuration).
    
    Retryable (transient condition).
    """
    
    def __init__(
        self,
        agent_id: str,
        timeout_seconds: int
    ):
        self.agent_id = agent_id
        self.timeout_seconds = timeout_seconds
        message = f"Subagent '{agent_id}' timed out after {timeout_seconds}s."
        super().__init__(message)