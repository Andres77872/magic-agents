"""Task subagents package for magic-agents — THIN WRAPPER architecture.

magic-agents owns ONLY:
- Manifest discovery/loading (YAML discovery from graph directories)
- JSON workflow/agent parsing (converting manifests to runtime contracts)
- Registration/invocation wiring (calling MagicLLM.register_task())
- Graph-side callable implementations (where unavoidable)

magic-llm owns ALL runtime behavior:
- Depth tracking (ContextVar in TaskExecutor)
- Timeout enforcement (asyncio.wait_for in TaskExecutor)
- Concurrency control (asyncio.Semaphore in TaskExecutor)
- Result normalization (ResultNormalizer in magic-llm)
- Structured result contracts (TaskResult, TaskError in magic-llm)

Usage:
    # 1. Define manifest YAML (subagents/research.web.agent.yaml)
    apiVersion: magic-agents/v1
    kind: TaskSubagent
    id: research.web
    name: Web Research
    description: Search and summarize web content
    version: 1.0.0
    input_schema:
      type: object
      properties:
        query: {type: string}
      required: [query]
    timeout_seconds: 30
    max_concurrency: 5
    max_depth: 3
    
    # 2. Implement callable with decorator
    from magic_agents.subagents import task_subagent
    
    @task_subagent("research.web")
    async def research_web(query: str) -> str:
        # Agent loop execution
        result = await execute_agent_loop(...)
        return result
    
    # 3. Enable feature flag and registry initialization at startup
    from magic_agents.subagents import enable_task_subagents, init_registry
    from pathlib import Path
    
    enable_task_subagents()
    await init_registry(Path("subagents"))
    
    # Registration happens automatically in NodeLLM.process():
    # - _collect_tools() collects manifests + callables
    # - _register_task_subagents() calls MagicLLM.register_task()
    # - TaskExecutor wraps with safeguards and executes

DEPRECATED IMPORTS (use magic-llm instead):
    - TaskResult → magic_llm.agent.TaskResult
    - TaskError → magic_llm.agent.TaskError
"""

# === DEPRECATED: Import from magic-llm instead ===
# Backward-compatible import aliases with deprecation warning
import warnings

# TaskResult and TaskError now live in magic-llm
# These aliases exist for transition period (2 releases)
warnings.warn(
    "Importing TaskResult/TaskError from magic_agents.subagents is deprecated. "
    "Import from magic_llm.agent instead:\n"
    "    from magic_llm.agent import TaskResult, TaskError\n"
    "These aliases will be removed after 2 releases.",
    DeprecationWarning,
    stacklevel=2,
)

# Import from magic-llm for backward compatibility
try:
    from magic_llm.agent import TaskResult, TaskError
except ImportError:
    # magic-llm version without task-runtime exports
    # This should not happen with correct dependency version
    warnings.warn(
        "magic-llm does not export TaskResult/TaskError. "
        "Upgrade magic-llm to version with TaskExecutor support.",
        ImportError,
        stacklevel=2,
    )
    TaskResult = None  # type: ignore
    TaskError = None  # type: ignore

# === Local exports (thin wrapper layer) ===

# Models (wrapper layer only)
from .models import (
    SubagentManifest,
    BoundSubagent,
)

# Errors (wrapper-specific errors, not runtime errors)
from .errors import (
    DuplicateSubagentError,
    SubagentValidationError,
    UnknownSubagentError,
    # NOTE: DepthLimitError and SubagentTimeoutError are kept for
    # backward compatibility with existing error handling code.
    # These are wrapper-specific errors, not runtime execution errors.
    # Runtime depth/timeout errors are now TaskError.DEPTH_LIMIT/TIMEOUT.
    DepthLimitError,
    SubagentTimeoutError,
)

# Registry (wrapper layer)
from .registry import (
    RegistryBackend,
    StaticManifestRegistry,
    init_registry,
    get_registry,
    reset_registry,
)

# Decorator (registration mechanism)
from .decorator import (
    task_subagent,
    get_code_registry,
)

# Config (wrapper config)
from .config import (
    ENABLE_TASK_SUBAGENTS,
    MAX_SUMMARY_LENGTH,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DEPTH,
    is_task_subagents_enabled,
    enable_task_subagents,
    disable_task_subagents,
)

# Bundle (schema/callable container)
from .bundle import TaskToolBundle

# Binder (manifest + callable join)
from .binder import Binder

__all__ = [
    # === DEPRECATED: Import from magic-llm instead ===
    "TaskResult",  # DEPRECATED — use magic_llm.agent.TaskResult
    "TaskError",   # DEPRECATED — use magic_llm.agent.TaskError
    
    # === Models (wrapper layer) ===
    "SubagentManifest",
    "BoundSubagent",
    
    # === Errors (wrapper-specific) ===
    "DuplicateSubagentError",
    "SubagentValidationError",
    "DepthLimitError",       # Wrapper-specific, kept for compatibility
    "UnknownSubagentError",
    "SubagentTimeoutError",  # Wrapper-specific, kept for compatibility
    
    # === Registry (wrapper layer) ===
    "RegistryBackend",
    "StaticManifestRegistry",
    "init_registry",
    "get_registry",
    "reset_registry",
    
    # === Decorator (registration mechanism) ===
    "task_subagent",
    "get_code_registry",
    
    # === Config (wrapper config) ===
    "ENABLE_TASK_SUBAGENTS",
    "MAX_SUMMARY_LENGTH",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_DEPTH",
    "is_task_subagents_enabled",
    "enable_task_subagents",
    "disable_task_subagents",
    
    # === Bundle (schema/callable container) ===
    "TaskToolBundle",
    
    # === Binder (manifest + callable join) ===
    "Binder",
]