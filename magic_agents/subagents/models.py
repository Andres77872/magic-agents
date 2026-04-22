"""Subagent models: Manifest and BoundSubagent protocol.

YAML manifest is the machine source of truth for identity, schema, and policies.

NOTE: TaskResult and TaskError moved to magic_llm.agent.types.
Import from magic_llm instead:
    from magic_llm.agent import TaskResult, TaskError
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol
from pydantic import BaseModel, Field, field_validator


class SubagentManifest(BaseModel):
    """YAML manifest for task-backed subagent registration.
    
    This is the machine source of truth for identity, schema, and policies.
    Markdown files are supportive material, not runtime source.
    
    Loaded from YAML files matching pattern: *.agent.yaml
    """
    
    # Identity
    apiVersion: Literal["magic-agents/v1"] = "magic-agents/v1"
    kind: Literal["TaskSubagent"] = "TaskSubagent"
    id: str = Field(..., pattern=r'^[a-z0-9._-]+$')  # Stable registry ID
    name: str  # Human-readable name
    description: str  # When-to-use summary for routing/delegation
    version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')  # Semver
    
    # Schema
    input_schema: dict[str, Any]  # JSON Schema for input validation
    output_schema: Optional[dict[str, Any]] = None  # Optional output schema
    
    # Execution Policy
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_concurrency: int = Field(default=5, ge=1, le=20)
    max_depth: int = Field(default=3, ge=1, le=10)
    
    # Optional Overrides
    model_override: Optional[str] = None  # e.g., "gpt-4.1-mini"
    instruction_file: Optional[Path] = None  # Markdown reference (supportive)
    
    # State
    enabled: bool = True
    
    # Source tracking (set by loader, not in YAML)
    source_file: Optional[Path] = None
    
    @property
    def tool_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible tool schema.
        
        Tool name = manifest id (stable registry key).
        """
        return {
            "type": "function",
            "function": {
                "name": self.id,
                "description": self.description,
                "parameters": self.input_schema
            }
        }
    
    @field_validator('instruction_file', mode='before')
    @classmethod
    def validate_instruction_file(cls, v: Any) -> Optional[Path]:
        """Convert string to Path if provided."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v


class BoundSubagent(Protocol):
    """Protocol for bound subagent exposed as tool.
    
    Follows FetchToolCallable pattern: tool_schema + tool_callable.
    TaskToolCallable implements this protocol.
    """
    
    @property
    def tool_schema(self) -> dict[str, Any]:
        """OpenAI-compatible tool schema."""
        ...
    
    @property
    def tool_callable(self) -> Callable:
        """Async callable for execution."""
        ...