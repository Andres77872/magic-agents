from typing import Optional

from pydantic import Field

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class PythonExecNodeModel(BaseNodeModel):
    """Pydantic model for Python execution node configuration."""

    safety_mode: str = Field(default="subprocess", description="Safety mode for code execution: subprocess, restricted_builtins, or in_process")
    timeout: float = Field(default=30.0, description="Maximum execution time in seconds (applies to subprocess mode)")
    max_output_chars: int = Field(default=8000, description="Maximum output length before truncation")
