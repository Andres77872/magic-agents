from typing import Optional

from pydantic import Field

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class PythonExecNodeModel(BaseNodeModel):
    """Pydantic model for Python execution node configuration.

    Dual-mode: tool mode (code=None) yields a PythonExecutor callable as an LLM tool.
    Node mode (code=string) executes user-provided Python via the run(handler) contract.
    """

    safety_mode: str = Field(default="subprocess", description="Safety mode for code execution: subprocess, restricted_builtins, or in_process")
    timeout: float = Field(default=30.0, description="Maximum execution time in seconds (applies to subprocess mode)")
    max_output_chars: int = Field(default=8000, description="Maximum output length before truncation")

    # NEW: User's Python source containing run(handler) entrypoint for node mode
    code: Optional[str] = Field(
        default=None,
        description="Python code with run(handler) function for node mode execution. "
                    "When set, the node executes user-defined Python as a graph node. "
                    "When omitted (None), the node provides PythonExecutor as an LLM tool."
    )

    # NEW: Handle name overrides (standard pattern, see ChatNodeModel)
    handles: Optional[dict[str, str]] = Field(
        default=None,
        description="Handle name overrides for custom edge routing. "
                    "Supports keys: safety_mode, timeout, max_output_chars, output."
    )
