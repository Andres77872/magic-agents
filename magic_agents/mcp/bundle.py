"""MCP Tool Bundle dataclass.

Container for MCP tools to yield to NodeLLM.
"""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class MCPToolBundle:
    """Container for MCP tools to yield to NodeLLM.
    
    Implements extended ToolProvider protocol:
      - tool_schemas: list of OpenAI-compatible tool definitions
      - tool_functions: dict of async callable wrappers
    
    NodeLLM._collect_tools() flattens bundles into unified registry.
    """
    
    # Required: tool definitions
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    tool_functions: dict[str, Callable] = field(default_factory=dict)
    
    # Metadata for debugging/observability
    server_key: str = ""  # Identifier for source MCP server
    node_id: str = ""  # Graph node ID
    discovered_count: int = 0  # Number of tools discovered (before filtering)
    filtered_count: int = 0  # Number of tools after filtering
    prefix: str = ""  # Applied prefix
    
    def __post_init__(self):
        """Validate bundle consistency."""
        # Ensure schema count matches function count
        if len(self.tool_schemas) != len(self.tool_functions):
            # This is a warning, not an error - schemas may be provided without functions
            pass
        
        # Set filtered_count if not provided
        if self.filtered_count == 0:
            self.filtered_count = len(self.tool_schemas)