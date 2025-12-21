from typing import Any, Optional, Dict, List

from pydantic import BaseModel, ConfigDict, PrivateAttr

from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.debug.config import DebugConfig


class AgentFlowModel(BaseModel):
    """
    Model representing an agent flow graph.
    
    Attributes:
        type: Graph type identifier (e.g., 'chat', 'graph')
        debug: Whether debug mode is enabled
        debug_config: Optional debug configuration
        nodes: Dictionary of node_id -> node instance
        edges: List of edge connections
        _validation_errors: Internal list of validation errors (not persisted)
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='ignore'
    )
    
    type: str = "chat"
    debug: bool = False
    debug_config: Optional[Dict[str, Any]] = None
    nodes: dict[str, Any]
    edges: list[EdgeNodeModel]
    
    # Private attribute for storing validation errors
    _validation_errors: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)
    
    @property
    def resolved_debug_config(self) -> Optional[DebugConfig]:
        """
        Get the resolved DebugConfig from the debug_config dict.
        
        Returns:
            DebugConfig instance if debug is enabled, None otherwise
        """
        if not self.debug:
            return None
        
        if self.debug_config:
            return DebugConfig.from_dict(self.debug_config.copy())
        
        return DebugConfig()
