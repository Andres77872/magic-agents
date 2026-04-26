"""
AgentFlowModel - Model representing an agent flow graph.

NEW (Phase 3): Added ContractConfig and GraphContractReport for validation
mode integration and diagnostic reporting.
"""
from typing import Any, Optional, Dict, List, Literal
from datetime import datetime

from pydantic import BaseModel, ConfigDict, PrivateAttr, Field

from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.debug.config import DebugConfig


class ContractConfig(BaseModel):
    """
    Validation enforcement configuration (Phase 3).
    
    Controls how validation diagnostics are handled during build and execution.
    
    Attributes:
        mode: Validation mode - "off", "shadow", "warn", or "strict"
            - off: Disable all contract validation (rollback path)
            - shadow: Compute diagnostics, attach to report, no surfacing
            - warn: Surface diagnostics as warnings, execution proceeds (default)
            - strict: Block on errors (deferred to follow-up)
        strict_runtime: Runtime delivery validation (Phase 4, deferred)
    """
    mode: Literal["off", "shadow", "warn", "strict"] = Field(default="warn")
    strict_runtime: bool = Field(default=False)
    
    @property
    def is_off(self) -> bool:
        """Check if mode is off (no validation)."""
        return self.mode == "off"
    
    @property
    def is_shadow(self) -> bool:
        """Check if mode is shadow (compute but don't surface)."""
        return self.mode == "shadow"
    
    @property
    def is_warn(self) -> bool:
        """Check if mode is warn (surface but don't block)."""
        return self.mode == "warn"
    
    @property
    def is_strict(self) -> bool:
        """Check if mode is strict (block on errors)."""
        return self.mode == "strict"


class GraphContractReport(BaseModel):
    """
    Validation report attached to built graph model (Phase 3).
    
    Contains all validation diagnostics computed during build, enabling
    CI baseline comparison and diff detection.
    
    Attributes:
        diagnostics: List of validation diagnostics
        mode: Validation mode used
        timestamp: Report generation timestamp (excluded from normalized serialization)
        edge_count: Total edges validated
        node_count: Total nodes validated
    """
    model_config = ConfigDict(
        extra='ignore',
        # Don't serialize timestamps/UUIDs for CI comparison
    )
    
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    mode: str = Field(default="warn")
    edge_count: int = Field(default=0)
    node_count: int = Field(default=0)
    # Timestamp excluded from normalized serialization for CI stability
    _timestamp: Optional[datetime] = PrivateAttr(default_factory=datetime.now)
    
    def add_diagnostic(self, diagnostic: Dict[str, Any]) -> None:
        """Add a diagnostic to the report."""
        self.diagnostics.append(diagnostic)
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all diagnostics with severity 'error'."""
        return [d for d in self.diagnostics if d.get('severity') == 'error']
    
    def get_warnings(self) -> List[Dict[str, Any]]:
        """Get all diagnostics with severity 'warning'."""
        return [d for d in self.diagnostics if d.get('severity') == 'warning']
    
    def get_info(self) -> List[Dict[str, Any]]:
        """Get all diagnostics with severity 'info'."""
        return [d for d in self.diagnostics if d.get('severity') == 'info']
    
    def has_errors(self) -> bool:
        """Check if report contains any error-level diagnostics."""
        return len(self.get_errors()) > 0
    
    def to_normalized_dict(self) -> Dict[str, Any]:
        """
        Convert to normalized dict for CI baseline comparison.
        
        Excludes timestamps and sorts diagnostics by code for stability.
        """
        # Sort diagnostics by code for stable comparison
        sorted_diags = sorted(
            self.diagnostics,
            key=lambda d: (d.get('code', ''), d.get('edge_id', ''), d.get('type', ''))
        )
        return {
            "mode": self.mode,
            "edge_count": self.edge_count,
            "node_count": self.node_count,
            "diagnostic_count": len(sorted_diags),
            "error_count": len(self.get_errors()),
            "warning_count": len(self.get_warnings()),
            "diagnostics": sorted_diags,
        }


class AgentFlowModel(BaseModel):
    """
    Model representing an agent flow graph.
    
    NEW (Phase 3): Added ContractConfig and GraphContractReport for validation
    mode integration and diagnostic reporting.
    
    Attributes:
        type: Graph type identifier (e.g., 'chat', 'graph')
        debug: Whether debug mode is enabled
        debug_config: Optional debug configuration
        nodes: Dictionary of node_id -> node instance
        edges: List of edge connections
        contract_config: Validation configuration (Phase 3)
        _validation_errors: Internal list of validation errors (not persisted)
        _contract_report: Internal validation report (Phase 3)
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
    
    # NEW: Validation configuration (Phase 3)
    contract_config: ContractConfig = Field(default_factory=ContractConfig)
    
    # Private attribute for storing validation errors
    _validation_errors: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)
    
    # NEW: Private attribute for contract report (Phase 3)
    _contract_report: Optional[GraphContractReport] = PrivateAttr(default=None)
    
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
    
    @property
    def contract_report(self) -> Optional[GraphContractReport]:
        """
        Get the contract validation report.
        
        Returns:
            GraphContractReport if validation was run, None otherwise
        """
        return self._contract_report
    
    def get_contract_errors(self) -> List[Dict[str, Any]]:
        """Get all error-level diagnostics from contract report."""
        if self._contract_report:
            return self._contract_report.get_errors()
        return []
    
    def get_contract_warnings(self) -> List[Dict[str, Any]]:
        """Get all warning-level diagnostics from contract report."""
        if self._contract_report:
            return self._contract_report.get_warnings()
        return []
