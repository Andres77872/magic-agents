"""
Graph Validator - Build-time validation for agent flow graphs.

This module provides validation functions to ensure graph integrity,
particularly for conditional nodes and their edge connections.

Handle validation enforces the clean-break policy: legacy handles like
handle_generated_end are rejected at build time.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
    from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel

from magic_agents.util.handle_registry import (
    is_valid_source_handle,
    is_legacy_rejected_handle,
    get_node_output_handles_from_instance,
    LEGACY_REJECTED_HANDLES,
)

logger = logging.getLogger(__name__)


class ConditionalEdgeValidator:
    """
    Validates conditional node edges at build time.
    
    Ensures:
    1. Declared output handles have matching edges
    2. Default handle has an edge if specified
    3. Warns about undeclared outputs
    """
    
    @staticmethod
    def validate(
        nodes: Dict[str, Any],
        edges: List['EdgeNodeModel']
    ) -> List[Dict[str, Any]]:
        """
        Validate conditional nodes have proper edge connections.
        
        Args:
            nodes: Dictionary of node_id -> Node instance
            edges: List of EdgeNodeModel defining connections
        
        Returns:
            List of validation errors/warnings (empty if valid)
        """
        from magic_agents.node_system import NodeConditional
        
        errors = []
        
        for node_id, node in nodes.items():
            if not isinstance(node, NodeConditional):
                continue
            
            # Get declared output handles from node
            declared = getattr(node, 'output_handles', None)
            default_handle = getattr(node, 'default_handle', None)
            
            # Get actual outgoing edge handles for this conditional
            outgoing_edges = [e for e in edges if e.source == node_id]
            edge_handles = {e.sourceHandle for e in outgoing_edges}
            
            # Validation 1: Check declared outputs have edges
            if declared:
                missing = set(declared) - edge_handles
                if missing:
                    errors.append({
                        "type": "MissingConditionalEdge",
                        "severity": "error",
                        "node_id": node_id,
                        "error_message": (
                            f"Conditional '{node_id}' declares outputs {list(declared)} "
                            f"but missing edges for: {list(missing)}"
                        ),
                        "declared_handles": list(declared),
                        "actual_handles": list(edge_handles),
                        "missing_handles": list(missing),
                        "suggestion": (
                            f"Add edges with sourceHandle={list(missing)} from '{node_id}', "
                            "or remove unused handles from output_handles declaration."
                        )
                    })
            
            # Validation 2: Check default handle has edge (if specified)
            if default_handle and default_handle not in edge_handles:
                errors.append({
                    "type": "MissingDefaultEdge",
                    "severity": "error",
                    "node_id": node_id,
                    "error_message": (
                        f"Conditional '{node_id}' specifies default_handle='{default_handle}' "
                        "but no edge matches this handle"
                    ),
                    "default_handle": default_handle,
                    "actual_handles": list(edge_handles),
                    "suggestion": (
                        f"Add an edge with sourceHandle='{default_handle}' from '{node_id}', "
                        "or change default_handle to one of the existing handles."
                    )
                })
            
            # Validation 3: Warning when no declared outputs (can't fully validate)
            if not declared and len(edge_handles) > 0:
                errors.append({
                    "type": "UndeclaredOutputs",
                    "severity": "warning",
                    "node_id": node_id,
                    "error_message": (
                        f"Conditional '{node_id}' has edges with handles {list(edge_handles)} "
                        "but no output_handles declared. Runtime validation only."
                    ),
                    "actual_handles": list(edge_handles),
                    "suggestion": (
                        "Consider adding output_handles to the conditional data for "
                        "build-time validation. Example: \"output_handles\": " 
                        f"{list(edge_handles)}"
                    )
                })
            
            # Validation 4: Log fan-out information for debugging
            handle_targets: Dict[str, List[str]] = {}
            for edge in outgoing_edges:
                handle_targets.setdefault(edge.sourceHandle, []).append(edge.target)
            
            for handle, targets in handle_targets.items():
                if len(targets) > 1:
                    logger.debug(
                        "Conditional '%s' has fan-out on handle '%s' -> %d targets: %s",
                        node_id, handle, len(targets), targets
                    )
        
        return errors


def validate_graph_conditionals(graph: 'AgentFlowModel') -> List[Dict[str, Any]]:
    """
    Convenience function to validate all conditional nodes in a graph.
    
    Args:
        graph: The agent flow model to validate
        
    Returns:
        List of validation errors/warnings
    """
    return ConditionalEdgeValidator.validate(graph.nodes, graph.edges)


def validate_edge_connectivity(
    nodes: Dict[str, Any],
    edges: List['EdgeNodeModel']
) -> List[Dict[str, Any]]:
    """
    Validate basic edge connectivity in the graph.
    
    Checks:
    1. Source and target nodes exist
    2. No duplicate edges
    3. No self-loops
    
    Args:
        nodes: Dictionary of node_id -> Node instance
        edges: List of EdgeNodeModel
        
    Returns:
        List of validation errors
    """
    errors = []
    seen_edges = set()
    
    for edge in edges:
        # Check source exists
        if edge.source not in nodes:
            errors.append({
                "type": "InvalidEdgeSource",
                "severity": "error",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "error_message": f"Edge references non-existent source node: '{edge.source}'",
                "source": edge.source,
                "target": edge.target
            })
        
        # Check target exists
        if edge.target not in nodes:
            errors.append({
                "type": "InvalidEdgeTarget",
                "severity": "error",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "error_message": f"Edge references non-existent target node: '{edge.target}'",
                "source": edge.source,
                "target": edge.target
            })
        
        # Check for self-loops
        if edge.source == edge.target:
            errors.append({
                "type": "SelfLoopEdge",
                "severity": "warning",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "error_message": f"Edge creates a self-loop on node: '{edge.source}'",
                "node_id": edge.source
            })
        
        # Check for duplicates
        edge_key = (edge.source, edge.target, edge.sourceHandle, edge.targetHandle)
        if edge_key in seen_edges:
            errors.append({
                "type": "DuplicateEdge",
                "severity": "warning",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "error_message": (
                    f"Duplicate edge: {edge.source}.{edge.sourceHandle} -> "
                    f"{edge.target}.{edge.targetHandle}"
                ),
                "source": edge.source,
                "target": edge.target,
                "sourceHandle": edge.sourceHandle,
                "targetHandle": edge.targetHandle
            })
        seen_edges.add(edge_key)
    
    return errors


def validate_edge_handles(
    nodes: Dict[str, Any],
    edges: List['EdgeNodeModel']
) -> List[Dict[str, Any]]:
    """
    Validate edge sourceHandle values against node output handle contracts.
    
    Enforces clean-break policy: legacy handles (handle_generated_end) are rejected.
    
    Checks:
    1. sourceHandle must be valid for the source node's output contract
    2. Legacy rejected handles fail validation with clear error
    3. Dynamic handles (handle-tool-definition-*, handle_parser_input_*) pass
    4. Nodes with custom handles (data.handles) validated against instance
    
    Args:
        nodes: Dictionary of node_id -> Node instance
        edges: List of EdgeNodeModel
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for edge in edges:
        source_handle = edge.sourceHandle
        
        # Skip edges without sourceHandle (implicit default routing)
        if not source_handle:
            continue
        
        # Get source node
        source_node = nodes.get(edge.source)
        if not source_node:
            # Already handled by validate_edge_connectivity
            continue
        
        # Get source node type
        source_node_type = getattr(source_node, 'node_type', None)
        if not source_node_type:
            # Node might be stub or error node - skip handle validation
            continue
        
        # Check for legacy rejected handles first (explicit rejection)
        if is_legacy_rejected_handle(source_handle):
            errors.append({
                "type": "LegacyHandleRejected",
                "severity": "error",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "source": edge.source,
                "target": edge.target,
                "sourceHandle": source_handle,
                "targetHandle": edge.targetHandle,
                "error_message": (
                    f"Edge uses legacy handle '{source_handle}' which is no longer emitted "
                    f"by node '{edge.source}' (type: {source_node_type}). "
                    f"Use canonical handles instead."
                ),
                "legacy_handle": source_handle,
                "node_type": source_node_type,
                "suggestion": (
                    f"Reconnect edge using canonical handle for {source_node_type}. "
                    f"See docs/JSON_CONTRACT.md for valid handles."
                )
            })
            continue
        
        # Get output handles from the node instance
        instance_handles = get_node_output_handles_from_instance(source_node)
        
        # Validate the source handle
        is_valid, reason = is_valid_source_handle(
            source_handle,
            source_node_type,
            instance_handles
        )
        
        if not is_valid:
            errors.append({
                "type": "InvalidSourceHandle",
                "severity": "error",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "source": edge.source,
                "target": edge.target,
                "sourceHandle": source_handle,
                "targetHandle": edge.targetHandle,
                "error_message": (
                    f"Edge sourceHandle '{source_handle}' is not valid for "
                    f"node '{edge.source}' (type: {source_node_type}). {reason}"
                ),
                "node_type": source_node_type,
                "reason": reason,
                "suggestion": (
                    f"Check node's output handles or use canonical handles for {source_node_type}. "
                    f"See docs/JSON_CONTRACT.md for valid handles."
                )
            })
    
    return errors


def run_all_validations(graph: 'AgentFlowModel') -> List[Dict[str, Any]]:
    """
    Run all graph validations.
    
    Args:
        graph: The agent flow model to validate
        
    Returns:
        Combined list of all validation errors/warnings
    """
    errors = []
    
    # Basic edge connectivity
    errors.extend(validate_edge_connectivity(graph.nodes, graph.edges))
    
    # Edge handle validation (clean-break: reject legacy handles)
    errors.extend(validate_edge_handles(graph.nodes, graph.edges))
    
    # Conditional-specific validation
    errors.extend(validate_graph_conditionals(graph))
    
    return errors
