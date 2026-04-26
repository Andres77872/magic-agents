"""
Graph Validator - Build-time validation for agent flow graphs.

This module provides validation functions to ensure graph integrity,
particularly for conditional nodes and their edge connections.

Handle validation enforces the clean-break policy: legacy handles like
handle_generated_end are rejected at build time.

NEW (Phase 1): Added targetHandle validation via validate_edge_target_handles().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
    from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel

from magic_agents.util.handle_registry import (
    is_valid_source_handle,
    is_valid_target_handle,
    is_legacy_rejected_handle,
    get_node_output_handles_from_instance,
    get_node_input_handles_from_instance,
    get_port_cardinality,
    is_multi_compatible_port,
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


def validate_edge_target_handles(
    nodes: Dict[str, Any],
    edges: List['EdgeNodeModel'],
    mode: str = "warn"
) -> List[Dict[str, Any]]:
    """
    Validate edge targetHandle values against node input handle contracts.
    
    NEW function (Phase 1) - parallel to validate_edge_handles() for input side.
    
    Validation mode: shadow + warn by default (first release).
    - shadow/warn: unknown handles pass with warning (legacy synthesis)
    - strict: unknown handles rejected (deferred to follow-up)
    
    Checks:
    1. targetHandle must be valid for the target node's input contract
    2. Dynamic handles (handle-tool-definition-*, handle_parser_input_*) pass
    3. VOID nodes accept any handle (sink node)
    4. Nodes with custom handles validated against instance
    
    Args:
        nodes: Dictionary of node_id -> Node instance
        edges: List of EdgeNodeModel
        mode: Validation mode ("shadow", "warn", "strict")
        
    Returns:
        List of validation errors/warnings
    """
    errors = []
    
    for edge in edges:
        target_handle = edge.targetHandle
        
        # Skip edges without targetHandle (implicit routing or VOID)
        if not target_handle:
            continue
        
        # Get target node
        target_node = nodes.get(edge.target)
        if not target_node:
            # Already handled by validate_edge_connectivity
            continue
        
        # Get target node type
        target_node_type = getattr(target_node, 'node_type', None)
        if not target_node_type:
            # Node might be stub or error node - skip handle validation
            continue
        
        # Get input handles from the node instance (if available)
        instance_handles = get_node_input_handles_from_instance(target_node)
        
        # Validate the target handle
        is_valid, reason = is_valid_target_handle(
            target_handle,
            target_node_type,
            instance_handles if instance_handles else None,
            mode
        )
        
        if not is_valid:
            # In shadow/warn mode, this produces a warning (not blocking)
            # In strict mode, this would be an error (deferred)
            severity = "warning" if mode in ("shadow", "warn") else "error"
            errors.append({
                "type": "InvalidTargetHandle",
                "severity": severity,
                "code": "PORT_TARGET_HANDLE_UNKNOWN",
                "edge_id": getattr(edge, 'id', 'unknown'),
                "source": edge.source,
                "target": edge.target,
                "sourceHandle": edge.sourceHandle,
                "targetHandle": target_handle,
                "error_message": (
                    f"Edge targetHandle '{target_handle}' is not a recognized input "
                    f"for node '{edge.target}' (type: {target_node_type}). {reason}"
                ),
                "node_type": target_node_type,
                "reason": reason,
                "suggestion": (
                    f"Check node's input handles or use canonical handles for {target_node_type}. "
                    f"See docs/JSON_CONTRACT.md for valid handles."
                )
            })
        elif "Opaque/legacy" in reason:
            # Valid but opaque/legacy - produce advisory warning in warn mode
            if mode == "warn":
                errors.append({
                    "type": "OpaqueTargetHandle",
                    "severity": "warning",
                    "code": "PORT_TARGET_HANDLE_LEGACY",
                    "edge_id": getattr(edge, 'id', 'unknown'),
                    "source": edge.source,
                    "target": edge.target,
                    "sourceHandle": edge.sourceHandle,
                    "targetHandle": target_handle,
                    "error_message": (
                        f"Edge targetHandle '{target_handle}' is not in canonical registry "
                        f"for node '{edge.target}' (type: {target_node_type}). "
                        f"Using legacy synthesis - consider declaring input handles."
                    ),
                    "node_type": target_node_type,
                    "reason": reason,
                    "suggestion": (
                        f"Consider adding '{target_handle}' to the input handle registry "
                        f"for {target_node_type} for explicit validation."
                    )
                })
    
    return errors


def validate_edge_fan_in_compatibility(
    nodes: Dict[str, Any],
    edges: List['EdgeNodeModel'],
    mode: str = "warn"
) -> List[Dict[str, Any]]:
    """
    Validate fan-in compatibility for multi-edge scenarios.
    
    NEW function (Phase 2) - validates cardinality and multi-compatibility.
    
    Rules:
    1. Multi-edge fan-in to the same targetHandle is SUPPORTED (not banned)
    2. Exclusive ports (cardinality="one", exclusive=True) with multiple edges: warning/error
    3. Ambiguous ports (no declaration) with multiple edges: advisory warning
    4. Multi-compatible ports (cardinality="many", multi_compatible=True): pass
    5. Mixed-type fan-in allowed ONLY with explicit multi-compatibility declaration
    
    Args:
        nodes: Dictionary of node_id -> Node instance
        edges: List of EdgeNodeModel
        mode: Validation mode ("shadow", "warn", "strict")
        
    Returns:
        List of validation errors/warnings
    """
    errors = []
    
    # Group edges by target node and targetHandle
    fan_in_groups: Dict[str, Dict[str, List['EdgeNodeModel']]] = {}
    
    for edge in edges:
        target_handle = edge.targetHandle
        if not target_handle:
            continue
        
        target_node = nodes.get(edge.target)
        if not target_node:
            continue
        
        # Group by (target_node_id, targetHandle)
        target_node_id = edge.target
        fan_in_groups.setdefault(target_node_id, {})
        fan_in_groups[target_node_id].setdefault(target_handle, []).append(edge)
    
    # Check each fan-in group
    for target_node_id, handle_groups in fan_in_groups.items():
        target_node = nodes.get(target_node_id)
        if not target_node:
            continue
        
        target_node_type = getattr(target_node, 'node_type', None)
        if not target_node_type:
            continue
        
        for target_handle, edges_to_handle in handle_groups.items():
            # Single edge - no fan-in concern
            if len(edges_to_handle) <= 1:
                continue
            
            # Get cardinality info for this port
            cardinality = get_port_cardinality(target_node_type, target_handle)
            
            # Case 1: Exclusive port with multiple edges
            if cardinality.exclusive and cardinality.cardinality == "one":
                severity = "error" if mode == "strict" else "warning"
                errors.append({
                    "type": "PortCardinalityViolation",
                    "severity": severity,
                    "code": "PORT_CARDINALITY_EXCLUSIVE_VIOLATION",
                    "target_node": target_node_id,
                    "target_handle": target_handle,
                    "node_type": target_node_type,
                    "edge_count": len(edges_to_handle),
                    "edge_ids": [e.id for e in edges_to_handle],
                    "error_message": (
                        f"Port '{target_handle}' on node '{target_node_id}' (type: {target_node_type}) "
                        f"is declared as exclusive (cardinality='one') but has {len(edges_to_handle)} "
                        f"incoming edges. Exclusive ports should have at most one edge."
                    ),
                    "cardinality_info": {
                        "cardinality": cardinality.cardinality,
                        "exclusive": cardinality.exclusive,
                        "multi_compatible": cardinality.multi_compatible,
                    },
                    "suggestion": (
                        f"Either change the port to cardinality='many' with multi_compatibility=true, "
                        f"or reduce edges to this port to 1."
                    )
                })
            
            # Case 2: Ambiguous cardinality with multiple edges
            elif cardinality.cardinality == "ambiguous":
                # Advisory warning - no explicit declaration
                errors.append({
                    "type": "FanInCardinalityAmbiguous",
                    "severity": "warning",
                    "code": "PORT_FAN_IN_CARDINALITY_AMBIGUOUS",
                    "target_node": target_node_id,
                    "target_handle": target_handle,
                    "node_type": target_node_type,
                    "edge_count": len(edges_to_handle),
                    "edge_ids": [e.id for e in edges_to_handle],
                    "error_message": (
                        f"Port '{target_handle}' on node '{target_node_id}' (type: {target_node_type}) "
                        f"has {len(edges_to_handle)} incoming edges but no explicit cardinality declaration. "
                        f"Fan-in behavior is ambiguous - consider declaring cardinality for clarity."
                    ),
                    "suggestion": (
                        f"Declare cardinality for port '{target_handle}' in PORT_CARDINALITY registry: "
                        f"cardinality='many' and multi_compatible=true for safe multi-edge fan-in, "
                        f"or cardinality='one' and exclusive=true to enforce single-edge constraint."
                    )
                })
            
            # Case 3: Multi-compatible port - safe fan-in
            elif cardinality.multi_compatible:
                # Safe - multi-edge fan-in is explicitly supported
                logger.debug(
                    "Safe fan-in: %d edges to %s.%s (cardinality=%s, multi_compatible=true)",
                    len(edges_to_handle), target_node_id, target_handle, cardinality.cardinality
                )
                # Optionally: produce informational message in shadow mode
                if mode == "shadow":
                    errors.append({
                        "type": "FanInInfo",
                        "severity": "info",
                        "code": "PORT_FAN_IN_DECLARED_MULTI",
                        "target_node": target_node_id,
                        "target_handle": target_handle,
                        "node_type": target_node_type,
                        "edge_count": len(edges_to_handle),
                        "edge_ids": [e.id for e in edges_to_handle],
                        "error_message": (
                            f"Port '{target_handle}' on node '{target_node_id}' has {len(edges_to_handle)} "
                            f"incoming edges (safe: port declares multi_compatibility=true)."
                        ),
                        "merge_policy": cardinality.merge_policy,
                    })
            
            # Case 4: Non-exclusive, non-multi-compatible with multiple edges (unusual)
            elif not cardinality.exclusive and not cardinality.multi_compatible:
                # Warning - this configuration is unusual
                errors.append({
                    "type": "FanInConfigurationUnusual",
                    "severity": "warning",
                    "code": "PORT_FAN_IN_NON_EXCLUSIVE_NOT_MULTI",
                    "target_node": target_node_id,
                    "target_handle": target_handle,
                    "node_type": target_node_type,
                    "edge_count": len(edges_to_handle),
                    "edge_ids": [e.id for e in edges_to_handle],
                    "error_message": (
                        f"Port '{target_handle}' on node '{target_node_id}' has {len(edges_to_handle)} "
                        f"incoming edges. Port is not exclusive but also not explicitly multi-compatible. "
                        f"Recommend declaring multi_compatibility=true for clarity."
                    ),
                    "suggestion": (
                        f"Set multi_compatibility=true in PORT_CARDINALITY for '{target_handle}' "
                        f"to explicitly allow multi-edge fan-in."
                    )
                })
    
    return errors


def run_all_validations(graph: 'AgentFlowModel', mode: str = "warn") -> List[Dict[str, Any]]:
    """
    Run all graph validations.
    
    NEW (Phase 2): Includes fan-in compatibility validation.
    
    Validation order:
    1. Edge connectivity (structural)
    2. Source handle validation (clean-break policy)
    3. Target handle validation (Phase 1)
    4. Fan-in compatibility validation (Phase 2 NEW)
    5. Conditional-specific validation
    
    Args:
        graph: The agent flow model to validate
        mode: Validation mode ("shadow", "warn", "strict")
        
    Returns:
        Combined list of all validation errors/warnings
    """
    errors = []
    
    # Basic edge connectivity
    errors.extend(validate_edge_connectivity(graph.nodes, graph.edges))
    
    # Edge handle validation (clean-break: reject legacy handles)
    errors.extend(validate_edge_handles(graph.nodes, graph.edges))
    
    # Target handle validation (Phase 1)
    errors.extend(validate_edge_target_handles(graph.nodes, graph.edges, mode))
    
    # NEW: Fan-in compatibility validation (Phase 2)
    errors.extend(validate_edge_fan_in_compatibility(graph.nodes, graph.edges, mode))
    
    # Conditional-specific validation
    errors.extend(validate_graph_conditionals(graph))
    
    return errors
