"""
Canonical Handle Registry - Maps node types to their valid handles.

This registry is used by graph validation to reject non-canonical handle names
in edge sourceHandle and targetHandle fields. Legacy handles like handle_generated_end
are explicitly rejected.

The registry defines:
- CANONICAL_OUTPUT_HANDLES: Output handles per node type (original)
- CANONICAL_INPUT_HANDLES: Input handles per node type (Phase 1)
- PORT_CARDINALITY: Fan-in cardinality metadata per input port (Phase 2)

Individual node instances may override handles via data.handles in JSON, which
is validated separately after node instantiation.
"""
from typing import Dict, List, Set, Optional, Literal
from dataclasses import dataclass

from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel


@dataclass
class CardinalityInfo:
    """
    Cardinality metadata for a port (Phase 2).
    
    Defines fan-in semantics for input ports:
    - cardinality: "one" (exclusive) or "many" (multi-compatible)
    - exclusive: True means max one incoming edge
    - merge_policy: How to handle multiple inputs when cardinality="many"
    
    Defaults: implicit cardinality treated as "ambiguous" in warn mode.
    """
    cardinality: Literal["one", "many", "ambiguous"] = "ambiguous"
    exclusive: bool = True  # True = max one edge, False = multi-compatible
    merge_policy: Literal["first-wins", "collect", "merge"] = "first-wins"
    multi_compatible: bool = False  # True = explicit multi-compatibility declaration


# Canonical output handles per node type (defaults, not including overrides)
# These are the handles nodes emit by default. Custom handles via data.handles
# are validated against the actual node instance after creation.
CANONICAL_OUTPUT_HANDLES: Dict[str, Set[str]] = {
    # Source/output handles per node type
    ModelAgentFlowTypesModel.USER_INPUT: {
        'handle_user_message',
        'handle_user_files',
        'handle_user_images',
        'handle_client_extras',
    },
    ModelAgentFlowTypesModel.TEXT: {
        'handle_text_output',
    },
    ModelAgentFlowTypesModel.CONSTANT: {
        'handle_constant_output',
    },
    ModelAgentFlowTypesModel.PARSER: {
        'handle_parser_output',
    },
    ModelAgentFlowTypesModel.FETCH: {
        'handle_fetch_output',
        'handle-tool-definition',  # In tool_mode
    },
    ModelAgentFlowTypesModel.CLIENT: {
        'handle-client-provider',
    },
    ModelAgentFlowTypesModel.LLM: {
        'handle_streaming_content',
        'handle_generated_content',
        'handle-tool-calls',
    },
    ModelAgentFlowTypesModel.CHAT: {
        'handle_chat_output',
    },
    ModelAgentFlowTypesModel.SEND_MESSAGE: {
        'handle_message_output',
        'content',  # Internal streaming handle
    },
    ModelAgentFlowTypesModel.LOOP: {
        'handle_item',
        'handle_end',
    },
    ModelAgentFlowTypesModel.CONDITIONAL: set(),  # Dynamic - validated via output_handles declaration
    ModelAgentFlowTypesModel.INNER: {
        'handle_content_stream',
        'handle_execution_content',
        'handle_execution_extras',
    },
    ModelAgentFlowTypesModel.END: {
        'handle_end_output',
    },
    ModelAgentFlowTypesModel.VOID: set(),  # Terminal node - no outputs
    ModelAgentFlowTypesModel.PYTHON_EXEC: {
        'handle-tool-definition',
    },
    ModelAgentFlowTypesModel.MCP: {
        'handle-tool-definition',
    },
    # Phase 6: NodeHook output handles
    ModelAgentFlowTypesModel.HOOK: {
        'handle-user-output',
        'handle-debug-output',
        'handle-feedback-output',
    },
}

# NEW: Canonical input handles per node type (Phase 1)
# These are the handles nodes consume as inputs.
CANONICAL_INPUT_HANDLES: Dict[str, Set[str]] = {
    # Input handles per node type
    ModelAgentFlowTypesModel.USER_INPUT: set(),  # Source node - no inputs
    ModelAgentFlowTypesModel.TEXT: {
        'handle_flow_input',
    },
    ModelAgentFlowTypesModel.CONSTANT: set(),  # No inputs
    ModelAgentFlowTypesModel.PARSER: {
        'handle_parser_input',  # Primary input (templates may have multiple)
    },
    ModelAgentFlowTypesModel.FETCH: {
        'handle_fetch_input',
    },
    ModelAgentFlowTypesModel.CLIENT: set(),  # No inputs (provides client to LLM)
    ModelAgentFlowTypesModel.LLM: {
        'handle_user_message',        # Primary user message input
        'handle-client-provider',     # Client provider input (actual handle)
        'handle-tool-definition',     # Tool definition inputs (variadic)
        'handle-chat',                # Chat history input
        'handle-system-context',      # System context input
    },
    ModelAgentFlowTypesModel.CHAT: set(),  # Source node variant
    ModelAgentFlowTypesModel.SEND_MESSAGE: {
        'handle_send_extra',  # Extra context input
    },
    ModelAgentFlowTypesModel.LOOP: {
        'handle_loop_input',  # Primary input for iteration
    },
    ModelAgentFlowTypesModel.CONDITIONAL: {
        'handle_context',  # Context input for evaluation
    },
    ModelAgentFlowTypesModel.INNER: {
        'handle_inner_input',  # Primary input for inner graph
    },
    ModelAgentFlowTypesModel.END: {
        'handle_flow_input',  # Standard input for END nodes
    },
    ModelAgentFlowTypesModel.VOID: set(),  # Terminal sink - no inputs validated
    ModelAgentFlowTypesModel.PYTHON_EXEC: set(),  # No input handles
    ModelAgentFlowTypesModel.MCP: set(),  # No input handles
    # Phase 6: NodeHook input handles
    ModelAgentFlowTypesModel.HOOK: {
        'handle-hook-context',
    },
}

# NEW: Port cardinality metadata (Phase 2)
# Defines fan-in semantics for input ports per node type.
# Ports without explicit declaration default to CardinalityInfo(cardinality="ambiguous")
PORT_CARDINALITY: Dict[str, Dict[str, CardinalityInfo]] = {
    # LLM node: tool definition handles accept multiple tools (multi-compatible)
    ModelAgentFlowTypesModel.LLM: {
        'handle_user_message': CardinalityInfo(cardinality="one", exclusive=True),
        'handle-client-provider': CardinalityInfo(cardinality="one", exclusive=True),
        'handle-tool-definition': CardinalityInfo(
            cardinality="many", 
            exclusive=False, 
            multi_compatible=True,
            merge_policy="collect"
        ),
        'handle-chat': CardinalityInfo(cardinality="one", exclusive=True),
        'handle-system-context': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Text node: single input
    ModelAgentFlowTypesModel.TEXT: {
        'handle_flow_input': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Parser node: can accept multiple template inputs
    ModelAgentFlowTypesModel.PARSER: {
        'handle_parser_input': CardinalityInfo(cardinality="many", exclusive=False, multi_compatible=True),
    },
    # END node: single input
    ModelAgentFlowTypesModel.END: {
        'handle_flow_input': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Conditional: single context input
    ModelAgentFlowTypesModel.CONDITIONAL: {
        'handle_context': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Loop: single input for iteration
    ModelAgentFlowTypesModel.LOOP: {
        'handle_loop_input': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Inner: single input
    ModelAgentFlowTypesModel.INNER: {
        'handle_inner_input': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Send message: single extra input
    ModelAgentFlowTypesModel.SEND_MESSAGE: {
        'handle_send_extra': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # Fetch: single input
    ModelAgentFlowTypesModel.FETCH: {
        'handle_fetch_input': CardinalityInfo(cardinality="one", exclusive=True),
    },
    # VOID: sink node, no cardinality restrictions
    ModelAgentFlowTypesModel.VOID: {},
    # Source nodes (user_input, client, constant, chat): no inputs
    ModelAgentFlowTypesModel.USER_INPUT: {},
    ModelAgentFlowTypesModel.CLIENT: {},
    ModelAgentFlowTypesModel.CONSTANT: {},
    ModelAgentFlowTypesModel.CHAT: {},
    ModelAgentFlowTypesModel.PYTHON_EXEC: {},
    ModelAgentFlowTypesModel.MCP: {},
    # Phase 6: NodeHook cardinality
    ModelAgentFlowTypesModel.HOOK: {
        'handle-hook-context': CardinalityInfo(
            cardinality="one", exclusive=True
        ),
    },
}


def get_port_cardinality(node_type: str, handle: str) -> CardinalityInfo:
    """
    Get cardinality metadata for a port.
    
    NEW function (Phase 2) - returns CardinalityInfo for fan-in validation.
    
    Args:
        node_type: Node type key
        handle: Input handle name
        
    Returns:
        CardinalityInfo for the port, or default ambiguous if not declared
    """
    node_cardinality = PORT_CARDINALITY.get(node_type, {})
    if handle in node_cardinality:
        return node_cardinality[handle]
    # Check for dynamic pattern match (handle-tool-definition-0 matches handle-tool-definition)
    for pattern in DYNAMIC_HANDLE_PATTERNS:
        if handle.startswith(pattern.rstrip('-').rstrip('_')):
            # Check if base pattern exists
            base_handle = pattern.rstrip('-').rstrip('_')
            if base_handle in node_cardinality:
                return node_cardinality[base_handle]
    # Default: ambiguous cardinality
    return CardinalityInfo(cardinality="ambiguous", exclusive=False, multi_compatible=False)


def is_multi_compatible_port(node_type: str, handle: str) -> bool:
    """
    Check if a port explicitly declares multi-compatibility for fan-in.
    
    NEW function (Phase 2) - determines if mixed-type fan-in is allowed.
    
    Args:
        node_type: Node type key
        handle: Input handle name
        
    Returns:
        True if port is explicitly multi-compatible
    """
    cardinality = get_port_cardinality(node_type, handle)
    return cardinality.multi_compatible

# Legacy handles that must be rejected (clean-break policy)
LEGACY_REJECTED_HANDLES: Set[str] = {
    'handle_generated_end',
}

# Handle patterns that are dynamically generated and should pass validation
# These patterns are prefixes that can have arbitrary suffixes
DYNAMIC_HANDLE_PATTERNS: List[str] = [
    'handle-tool-definition-',  # Tool handles auto-generated: handle-tool-definition-0, -1, etc.
    'handle_parser_input_',     # Parser input handles: handle_parser_input_0, handle_parser_input_1, etc.
    'handle-tool-',             # General tool input prefix for LLM nodes
]


def is_dynamic_handle(handle: str) -> bool:
    """Check if a handle matches a dynamic pattern.
    
    Dynamic handles are auto-generated at runtime and should pass validation
    even if not in the canonical list.
    
    Args:
        handle: Handle name to check
        
    Returns:
        True if handle matches a dynamic pattern
    """
    for pattern in DYNAMIC_HANDLE_PATTERNS:
        if handle.startswith(pattern):
            return True
    return False


def is_legacy_rejected_handle(handle: str) -> bool:
    """Check if a handle is a legacy handle that must be rejected.
    
    Args:
        handle: Handle name to check
        
    Returns:
        True if handle is a legacy rejected handle
    """
    return handle in LEGACY_REJECTED_HANDLES


def get_canonical_output_handles(node_type: str) -> Set[str]:
    """Get canonical output handles for a node type.
    
    Args:
        node_type: Canonical node type key
        
    Returns:
        Set of canonical output handle names, or empty set if unknown type
    """
    return CANONICAL_OUTPUT_HANDLES.get(node_type, set())


def get_canonical_input_handles(node_type: str) -> Set[str]:
    """Get canonical input handles for a node type.
    
    Args:
        node_type: Canonical node type key
        
    Returns:
        Set of canonical input handle names, or empty set if unknown type
    """
    return CANONICAL_INPUT_HANDLES.get(node_type, set())


def is_valid_source_handle(
    source_handle: str,
    source_node_type: str,
    source_node_instance_handles: Set[str] = None
) -> tuple[bool, str]:
    """Validate if a source handle is valid for a node type.
    
    Validation rules:
    1. Legacy rejected handles always fail (handle_generated_end)
    2. Dynamic pattern handles pass (handle-tool-definition-*, handle_parser_input_*)
    3. If node instance handles provided, check against instance handles
    4. Otherwise check against canonical handles for the node type
    5. Conditional nodes: check against declared output_handles or skip
    
    Args:
        source_handle: The sourceHandle from an edge
        source_node_type: The canonical type key of the source node
        source_node_instance_handles: Actual output handles from node instance (optional)
        
    Returns:
        Tuple of (is_valid, reason)
    """
    # Rule 1: Reject legacy handles
    if is_legacy_rejected_handle(source_handle):
        return False, f"Legacy handle '{source_handle}' is rejected. Use canonical handles instead."
    
    # Rule 2: Dynamic pattern handles pass
    if is_dynamic_handle(source_handle):
        return True, "Dynamic pattern handle"
    
    # Rule 3: Check against instance handles if provided
    if source_node_instance_handles is not None:
        if source_handle in source_node_instance_handles:
            return True, "Valid instance handle"
        # For conditional nodes, if output_handles not declared, we can't validate
        if source_node_type == ModelAgentFlowTypesModel.CONDITIONAL:
            if not source_node_instance_handles:
                return True, "Conditional without declared output_handles"
        return False, f"Handle '{source_handle}' not in node's output handles: {list(source_node_instance_handles)}"
    
    # Rule 4: Check against canonical handles
    canonical_handles = get_canonical_output_handles(source_node_type)
    
    # Special case: Conditional nodes have dynamic handles
    if source_node_type == ModelAgentFlowTypesModel.CONDITIONAL:
        # Conditionals emit whatever handle the condition evaluates to
        # Without output_handles declaration, we can't validate statically
        return True, "Conditional dynamic output"
    
    # Special case: Parser nodes accept arbitrary template inputs
    # But for OUTPUT (sourceHandle), they only emit handle_parser_output
    # However, template inputs like handle_parser_input_0 are handled above via dynamic patterns
    
    # Special case: void nodes don't emit
    if source_node_type == ModelAgentFlowTypesModel.VOID:
        return False, f"Void nodes do not emit outputs. Handle '{source_handle}' is invalid."
    
    # Check canonical handles
    if source_handle in canonical_handles:
        return True, "Canonical handle"
    
    # Not in canonical handles and not a dynamic pattern
    return False, (
        f"Handle '{source_handle}' is not valid for node type '{source_node_type}'. "
        f"Valid handles: {list(canonical_handles)}"
    )


def is_valid_target_handle(
    target_handle: str,
    target_node_type: str,
    target_node_instance_handles: Set[str] = None,
    mode: str = "warn"
) -> tuple[bool, str]:
    """Validate if a target handle is valid for a node type.
    
    NEW function (Phase 1) - parallel to is_valid_source_handle() for input side.
    
    Validation rules:
    1. Dynamic pattern handles pass (handle-tool-definition-*, handle_parser_input_*)
    2. If node instance handles provided, check against instance handles
    3. Check against canonical input handles for the node type
    4. VOID nodes: no target validation (sink node)
    5. Legacy synthesis: unknown handles pass with warning in shadow/warn mode
    
    Args:
        target_handle: The targetHandle from an edge
        target_node_type: The canonical type key of the target node
        target_node_instance_handles: Actual input handles from node instance (optional)
        mode: Validation mode ("shadow", "warn", "strict") - affects legacy synthesis
        
    Returns:
        Tuple of (is_valid, reason)
    """
    # Rule 1: Dynamic pattern handles pass
    if is_dynamic_handle(target_handle):
        return True, "Dynamic pattern handle"
    
    # Rule 2: Check against instance handles if provided
    if target_node_instance_handles is not None:
        if target_handle in target_node_instance_handles:
            return True, "Valid instance input handle"
        return False, f"Handle '{target_handle}' not in node's input handles: {list(target_node_instance_handles)}"
    
    # Rule 3: Check against canonical input handles
    canonical_handles = get_canonical_input_handles(target_node_type)
    
    # Rule 4: VOID nodes accept anything (sink node, no validation)
    if target_node_type == ModelAgentFlowTypesModel.VOID:
        return True, "Void sink node - accepts any handle"
    
    # Check canonical handles
    if target_handle in canonical_handles:
        return True, "Canonical input handle"
    
    # Rule 5: Legacy synthesis - unknown handles pass with warning in shadow/warn mode
    # In strict mode, this would be False (deferred to follow-up)
    if mode in ("shadow", "warn"):
        return True, f"Opaque/legacy handle (not in registry for {target_node_type})"
    
    # Strict mode (deferred): reject unknown handles
    return False, (
        f"Handle '{target_handle}' is not valid input for node type '{target_node_type}'. "
        f"Valid input handles: {list(canonical_handles)}"
    )


def resolve_handle_to_port(handle: str, node_type: str, direction: str) -> str:
    """
    Resolve a handle string to internal port identity.
    
    NEW function (Phase 1) - placeholder for alias resolution.
    
    For shadow/warn mode: handle resolves to itself (identity passthrough).
    Future: map handle aliases to canonical port identifiers (P2 deferred).
    
    Args:
        handle: Handle name to resolve
        node_type: Node type for context
        direction: "input" or "output"
        
    Returns:
        Resolved port identifier (currently returns handle itself)
    """
    # Phase 1: Identity passthrough - handle resolves to itself
    # Phase 2 (deferred): Map handle aliases to canonical port IDs
    return handle


def get_node_output_handles_from_instance(node_instance) -> Set[str]:
    """Extract output handles from a node instance.
    
    Scans the node instance for OUTPUT_HANDLE attributes and returns them.
    This handles nodes with multiple output handles (like NodeLLM).
    
    Args:
        node_instance: A Node subclass instance
        
    Returns:
        Set of output handle names from the instance
    """
    handles: Set[str] = set()
    
    # Check common output handle attribute patterns
    for attr_name in dir(node_instance):
        # Look for OUTPUT_HANDLE attributes
        if 'OUTPUT_HANDLE' in attr_name.upper() or 'OUTPUT' in attr_name.upper():
            attr_value = getattr(node_instance, attr_name, None)
            if isinstance(attr_value, str) and attr_value:
                handles.add(attr_value)
    
    # Also check node.outputs keys (populated after execution, but useful for validation)
    if hasattr(node_instance, 'outputs') and isinstance(node_instance.outputs, dict):
        handles.update(node_instance.outputs.keys())
    
    # For conditionals, check output_handles declaration
    if hasattr(node_instance, 'output_handles') and node_instance.output_handles:
        handles.update(node_instance.output_handles)
    
    return handles


def get_node_input_handles_from_instance(node_instance) -> Set[str]:
    """
    Extract input handles from a node instance.
    
    NEW function (Phase 1) - parallel to get_node_output_handles_from_instance.
    
    Scans the node instance for input handle patterns.
    
    Args:
        node_instance: A Node subclass instance
        
    Returns:
        Set of input handle names from the instance
    """
    handles: Set[str] = set()
    
    # Check common input handle attribute patterns
    for attr_name in dir(node_instance):
        if 'INPUT_HANDLE' in attr_name.upper() or 'INPUT' in attr_name.upper():
            attr_value = getattr(node_instance, attr_name, None)
            if isinstance(attr_value, str) and attr_value:
                handles.add(attr_value)
    
    # Also check node.inputs keys (populated at runtime)
    if hasattr(node_instance, 'inputs') and isinstance(node_instance.inputs, dict):
        handles.update(node_instance.inputs.keys())
    
    return handles
