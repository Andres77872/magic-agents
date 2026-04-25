"""
Canonical Handle Registry - Maps node types to their valid output handles.

This registry is used by graph validation to reject non-canonical handle names
in edge sourceHandle fields. Legacy handles like handle_generated_end are 
explicitly rejected.

The registry defines DEFAULT output handles per node type. Individual node 
instances may override handles via data.handles in JSON, which is validated
separately after node instantiation.
"""
from typing import Dict, List, Set

from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel


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
}

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