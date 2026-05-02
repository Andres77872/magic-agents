from typing import List, Dict, Tuple, Optional, Any

import networkx as nx

from magic_agents.node_system.NodeChat import NodeChat


# ============================================================================
# LAYOUT PRESETS - CRITICAL: Must match client LayoutPresets EXACTLY
# ============================================================================

# Per spec/design: Backend LAYOUT_PRESETS MUST match client LayoutPresets (autoLayout.ts:297-318)
# This ensures identical layout behavior whether server or client computes positions.
# DO NOT modify these values without updating client LayoutPresets in parallel.

LAYOUT_PRESETS = {
    "standard": {
        "direction": "LR",
        "horizontalSpacing": 500,
        "verticalSpacing": 400,
    },
    "compact": {
        "direction": "LR",
        "horizontalSpacing": 420,
        "verticalSpacing": 320,
    },
    "vertical": {
        "direction": "TB",  # Top-to-bottom
        "horizontalSpacing": 400,
        "verticalSpacing": 450,
    },
    "spacious": {
        "direction": "LR",
        "horizontalSpacing": 650,
        "verticalSpacing": 500,
    },
}

ALLOWED_PRESETS = list(LAYOUT_PRESETS.keys())  # ["standard", "compact", "vertical", "spacious"]


def resolve_preset(preset_name: str) -> Dict[str, Any]:
    """
    Resolve preset name to layout options dict.
    
    Args:
        preset_name: One of "standard", "compact", "vertical", "spacious"
        
    Returns:
        Dict with direction, horizontalSpacing, verticalSpacing
        
    Raises:
        ValueError: If preset_name not in ALLOWED_PRESETS
    """
    if preset_name not in ALLOWED_PRESETS:
        raise ValueError(
            f"Invalid preset '{preset_name}'. Valid presets: {', '.join(ALLOWED_PRESETS)}"
        )
    return LAYOUT_PRESETS[preset_name]
from magic_agents.node_system.NodeClientLLM import NodeClientLLM
from magic_agents.node_system.NodeConstant import NodeConstant
from magic_agents.node_system.NodeEND import NodeEND
from magic_agents.node_system.NodeFetch import NodeFetch
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.node_system.NodeLoop import NodeLoop
from magic_agents.node_system.NodeConditional import NodeConditional
from magic_agents.node_system.NodeInner import NodeInner
from magic_agents.node_system.NodeParser import NodeParser
from magic_agents.node_system.NodeSendMessage import NodeSendMessage
from magic_agents.node_system.NodeText import NodeText
from magic_agents.node_system.NodeUserInput import NodeUserInput
from magic_agents.node_system.NodePythonExec import NodePythonExec
from magic_agents.node_system.NodeHook import NodeHook

_NodeMcp_class = None

def get_node_mcp():
    global _NodeMcp_class
    if _NodeMcp_class is None:
        from magic_agents.node_system.NodeMcp import NodeMcp
        _NodeMcp_class = NodeMcp
    return _NodeMcp_class

class NodeMcpProxy:
    def __new__(cls, *args, **kwargs):
        return get_node_mcp()(*args, **kwargs)

NodeMcp = NodeMcpProxy


def build_graph(edges: List[Dict]) -> nx.DiGraph:
    """Create directed graph from edges."""
    graph = nx.DiGraph()
    for edge in edges:
        graph.add_edge(edge['source'], edge['target'])
    return graph


def detect_cycles(graph: nx.DiGraph):
    """Detects cycles and raises an exception if found."""
    try:
        cycle = nx.find_cycle(graph)
        raise ValueError(f"Cycle detected in the graph: {cycle}")
    except nx.NetworkXNoCycle:
        pass  # no cycle detected


def perform_topological_sort(graph: nx.DiGraph) -> List[str]:
    """Performs topological sort using networkx."""
    try:
        # allow cycles for specialized nodes (e.g., loop), fallback to insertion order on cycle
        return list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        # cycle(s) detected: skip strict topological sort, use current node order
        return list(graph.nodes())


def sort_edges_by_nodes_order(edges: List[Dict], sorted_node_ids: List[str]) -> List[Dict]:
    """Sort edges according to sorted node order (based on source node)."""
    node_order = {node_id: index for index, node_id in enumerate(sorted_node_ids)}
    return sorted(edges, key=lambda edge: node_order[edge['source']])


def assign_node_positions(nodes: List[Dict], graph: nx.DiGraph, sorted_nodes: List[str]) -> List[Dict]:
    """Assign x,y positions to nodes based on topological ordering and node level (distance from starting node)."""
    x_spacing = 300
    y_spacing = 100

    # Compute levels using longest path length from sources
    levels = {}
    for node in sorted_nodes:
        preds = list(graph.predecessors(node))
        if not preds:
            levels[node] = 0
        else:
            levels[node] = 1 + max(levels.get(pred, 0) for pred in preds)

    node_positions = {
        node_id: {
            'x': index * x_spacing,
            'y': levels[node_id] * y_spacing
        }
        for index, node_id in enumerate(sorted_nodes)
    }

    node_dict = {node['id']: node for node in nodes}
    for node_id, pos in node_positions.items():
        if 'position' not in node_dict[node_id] or node_dict[node_id]['position'] == {'x': 0, 'y': 0}:
            node_dict[node_id]['position'] = pos

    return list(node_dict.values())


def arrange_with_sizes(
    nodes: List[Dict],
    edges: List[Dict],
    sizes: Optional[Dict[str, Dict[str, int]]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, int]]:
    """
    Compute node positions using size-aware topological layout.
    
    Args:
        nodes: List of node dicts (must include 'id' field)
        edges: List of edge dicts (must include 'source', 'target')
        sizes: Optional mapping {node_id: {'width': int, 'height': int}}
        options: Optional dict with:
            - direction: "LR" (left-right) or "TB" (top-bottom)
            - horizontalSpacing: int (px between columns/rows)
            - verticalSpacing: int (px between nodes in same layer)
            
    Returns:
        Mapping {node_id: {'x': int, 'y': int}}
        
    Direction behavior:
        - "LR": Layers are columns; X increases per layer; Y varies within layer
        - "TB": Layers are rows; Y increases per layer; X varies within layer
        - Coordinate assignment swaps X/Y based on direction
        
    Size-aware spacing:
        - Horizontal spacing = max(prev_layer_max_width, options.horizontalSpacing) + SPACING_BUFFER
        - Vertical spacing = node_height + options.verticalSpacing
        
    Default options (when options omitted):
        - direction: "LR"
        - horizontalSpacing: 250
        - verticalSpacing: 100
    """
    # Extract options or use defaults
    direction = options.get('direction', 'LR') if options else 'LR'
    horizontal_spacing = options.get('horizontalSpacing', 250) if options else 250
    vertical_spacing = options.get('verticalSpacing', 100) if options else 100
    
    # Legacy default spacing values (for backward compatibility)
    DEFAULT_HORIZONTAL = horizontal_spacing
    DEFAULT_VERTICAL = 100  # Base height estimate
    SPACING_BUFFER = 50
    
    # Handle empty nodes list
    if len(nodes) == 0:
        return {}
    
    # Build graph from edges
    graph = build_graph(edges)
    
    # Get topological order (nodes in edges)
    sorted_node_ids = perform_topological_sort(graph)
    
    # Collect all node IDs from input
    all_node_ids = set(n.get('id') for n in nodes)
    
    # If no edges, all nodes are disconnected - position them linearly
    if len(sorted_node_ids) == 0 and len(nodes) > 0:
        positions: Dict[str, Dict[str, int]] = {}
        current_x = 0
        for node in nodes:
            node_id = node.get('id')
            node_height = DEFAULT_VERTICAL
            if sizes and node_id in sizes:
                node_height = sizes[node_id].get('height', DEFAULT_VERTICAL)
            positions[node_id] = {
                'x': int(current_x),
                'y': int(0)
            }
            current_x += DEFAULT_HORIZONTAL + SPACING_BUFFER
        return positions
    
    # Compute levels using longest path length from sources
    levels: Dict[str, int] = {}
    for node_id in sorted_node_ids:
        preds = list(graph.predecessors(node_id))
        if not preds:
            levels[node_id] = 0
        else:
            levels[node_id] = 1 + max(levels.get(pred, 0) for pred in preds)
    
    # Group nodes by level (layer)
    layer_groups: Dict[int, List[str]] = {}
    for node_id in sorted_node_ids:
        level = levels[node_id]
        if level not in layer_groups:
            layer_groups[level] = []
        layer_groups[level].append(node_id)
    
    # Compute max width per layer for size-aware horizontal spacing
    layer_max_widths: Dict[int, int] = {}
    for level, node_ids in layer_groups.items():
        max_width = DEFAULT_HORIZONTAL
        for node_id in node_ids:
            if sizes and node_id in sizes:
                node_width = sizes[node_id].get('width', DEFAULT_HORIZONTAL)
                max_width = max(max_width, node_width)
        layer_max_widths[level] = max_width
    
    # Calculate positions with direction-aware coordinate assignment
    # For LR: layers are columns (X increases per layer, Y within layer)
    # For TB: layers are rows (Y increases per layer, X within layer)
    positions: Dict[str, Dict[str, int]] = {}
    layer_distance = 0  # Distance between layers (X for LR, Y for TB)
    
    # Process layers in order
    max_level = max(layer_groups.keys()) if layer_groups else 0
    for level in range(max_level + 1):
        if level not in layer_groups:
            continue
        
        node_ids_in_layer = layer_groups[level]
        
        # Calculate within-layer positions (Y for LR, X for TB)
        within_layer_pos = 0
        for node_id in node_ids_in_layer:
            # Get node height for size-aware spacing
            node_height = DEFAULT_VERTICAL
            if sizes and node_id in sizes:
                node_height = sizes[node_id].get('height', DEFAULT_VERTICAL)
            
            # Apply direction-aware coordinate assignment
            if direction == 'LR':
                # Left-to-right: layers are columns
                positions[node_id] = {
                    'x': int(layer_distance),
                    'y': int(within_layer_pos)
                }
            else:
                # Top-to-bottom: layers are rows (swap X/Y)
                positions[node_id] = {
                    'x': int(within_layer_pos),
                    'y': int(layer_distance)
                }
            
            # Next node in same layer: add spacing
            within_layer_pos += node_height + vertical_spacing
        
        # Next layer: add layer spacing (horizontal spacing between layers)
        layer_distance += (layer_max_widths[level] if direction == 'LR' else DEFAULT_VERTICAL) + horizontal_spacing
    
    # Handle disconnected nodes (nodes not in graph edges)
    node_ids_in_graph = set(graph.nodes())
    disconnected_nodes = all_node_ids - node_ids_in_graph
    
    # Position disconnected nodes after the main graph
    offset = layer_distance
    for node_id in disconnected_nodes:
        if direction == 'LR':
            positions[node_id] = {
                'x': int(offset),
                'y': int(0)
            }
        else:
            positions[node_id] = {
                'x': int(0),
                'y': int(offset)
            }
        offset += DEFAULT_HORIZONTAL + SPACING_BUFFER
    
    return positions


def sort_nodes(nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Main function to sort nodes ensuring correct execution order and positioning."""
    graph = build_graph(edges)
    sorted_node_ids = perform_topological_sort(graph)
    sorted_edges = sort_edges_by_nodes_order(edges, sorted_node_ids)
    sorted_nodes_with_positions = assign_node_positions(nodes, graph, sorted_node_ids)

    return sorted_nodes_with_positions, sorted_edges


# Expose only the main node classes and utilities
__all__ = [
    "NodeChat",
    "NodeClientLLM",
    "NodeConstant",
    "NodeEND",
    "NodeFetch",
    "NodeLLM",
    "NodeParser",
    "NodeSendMessage",
    "NodeText",
    "NodeUserInput",
    "NodeLoop",
    "NodeInner",
    "NodeConditional",
    "NodePythonExec",
    "NodeMcp",
    "NodeHook",
    "build_graph",
    "detect_cycles",
    "perform_topological_sort",
    "sort_edges_by_nodes_order",
    "assign_node_positions",
    "sort_nodes",
    "arrange_with_sizes",
    "LAYOUT_PRESETS",
    "ALLOWED_PRESETS",
    "resolve_preset",
]
