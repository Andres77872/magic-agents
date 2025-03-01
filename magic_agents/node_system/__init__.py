from typing import List, Dict, Tuple

import networkx as nx

from magic_agents.node_system.NodeChat import NodeChat
from magic_agents.node_system.NodeClientLLM import NodeClientLLM
from magic_agents.node_system.NodeEND import NodeEND
from magic_agents.node_system.NodeFetch import NodeFetch
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.node_system.NodeParser import NodeParser
from magic_agents.node_system.NodeSendMessage import NodeSendMessage
from magic_agents.node_system.NodeText import NodeText
from magic_agents.node_system.NodeUserInput import NodeUserInput


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
    detect_cycles(graph)
    return list(nx.topological_sort(graph))


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
            levels[node] = 1 + max(levels[pred] for pred in preds)

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


def sort_nodes(nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Main function to sort nodes ensuring correct execution order and positioning."""
    graph = build_graph(edges)
    sorted_node_ids = perform_topological_sort(graph)
    sorted_edges = sort_edges_by_nodes_order(edges, sorted_node_ids)
    sorted_nodes_with_positions = assign_node_positions(nodes, graph, sorted_node_ids)

    return sorted_nodes_with_positions, sorted_edges
