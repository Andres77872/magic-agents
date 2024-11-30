import functools
from collections import defaultdict
from typing import Callable

# from magic_agents.node_system.NodeBrowsing import NodeBrowsing
from magic_agents.node_system.NodeChat import NodeChat
from magic_agents.node_system.NodeEND import NodeEND
# from magic_agents.node_system.NodeFindit import NodeFindit
# from magic_agents.node_system.NodeGenerator import NodeGenerator
from magic_agents.node_system.NodeLLM import NodeLLM
# from magic_agents.node_system.NodeMerger import NodeMerger
from magic_agents.node_system.NodeText import NodeText
from magic_agents.node_system.NodeUserInput import NodeUserInput
# from magic_agents.node_system.NodeArxiv import NodeArxiv
# from magic_agents.node_system.NodeParser import NodeParser
from magic_agents.node_system.NodeFetch import NodeFetch


def find_starting_nodes(edges):
    targets = {edge['target'] for edge in edges}
    sources = {edge['source'] for edge in edges}
    # Starting nodes are those that are sources but not targets
    return list(sources - targets)


def topological_sort_util(node, visited, stack, graph):
    visited[node] = True
    if node in graph:
        for child in graph[node]:
            if not visited[child]:
                topological_sort_util(child, visited, stack, graph)
    stack.insert(0, node)  # Add the node to the stack


def topological_sort(edges):
    graph = defaultdict(list)
    for edge in edges:
        graph[edge['source']].append(edge['target'])
    all_nodes = set(graph.keys()) | {target for targets in graph.values() for target in targets}
    visited = {node: False for node in all_nodes}
    stack = []
    for node in all_nodes:
        if not visited[node]:
            topological_sort_util(node, visited, stack, graph)
    return stack


def sort_edges(edges: list[dict], node_order) -> list[dict]:
    node_position = {node: pos for pos, node in enumerate(node_order)}
    return sorted(edges, key=lambda edge: node_position[edge['source']])


def assign_positions(nodes, sorted_nodes):
    x_spacing = 300  # Horizontal spacing
    y_spacing = 100  # Vertical spacing
    positions = {node_id: index for index, node_id in enumerate(sorted_nodes)}

    # Calculate the level of each node for y-positioning
    level = {node_id: 0 for node_id in sorted_nodes}
    for node in nodes:
        for edge in node.get('edges', []):
            child = edge['target']
            parent = edge['source']
            level[child] = max(level[child], level[parent] + 1)

    for node in nodes:
        node_id = node['id']
        # x-coordinate is based on topological position
        if 'position' not in node:
            node['position'] = {
                'x': 0,
                'y': 0
            }
        xsp = (positions[node_id] * x_spacing) if node['position']['x'] == 0 else node['position']['x']
        ysp = (level[node_id] * y_spacing) if node['position']['y'] == 0 else node['position']['y']
        node['position'] = {'x': xsp,
                            # y-coordinate is based on level
                            'y': ysp}
    return nodes


def sort_nodes(nodes: list[dict], edges: list[dict]):
    # Get topological sort order of nodes
    sorted_nodes = topological_sort(edges)
    # Assign positions to nodes
    nodes_with_positions = assign_positions(nodes, sorted_nodes)
    # Sort edges based on node order
    sorted_edges = sort_edges(edges, sorted_nodes)
    return nodes_with_positions, sorted_edges
