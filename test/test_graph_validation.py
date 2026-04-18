"""
Tests for graph validation in agt_flow module.

validate_graph() returns a dict {"valid": bool, "errors": [...]} —
it NEVER raises exceptions. All tests assert on the return value.
"""
import pytest
from magic_agents.agt_flow import validate_graph
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel


class TestGraphValidation:
    """Test suite for graph validation"""

    def test_valid_graph_with_single_user_input(self):
        """Test that a valid graph with single USER_INPUT passes validation"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2'},
            {'id': 'edge2', 'source': 'node2', 'target': 'node3'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_valid_graph_with_multiple_end_nodes(self):
        """Test that multiple END nodes are allowed"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END},
            {'id': 'node4', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2'},
            {'id': 'edge2', 'source': 'node2', 'target': 'node3'},
            {'id': 'edge3', 'source': 'node2', 'target': 'node4'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_user_input_node(self):
        """Test that missing USER_INPUT node returns validation error"""
        nodes = [
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node2', 'target': 'node3'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["error_type"] == "GraphValidationError"
        assert "Graph must contain exactly one USER_INPUT node" in error["error_message"]
        assert "Found: 0" in error["error_message"]

    def test_multiple_user_input_nodes(self):
        """Test that multiple USER_INPUT nodes return validation error"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node4', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node3'},
            {'id': 'edge2', 'source': 'node2', 'target': 'node3'},
            {'id': 'edge3', 'source': 'node3', 'target': 'node4'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert "Graph must contain exactly one USER_INPUT node" in error["error_message"]
        assert "Found 2" in error["error_message"]
        assert "node1" in error["context"]["node_ids"]
        assert "node2" in error["context"]["node_ids"]

    def test_duplicate_edges_same_source_target(self):
        """Test that duplicate edges with same source and target are detected"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2'},
            {'id': 'edge2', 'source': 'node1', 'target': 'node2'},  # Duplicate
            {'id': 'edge3', 'source': 'node2', 'target': 'node3'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["error_type"] == "GraphValidationError"
        assert "Found duplicate edges with same source, target, and handles" in error["error_message"]
        duplicate_ids = [d["edge_id"] for d in error["context"]["duplicate_edges"]]
        assert "edge2" in duplicate_ids

    def test_multiple_duplicate_edges(self):
        """Test detection of multiple duplicate edges"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2'},
            {'id': 'edge2', 'source': 'node1', 'target': 'node2'},  # Duplicate
            {'id': 'edge3', 'source': 'node2', 'target': 'node3'},
            {'id': 'edge4', 'source': 'node2', 'target': 'node3'}   # Duplicate
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error["error_type"] == "GraphValidationError"
        duplicates = error["context"]["duplicate_edges"]
        assert error["context"]["duplicate_count"] == 2
        duplicate_ids = {d["edge_id"] for d in duplicates}
        assert "edge2" in duplicate_ids
        assert "edge4" in duplicate_ids

    def test_different_handles_not_considered_duplicate(self):
        """Test that edges with same source/target but different handles are NOT duplicates"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2', 'sourceHandle': 'handle1', 'targetHandle': 'input1'},
            {'id': 'edge2', 'source': 'node1', 'target': 'node2', 'sourceHandle': 'handle2', 'targetHandle': 'input2'},
            {'id': 'edge3', 'source': 'node2', 'target': 'node3'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_exact_duplicate_edges_with_handles(self):
        """Test that edges with same source, target, AND handles are detected as duplicates"""
        nodes = [
            {'id': 'node1', 'type': ModelAgentFlowTypesModel.USER_INPUT},
            {'id': 'node2', 'type': ModelAgentFlowTypesModel.LLM},
            {'id': 'node3', 'type': ModelAgentFlowTypesModel.END}
        ]
        edges = [
            {'id': 'edge1', 'source': 'node1', 'target': 'node2', 'sourceHandle': 'out1', 'targetHandle': 'in1'},
            {'id': 'edge2', 'source': 'node1', 'target': 'node2', 'sourceHandle': 'out1', 'targetHandle': 'in1'},  # Exact duplicate
            {'id': 'edge3', 'source': 'node2', 'target': 'node3'}
        ]

        result = validate_graph(nodes, edges)
        assert result["valid"] is False
        assert len(result["errors"]) == 1
        error = result["errors"][0]
        duplicates = error["context"]["duplicate_edges"]
        assert len(duplicates) == 1
        assert duplicates[0]["edge_id"] == "edge2"
        assert duplicates[0]["sourceHandle"] == "out1"
        assert duplicates[0]["targetHandle"] == "in1"
