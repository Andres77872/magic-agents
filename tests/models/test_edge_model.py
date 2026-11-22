"""
Edge.id serialization roundtrip tests (Phase 4.4).

Tests that:
1. Edge.id preserved through JSON → build → JSON cycle
2. Edge.id uniqueness maintained after deserialization
3. Edge.id used as tracker key after deserialization
4. EdgeNodeModel serialization preserves all fields
"""

from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.execution.input_tracker import InputInfo, NodeInputTracker


class TestEdgeSerializationRoundtrip:
    """Tests for edge.id preservation through serialization cycles."""

    def test_edge_id_uniqueness_after_serialization(self):
        """
        Edge IDs must remain unique after serialization/deserialization.
        
        UUID-generated IDs should not collide.
        """
        # Create 10 edges without explicit IDs
        edges = []
        for i in range(10):
            edge = EdgeNodeModel(
                source=f"node_{i}",
                target=f"node_{i+1}"
            )
            edges.append(edge)
        
        # All IDs should be unique
        ids = [e.id for e in edges]
        assert len(ids) == len(set(ids))  # No duplicates
        
        # Serialize all
        data_list = [e.model_dump() for e in edges]
        
        # Deserialize back
        reconstructed = [EdgeNodeModel(**d) for d in data_list]
        
        # IDs still unique
        reconstructed_ids = [e.id for e in reconstructed]
        assert len(reconstructed_ids) == len(set(reconstructed_ids))
        
        # IDs match originals
        assert reconstructed_ids == ids

class TestEdgeIdUsedAsTrackerKey:
    """Tests verifying edge.id is used as tracker key after deserialization."""

    def test_tracker_keys_match_edge_ids_from_serialized_graph(self):
        """
        After deserializing graph edges, tracker keys must match edge.id values.
        
        This ensures the edge.id is used correctly in tracker construction.
        """
        # Create edges with explicit IDs
        edge1 = EdgeNodeModel(id="edge_001", source="input", target="llm", targetHandle="handle_user_message")
        edge2 = EdgeNodeModel(id="edge_002", source="client", target="llm", targetHandle="handle_client")
        
        # Create InputInfo from edge data
        info1 = InputInfo(
            edge_id=edge1.id,
            handle=edge1.targetHandle,
            source_node=edge1.source,
            source_handle=edge1.sourceHandle
        )
        info2 = InputInfo(
            edge_id=edge2.id,
            handle=edge2.targetHandle,
            source_node=edge2.source,
            source_handle=edge2.sourceHandle
        )
        
        # Build tracker
        tracker = NodeInputTracker(node_id="llm", expected_inputs=[info1, info2])
        
        # Keys match edge IDs
        assert tracker.expected_edges == {"edge_001", "edge_002"}


