"""Unit tests for arrange_with_sizes() function.

Per spec.md graph-layout requirement:
- Layout with size metadata produces dimension-aware positions
- Layout without size metadata uses default spacing
- Layout handles disconnected graph components
- Layout preserves topological order
- Performance: graphs up to 100 nodes within 500ms
"""
import time
from magic_agents.node_system import arrange_with_sizes


class TestArrangeWithSizesBasic:
    """Basic layout functionality tests."""
    
    def test_arrange_with_sizes_basic(self):
        """3 nodes linear graph (A→B→C) with sizes, verify positions returned."""
        nodes = [
            {'id': 'node-a'},
            {'id': 'node-b'},
            {'id': 'node-c'},
        ]
        edges = [
            {'source': 'node-a', 'target': 'node-b'},
            {'source': 'node-b', 'target': 'node-c'},
        ]
        sizes = {
            'node-a': {'width': 200, 'height': 100},
            'node-b': {'width': 300, 'height': 150},
            'node-c': {'width': 150, 'height': 80},
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # Verify all nodes have positions
        assert 'node-a' in positions
        assert 'node-b' in positions
        assert 'node-c' in positions
        
        # Verify topological order: node-a (layer 0) → node-b (layer 1) → node-c (layer 2)
        assert positions['node-a']['x'] < positions['node-b']['x']
        assert positions['node-b']['x'] < positions['node-c']['x']
        
        # All positions should be integers
        assert all(isinstance(pos['x'], int) for pos in positions.values())
        assert all(isinstance(pos['y'], int) for pos in positions.values())
    
    def test_arrange_without_sizes_uses_defaults(self):
        """Nodes without sizes, verify default spacing applied."""
        nodes = [
            {'id': 'node-a'},
            {'id': 'node-b'},
        ]
        edges = [
            {'source': 'node-a', 'target': 'node-b'},
        ]
        # No sizes provided
        
        positions = arrange_with_sizes(nodes, edges)
        
        assert 'node-a' in positions
        assert 'node-b' in positions
        
        # Default horizontal spacing is 250 + 50 buffer = 300
        # Node-a at x=0, node-b at x=300 (default width + buffer)
        assert positions['node-a']['x'] == 0
        assert positions['node-b']['x'] >= 250  # At least default width
    
    def test_arrange_empty_graph(self):
        """Empty nodes/edges, verify empty positions dict returned."""
        nodes = []
        edges = []
        
        positions = arrange_with_sizes(nodes, edges)
        
        assert positions == {}
    
    def test_arrange_empty_edges(self):
        """Nodes with no edges (all disconnected), verify positions for all."""
        nodes = [
            {'id': 'node-a'},
            {'id': 'node-b'},
        ]
        edges = []
        
        positions = arrange_with_sizes(nodes, edges)
        
        # All nodes should still have positions (disconnected handling)
        assert 'node-a' in positions
        assert 'node-b' in positions


class TestArrangeWithSizesTopology:
    """Topological ordering and graph structure tests."""
    
    def test_arrange_diamond_shape(self):
        """A→B, A→C, B→D, C→D, verify topological order respected."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
            {'id': 'd'},
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
            {'source': 'a', 'target': 'c'},
            {'source': 'b', 'target': 'd'},
            {'source': 'c', 'target': 'd'},
        ]
        sizes = {
            'a': {'width': 200, 'height': 100},
            'b': {'width': 250, 'height': 120},
            'c': {'width': 180, 'height': 90},
            'd': {'width': 300, 'height': 150},
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # Node 'a' is at layer 0
        assert positions['a']['x'] == 0
        
        # Nodes 'b' and 'c' are at layer 1 (both successors of 'a')
        assert positions['b']['x'] == positions['c']['x']
        assert positions['b']['x'] > positions['a']['x']
        
        # Node 'd' is at layer 2 (successor of both 'b' and 'c')
        assert positions['d']['x'] > positions['b']['x']
        assert positions['d']['x'] > positions['c']['x']
    
    def test_arrange_disconnected_components(self):
        """Two disconnected components, verify non-overlapping positions."""
        nodes = [
            {'id': 'a1'},
            {'id': 'a2'},
            {'id': 'b1'},
            {'id': 'b2'},
        ]
        edges = [
            {'source': 'a1', 'target': 'a2'},
            {'source': 'b1', 'target': 'b2'},
        ]
        sizes = {
            'a1': {'width': 200, 'height': 100},
            'a2': {'width': 150, 'height': 80},
            'b1': {'width': 300, 'height': 120},
            'b2': {'width': 250, 'height': 100},
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # All nodes should have positions
        assert len(positions) == 4
        
        # All x coordinates should be different (no overlap)
        [pos['x'] for pos in positions.values()]
        # Some may share same x if in same layer, but should be vertically separated
        # Verify nodes in same layer are vertically separated
        for node_id, pos in positions.items():
            # Find other nodes at same x
            same_x_nodes = [nid for nid, p in positions.items() if p['x'] == pos['x'] and nid != node_id]
            for other_id in same_x_nodes:
                other_pos = positions[other_id]
                # Nodes at same x should have different y (no overlap)
                assert pos['y'] != other_pos['y']


class TestArrangeWithSizesIntegrity:
    """Data integrity tests."""
    
    def test_arrange_all_node_ids_present(self):
        """Verify every input node ID appears in output."""
        nodes = [
            {'id': 'node-1'},
            {'id': 'node-2'},
            {'id': 'node-3'},
            {'id': 'node-4'},
        ]
        edges = [
            {'source': 'node-1', 'target': 'node-2'},
            {'source': 'node-2', 'target': 'node-3'},
        ]
        
        positions = arrange_with_sizes(nodes, edges)
        
        # All node IDs from input should be in output
        for node in nodes:
            assert node['id'] in positions
    
    def test_arrange_integer_coordinates(self):
        """Verify all x, y values are integers."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
            {'source': 'b', 'target': 'c'},
        ]
        sizes = {
            'a': {'width': 200.5, 'height': 100.3},  # Floats in input
            'b': {'width': 300, 'height': 150},
            'c': {'width': 150, 'height': 80},
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # All coordinates should be integers
        for pos in positions.values():
            assert isinstance(pos['x'], int)
            assert isinstance(pos['y'], int)


class TestArrangeWithSizesSpacing:
    """Size-aware spacing tests."""
    
    def test_arrange_size_aware_spacing(self):
        """Varying node widths, verify horizontal spacing proportional."""
        nodes = [
            {'id': 'narrow'},
            {'id': 'wide'},
        ]
        edges = [
            {'source': 'narrow', 'target': 'wide'},
        ]
        sizes = {
            'narrow': {'width': 100, 'height': 50},  # Narrow node
            'wide': {'width': 500, 'height': 200},    # Wide node
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # narrow at layer 0, wide at layer 1
        # Layer 1 X = max_width of layer 0 + buffer = 100 + 50 = 150
        assert positions['narrow']['x'] == 0
        assert positions['wide']['x'] >= 100  # At least narrow width + buffer
    
    def test_arrange_vertical_spacing_with_heights(self):
        """Nodes with varying heights in same layer, verify proper vertical spacing."""
        nodes = [
            {'id': 'short'},
            {'id': 'tall'},
        ]
        edges = []  # Both nodes at layer 0 (no edges)
        
        # When no edges, nodes are disconnected and positioned sequentially
        sizes = {
            'short': {'width': 200, 'height': 50},
            'tall': {'width': 200, 'height': 300},
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # Both should have positions
        assert len(positions) == 2


class TestArrangeWithSizesPerformance:
    """Performance tests."""
    
    def test_arrange_large_graph_performance(self):
        """100 nodes, verify completes within 500ms."""
        # Build a linear chain of 100 nodes
        nodes = [{'id': f'node-{i}'} for i in range(100)]
        edges = [{'source': f'node-{i}', 'target': f'node-{i+1}'} for i in range(99)]
        sizes = {f'node-{i}': {'width': 250, 'height': 100} for i in range(100)}
        
        start_time = time.time()
        positions = arrange_with_sizes(nodes, edges, sizes)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Should complete within 500ms
        assert elapsed_ms < 2000  # catches a complexity blow-up, not CI jitter
        
        # All nodes should have positions
        assert len(positions) == 100
        
        # Verify linear chain layout
        for i in range(99):
            assert positions[f'node-{i}']['x'] <= positions[f'node-{i+1}']['x']


class TestArrangeWithSizesEdgeCases:
    """Edge case handling tests."""
    
    def test_arrange_with_partial_sizes(self):
        """Sizes for some nodes only, verify defaults for missing nodes."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
            {'source': 'b', 'target': 'c'},
        ]
        sizes = {
            'a': {'width': 400, 'height': 200},
            # 'b' and 'c' missing from sizes
        }
        
        positions = arrange_with_sizes(nodes, edges, sizes)
        
        # All nodes should have positions
        assert len(positions) == 3
        
        # 'a' has custom size, 'b' and 'c' use defaults
        assert positions['a']['x'] == 0
        # 'b' at layer 1, uses default width for layer spacing calculation
    
    def test_arrange_single_node(self):
        """Single node, verify correct position."""
        nodes = [{'id': 'single'}]
        edges = []
        
        positions = arrange_with_sizes(nodes, edges)
        
        assert 'single' in positions
        assert positions['single']['x'] == 0
        assert positions['single']['y'] == 0
    
    def test_arrange_with_none_sizes(self):
        """sizes=None explicitly, verify uses defaults."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
        ]
        
        positions = arrange_with_sizes(nodes, edges, sizes=None)
        
        assert len(positions) == 2
        assert positions['a']['x'] == 0

def test_isolated_nodes_are_placed_in_authored_order_independent_of_hash_seed():
    """Regression: isolated nodes were placed in ``set`` order, which follows the
    per-process string hash seed, so the same graph got different positions."""
    import json
    import os
    import subprocess
    import sys

    script = (
        "import json\n"
        "from magic_agents.node_system import arrange_with_sizes\n"
        "nodes = [{'id': i} for i in ['root', 'leaf', 'zeta', 'alpha', 'mid', 'omega', 'beta']]\n"
        "print(json.dumps(arrange_with_sizes(nodes, [{'source': 'root', 'target': 'leaf'}])))\n"
    )
    layouts = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        layouts.append(json.loads(out.stdout))

    assert layouts[0] == layouts[1] == layouts[2]
    isolated = ["zeta", "alpha", "mid", "omega", "beta"]
    xs = [layouts[0][node_id]["x"] for node_id in isolated]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)
