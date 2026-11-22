from importlib.metadata import requires

from magic_agents.node_system.NodeMcp import NodeMcp


def test_install_requires_qualified_http_transport():
    declared = set(requires('magic-agents') or ())
    qualified = NodeMcp.coordination_external_transport_versions

    assert qualified.keys() == {'mcp', 'httpx2', 'mcp-types'}
    assert {f'{name}=={version}' for name, version in qualified.items()} <= declared
