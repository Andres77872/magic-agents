"""Tests for MCPToolNamespace prefixing and collision detection."""
import pytest
from magic_agents.mcp.namespace import MCPToolNamespace
from magic_agents.mcp.errors import MCPToolNameCollisionError


def test_prefix_normalization_lowercase():
    """Test prefix normalization converts to lowercase."""
    ns = MCPToolNamespace(prefix="FileSystem")
    assert ns.prefix == "filesystem"


def test_prefix_normalization_hyphen_to_underscore():
    """Test prefix normalization converts hyphens to underscores."""
    ns = MCPToolNamespace(prefix="mcp-server-filesystem")
    assert ns.prefix == "mcp_server_filesystem"


def test_prefix_normalization_spaces_to_underscore():
    """Test prefix normalization converts spaces to underscores."""
    ns = MCPToolNamespace(prefix="My Server")
    assert ns.prefix == "my_server"


def test_prefix_normalization_remove_special_chars():
    """Test prefix normalization removes special characters."""
    ns = MCPToolNamespace(prefix="File@System#123")
    assert ns.prefix == "filesystem123"


def test_prefix_normalization_strip_underscores():
    """Test prefix normalization strips leading/trailing underscores."""
    ns = MCPToolNamespace(prefix="_FileSystem_")
    assert ns.prefix == "filesystem"


def test_prefix_normalization_empty_fallback():
    """Test empty prefix falls back to 'mcp'."""
    ns = MCPToolNamespace(prefix="")
    assert ns.prefix == "mcp"


def test_apply_prefix_single_tool():
    """Test prefixing a single tool."""
    tools = [
        {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}}
    ]
    
    ns = MCPToolNamespace(prefix="filesystem")
    mapped = ns.apply_prefix(tools)
    
    assert len(mapped) == 1
    assert mapped[0].local_name == "filesystem_read_file"
    assert mapped[0].remote_name == "read_file"
    assert mapped[0].description == "Read a file"


def test_apply_prefix_multiple_tools():
    """Test prefixing multiple tools."""
    tools = [
        {"name": "read_file", "description": "Read", "inputSchema": {}},
        {"name": "write_file", "description": "Write", "inputSchema": {}},
        {"name": "list_directory", "description": "List", "inputSchema": {}},
    ]
    
    ns = MCPToolNamespace(prefix="fs")
    mapped = ns.apply_prefix(tools)
    
    assert len(mapped) == 3
    assert mapped[0].local_name == "fs_read_file"
    assert mapped[1].local_name == "fs_write_file"
    assert mapped[2].local_name == "fs_list_directory"


def test_apply_prefix_collision_detection():
    """Test collision detection within namespace."""
    # Two tools with same remote name would cause collision after prefixing
    tools = [
        {"name": "read", "description": "Read 1", "inputSchema": {}},
        {"name": "read", "description": "Read 2", "inputSchema": {}},  # Duplicate!
    ]
    
    ns = MCPToolNamespace(prefix="fs")
    
    with pytest.raises(MCPToolNameCollisionError) as exc_info:
        ns.apply_prefix(tools)
    
    assert "fs_read" in str(exc_info.value)


def test_filter_tools_no_filtering():
    """Test filter_tools with no allowlist/denylist."""
    tools = [
        {"name": "read_file", "description": "Read"},
        {"name": "write_file", "description": "Write"},
        {"name": "delete_file", "description": "Delete"},
    ]
    
    ns = MCPToolNamespace(prefix="fs")
    filtered = ns.filter_tools(tools)
    
    assert len(filtered) == 3


def test_filter_tools_allowlist():
    """Test filter_tools with allowlist."""
    tools = [
        {"name": "read_file", "description": "Read"},
        {"name": "write_file", "description": "Write"},
        {"name": "delete_file", "description": "Delete"},
    ]
    
    ns = MCPToolNamespace(prefix="fs", allowed=["read_file"])
    filtered = ns.filter_tools(tools)
    
    assert len(filtered) == 1
    assert filtered[0]["name"] == "read_file"


def test_filter_tools_denylist():
    """Test filter_tools with denylist."""
    tools = [
        {"name": "read_file", "description": "Read"},
        {"name": "write_file", "description": "Write"},
        {"name": "delete_file", "description": "Delete"},
    ]
    
    ns = MCPToolNamespace(prefix="fs", denied=["delete_file"])
    filtered = ns.filter_tools(tools)
    
    assert len(filtered) == 2
    assert "delete_file" not in [t["name"] for t in filtered]


def test_filter_tools_allowlist_and_denylist():
    """Test filter_tools with both allowlist and denylist."""
    tools = [
        {"name": "read_file", "description": "Read"},
        {"name": "write_file", "description": "Write"},
        {"name": "delete_file", "description": "Delete"},
    ]
    
    # Allowlist wins - if tool not in allowlist, it's excluded
    ns = MCPToolNamespace(prefix="fs", allowed=["read_file", "write_file"], denied=["write_file"])
    filtered = ns.filter_tools(tools)
    
    # write_file is in allowlist but also in denylist
    # Current logic: check denylist first, then allowlist
    assert len(filtered) == 1  # Only read_file passes


def test_get_remote_name():
    """Test getting remote name from local name."""
    tools = [{"name": "read_file", "description": "Read", "inputSchema": {}}]
    
    ns = MCPToolNamespace(prefix="fs")
    ns.apply_prefix(tools)
    
    assert ns.get_remote_name("fs_read_file") == "read_file"
    assert ns.get_remote_name("unknown_tool") is None


def test_get_local_name():
    """Test getting local name from remote name."""
    ns = MCPToolNamespace(prefix="filesystem")
    
    assert ns.get_local_name("read_file") == "filesystem_read_file"


def test_schema_preserved_in_mapped_tool():
    """Test that inputSchema is preserved without transformation."""
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "encoding": {"type": "string", "default": "utf-8"}
        },
        "required": ["path"]
    }
    
    tools = [{"name": "read", "description": "Read", "inputSchema": input_schema}]
    
    ns = MCPToolNamespace(prefix="fs")
    mapped = ns.apply_prefix(tools)
    
    # Schema should be preserved exactly
    assert mapped[0].input_schema == input_schema
    assert mapped[0].input_schema["required"] == ["path"]