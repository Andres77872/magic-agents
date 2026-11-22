import pytest

from magic_agents.models.factory.Nodes.ToolNodeModel import ToolNodeModel
from magic_agents.node_system.NodeTool import NodeTool


def tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "client_lookup_order",
            "description": "Ask the client application to look up an order.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


@pytest.mark.asyncio
async def test_node_tool_emits_exact_raw_dict_on_tool_definition_handle():
    model = ToolNodeModel(tool=tool_schema())
    node = NodeTool(data=model, node_id="tool-1")

    outputs = [event async for event in node.process([])]

    tool_outputs = [event for event in outputs if event["type"] == "handle-tool-definition"]
    assert len(tool_outputs) == 1
    assert tool_outputs[0]["content"]["content"] is model.tool


def test_node_tool_exposes_no_server_executor_surface():
    node = NodeTool(data=ToolNodeModel(tool=tool_schema()), node_id="tool-1")

    assert not hasattr(node, "tool_callable")
    assert not hasattr(node, "tool_functions")
    assert not hasattr(node, "client_lookup_order")
