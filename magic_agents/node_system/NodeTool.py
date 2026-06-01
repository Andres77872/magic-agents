from typing import Any

from magic_agents.models.factory.Nodes.ToolNodeModel import ToolNodeModel
from magic_agents.node_system.Node import Node


class NodeTool(Node):
    """Schema-only tool provider node.

    NodeTool emits a validated raw tool schema for NodeLLM. It intentionally has
    no callable/executor surface; clients execute emitted tool calls.
    """

    DEFAULT_OUTPUT_HANDLE = "handle-tool-definition"

    def __init__(self, data: ToolNodeModel, node_id: str, debug: bool = False, handles: dict | None = None, **kwargs: Any):
        super().__init__(debug=debug, node_id=node_id, **kwargs)
        handles = handles or {}
        self.OUTPUT_HANDLE_TOOL_DEFINITION = handles.get("output", self.DEFAULT_OUTPUT_HANDLE)
        self.tool = data.tool

    async def process(self, chat_log):
        yield self.yield_static(self.tool, content_type=self.OUTPUT_HANDLE_TOOL_DEFINITION)
