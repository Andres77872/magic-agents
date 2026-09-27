from typing import Any, Optional

from pydantic import Field, model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class InnerNodeModel(BaseNodeModel):
    """
    Node model for nested graph execution.
    The JSON definition is the source of truth.

    magic_flow/flow/graph: graph specification dict for an AgentFlowModel to execute.
    """
    magic_flow: Optional[dict[str, Any]] = None
    flow: Optional[dict[str, Any]] = None  # alias for magic_flow
    graph: Optional[dict[str, Any]] = None  # alias for magic_flow
    subgraph: Optional[dict[str, Any]] = None  # alias for magic_flow
    parent_state_mapping: Optional[dict[str, str]] = None  # Static key-path mapping from parent state to child extras
    tool_mode: bool = False
    tool_name: str = "run_flow"
    tool_description: str = "Run the configured flow with structured arguments and return its terminal result."
    tool_parameters: Optional[dict[str, Any]] = None
    tool_timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def resolve_aliases(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if self.magic_flow is None:
            if self.flow is not None:
                self.magic_flow = self.flow
            elif self.graph is not None:
                self.magic_flow = self.graph
            elif self.subgraph is not None:
                self.magic_flow = self.subgraph
        if self.tool_mode:
            from magic_agents.models.tool_schema import function_tool_schema
            function_tool_schema(self.tool_name, self.tool_description, self.tool_parameters)
            graph = self.magic_flow
            if isinstance(graph, dict) and isinstance(graph.get('content'), dict):
                graph = graph['content']
            if not isinstance(graph, dict) or not any(
                node.get('type') == 'end' for node in graph.get('nodes', []) if isinstance(node, dict)
            ):
                raise ValueError("An Inner tool requires a child graph with an End node for its result")
            def check_server_tools(definition):
                if isinstance(definition.get('content'), dict):
                    definition = definition['content']
                nodes = {node.get('id'): node for node in definition.get('nodes', []) if isinstance(node, dict)}
                for edge in definition.get('edges', []):
                    if (nodes.get(edge.get('source'), {}).get('type') == 'node_tool'
                            and nodes.get(edge.get('target'), {}).get('type') == 'llm'):
                        raise ValueError("Inner tools require executable server tools; schema-only client tools cannot resume inside a tool call")
                for node in nodes.values():
                    if node.get('type') != 'inner':
                        continue
                    data = node.get('data', {})
                    child = next((data[key] for key in ('magic_flow', 'flow', 'graph', 'subgraph') if data.get(key) is not None), None)
                    if isinstance(child, dict):
                        check_server_tools(child)
            check_server_tools(graph)
        return self
