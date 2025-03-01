import json
import logging
import uuid
from typing import Callable, Dict, Any, AsyncGenerator, Optional, Union

from pydantic import BaseModel, Field, ValidationError

from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import (
    NodeChat,
    NodeLLM,
    NodeEND,
    NodeText,
    NodeUserInput,
    NodeFetch,
    NodeClientLLM,
    NodeSendMessage,
    NodeParser
)
from magic_agents.util.const import HANDLE_VOID

logger = logging.getLogger(__name__)


class NodeTypes:
    CHAT = 'chat'
    LLM = 'llm'
    END = 'end'
    TEXT = 'text'
    USER_INPUT = 'user_input'
    PARSER = 'parser'
    FETCH = 'fetch'
    CLIENT = 'client'
    SEND_MESSAGE = 'send_message'
    VOID = 'void'


class GraphNode(BaseModel):
    type: str
    id: str
    data: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = HANDLE_VOID


class Graph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    debug: bool = False
    extras: Optional[Dict[str, Any]] = None


def create_node(node: GraphNode, load_chat: Callable, debug: bool = False):
    """Factory method to create node instances."""
    extra = {'debug': debug, 'node_id': node.id}
    match node.type:
        case NodeTypes.CHAT:
            return NodeChat(load_chat=load_chat, **extra, **node.data)
        case NodeTypes.LLM:
            return NodeLLM(**extra, **node.data)
        case NodeTypes.END:
            return NodeEND(**extra)
        case NodeTypes.TEXT:
            return NodeText(text=node.data.get('content', ''), **extra)
        case NodeTypes.USER_INPUT:
            return NodeUserInput(text=node.data.get('content', ''), **extra)
        case NodeTypes.PARSER:
            return NodeParser(**extra, **node.data)
        case NodeTypes.FETCH:
            return NodeFetch(**extra, **node.data)
        case NodeTypes.CLIENT:
            return NodeClientLLM(**extra, **node.data)
        case NodeTypes.SEND_MESSAGE:
            return NodeSendMessage(**extra, **node.data)
        case NodeTypes.VOID:
            return NodeEND(**extra)
        case _:
            raise ValueError(f"Unsupported node type: {node.type}")


async def execute_graph(graph_data: dict,
                        load_chat: Callable,
                        id_chat: Optional[Union[int, str]] = None,
                        id_thread: Optional[Union[int, str]] = None,
                        id_user: Optional[Union[int, str]] = None
                        ) -> AsyncGenerator[str, None]:
    try:
        graph = Graph(**graph_data)
    except ValidationError as e:
        logger.error(f"Validation error while loading graph: {e}")
        return

    nodes: Dict[str, Any] = {
        node.id: create_node(node, load_chat, graph.debug) for node in graph.nodes
    }

    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app='magic-research'
    )

    async def process_edge(edge: GraphEdge):
        source_node = nodes[edge.source]
        target_node = nodes[edge.target]

        if not source_node:
            logger.error(f"Source node {edge.source} not found.")
            return
        if not target_node:
            logger.error(f"Target node {edge.target} not found.")
            return

        # Execute source node (only if outputs not already computed)
        if not source_node.outputs:
            async for item in source_node(chat_log):
                if item["type"] == "end":
                    source_node.outputs[edge.sourceHandle] = item["content"]  # outputs must be structured as dict
                elif item["type"] == "content":
                    yield item["content"]

        # Pass output at source_handle to target_handle input
        source_handle = edge.sourceHandle
        target_handle = edge.targetHandle
        target_node.add_parent(source_node.outputs, source_handle, target_handle)

    for edge in graph.edges:
        async for result in process_edge(edge):
            yield result


async def run_agent(
        message: str,
        agt_data: dict,
        load_chat: Callable,
        id_chat: Optional[Union[int, str]] = None,
        id_thread: Optional[Union[int, str]] = None,
        id_user: Optional[Union[int, str]] = None,
        extras: Optional[str] = None) -> AsyncGenerator[str, None]:
    if extras:
        try:
            agt_data['extras'] = json.loads(extras)
        except json.JSONDecodeError:
            logger.error("Invalid extras format. Should be valid JSON.")
            return

    void_id = uuid.uuid4().hex
    agt_data['nodes'].append({'type': NodeTypes.VOID, 'id': void_id})

    # Prepare graph data
    for edge in agt_data['edges']:
        edge.setdefault('targetHandle', HANDLE_VOID)
        if edge['targetHandle'] == HANDLE_VOID:
            edge['target'] = void_id

    for node in agt_data['nodes']:
        if node['type'] in [NodeTypes.USER_INPUT, NodeTypes.CHAT]:
            node['data'] = node.get('data', {})
            node['data']['content' if node['type'] == NodeTypes.USER_INPUT else 'message'] = message
        elif node['type'] == NodeTypes.END:
            agt_data['edges'].append({
                "id": uuid.uuid4().hex,
                "source": node['id'],
                "target": void_id
            })

    async for result in execute_graph(
            graph_data=agt_data,
            id_chat=id_chat,
            id_thread=id_thread,
            id_user=id_user,
            load_chat=load_chat):
        yield result
