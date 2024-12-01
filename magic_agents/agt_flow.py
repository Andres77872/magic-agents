import json
import uuid
from typing import Callable

from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import (NodeChat,
                                      NodeLLM,
                                      NodeEND,
                                      NodeText,
                                      NodeUserInput,
                                      NodeFetch,
                                      NodeClientLLM,
                                      NodeSendMessage,
                                      )
from magic_agents.node_system.NodeParser import NodeParser
from magic_agents.util.const import HANDLE_VOID


async def execute_graph(graph: dict, load_chat: Callable):
    # Prepare nodes and edges
    nodes = {}
    edges = graph["edges"]

    # Initialize the logs for execution tracking
    chat_completion_log = {
        'id_chat': 'chat_id',
        'id_app': 'magic-research',
        'chat_system': '',
        'execution': [],
        'execution_time': 0,
    }
    chat_log = ModelAgentRunLog(**chat_completion_log)

    # Helper function to instantiate node objects
    def create_node(node_data):
        """Create a node instance based on its type."""
        nodo_tipo = node_data['type']
        data = node_data.get('data', {})

        if nodo_tipo == 'chat':
            return NodeChat(load_chat=load_chat, debug=debug, **data)
        elif nodo_tipo == 'llm':
            return NodeLLM(debug=debug, **data)
        elif nodo_tipo == 'end':
            return NodeEND(debug=debug)
        elif nodo_tipo == 'text':
            return NodeText(text=data['content'], debug=debug)
        elif nodo_tipo == 'user_input':
            return NodeUserInput(text=data['content'], debug=debug)
        elif nodo_tipo == 'parser':
            return NodeParser(debug=debug, **data)
        elif nodo_tipo == 'fetch':
            return NodeFetch(debug=debug, **data)
        elif nodo_tipo == 'client':
            return NodeClientLLM(debug=debug, **data)
        elif nodo_tipo == 'send_message':
            return NodeSendMessage(debug=debug, **data)
        elif nodo_tipo == 'void':  # 'Void' node does nothing
            return lambda _: None
        else:
            raise ValueError(f"Unsupported node type: {nodo_tipo}")

    # Initialize all nodes based on the graph
    debug = graph.get('debug', False)
    for nd in graph['nodes']:
        try:
            nodes[nd['id']] = create_node(nd)
        except ValueError as e:
            print(f"Error creating node: {e}")
            continue  # Skip unsupported nodes

    # Helper function to handle node execution
    async def process_node(source_id, target_id, target_handle=None):
        """Executes a source node and forwards its output to the target node."""
        source_node = nodes[source_id]
        target_node = nodes[target_id]

        output = None
        async for item in source_node(chat_log):
            if item['type'] == 'end':
                output = item['content']
            elif item['type'] == 'content':
                yield item['content']

        # Send the output of the source node to the target node
        if target_handle and output:
            target_node.add_parent(output, target_handle)

    # Iterate through edges and connect source to targets
    for edge in edges:
        # Trigger processing of source-to-target execution
        async for result in process_node(edge["source"], edge["target"], edge.get('targetHandle')):
            yield result


async def run_agent(message: str,
                    agt: dict,
                    load_chat: Callable,
                    extras: str = None):
    if extras:
        agt.update({
            'extras': json.loads(extras)
        })
    void_id = uuid.uuid4().hex
    agt['nodes'].append({
        'type': 'void',
        'id': void_id
    })

    for i in agt['edges']:
        if 'targetHandle' not in i:
            i['targetHandle'] = HANDLE_VOID
        if i['targetHandle'] == HANDLE_VOID:
            i['target'] = void_id

    for i in agt['nodes']:
        if i['type'] == 'user_input':
            i['data'] = {'content': message}
        if i['type'] == 'chat':
            i['data'].update({'message': message})
        if i['type'] == 'end':
            agt['edges'].append({
                "id": uuid.uuid4().hex,
                "source": i['id'],
                "target": void_id
            })

    r = execute_graph(agt, load_chat)
    async for i in r:
        yield i
