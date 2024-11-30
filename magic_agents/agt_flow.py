import json
import uuid
from typing import Callable

from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import (NodeChat,
                                      NodeLLM,
                                      NodeEND,
                                      NodeText,
                                      NodeUserInput, NodeFetch,
                                      )
from magic_agents.node_system.NodeParser import NodeParser
from magic_agents.util.const import HANDLE_VOID


# Updated graph execution function to accumulate inputs for each node
async def execute_graph(graph: dict, load_chat: Callable, get_client: Callable):
    # nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph["edges"]
    nodes = {}
    chat_completion_log = {
        'id_chat': 'chat_id',
        'id_app': 'magic-research',
        'chat_system': '',
        'execution': [],
        'execution_time': 0,
    }

    chat_log = ModelAgentRunLog(**chat_completion_log)

    for node_data in graph['nodes']:
        nodo_tipo = node_data['type']
        if nodo_tipo == 'chat':
            nodes[node_data['id']] = NodeChat(load_chat=load_chat,
                                              debug=node_data.get('debug', False),
                                              **node_data['data'])
        elif nodo_tipo == 'llm':
            nodes[node_data['id']] = NodeLLM(data=node_data['data'],
                                             stream=node_data['data']['stream'],
                                             get_client=get_client,
                                             debug=node_data.get('debug', False))
        elif nodo_tipo == 'end':
            nodes[node_data['id']] = NodeEND(debug=node_data.get('debug', False))
        elif nodo_tipo == 'text':
            nodes[node_data['id']] = NodeText(text=node_data['data']['content'],
                                              debug=node_data.get('debug', False))
        elif nodo_tipo == 'user_input':
            nodes[node_data['id']] = NodeUserInput(text=node_data['data']['content'],
                                                   debug=node_data.get('debug', False))
        elif nodo_tipo == 'findit':
            pass
            # nodes[node_data['id']] = NodeFindit()
        elif nodo_tipo == 'generator':
            pass
            # nodes[node_data['id']] = NodeGenerator()
        elif nodo_tipo == 'merger':
            pass
            # nodes[node_data['id']] = NodeMerger(text=node_data['data']['content'])
        elif nodo_tipo == 'parser':
            nodes[node_data['id']] = NodeParser(debug=node_data.get('debug', False),
                                                **node_data['data'])
        elif nodo_tipo == 'browsing':
            pass
            # nodes[node_data['id']] = NodeBrowsing()
        elif nodo_tipo == 'arxiv':
            pass
            # nodes[node_data['id']] = NodeArxiv(extras=graph.get('extras'))
        elif nodo_tipo == 'fetch':
            nodes[node_data['id']] = NodeFetch(debug=node_data.get('debug', False),
                                               **node_data['data'])
        elif nodo_tipo == 'void':
            nodes[node_data['id']] = lambda x: None

    # First, execute all source nodes to accumulate inputs
    for edge in edges:
        source_id = edge["source"]
        target_id = edge["target"]

        # if not results[source_id]:  # Execute source node if not already done
        print('RUN source', nodes[source_id].__class__.__name__)
        print('RUN target', nodes[target_id].__class__.__name__)
        if nodes[source_id].__class__.__name__ == 'NodeLLM':
            async for i in nodes[source_id](chat_log):
                yield i
            print('NODE SOURCE', nodes[source_id].generated)
            nodes[target_id].add_parent({'NodeLLM': nodes[source_id].generated}, edge['targetHandle'])
        elif nodes[source_id].__class__.__name__ == 'NodeEND':
            print('NODE END')
            async for i in nodes[source_id](chat_log):
                yield i
        elif nodes[target_id].__class__.__name__ == 'function':
            nodes[source_id](chat_log)
        else:
            nodes[target_id].add_parent(await nodes[source_id](chat_log), edge['targetHandle'])


async def run_agent(message: str,
                    agt: dict,
                    load_chat: Callable,
                    get_client: Callable,
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

    print('VOID EDGES', void_id)

    r = execute_graph(agt, load_chat, get_client)
    async for i in r:
        yield i
