import copy
import json

import pytest

from magic_agents.agt_flow import build
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.util.graph_validator import validate_edge_fan_in_compatibility, validate_edge_target_handles


TOOL_GRAPH = {'type': 'chat',
 'timeout': 90,
 'debug': True,
 'debug_config': {'redact_sensitive': True,
                  'capture_inputs': True,
                  'capture_outputs': True,
                  'max_payload_length': 16000},
 'nodes': [{'id': 'input', 'type': 'user_input', 'data': {}},
           {'id': 'llm',
            'type': 'llm',
            'data': {'stream': True,
                     'temperature': 0,
                     'max_tokens': 900,
                     'iterate': True,
                     'handles': {'tool_prefix': 'tools/'},
                     'agent_config': {'max_iterations': 10,
                                      'wall_clock_timeout': 75,
                                      'per_tool_timeout': 15,
                                      'max_parallel_tools': 2,
                                      'max_output_chars': 12000,
                                      'deduplicate': False}}},
           {'id': 'end', 'type': 'end', 'data': {}},
           {'id': 'python',
            'type': 'python_exec',
            'data': {'tool_mode': True,
                     'code': 'def run(handler):\n'
                             '    return {"product": int(handler["left"]) * int(handler["right"])}',
                     'tool_name': 'multiply',
                     'tool_description': 'Execute the fixed implementation with exactly the '
                                         'provided arguments.',
                     'tool_parameters': {'type': 'object',
                                         'properties': {'left': {'type': 'integer'},
                                                        'right': {'type': 'integer'}},
                                         'required': ['left', 'right'],
                                         'additionalProperties': False},
                     'safety_mode': 'subprocess',
                     'timeout': 15,
                     'max_output_chars': 12000,
                     'handles': {'output': 'python-callable'}}},
           {'id': 'quote',
            'type': 'inner',
            'data': {'tool_mode': True,
                     'tool_name': 'quote_order',
                     'tool_description': 'Compute subtotal and total from quantity, unit_price and '
                                         'fee using the child graph.',
                     'tool_parameters': {'type': 'object',
                                         'properties': {'quantity': {'type': 'integer'},
                                                        'unit_price': {'type': 'integer'},
                                                        'fee': {'type': 'integer'}},
                                         'required': ['quantity', 'unit_price', 'fee'],
                                         'additionalProperties': False},
                     'tool_timeout': 10,
                     'handles': {'tool': 'quote-callable'},
                     'magic_flow': {'type': 'graph',
                                    'timeout': 5,
                                    'nodes': [{'id': 'child-input',
                                               'type': 'user_input',
                                               'data': {},
                                               'position': {'x': 0, 'y': 0}},
                                              {'id': 'calculate',
                                               'type': 'parser',
                                               'data': {'text': '{{ {"subtotal": request.quantity '
                                                                '* request.unit_price, "total": '
                                                                'request.quantity * '
                                                                'request.unit_price + request.fee} '
                                                                '| tojson }}'},
                                               'position': {'x': 300, 'y': 0}},
                                              {'id': 'child-end',
                                               'type': 'end',
                                               'data': {},
                                               'position': {'x': 600, 'y': 0}}],
                                    'edges': [{'id': 'child-input-handle_user_message-calculate-request',
                                               'source': 'child-input',
                                               'sourceHandle': 'handle_user_message',
                                               'target': 'calculate',
                                               'targetHandle': 'request'},
                                              {'id': 'calculate-handle_parser_output-child-end-handle_flow_input',
                                               'source': 'calculate',
                                               'sourceHandle': 'handle_parser_output',
                                               'target': 'child-end',
                                               'targetHandle': 'handle_flow_input'}]}}}],
 'edges': [{'id': 'llm-handle_generated_content-end-handle_flow_input',
            'source': 'llm',
            'sourceHandle': 'handle_generated_content',
            'target': 'end',
            'targetHandle': 'handle_flow_input'},
           {'id': 'python-python-callable-llm-tools/definition-0',
            'source': 'python',
            'sourceHandle': 'python-callable',
            'target': 'llm',
            'targetHandle': 'tools/definition-0'},
           {'id': 'quote-quote-callable-llm-tools/definition-0',
            'source': 'quote',
            'sourceHandle': 'quote-callable',
            'target': 'llm',
            'targetHandle': 'tools/definition-0'}]}

@pytest.mark.asyncio
@pytest.mark.parametrize('prefix', ['handle-tool-', 'tools/'])
@pytest.mark.parametrize('reverse', [False, True])
async def test_real_tool_edges_on_one_slot_are_collected_and_executable(monkeypatch, prefix, reverse):
    definition = copy.deepcopy(TOOL_GRAPH)
    # Exercise real source nodes, graph-tool factory and dispatcher while keeping
    # provider nondeterminism outside this routing regression.
    definition['nodes'] = [n for n in definition['nodes'] if n['id'] not in {'client','chat','system'}]
    definition['edges'] = [e for e in definition['edges'] if e['source'] in {'python','quote','llm'}]
    for node in definition['nodes']:
        if node['id'] == 'llm': node['data']['handles']['tool_prefix'] = prefix
    for edge in definition['edges']:
        if edge['target'] == 'llm': edge['targetHandle'] = prefix+'definition-0'
    if reverse: definition['edges'].reverse()
    observed = {}
    async def process(self, chat_log):
        schemas, functions, _ = await self._collect_tools()
        observed['names'] = set(functions)
        assert set(functions) == {'multiply','quote_order'}
        observed['python'] = json.loads(await functions['multiply'](left=6,right=7))
        observed['graph'] = json.loads(await functions['quote_order'](quantity=2,unit_price=42,fee=7))
        yield self.yield_static(json.dumps(observed['graph']), content_type=self.OUTPUT_HANDLE_GENERATED)
    monkeypatch.setattr(NodeLLM,'process',process)
    graph = build(definition,message='test')
    errors = validate_edge_fan_in_compatibility(graph.nodes,graph.edges,mode='strict')
    assert not errors, errors
    assert not validate_edge_target_handles(graph.nodes, graph.edges, mode='strict')
    events = [e async for e in execute_graph_reactive(graph)]
    failures = [e for e in events if e.get('type')=='debug' and e.get('content',{}).get('error_type')]
    assert not failures, failures
    assert observed == {'names':{'multiply','quote_order'},'python':{'product':42},'graph':{'subtotal':84,'total':91}}


@pytest.mark.asyncio
async def test_repeated_edge_and_bypass_do_not_duplicate_or_invent_tools():
    definition={'nodes':[{'id':'llm','type':'llm','data':{'handles':{'tool_prefix':'tools/'}}},
        {'id':'one','type':'node_tool','data':{'tool':{'type':'function','function':{'name':'one','description':'Tool one','parameters':{'type':'object','properties':{}}}}}},
        {'id':'two','type':'node_tool','data':{'tool':{'type':'function','function':{'name':'two','description':'Tool two','parameters':{'type':'object','properties':{}}}}}}],
        'edges':[{'id':'one','source':'one','target':'llm','sourceHandle':'handle-tool-definition','targetHandle':'tools/definition-0'},
                 {'id':'two','source':'two','target':'llm','sourceHandle':'handle-tool-definition','targetHandle':'tools/definition-0'}]}
    definition['nodes'].extend([{'id':'input','type':'user_input','data':{}}, {'id':'end','type':'end','data':{}}])
    definition['edges'].append({'id':'root','source':'input','target':'end','sourceHandle':'handle_user_message','targetHandle':'handle_flow_input'})
    graph=build(definition,message='test')
    dispatcher=GraphEventDispatcher(graph.nodes,graph.edges)
    one={'type':'function','function':{'name':'one','description':'Tool one','parameters':{'type':'object','properties':{}}}}
    await dispatcher.dispatch_input('llm','tools/definition-0',one,edge_id='one')
    await dispatcher.dispatch_input('llm','tools/definition-0',copy.deepcopy(one),edge_id='one')
    await dispatcher.get_tracker('llm').receive_bypass(edge_id='two')
    schemas,_,_=await graph.nodes['llm']._collect_tools()
    assert schemas == [one]
