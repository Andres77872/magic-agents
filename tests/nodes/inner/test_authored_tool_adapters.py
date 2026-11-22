import asyncio
import json
from types import SimpleNamespace

import pytest

from magic_agents.agt_flow import build, _assign_tool_handles
from magic_agents.models.factory.Nodes.InnerNodeModel import InnerNodeModel
from magic_agents.models.factory.Nodes.PythonExecNodeModel import PythonExecNodeModel
from magic_agents.node_system.NodePythonExec import NodePythonExec
from magic_agents.util.aio import wait_for


PARAMETERS = {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x'], 'additionalProperties': False}
LOG = SimpleNamespace(id_chat='test-tools', id_thread=None, id_user='local-review', run_id='parent', flow_state={'region': 'west'})


def child_graph(template='{{ value.x * 2 }}'):
    return {'nodes': [
        {'id': 'input', 'type': 'user_input', 'data': {}},
        {'id': 'calculate', 'type': 'parser', 'data': {'template': template}},
        {'id': 'end', 'type': 'end', 'data': {}},
    ], 'edges': [
        {'source': 'input', 'target': 'calculate', 'sourceHandle': 'handle_user_message', 'targetHandle': 'value'},
        {'source': 'calculate', 'target': 'end', 'sourceHandle': 'handle_parser_output', 'targetHandle': 'handle_flow_input'},
    ]}


async def graph_tool(child=None, **config):
    graph = build({'nodes': [{'id': 'flow-tool', 'type': 'inner', 'data': {
        'magic_flow': child or child_graph(), 'tool_mode': True, 'tool_name': 'double',
        'tool_parameters': PARAMETERS, **config,
    }}, {'id': 'root-input', 'type': 'user_input', 'data': {}}, {'id': 'root-end', 'type': 'end', 'data': {}}], 'edges': [{'source': 'root-input', 'target': 'root-end', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'}]}, message='ignored')
    node = graph.nodes['flow-tool']
    events = [event async for event in node.process(LOG) if event['type'] == 'handle-tool-definition']
    assert len(events) == 1
    assert events[0]['type'] == 'handle-tool-definition'
    return events[0]['content']['content']


@pytest.mark.asyncio
async def test_graph_tool_repeated_and_parallel_calls_use_fresh_children():
    tool = await graph_tool()
    constructed = []
    factory = tool._node._tool_graph_factory
    def capture(*args):
        graph = factory(*args)
        constructed.append(graph)
        return graph
    tool._node._tool_graph_factory = capture
    assert await tool(x=2) == '4'
    assert await asyncio.gather(*(tool(x=x) for x in range(6))) == [str(x * 2) for x in range(6)]
    assert len({id(g.nodes['input']) for g in constructed}) == 7
    assert tool._node.inner_graph.nodes['input']._text == ''


@pytest.mark.asyncio
async def test_graph_tool_arguments_are_not_parent_message_and_parent_extras_are_copied():
    tool = await graph_tool(parent_state_mapping={'selected_region': 'region'})
    assert tool._extras == {'selected_region': 'west'}
    assert await tool(x=8) == '16'
    assert LOG.flow_state == {'region': 'west'}


@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [{}, {'x': True}, {'x': '2'}, {'x': 2, 'code': 'unexpected'}])
async def test_graph_tool_validates_arguments_before_building_child(arguments):
    tool = await graph_tool()
    def should_not_build(*args):
        pytest.fail('Invalid arguments must not execute a graph')
    tool._node._tool_graph_factory = should_not_build
    with pytest.raises(ValueError, match='Invalid tool arguments'):
        await tool(**arguments)


@pytest.mark.asyncio
async def test_graph_tool_has_no_public_child_output_and_returns_only_end():
    child = child_graph('{{ value | tojson }}')
    child['nodes'].append({'id': 'internal-message', 'type': 'send_message', 'data': {'message': 'INTERNAL SECRET STEP'}})
    tool = await graph_tool(child)
    assert json.loads(await tool(x=7)) == {'x': 7}
    assert tool.drain_events() == []


@pytest.mark.asyncio
@pytest.mark.parametrize('timeout', [True, False])
async def test_cancelled_or_timed_out_graph_tools_leave_no_owned_tasks(timeout):
    tool = await graph_tool(tool_timeout=0.02 if timeout else 10)
    stopped = asyncio.Event()
    started = asyncio.Event()
    factory = tool._node._tool_graph_factory
    def slow_graph(*args):
        graph = factory(*args)
        async def wait_forever(log):
            started.set()
            try:
                await asyncio.Event().wait()
                yield {}
            finally:
                stopped.set()
        graph.nodes['calculate'].process = wait_forever
        return graph
    tool._node._tool_graph_factory = slow_graph
    task = asyncio.create_task(tool(x=1))
    await started.wait()
    if not timeout:
        task.cancel()
    with pytest.raises(TimeoutError if timeout else asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith('node_') and not t.done()]


@pytest.mark.asyncio
async def test_graph_tool_failure_is_an_error_not_an_empty_success():
    from magic_agents.hooks.invocation_control import OperationFailure
    tool = await graph_tool(child_graph('{{ missing.invalid() }}'))
    with pytest.raises(OperationFailure, match="Inner tool failed: sub-flow step 'calculate' failed: UndefinedError") as raised:
        await tool(x=1)
    error = raised.value.outcome['error']
    assert error['code'] == 'INNER_FLOW_FAILED' and error['retryable'] is False
    assert error['details']['failed_nodes'] == ['calculate']
    assert error['details']['child_errors'][0] == {
        'node_path': ['calculate'], 'error_type': 'UndefinedError', 'error_message': "'missing' is undefined"}
    assert error['details']['partial_content'] == ''


@pytest.mark.asyncio
async def test_graph_tool_recovered_child_diagnostic_is_not_a_failure():
    """A child diagnostic whose node still completed (recovered) is not a tool failure."""
    tool = await graph_tool()
    factory = tool._node._tool_graph_factory
    def with_diagnostic(*args):
        graph = factory(*args)
        parser = graph.nodes['calculate']
        async def diagnose_then_answer(log):
            yield parser.yield_debug_error(error_type='JSONParseError', error_message='not JSON')
            yield parser.yield_static('recovered value', content_type='handle_parser_output')
        parser.process = diagnose_then_answer
        return graph
    tool._node._tool_graph_factory = with_diagnostic
    assert await tool(x=1) == 'recovered value'


@pytest.mark.asyncio
async def test_fixed_python_tool_invokes_author_code_and_preserves_falsy_json():
    node = NodePythonExec(data=PythonExecNodeModel(
        code='def run(handler):\n    return {"doubled": handler["x"] * 2, "valid": False}',
        tool_mode=True, tool_name='double_python', tool_parameters=PARAMETERS,
        safety_mode='restricted_builtins',
    ), node_id='fixed')
    events = [event async for event in node.process(LOG) if event['type'] == 'handle-tool-definition']
    tool = events[0]['content']['content']
    assert events[0]['type'] == 'handle-tool-definition'
    assert not node.accepts_dynamic_input_handles
    assert json.loads(await tool(x=0)) == {'doubled': 0, 'valid': False}
    assert [json.loads(value)['doubled'] for value in await asyncio.gather(tool(x=3), tool(x=9))] == [6, 18]
    with pytest.raises(ValueError, match='Invalid tool arguments'):
        await tool(x=1, code='def run(handler): return "override"')


@pytest.mark.asyncio
async def test_fixed_python_errors_are_reported_as_tool_failures():
    node = NodePythonExec(data=PythonExecNodeModel(code='def run(handler): return 1 / 0', tool_mode=True,
        safety_mode='restricted_builtins'), node_id='broken')
    tool = [event async for event in node.process(LOG) if event['type'] == 'handle-tool-definition'][0]['content']['content']
    with pytest.raises(RuntimeError, match='division by zero'):
        await tool()


@pytest.mark.parametrize('mode,code,expected', [(None, None, 'handle-tool-definition'), (None, 'def run(handler): return 9', 'handle-python_exec-result'), (False, 'def run(handler): return 9', 'handle-python_exec-result')])
@pytest.mark.asyncio
async def test_python_existing_modes_remain_compatible(mode, code, expected):
    node = NodePythonExec(data=PythonExecNodeModel(tool_mode=mode, code=code, safety_mode='restricted_builtins'), node_id='legacy')
    assert any(event['type'] == expected for event in [event async for event in node.process(LOG)])


@pytest.mark.parametrize('parameters', [{'type': 'array'}, {'type':'object','properties':{'x':{'type':'bogus'}}}, {'type':'object','$ref':'https://example.com/schema.json'}])
def test_invalid_custom_tool_schemas_fail_at_build(parameters):
    with pytest.raises(ValueError):
        InnerNodeModel(tool_mode=True, magic_flow=child_graph(), tool_parameters=parameters)
    with pytest.raises(ValueError):
        PythonExecNodeModel(tool_mode=True, code='def run(handler): return handler', tool_parameters=parameters)


def test_graph_tool_requires_terminal_result_contract():
    with pytest.raises(ValueError, match='End node'):
        InnerNodeModel(tool_mode=True, magic_flow={'nodes': [], 'edges': []})


def test_graph_tools_reject_unresolved_client_tool_handoffs():
    child = child_graph()
    child['nodes'].extend([{'id': 'schema', 'type': 'node_tool'}, {'id': 'llm', 'type': 'llm'}])
    child['edges'].append({'source': 'schema', 'target': 'llm'})
    with pytest.raises(ValueError, match='schema-only client tools'):
        InnerNodeModel(tool_mode=True, magic_flow=child)


@pytest.mark.asyncio
@pytest.mark.parametrize('fails', [True, False])
async def test_graph_tool_keeps_usage_on_success_or_error_without_publishing_child_text(fails):
    from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
    tool = await graph_tool()
    factory = tool._node._tool_graph_factory
    def with_provider(*args):
        graph = factory(*args)
        parser = graph.nodes['calculate']
        async def emit_usage(log):
            yield parser.yield_static(ChatCompletionModel(
                id='child-response', model='test-model', choices=[ChoiceModel(delta=DeltaModel(content='private reasoning'))],
                usage={'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8},
            ), content_type='content')
            if fails:
                raise ValueError('simulated child failure')
            yield parser.yield_static('terminal result', content_type='handle_parser_output')
        parser.process = emit_usage
        return graph
    tool._node._tool_graph_factory = with_provider
    if fails:
        from magic_agents.hooks.invocation_control import OperationFailure
        with pytest.raises(OperationFailure, match='Inner tool failed: .*simulated child failure'):
            await tool(x=1)
    else:
        assert await tool(x=1) == 'terminal result'
    events = tool.drain_events()
    assert len(events) == 1
    assert events[0]['data']['usage']['total_tokens'] == 8
    assert events[0]['source_node_path'] == ['flow-tool', 'calculate']
    assert 'private reasoning' not in json.dumps(events)
    assert tool.drain_events() == []


@pytest.mark.asyncio
async def test_fixed_python_subprocess_cancellation_terminates_child():
    import multiprocessing
    from magic_agents.node_system.authored_tools import FixedPythonTool
    baseline = {p.pid for p in multiprocessing.active_children()}
    tool = FixedPythonTool(code='import time\ndef run(handler):\n    time.sleep(20)\n    return 1', name='slow',
        description='wait', parameters=None, node_id='slow', safety_mode='subprocess', timeout=25, max_output_chars=100)
    task = asyncio.create_task(tool())
    for _ in range(100):
        if {p.pid for p in multiprocessing.active_children()} - baseline:
            break
        await asyncio.sleep(0.01)
    else:
        task.cancel()
        pytest.fail('Python subprocess did not start')
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait_for(task, timeout=2)
    assert not ({p.pid for p in multiprocessing.active_children()} - baseline)


def test_assign_tool_handles_preserves_explicit_slots_and_resolves_new_custom_sources():
    nodes = [{'id':'llm','type':'llm','data':{'handles':{'tool_prefix':'custom-'}}},
        {'id':'flow','type':'inner','data':{'tool_mode':True,'handles':{'tool':'flow-tool'}}},
        {'id':'python','type':'python_exec','data':{'tool_mode':True,'code':'def run(handler): return handler'}}]
    edges = [{'source':'flow','target':'llm','targetHandle':'custom-definition-0'}, {'source':'python','target':'llm'}]
    _assign_tool_handles(nodes, edges)
    assert edges[0]['sourceHandle']=='flow-tool'
    assert edges[1]['targetHandle']=='custom-definition-1'
