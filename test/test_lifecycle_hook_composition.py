"""Actual Hook child operations and isolated terminal edge deliveries."""
import asyncio
import pytest
from magic_agents.agt_flow import build, run_agent

pytestmark = pytest.mark.asyncio

def edge(identifier, source, target, source_handle, target_handle, hook=None):
    value = {"id":identifier,"source":source,"target":target,"sourceHandle":source_handle,"targetHandle":target_handle}
    if hook:
        value["hooks"] = {"hook_node_id":hook,"enabled":True}
    return value

async def collect(graph):
    return [event async for event in run_agent(graph)]

@pytest.mark.parametrize('loop', [False, True])
@pytest.mark.parametrize('converged', [False, True])
async def test_terminal_delivery_suppresses_only_its_edge_before_target_core(loop, converged):
    nodes=[{"id":"input","type":"user_input"},
           {"id":"blocked","type":"parser","data":{"text":"blocked:{{ value }}"}},
           {"id":"end","type":"end"},
           {"id":"suppress","type":"hook","data":{"lifecycle_event":"onDeliver", "function_template":
                'async def suppress(context, chat_log): return {"action":"outcome","outcome":{"status":"success","content":"delivery answered"}}'}}]
    edges=[]
    if loop:
        nodes.append({"id":"loop","type":"loop"})
        edges += [edge('input-loop','input','loop','handle_user_message','handle_list'),
                  edge('loop-blocked','loop','blocked','handle_item','value','suppress'),
                  edge('blocked-loop','blocked','loop','handle_parser_output','handle_loop'),
                  edge('loop-end','loop','end','handle_end','handle_flow_input')]
        if converged:
            edges.append(edge('active-input','loop','blocked','handle_item','alternative'))
        message='["a","b"]'
    else:
        edges += [edge('input-blocked','input','blocked','handle_user_message','value','suppress'),
                  edge('blocked-end','blocked','end','handle_parser_output','handle_flow_input')]
        if converged:
            edges.append(edge('active-input','input','blocked','handle_user_message','alternative'))
        message='a'
    graph=build({"type":"graph","timeout":2,"nodes":nodes,"edges":edges},message=message)
    await asyncio.wait_for(collect(graph),3)
    blocked=graph.nodes['blocked']
    if converged:
        assert blocked.outputs
        assert 'value' not in blocked.inputs
        assert blocked.inputs['alternative']==('b' if loop else 'a')
    else:
        assert not blocked.outputs and blocked._response is None
    deliveries=[record for record in blocked._invocation_control.records if record.get('operation')=='delivery']
    assert len(deliveries)==(2 if loop else 1)
    assert all(not record['executed'] and record['outcome']['content']=='delivery answered' for record in deliveries)

@pytest.mark.parametrize('tool_mode', [False, True])
async def test_real_inner_child_operation_runs_its_built_graph(tool_mode):
    child_graph={"type":"graph","nodes":[{"id":"inside-input","type":"user_input"},
        {"id":"inside-parser","type":"parser","data":{"text":"inner:{{ value }}"}},
        {"id":"inside-end","type":"end"}],"edges":[
        edge('inner-in','inside-input','inside-parser','handle_user_message','value'),
        edge('inner-out','inside-parser','inside-end','handle_parser_output','handle_flow_input')]}
    inner_data={"magic_flow":child_graph,"tool_mode":tool_mode}
    content='"prepared"' if not tool_mode else '{"value":"prepared"}'
    if tool_mode:
        inner_data.update(tool_name='inside',tool_parameters={"type":"object","properties":{"value":{"type":"string"}},"required":["value"]})
        child_graph['nodes'][1]['data']['text']='inner:{{ value.value }}'
    graph=build({"type":"graph","timeout":2,"nodes":[{"id":"input","type":"user_input"},
        {"id":"target","type":"parser","data":{"text":"original"}},
        {"id":"inside","type":"inner","data":inner_data},
        {"id":"control","type":"hook","data":{"lifecycle_event":"onStart","target_node_id":"target","function_template":
            'async def control(context, chat_log):\n    child = await context.call("control-inside", '+content+')\n    if child["outcome"]["status"] != "success":\n        return {"action":"pass"}\n    value = child["outcome"]["content"]\n    if isinstance(value, dict):\n        value = value["handle_execution_content"]\n    return {"action":"outcome","outcome":{"status":"success","content":{"handle_parser_output":value}}}'}},
        {"id":"end","type":"end"}],"edges":[edge('input-target','input','target','handle_user_message','value'),
            edge('control-inside','control','inside','handle-child-call','handle_user_message'),
            edge('target-end','target','end','handle_parser_output','handle_flow_input')]}, message='original')
    await asyncio.wait_for(collect(graph),3)
    assert graph.nodes['end'].inputs['handle_flow_input']=='inner:prepared'
    frame=graph.nodes['target']._last_invocation_record
    assert not frame['executed']
    child=frame['child'][0]['child'][0]
    assert child['node_id']=='inside' and child['executed'] and child['outcome']['status']=='success'

async def test_concurrent_independent_child_cancellation_keeps_each_record_identity(monkeypatch):
    from magic_agents.node_system.NodeParser import NodeParser
    b_started, a_done = asyncio.Event(), asyncio.Event()
    original = NodeParser.process

    async def process(self, chat_log):
        if self.node_id == 'child-a':
            await b_started.wait()
            try:
                raise asyncio.CancelledError()
            finally:
                a_done.set()
        if self.node_id == 'child-b':
            b_started.set()
            await a_done.wait()
        async for event in original(self, chat_log):
            yield event

    monkeypatch.setattr(NodeParser, 'process', process)
    graph=build({"type":"graph","timeout":2,"nodes":[
        {"id":"input","type":"user_input"},
        {"id":"target","type":"parser","data":{"text":"original:{{ value }}"}},
        {"id":"child-a","type":"parser","data":{"text":"a:{{ value }}"}},
        {"id":"child-b","type":"parser","data":{"text":"b:{{ value }}"}},
        {"id":"control","type":"hook","data":{"lifecycle_event":"onStart","target_node_id":"target","function_template":'''async def control(context, chat_log):
    children = await context.call_many([{"connection":"call-a","content":"a"}, {"connection":"call-b","content":"b"}], mode="parallel")
    context.emit.debug({"ids":[child["node_id"] for child in children], "statuses":[child["outcome"]["status"] for child in children]})
    return {"action":"pass"}
'''}}, {"id":"end","type":"end"}],"edges":[
        edge('input-target','input','target','handle_user_message','value'),
        edge('call-a','control','child-a','handle-child-call','value'),
        edge('call-b','control','child-b','handle-child-call','value'),
        edge('target-end','target','end','handle_parser_output','handle_flow_input')]}, message='source')
    await asyncio.wait_for(collect(graph),3)
    root=graph.nodes['target']._last_invocation_record
    hook=root['child'][0]
    assert hook['side_events']==[{"type":"debug","content":{"ids":["child-a","child-b"],"statuses":["cancelled","success"]}}]
    assert [child['node_id'] for child in hook['child']]==['child-a','child-b']
    assert hook['child'][0]['outcome']=={"status":"cancelled"}
    assert hook['child'][1]['outcome']['status']=='success'
    assert root['outcome']['status']=='success'
    assert graph.nodes['end'].inputs['handle_flow_input']=='original:source'
