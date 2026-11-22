"""Edge timeouts take effect per invocation in both graph executors."""
import pytest
from magic_agents.agt_flow import build,run_agent


def edge(a,b,s,t,hook=False,override=None):
    value={'id':f'{a}-{s}-{b}-{t}','source':a,'target':b,'sourceHandle':s,'targetHandle':t}
    if hook:
        value['hooks']={'enabled':True,'hook_node_id':'hook'}
        if override is not None: value['hooks']['timeout_override']=override
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize('loop',[False,True])
@pytest.mark.parametrize('node_timeout,edge_timeout,expected_timeout',[(2,0,True),(0,2,False),(0,None,True),(2,None,False)])
async def test_edge_hook_timeout_overrides_node_without_mutating_config(loop,node_timeout,edge_timeout,expected_timeout):
    nodes=[{'id':'input','type':'user_input'},{'id':'end','type':'end'},
           {'id':'hook','type':'hook','data':{'timeout_override':node_timeout,'function_template':"async def hook(ctx, log):\n    return emit.user(str(ctx.inputs['content']))"}}]
    if loop:
        nodes += [{'id':'loop','type':'loop'},{'id':'echo','type':'parser','data':{'template':'{{ item }}'}}]
        edges=[edge('input','loop','handle_user_message','handle_list'),edge('loop','echo','handle_item','item',True,edge_timeout),
               edge('echo','loop','handle_parser_output','handle_loop'),edge('loop','end','handle_end','handle_flow_input')]
    else:
        edges=[edge('input','end','handle_user_message','handle_flow_input',True,edge_timeout)]
    graph=build({'nodes':nodes,'edges':edges},message='[1,2]' if loop else 'payload')
    events=[e async for e in run_agent(graph)]
    errors=[e['content'] for e in events if e.get('type')=='debug' and e.get('content',{}).get('error_type')=='HookTimeout']
    content=[e['content'].choices[0].delta.content for e in events if e.get('type')=='content' and e['content'].choices[0].delta.content]
    if expected_timeout:
        assert len(errors)==(2 if loop else 1)
        assert all('0s timeout' in error['error_message'] for error in errors)
        assert content==[]
    else:
        assert errors==[]
        assert content==(['1','2'] if loop else ['payload'])
    assert graph.nodes['hook']._timeout_seconds==node_timeout
