"""INNER is a fresh child invocation on every parent-loop iteration."""
import json

import pytest
from magic_agents.agt_flow import build, run_agent
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


def edge(a,b,s,t):
    return {'id':f'{a}-{s}-{b}-{t}','source':a,'target':b,'sourceHandle':s,'targetHandle':t}


def child():
    return {'nodes':[{'id':'input','type':'user_input'},
                     {'id':'route','type':'conditional','data':{'condition':"{{ 'positive' if value > 0 else 'other' }}",'output_handles':['positive','other']}},
                     {'id':'positive','type':'parser','data':{'template':'{{ item.value * 2 }}'}},
                     {'id':'other','type':'parser','data':{'template':'{{ item.value - 1 }}'}},
                     {'id':'merge','type':'parser','data':{'template':'{{ result }}'}},{'id':'end','type':'end'}],
            'edges':[edge('input','route','handle_user_message','handle_input'),edge('route','positive','positive','item'),edge('route','other','other','item'),
                     edge('positive','merge','handle_parser_output','result'),edge('other','merge','handle_parser_output','result'),edge('merge','end','handle_parser_output','handle_flow_input')]}


@pytest.mark.asyncio
async def test_parent_loop_reexecutes_nested_conditional_and_clears_stale_branches():
    definition={'nodes':[{'id':'input','type':'user_input'},{'id':'loop','type':'loop'},
                         {'id':'inner','type':'inner','data':{'magic_flow':child()}},{'id':'end','type':'end'}],
                'edges':[edge('input','loop','handle_user_message','handle_list'),edge('loop','inner','handle_item','handle_user_message'),
                         edge('inner','loop','handle_execution_content','handle_loop'),edge('loop','end','handle_end','handle_flow_input')]}
    graph=build(definition,message='[2,-3,0,4]')
    events=[e async for e in run_agent(graph)]
    assert graph.nodes['loop'].outputs['handle_end']['content']==['4','-4','-1','8']
    errors=[e for e in events if e.get('type')=='debug' and e.get('content',{}).get('error_type')]
    assert errors==[]
    nested=graph.nodes['inner'].inner_graph
    assert nested.nodes['input']._text==4
    assert nested.nodes['route'].selected_handle=='positive'
    assert nested.nodes['other'].outputs=={}  # Prior selected branch is not cached.


@pytest.mark.asyncio
async def test_repeated_inner_invocations_receive_current_extras():
    nested={'nodes':[{'id':'input','type':'user_input'}, {'id':'render','type':'parser','data':{'template':'{{ {"message": message, "extra": extra.tag} | tojson }}'}}, {'id':'end','type':'end'}],
            'edges':[edge('input','render','handle_user_message','message'),edge('input','render','handle_client_extras','extra'),edge('render','end','handle_parser_output','handle_flow_input')]}
    graph=build({'nodes':[{'id':'input','type':'user_input'},{'id':'inner','type':'inner','data':{'magic_flow':nested}}],'edges':[]},message='start')
    inner=graph.nodes['inner']
    log=ModelAgentRunLog(id_chat='chat',id_thread='thread',id_user='user')
    for message,tag in [('first','one'),('second','two')]:
        inner.inputs[inner.INPUT_HANDLE]=message
        inner.inputs[inner.HANDLER_CLIENT_EXTRAS]={'tag':tag}
        events=[e async for e in inner.process(log)]
        result=next(e['content']['content'] for e in events if e.get('type')==inner.HANDLER_EXECUTION_CONTENT)
        assert json.loads(result)=={'message':message,'extra':tag}
