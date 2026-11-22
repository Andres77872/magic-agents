import json

import pytest

from magic_agents.node_system.NodePythonExec import PythonExecToolWrapper
from magic_agents.node_system.python_code_runner import CodeRunner
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.util.python_executor import PythonExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize('handler',[False,True])
async def test_python_tool_errors_are_failed_results_and_success_payload_is_compatible(handler):
    wrapper=PythonExecToolWrapper(executor=PythonExecutor(),code_runner=CodeRunner())
    executor=ToolExecutor()
    executor.register('execute_python',wrapper)
    arguments={'code':'def run(handler): return 1 / 0','handler':{}} if handler else {'code':'1 / 0'}
    results=await executor.execute_parallel_async([CanonicalToolCall(id='failed',name='execute_python',arguments=arguments)])
    assert results[0].is_error is True
    assert 'division by zero' in results[0].content
    arguments={'code':'def run(handler): return handler["value"] * 2','handler':{'value':7}} if handler else {'code':'print(7 * 2)'}
    results=await executor.execute_parallel_async([CanonicalToolCall(id='success',name='execute_python',arguments=arguments)])
    assert results[0].is_error is False
    assert json.loads(results[0].content)==(json.dumps({'result':14}) if handler else '14\n')
