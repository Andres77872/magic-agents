"""Real graph coverage for PythonExec node mode.

These tests deliberately use the production CodeRunner and graph executor.
They protect the boundary that mock-heavy unit tests cannot prove: arbitrary
edge target handles must become handler keys and execute end to end.
"""

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.node_system import NodePythonExec


@pytest.mark.asyncio
async def test_python_exec_runs_real_code_with_dynamic_handler_input():
    graph = build(
        {
            "type": "graph",
            "contract_config": {"mode": "strict"},
            "nodes": [
                {
                    "id": "input",
                    "type": "user_input",
                    "data": {"text": "hello"},
                },
                {
                    "id": "python",
                    "type": "python_exec",
                    "data": {
                        "code": (
                            "def run(handler):\n"
                            "    return {'upper': handler['message'].upper()}"
                        ),
                    },
                },
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {
                    "id": "input-python",
                    "source": "input",
                    "target": "python",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "message",
                },
                {
                    "id": "python-end",
                    "source": "python",
                    "target": "end",
                    "sourceHandle": "handle-python_exec-result",
                    "targetHandle": "handle_flow_input",
                },
            ],
        },
        message="hello",
    )

    target_handle_errors = [
        diagnostic
        for diagnostic in graph.contract_report.diagnostics
        if diagnostic.get("code") == "PORT_TARGET_HANDLE_UNKNOWN"
    ]
    assert target_handle_errors == []
    assert isinstance(graph.nodes["python"], NodePythonExec)

    async for _ in run_agent(graph):
        pass

    assert graph.nodes["python"].outputs["handle-python_exec-result"]["content"] == {
        "upper": "HELLO"
    }
    assert "handle_end_output" in graph.nodes["end"].outputs
