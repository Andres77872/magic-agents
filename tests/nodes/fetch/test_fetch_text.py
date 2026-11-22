"""Real HTTP regression for the text/plain llms.txt failure."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeFetch import NodeFetch


@pytest.mark.asyncio
@pytest.mark.parametrize('plain_text', [True, False])
async def test_fetch_routes_plain_documentation_and_json(plain_text):
    expected = '# Official documentation\nGET /search' if plain_text else {'documentation': 'ready'}
    async def handler(_request):
        return web.Response(text=expected) if plain_text else web.json_response(expected)
    app = web.Application()
    app.router.add_get('/llms.txt', handler)
    async with TestServer(app) as server:
        node = NodeFetch(FetchNodeModel(url=str(server.make_url('/llms.txt')), method='GET'), node_id='fetch')
        node.inputs[node.INPUT_HANDLE_TEMPLATE_CONTEXT] = 'run'
        events = [event async for event in node.process(ModelAgentRunLog()) if event['type'] == node.OUTPUT_HANDLE]
    assert len(events) == 1
    assert events[0]['content']['content'] == expected
