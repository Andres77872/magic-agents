"""Loopback HTTP contract tests for NodeFetch and FetchToolCallable.

The server and client are both real aiohttp components. Only the external
network is replaced by a local deterministic endpoint.
"""

import json
import logging
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeFetch import FetchToolCallable, NodeFetch


@asynccontextmanager
async def running_server(*routes):
    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


def server_url(server: TestServer, path: str) -> str:
    return f"{str(server.make_url('/')).rstrip('/')}{path}"


@pytest.mark.asyncio
async def test_fetch_tool_real_get_renders_path_query_and_headers():
    async def echo(request):
        return web.json_response(
            {
                "item": request.match_info["item"],
                "query": request.query.get("q"),
                "token": request.headers.get("X-Test-Token"),
            }
        )

    async with running_server(("GET", "/items/{item}", echo)) as server:
        tool = FetchToolCallable(
            url_template=server_url(server, "/items/{{item}}"),
            method="get",
            headers={"X-Test-Token": "{{token}}"},
            params={"q": "{{query}}"},
            tool_name="lookup",
        )

        result = json.loads(
            await tool(item="widget", query="active", token="local-token")
        )

    assert result == {
        "item": "widget",
        "query": "active",
        "token": "local-token",
    }


@pytest.mark.asyncio
async def test_fetch_tool_real_post_preserves_raw_string_body():
    async def echo_body(request):
        return web.json_response({"body": await request.text()})

    async with running_server(("POST", "/submit", echo_body)) as server:
        tool = FetchToolCallable(
            url_template=server_url(server, "/submit"),
            method="post",
            data="value={{value}}",
        )

        result = json.loads(await tool(value="hello"))

    assert result == {"body": "value=hello"}


@pytest.mark.asyncio
async def test_fetch_tool_real_call_applies_tool_parameter_request_overrides():
    """Tool-call templates and constants reach a real HTTP request together."""
    async def inspect_request(request):
        return web.json_response(
            {
                "item": request.match_info["item"],
                "mode": request.headers.get("X-Mode"),
                "fixed": request.query.get("fixed"),
                "body": await request.json(),
            }
        )

    async with running_server(("POST", "/tool/{item}", inspect_request)) as server:
        tool = FetchToolCallable(
            url_template="https://unused.invalid/default",
            method="GET",
            headers={"X-Base": "preserved"},
            tool_name="override_request",
            tool_parameters={
                "url": server_url(server, "/tool/{{item}}"),
                "method": "post",
                "headers": {"X-Mode": "tool-override"},
                "params": {"fixed": "yes"},
                "json_data": {"base": True},
                "query": "{{query}}",
            },
        )

        result = json.loads(await tool(item="widget", query="from-tool-call"))

    assert result == {
        "item": "widget",
        "mode": "tool-override",
        "fixed": "yes",
        "body": {"base": True, "query": "from-tool-call"},
    }


@pytest.mark.asyncio
async def test_node_fetch_real_request_uses_template_input_and_routes_json():
    async def node_response(request):
        return web.json_response({"received": request.match_info["value"]})

    async with running_server(("GET", "/node/{value}", node_response)) as server:
        node = NodeFetch(
            data=FetchNodeModel(
                url=server_url(server, "/node/{{handle_fetch_input}}"),
                method="GET",
            ),
            node_id="fetch-real",
        )
        node.inputs[node.INPUT_HANDLE_TEMPLATE_CONTEXT] = "from-graph"

        events = [
            event
            async for event in node.process(ModelAgentRunLog())
            if event["type"] == node.OUTPUT_HANDLE
        ]

    assert len(events) == 1
    assert events[0]["content"]["content"] == {"received": "from-graph"}


@pytest.mark.asyncio
async def test_fetch_tool_non_2xx_returns_body_but_redacts_log(caplog):
    async def unauthorized(_request):
        return web.json_response(
            {
                "message": "denied",
                "api_key": "sk-local-secret",
                "nested": {"authorization": "Bearer nested-local-secret"},
                "padding": "x" * 800,
            },
            status=401,
        )

    async with running_server(("GET", "/private", unauthorized)) as server:
        tool = FetchToolCallable(server_url(server, "/private"), tool_name="private")

        with caplog.at_level(logging.WARNING):
            result = await tool(parameters="unused")

    assert "HTTP 401" in result
    assert "sk-local-secret" in result
    assert "nested-local-secret" in result
    assert "[REDACTED]" in caplog.text
    assert "sk-local-secret" not in caplog.text
    assert "nested-local-secret" not in caplog.text
