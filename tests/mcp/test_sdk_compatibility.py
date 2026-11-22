"""MCP transport naming changes must not break all MCP JSON graphs."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from magic_agents.mcp.session import streamablehttp_client


@pytest.mark.asyncio
async def test_modern_only_sdk_receives_headers_on_managed_client(monkeypatch):
    import mcp.client
    import mcp.shared._httpx_utils
    events = []
    streams = ('read', 'write')
    client = object()

    @asynccontextmanager
    async def http_factory(**kwargs):
        events.append(('open', kwargs))
        yield client
        events.append(('close',))

    @asynccontextmanager
    async def modern(url, *, http_client):
        assert url == 'http://mcp.test'
        assert http_client is client
        yield streams

    monkeypatch.setattr(mcp.client, 'streamable_http', SimpleNamespace(streamable_http_client=modern))
    monkeypatch.setattr(mcp.shared._httpx_utils, 'create_mcp_http_client', http_factory)
    async with streamablehttp_client('http://mcp.test', headers={'Authorization': 'Bearer fixture'}) as result:
        assert result is streams
    assert events == [('open', {'headers': {'Authorization': 'Bearer fixture'}}), ('close',)]


@pytest.mark.asyncio
async def test_legacy_sdk_retains_headers_and_session_id_callback(monkeypatch):
    import mcp.client
    callback = lambda: 'session-id'
    streams = ('read', 'write', callback)

    @asynccontextmanager
    async def legacy(url, *, headers):
        assert url == 'http://mcp.test'
        assert headers == {'X-Test': 'fixture'}
        yield streams

    monkeypatch.setattr(mcp.client, 'streamable_http', SimpleNamespace(streamablehttp_client=legacy))
    async with streamablehttp_client('http://mcp.test', headers={'X-Test': 'fixture'}) as result:
        assert result is streams
