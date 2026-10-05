"""Real installed MCP ClientSession/HTTP transport, fake raw socket adapter."""
import asyncio
import json
from decimal import Decimal

import httpx2
import pytest

from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.dispatch import DispatchSession, invocation_charge
from magic_agents.coordination.service import CoordinationError
from magic_agents.mcp import controlled_http
from magic_agents.mcp.session import MCPSessionManager
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig
from test.test_coordination_dispatch import scope

pytestmark = pytest.mark.asyncio


class Wire(httpx2.AsyncBaseTransport):
    def __init__(self, handler=None):
        self.requests, self.closed = [], False
        self.handler = handler

    async def handle_async_request(self, request):
        body = json.loads(await request.aread())
        self.requests.append((request.method, str(request.url), body, dict(request.headers)))
        if self.handler is not None:
            response = await self.handler(request, body)
            if response is not None: return response
        if body['method'] == 'initialize':
            result = {'protocolVersion': body['params']['protocolVersion'], 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'bounded-fake', 'version': '1'}}
        elif body['method'] == 'tools/list':
            result = {'tools': [{'name': 'lookup', 'inputSchema': {'type': 'object', 'properties': {}}}]}
        elif body['method'] == 'tools/call':
            result = {'content': [{'type': 'text', 'text': 'private lookup result'}], 'isError': False}
        else:
            return httpx2.Response(202)
        return httpx2.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': result})

    async def aclose(self): self.closed = True


def setup(monkeypatch, handler=None, **limits):
    context = scope(maxConcurrentJobs=1, **limits)
    context.runtime.external_estimate = lambda *args: UsageBound(tool_calls=1, cost=Decimal('.1'))
    context.runtime.external_usage = lambda *args: UsageBound(tool_calls=1, cost=Decimal('.01'))
    wire = Wire(handler)
    monkeypatch.setattr(controlled_http.httpx2, 'AsyncHTTPTransport', lambda **kwargs: wire)
    manager = MCPSessionManager(MCPServerConfig(transport='http', url='https://example.invalid/mcp',
        headers={'authorization': 'private-bearer'}, init_timeout=1), node_id='mcp',
        dispatch_session=DispatchSession(context, 'mcp'))
    return context, wire, manager


async def close_failed(manager):
    try: await manager.cleanup()
    except (CoordinationError, asyncio.CancelledError): pass


async def test_real_sdk_initialize_list_call_notification_and_one_slot_invocation_credit(monkeypatch):
    context, wire, manager = setup(monkeypatch)
    seen = []
    manager._dispatch.authorize = lambda path, kind, request: seen.append((path, kind, request))
    async with invocation_charge(context, 'native-mcp-call', context.capture_guard('mcp')):
        try:
            await manager.connect()
            assert manager.server_info['name'] == 'bounded-fake'
            assert 'lookup' in repr(await manager.list_tools())
            assert 'private lookup result' in repr(await manager.call_tool('lookup', {}))
        finally: await manager.cleanup()
    assert [item[2]['method'] for item in wire.requests] == ['initialize', 'notifications/initialized', 'tools/list', 'tools/call']
    assert all(item[0] == 'POST' for item in wire.requests) and wire.closed
    totals = await context.budget.snapshot()
    assert totals['activeJobs'] == 0 and totals['spent']['tool_calls'] == 4
    assert {entry['kind'] for entry in context._external_operations.values()} == {'mcp.http.request'}
    assert all(kind == 'mcp.http.request' and request['profile'] == controlled_http.PROFILE for _, kind, request in seen)
    assert seen[0][2]['headers'] and seen[0][2]['body']['method'] == 'initialize'
    assert 'private-bearer' not in repr(context._external_operations)


@pytest.mark.parametrize('mode', ['complete_sse', 'incomplete_sse', 'stateful', 'gzip', 'disconnect'])
async def test_installed_sdk_bounded_response_profiles_never_reconnect_or_retry(monkeypatch, mode):
    async def handler(request, body):
        if body['method'] != 'tools/call': return None
        if mode == 'disconnect': raise httpx2.ConnectError('connection refused')
        if mode == 'stateful': return httpx2.Response(200, headers={'mcp-session-id': 'unowned-remote-lease'})
        if mode == 'gzip': return httpx2.Response(200, headers={'content-encoding': 'gzip'}, content=b'not inflated')
        data = ('id: event-1\nretry: 0\ndata: ' + json.dumps({'jsonrpc': '2.0', 'id': body['id'],
            'result': {'content': [{'type': 'text', 'text': 'finite SSE result'}]}}) + '\n\n')
        if mode == 'incomplete_sse': data = 'id: event-1\nretry: 0\ndata: {}\n\n'
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=data.encode())
    context, wire, manager = setup(monkeypatch, handler)
    try:
        await manager.connect()
        if mode == 'complete_sse':
            assert 'finite SSE result' in repr(await manager.call_tool('lookup', {}))
        else:
            with pytest.raises(CoordinationError):
                await asyncio.wait_for(manager.call_tool('lookup', {}), 2)
            with pytest.raises(CoordinationError): await manager.list_tools()
            assert (await context.budget.snapshot())['uncertain']['cost'] != '0'
    finally:
        await close_failed(manager)
    assert [item[0] for item in wire.requests] == ['POST'] * (4 if mode == 'complete_sse' else 3)
    if mode == 'complete_sse':
        # The installed SDK fetches tool schemas lazily for result validation.
        assert wire.requests[-1][2]['method'] == 'tools/list'
        assert (await context.budget.snapshot())['spent']['tool_calls'] == 4
    assert wire.closed and (await context.budget.snapshot())['activeJobs'] == 0


@pytest.mark.parametrize('revoke', [False, True])
async def test_each_sdk_same_origin_redirect_is_admitted_and_original_owner_can_deny_it(monkeypatch, revoke):
    async def handler(request, body):
        if str(request.url).endswith('/mcp'):
            if revoke: context.owner[0] = False
            return httpx2.Response(307, headers={'location': '/resolved'})
    context, wire, manager = setup(monkeypatch, handler)
    try:
        if revoke:
            with pytest.raises(CoordinationError): await manager.connect()
        else:
            await manager.connect()
    finally: await close_failed(manager)
    if revoke:
        assert len(wire.requests) == 1
    else:
        assert [item[1] for item in wire.requests] == ['https://example.invalid/mcp', 'https://example.invalid/resolved'] * 2
        assert (await context.budget.snapshot())['spent']['tool_calls'] == 4


async def test_cross_origin_redirect_never_forwards_credentials(monkeypatch):
    async def handler(request, body):
        return httpx2.Response(307, headers={'location': 'https://foreign.invalid/mcp'})
    _, wire, manager = setup(monkeypatch, handler)
    try:
        with pytest.raises(Exception): await manager.connect()
    finally: await close_failed(manager)
    assert len(wire.requests) == 1 and wire.requests[0][1] == 'https://example.invalid/mcp'


async def test_raw_body_byte_ceiling_stops_incremental_read_before_sdk_parse(monkeypatch):
    reads, closes = [], []
    class Stream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            reads.append(1); yield b'x' * (controlled_http.MAX_RESPONSE_BYTES + 1)
            reads.append(2); yield b'never read'
        async def aclose(self): closes.append(True)
    async def handler(request, body): return httpx2.Response(200, stream=Stream())
    context, wire, manager = setup(monkeypatch, handler)
    try:
        with pytest.raises(CoordinationError, match='physical byte ceiling'):
            await manager.connect()
    finally: await close_failed(manager)
    assert reads == [1] and closes and len(wire.requests) == 1
    assert (await context.budget.snapshot())['uncertain']['cost'] != '0'


async def test_denied_physical_admission_escapes_sdk_exception_handler_without_wire_effect(monkeypatch):
    context, wire, manager = setup(monkeypatch)
    def denied(*args): raise CoordinationError('denied', 'Original tool authority denied')
    manager._dispatch.authorize = denied
    try:
        with pytest.raises(CoordinationError, match='Original tool authority'):
            async with asyncio.timeout(2):
                await manager.connect()
    finally: await close_failed(manager)
    assert not wire.requests and (await context.budget.snapshot())['uncertain']['cost'] == '0'


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('replace_owner', [False, True])
async def test_actual_actor_native_mcp_tool_inherits_original_owner_and_single_slot(monkeypatch, stream, replace_owner):
    from magic_agents.coordination.context import _Binding
    from magic_agents.coordination.service import ActorCaller
    from magic_agents.models.coordination import CoordinationPolicy, MessagingConfig
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
    from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
    from magic_agents.models.factory.Nodes.McpNodeModel import McpNodeModel
    from magic_agents.node_system.NodeMcp import NodeMcp
    from test.test_coordination_graph import node, runtime, limits, collect
    async def research(provider, chat):
        if len(provider.calls) == 1:
            return {'tool_calls': [{'id': 'lookup-1', 'type': 'function',
                    'function': {'name': 'mcp_lookup', 'arguments': '{}'}}]}
        assert 'private lookup result' in repr(chat.messages)
        return {'content': 'research complete'}
    async def author(provider, chat): return {'content': 'public final'}
    actor, provider = node('research', research, stream=stream)
    actor.messaging = MessagingConfig(enabled=True, role='research', peers=[])
    final, final_provider = node('author', author, stream=stream)
    mcp_node = NodeMcp(McpNodeModel(servers=[MCPServerConfig(transport='http', url='https://example.invalid/mcp')]),
                       node_id='mcp', node_type='mcp')
    rt = runtime(server_limits=limits(maxConcurrentJobs=1),
        authorize_external=lambda *args: None,
        external_estimate=lambda *args: UsageBound(tool_calls=1, cost=Decimal('.1')),
        external_usage=lambda *args: UsageBound(tool_calls=1, cost=Decimal('.01')))
    async def handler(request, body):
        if body['method'] == 'tools/call' and replace_owner:
            scope = rt.scopes[0]
            original = scope.bindings['research'].caller
            state = scope.service._actors[original.actor_id]
            state.owner_token, state.activation_id = object(), 'replacement-activation'
            replacement = ActorCaller(scope.service, original.actor_id, state.activation_id, state.owner_token)
            scope.bindings['research'] = _Binding(replacement)
    wire = Wire(handler)
    monkeypatch.setattr(controlled_http.httpx2, 'AsyncHTTPTransport', lambda **kwargs: wire)
    graph = AgentFlowModel(type='graph', nodes={'research': actor, 'author': final, 'mcp': mcp_node},
        coordination=CoordinationPolicy(enabled=True, allowedParticipants=['research']), edges=[
            EdgeNodeModel(id='mcp-research', source='mcp', target='research', sourceHandle=mcp_node.OUTPUT_HANDLE,
                          targetHandle=actor.INPUT_TOOL_PREFIX + 'lookup'),
            EdgeNodeModel(id='research-author', source='research', target='author', sourceHandle=actor.OUTPUT_HANDLE_GENERATED,
                          targetHandle=final.INPUT_HANDLER_SYSTEM_CONTEXT)])
    result, events = await collect(graph, rt)
    methods = [request[2]['method'] for request in wire.requests]
    assert methods.count('initialize') == 2 and methods.count('tools/call') == 1
    assert (await rt.budget.snapshot())['activeJobs'] == 0
    if replace_owner:
        assert result['has_errors'] and len(provider.calls) == 1 and not final_provider.calls, events
        assert (await rt.budget.snapshot())['uncertain']['cost'] != '0'
        assert methods[-1] == 'tools/call'
    else:
        assert not result['has_errors'], events
        assert len(provider.calls) == 2 and len(final_provider.calls) == 1
        assert (await rt.budget.snapshot())['spent']['tool_calls'] == len(wire.requests)


async def test_sdk_cancellation_joins_started_response_and_preserves_unknown_exposure(monkeypatch):
    started, joined = asyncio.Event(), asyncio.Event()
    async def handler(request, body):
        if body['method'] == 'tools/call':
            started.set()
            try: await asyncio.Event().wait()
            finally: joined.set()
    context, wire, manager = setup(monkeypatch, handler)
    await manager.connect()
    task = asyncio.create_task(manager.call_tool('lookup', {}))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    async with asyncio.timeout(2): await close_failed(manager)
    assert joined.is_set() and wire.closed
    totals = await context.budget.snapshot()
    assert totals['activeJobs'] == 0 and totals['uncertain']['cost'] != '0'
    assert [item[2]['method'] for item in wire.requests].count('tools/call') == 1
    assert not any(item[0] in ('GET', 'DELETE') for item in wire.requests)


async def test_total_response_deadline_bounds_an_unending_raw_stream(monkeypatch):
    closed = []
    class Stream(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield b'partial'
            await asyncio.Event().wait()
        async def aclose(self): closed.append(True)
    async def handler(request, body):
        if body['method'] == 'tools/call': return httpx2.Response(200, stream=Stream())
    context, wire, manager = setup(monkeypatch, handler)
    await manager.connect()
    manager._physical.timeout = .02
    try:
        async with asyncio.timeout(1):
            with pytest.raises(CoordinationError): await manager.call_tool('lookup', {})
    finally: await close_failed(manager)
    assert closed and wire.closed and len(wire.requests) == 3
    assert (await context.budget.snapshot())['uncertain']['cost'] != '0'


async def test_redirect_chain_is_finite_and_each_hop_consumes_admission(monkeypatch):
    async def handler(request, body): return httpx2.Response(307, headers={'location': '/again'})
    context, wire, manager = setup(monkeypatch, handler)
    try:
        with pytest.raises(Exception): await manager.connect()
    finally: await close_failed(manager)
    assert len(wire.requests) == controlled_http.MAX_REDIRECTS + 1
    assert (await context.budget.snapshot())['spent']['tool_calls'] == len(wire.requests)


async def test_sdk_extra_requests_cannot_exceed_shared_tool_cap(monkeypatch):
    from magic_agents.coordination.budget import BudgetError
    context, wire, manager = setup(monkeypatch, maxToolCalls=2)
    await manager.connect()
    try:
        with pytest.raises(BudgetError): await manager.list_tools()
    finally:
        try: await manager.cleanup()
        except BudgetError: pass
    assert len(wire.requests) == 2
    assert (await context.budget.snapshot())['spent']['tool_calls'] == 2


@pytest.mark.parametrize('method', ['GET', 'DELETE'])
async def test_unqualified_reconnect_or_session_cleanup_never_reaches_wire(monkeypatch, method):
    context, wire, _ = setup(monkeypatch)
    transport = controlled_http.BoundedMCPTransport(DispatchSession(context, 'mcp'), timeout=1, transport=wire)
    async with httpx2.AsyncClient(transport=transport) as client:
        response = await client.request(method, 'https://example.invalid/mcp')
        assert response.status_code == 400
        with pytest.raises(CoordinationError, match='stateless POST'): transport.check()
    assert not wire.requests and not context._external_operations


async def test_redirect_hops_share_one_wire_deadline_and_do_not_renew_allowance(monkeypatch):
    async def handler(request, body):
        if body['method'] == 'tools/call':
            await asyncio.sleep(.03)
            return httpx2.Response(307, headers={'location': '/next'})
    context, wire, manager = setup(monkeypatch, handler)
    await manager.connect()
    manager._physical.timeout = .05
    try:
        async with asyncio.timeout(1):
            with pytest.raises(CoordinationError): await manager.call_tool('lookup', {})
    finally: await close_failed(manager)
    requests = [item for item in wire.requests if item[2]['method'] == 'tools/call']
    assert 1 <= len(requests) <= 2 and wire.closed
    assert (await context.budget.snapshot())['uncertain']['cost'] != '0'


async def test_redirect_hops_share_one_incremental_response_byte_budget(monkeypatch):
    async def handler(request, body):
        if body['method'] == 'tools/call':
            return httpx2.Response(307, headers={'location': '/next'}, content=b'x' * 600000)
    context, wire, manager = setup(monkeypatch, handler)
    await manager.connect()
    try:
        with pytest.raises(CoordinationError, match='byte ceiling'): await manager.call_tool('lookup', {})
    finally: await close_failed(manager)
    assert len([item for item in wire.requests if item[2]['method'] == 'tools/call']) == 2
    assert wire.closed and (await context.budget.snapshot())['uncertain']['cost'] != '0'


async def test_sdk_notification_denial_is_retained_across_successful_initialize_reply(monkeypatch):
    context, wire, manager = setup(monkeypatch)
    def authorize(path, kind, request):
        if request['body']['method'] == 'notifications/initialized':
            raise CoordinationError('denied_notification', 'Notification authority denied')
    manager._dispatch.authorize = authorize
    try:
        with pytest.raises(CoordinationError, match='Notification authority'):
            await manager.connect()
            await manager.list_tools()
    finally: await close_failed(manager)
    assert len(wire.requests) == 1 and wire.closed


async def test_held_initialization_notification_settles_before_discovery(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    async def handler(request, body):
        if body['method'] == 'notifications/initialized':
            entered.set(); await release.wait()
    context, wire, manager = setup(monkeypatch, handler)
    async def peer():
        await entered.wait()
        assert (await context.budget.snapshot())['activeJobs'] == 1
        release.set()
    task = asyncio.create_task(peer())
    try:
        await manager.connect()
        await manager.list_tools()
    finally:
        await manager.cleanup()
        await task
    assert wire.closed and (await context.budget.snapshot())['activeJobs'] == 0


async def test_local_cleanup_failure_is_terminal_after_successful_wire_result(monkeypatch):
    context, wire, manager = setup(monkeypatch)
    await manager.connect()
    await manager.list_tools()
    async def broken_close():
        wire.closed = True
        raise RuntimeError('Local transport cleanup failed')
    wire.aclose = broken_close
    with pytest.raises(CoordinationError, match='cleanup did not complete'):
        await manager.cleanup()


@pytest.mark.parametrize('mode', ['malformed', 'duplicate', 'nonfinite', 'overflow', 'wrong_id',
    'wrong_id_type', 'missing_id', 'ambiguous', 'bad_result', 'accepted_only', 'wrong_content_type',
    'sse_wrong_id', 'sse_duplicate', 'sse_nonfinite'])
async def test_invalid_rpc_outcome_retains_exposure_before_sdk_parse(monkeypatch, mode):
    async def handler(request, body):
        if body['method'] != 'tools/call': return None
        reply = {'jsonrpc': '2.0', 'id': body['id'], 'result': {'content': []}}
        if mode == 'wrong_id': reply['id'] = body['id'] + 100
        if mode == 'wrong_id_type': reply['id'] = str(body['id'])
        if mode == 'missing_id': del reply['id']
        if mode == 'ambiguous': reply['error'] = {'code': -32603, 'message': 'ambiguous'}
        if mode == 'bad_result': reply['result'] = {'content': 'invalid'}
        content = json.dumps(reply)
        if mode == 'malformed': content = '{'
        if mode == 'duplicate': content = content[:-1] + ',"id":' + str(body['id']) + '}'
        if mode in ('nonfinite', 'overflow'): content = content[:-1] + ',"extra":' + ('NaN' if mode == 'nonfinite' else '1e9999') + '}'
        if mode == 'accepted_only': return httpx2.Response(202)
        if mode == 'wrong_content_type': return httpx2.Response(200, content=content, headers={'content-type': 'text/plain'})
        if mode.startswith('sse_'):
            if mode == 'sse_wrong_id': reply['id'] += 100
            content = json.dumps(reply)
            if mode == 'sse_duplicate': content = content[:-1] + ',"id":' + str(body['id']) + '}'
            if mode == 'sse_nonfinite': content = content[:-1] + ',"extra":Infinity}'
            return httpx2.Response(200, content='data: ' + content + '\n\n', headers={'content-type': 'text/event-stream'})
        return httpx2.Response(200, content=content, headers={'content-type': 'application/json'})
    context, wire, manager = setup(monkeypatch, handler)
    try:
        await manager.connect()
        with pytest.raises(CoordinationError):
            await asyncio.wait_for(manager.call_tool('lookup', {}), 1)
        before = len(wire.requests)
        with pytest.raises(CoordinationError): await manager.list_tools()
        assert len(wire.requests) == before == 3
        assert (await context.budget.snapshot())['uncertain']['cost'] != '0'
        assert list(context._external_operations.values())[-1]['state'] == 'unknown'
    finally:
        await close_failed(manager)
    assert wire.closed


async def test_actual_sdk_cleanup_cancels_session_tasks_and_bounds_local_transport_close(monkeypatch):
    _, wire, manager = setup(monkeypatch)
    entered, cancelled, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def held_session_task():
        entered.set()
        try: await asyncio.Event().wait()
        finally: cancelled.set()
    async def held_close():
        closing.set()
        await asyncio.Event().wait()
    await manager.connect()
    manager._session._task_group.start_soon(held_session_task)
    await entered.wait()
    wire.aclose = held_close
    monkeypatch.setattr(controlled_http, 'LOCAL_UNWIND_SECONDS', .05)
    with pytest.raises(CoordinationError, match='cleanup did not complete'):
        await asyncio.wait_for(manager.cleanup(), .5)
    assert cancelled.is_set() and closing.is_set()
    assert manager._session is manager._transport_context is None


async def test_request_unwind_deadline_is_finite_when_sdk_operation_suppresses_cancel(monkeypatch):
    _, wire, manager = setup(monkeypatch)
    await manager.connect()
    started, release, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def resistant_operation():
        started.set()
        try:
            while not release.is_set():
                try: await release.wait()
                except asyncio.CancelledError: pass
        finally: ended.set()
    # The native SDK request chain normally exits immediately on cancellation;
    # exercise the teardown bound independently of that implementation promise.
    monkeypatch.setattr(controlled_http, 'LOCAL_UNWIND_SECONDS', .02)
    task = asyncio.create_task(manager._physical.run(resistant_operation, timeout=.01))
    await started.wait()
    task.cancel()
    try:
        with pytest.raises(CoordinationError, match='local teardown bound'):
            await asyncio.wait_for(task, .5)
    finally:
        release.set()
        await asyncio.wait_for(ended.wait(), .5)
        await close_failed(manager)
    assert wire.closed
    assert wire.closed and not manager.is_healthy


@pytest.mark.parametrize('header', ['mcp-session-id', 'last-event-id'])
async def test_stateless_profile_never_adopts_authored_session_or_resume_identity(monkeypatch, header):
    context, wire, manager = setup(monkeypatch)
    manager._config.headers[header] = 'foreign-session-or-request'
    try:
        with pytest.raises(CoordinationError, match='cannot adopt'):
            await manager.connect()
    finally: await close_failed(manager)
    assert not wire.requests and not context._external_operations


async def test_even_empty_session_header_is_an_unsupported_remote_effect(monkeypatch):
    async def handler(request, body): return httpx2.Response(200, headers={'mcp-session-id': ''})
    context, wire, manager = setup(monkeypatch, handler)
    try:
        with pytest.raises(CoordinationError, match='session creation'):
            await manager.connect()
    finally: await close_failed(manager)
    assert len(wire.requests) == 1 and (await context.budget.snapshot())['uncertain']['cost'] != '0'


@pytest.mark.parametrize('package', ['mcp', 'httpx2', 'mcp-types'])
@pytest.mark.parametrize('missing', [False, True])
async def test_unqualified_sdk_version_rejects_before_transport_construction(monkeypatch, package, missing):
    import importlib.metadata
    from magic_agents.node_system.NodeMcp import NodeMcp
    context, wire, manager = setup(monkeypatch)
    original = importlib.metadata.version
    def checked(name):
        if name == package:
            if missing: raise importlib.metadata.PackageNotFoundError(name)
            return 'unqualified-new-version'
        return original(name)
    assert dict(NodeMcp.coordination_external_transport_versions) == {'mcp': '2.2.0', 'httpx2': '2.13.0', 'mcp-types': '2.2.0'}
    with pytest.raises(TypeError): NodeMcp.coordination_external_transport_versions[package] = 'authored-override'
    monkeypatch.setattr(importlib.metadata, 'version', checked)
    with pytest.raises(CoordinationError, match='dependencies have not been qualified'):
        await manager.connect()
    await manager.cleanup()
    assert not wire.requests and manager._physical is None and not context._external_operations
