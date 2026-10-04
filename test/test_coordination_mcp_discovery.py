"""Whole-discovery bounds through installed ClientSession and HTTP SDK."""
import asyncio

import httpx2
import pytest

from magic_agents.coordination.service import CoordinationError
from magic_agents.mcp import discovery as discovery_module
from magic_agents.mcp.discovery import MCPToolDiscovery
from test.test_coordination_mcp_http import close_failed, setup

pytestmark = pytest.mark.asyncio


def page(body, *, names, cursor=None, description=''):
    result = {'tools': [{'name': name, 'description': description,
                         'inputSchema': {'type': 'object'}} for name in names]}
    if cursor is not None: result['nextCursor'] = cursor
    return httpx2.Response(200, json={'jsonrpc': '2.0', 'id': body['id'], 'result': result})


async def test_complete_multipage_discovery_shares_identity_and_returns_bounded_cache(monkeypatch):
    async def handler(request, body):
        if body['method'] != 'tools/list': return None
        cursor = body.get('params', {}).get('cursor')
        return page(body, names=['first'] if cursor is None else ['second'],
                    cursor='page-2' if cursor is None else None)
    context, wire, manager = setup(monkeypatch, handler)
    admitted = []
    manager._dispatch.authorize = lambda path, kind, request: admitted.append(request)
    discovery = MCPToolDiscovery(manager)
    try:
        await manager.connect()
        tools = await discovery.list_tools()
        assert [tool['name'] for tool in tools] == ['first', 'second']
        assert await discovery.list_tools() is tools
        requests = [request for request in admitted if request['body']['method'] == 'tools/list']
        assert len({request['operation_id'] for request in requests}) == 1
        assert requests[-1]['timeout_seconds'] < requests[0]['timeout_seconds']
        assert [item[2].get('params', {}).get('cursor') for item in wire.requests if item[2]['method'] == 'tools/list'] == [None, 'page-2']
        assert (await context.budget.snapshot())['spent']['tool_calls'] == 4
        context.owner[0] = False
        with pytest.raises(CoordinationError): await discovery.list_tools()
        assert len(wire.requests) == 4
    finally:
        await close_failed(manager)
    assert wire.closed


@pytest.mark.parametrize('failure', ['combined_bytes', 'combined_time', 'repeated_cursor',
    'cursor_bytes', 'page_count', 'tool_count', 'cache_bytes', 'duplicate_tool'])
async def test_discovery_exhaustion_never_caches_a_partial_success(monkeypatch, failure):
    pages = []
    async def handler(request, body):
        if body['method'] != 'tools/list': return None
        pages.append(body.get('params', {}).get('cursor'))
        if failure == 'combined_time': await asyncio.sleep(.035)
        names = ['first'] if len(pages) == 1 else ['second']
        cursor = 'page-2' if len(pages) == 1 else None
        if failure == 'repeated_cursor': cursor = 'page-2'
        if failure == 'cursor_bytes': cursor = 'x' * (discovery_module.MAX_DISCOVERY_CURSOR_BYTES + 1)
        if failure == 'page_count': cursor = str(len(pages))
        if failure == 'duplicate_tool': names = ['same']
        description = 'x' * 600000 if failure == 'combined_bytes' else ''
        return page(body, names=names, cursor=cursor, description=description)
    context, wire, manager = setup(monkeypatch, handler)
    if failure == 'combined_time': manager._config.discovery_timeout = .06
    if failure == 'page_count': monkeypatch.setattr(discovery_module, 'MAX_DISCOVERY_PAGES', 2)
    if failure == 'tool_count': monkeypatch.setattr(discovery_module, 'MAX_DISCOVERY_TOOLS', 1)
    if failure == 'cache_bytes': monkeypatch.setattr(discovery_module, 'MAX_DISCOVERY_CACHE_BYTES', 10)
    discovery = MCPToolDiscovery(manager)
    try:
        await manager.connect()
        with pytest.raises(CoordinationError):
            await asyncio.wait_for(discovery.list_tools(), 2)
        assert discovery.tools is None
        dispatched = len(wire.requests)
        with pytest.raises(CoordinationError): await manager.list_tools()
        assert len(wire.requests) == dispatched
        assert len(pages) == (1 if failure in ('cursor_bytes', 'cache_bytes') else 2)
        if failure in ('combined_bytes', 'combined_time'):
            assert (await context.budget.snapshot())['uncertain']['cost'] != '0'
    finally:
        await close_failed(manager)
    assert wire.closed and (await context.budget.snapshot())['activeJobs'] == 0


@pytest.mark.parametrize('stop', ['cancel', 'owner'])
async def test_cancellation_or_owner_loss_between_pages_prevents_next_wire_request(monkeypatch, stop):
    async def handler(request, body):
        if body['method'] == 'tools/list': return page(body, names=['first'], cursor='page-2')
    context, wire, manager = setup(monkeypatch, handler)
    discovery = MCPToolDiscovery(manager)
    extract_cursor = discovery._extract_cursor
    def boundary(result):
        cursor = extract_cursor(result)
        if stop == 'cancel': asyncio.current_task().cancel()
        else: context.owner[0] = False
        return cursor
    discovery._extract_cursor = boundary
    try:
        await manager.connect()
        with pytest.raises(asyncio.CancelledError if stop == 'cancel' else CoordinationError):
            await discovery.list_tools()
        assert discovery.tools is None
        assert [item[2]['method'] for item in wire.requests] == ['initialize', 'notifications/initialized', 'tools/list']
        with pytest.raises(asyncio.CancelledError if stop == 'cancel' else CoordinationError):
            await manager.call_tool('lookup', {})
        assert len(wire.requests) == 3
    finally:
        await close_failed(manager)
    assert wire.closed


async def test_child_tasks_cannot_borrow_an_inherited_discovery_window(monkeypatch):
    async def handler(request, body):
        if body['method'] == 'tools/list': return page(body, names=['lookup'])
    _, wire, manager = setup(monkeypatch, handler)
    admitted = []
    manager._dispatch.authorize = lambda path, kind, request: admitted.append(request)
    try:
        await manager.connect()
        async def unrelated_calls():
            return await asyncio.gather(manager.list_tools(), manager.list_tools(cursor='different'))
        results = await manager._run_discovery(unrelated_calls, timeout=1)
        assert len(results) == 2
        requests = [request for request in admitted if request['body']['method'] == 'tools/list']
        assert len({request['operation_id'] for request in requests}) == 2
    finally:
        await manager.cleanup()
    assert wire.closed


async def test_cancelling_inflight_discovery_blocks_sdk_cancellation_post_and_joins_wire(monkeypatch):
    started, joined = asyncio.Event(), asyncio.Event()
    async def handler(request, body):
        if body['method'] == 'tools/list':
            started.set()
            try: await asyncio.Event().wait()
            finally: joined.set()
    context, wire, manager = setup(monkeypatch, handler)
    discovery = MCPToolDiscovery(manager)
    await manager.connect()
    task = asyncio.create_task(discovery.list_tools())
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError): await asyncio.wait_for(task, 1)
    finally:
        await close_failed(manager)
    assert joined.is_set() and wire.closed and discovery.tools is None
    assert [item[2]['method'] for item in wire.requests] == ['initialize', 'notifications/initialized', 'tools/list']
    totals = await context.budget.snapshot()
    assert totals['activeJobs'] == 0 and totals['uncertain']['cost'] != '0'


async def test_success_waits_for_queued_initialization_notification_to_settle(monkeypatch):
    entered, release, connected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def handler(request, body):
        if body['method'] == 'notifications/initialized':
            entered.set()
            await release.wait()
    context, wire, manager = setup(monkeypatch, handler)
    async def lifecycle():
        try:
            await manager.connect()
            connected.set()
            assert (await context.budget.snapshot())['activeJobs'] == 0
        finally:
            await close_failed(manager)
    task = asyncio.create_task(lifecycle())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(connected.wait(), .03)
        assert not task.done()
    finally:
        release.set()
        await asyncio.wait_for(task, 1)
    assert connected.is_set() and wire.closed
    assert [item[2]['method'] for item in wire.requests] == ['initialize', 'notifications/initialized']


@pytest.mark.parametrize('kind', ['initialized', 'cancelled'])
async def test_late_actual_sdk_notification_cannot_reuse_a_successfully_closed_window(monkeypatch, kind):
    from mcp_types import CancelledNotification, CancelledNotificationParams, InitializedNotification
    context, wire, manager = setup(monkeypatch)
    await manager.connect()
    release, denied = asyncio.Event(), asyncio.Event()
    original_fail = manager._physical.fail
    def record_failure(error):
        original_fail(error)
        denied.set()
    manager._physical.fail = record_failure
    late = None
    async def operation():
        nonlocal late
        async def background():
            await release.wait()
            # ContextSendStream carries the originating operation window into
            # the real SDK writer even though this task wakes much later.
            notification = InitializedNotification() if kind == 'initialized' else CancelledNotification(
                params=CancelledNotificationParams(request_id='retired-request'))
            await manager._session.send_notification(notification)
        late = asyncio.create_task(background())
        return 'complete'
    try:
        assert await manager._physical.run(operation, timeout=1) == 'complete'
        assert manager._physical.failure is None
        release.set()
        if kind == 'initialized':
            with pytest.raises(CoordinationError, match='outlived'): await asyncio.wait_for(late, 1)
        else:
            await asyncio.wait_for(late, 1)
        await asyncio.wait_for(denied.wait(), 1)
        assert manager._physical.failure.code == 'mcp_operation_closed'
        assert len(wire.requests) == 2
        assert (await context.budget.snapshot())['spent']['tool_calls'] == 2
        with pytest.raises(CoordinationError, match='outlived'):
            await manager.list_tools()
        assert len(wire.requests) == 2
    finally:
        if late is not None and not late.done(): late.cancel()
        if late is not None: await asyncio.gather(late, return_exceptions=True)
        await close_failed(manager)
    assert wire.closed


async def test_waiting_permit_cannot_admit_after_originating_operation_completed(monkeypatch):
    from mcp_types import CancelledNotification, CancelledNotificationParams
    from magic_agents.coordination.budget import UsageBound
    from magic_agents.mcp.controlled_http import _WINDOW
    context, wire, manager = setup(monkeypatch)
    await manager.connect()
    await context.budget.reserve('held-slot', UsageBound(tool_calls=1), kind='job')
    queued, origin_done = asyncio.Event(), asyncio.Event()
    def authorize(path, kind, request):
        if request['body']['method'] == 'notifications/cancelled': queued.set()
    manager._dispatch.authorize = authorize
    async def operation():
        _WINDOW.get().task.add_done_callback(lambda _: origin_done.set())
        await manager._session.send_notification(CancelledNotification(
            params=CancelledNotificationParams(request_id='retired-request')))
        await queued.wait()
        return 'logical-result'
    task = asyncio.create_task(manager._physical.run(operation, timeout=1))
    try:
        await asyncio.wait_for(queued.wait(), 1)
        await asyncio.wait_for(origin_done.wait(), 1)
        assert not task.done()  # It still owns the already queued HTTP handler.
        await context.budget.settle('held-slot', UsageBound(tool_calls=1))
        with pytest.raises(CoordinationError, match='before physical admission'):
            await asyncio.wait_for(task, 1)
        assert len(wire.requests) == 2
        assert list(context._external_operations.values())[-1]['state'] == 'not_dispatched'
        totals = await context.budget.snapshot()
        assert totals['spent']['tool_calls'] == 3 and totals['uncertain']['cost'] == '0'
    finally:
        if not task.done(): task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await close_failed(manager)
    assert wire.closed


@pytest.mark.parametrize('returned', [False, True, 0, 'allow', 'awaitable'])
async def test_physical_guard_requires_synchronous_none_success_before_any_admission(returned):
    from magic_agents.coordination.dispatch import DispatchSession
    from test.test_coordination_dispatch import scope
    context, callbacks = scope(), []
    context.runtime.authorize_external = lambda *args: callbacks.append('authorize')
    session = DispatchSession(context, 'fetch')
    async def asynchronous(): return None
    def guard(): return asynchronous() if returned == 'awaitable' else returned
    async def effect(): callbacks.append('effect'); return None
    with pytest.raises(CoordinationError) as denied:
        await session.call('http.fetch', {}, effect, admission_guard=guard)
    assert denied.value.code == 'invalid_external_adapter'
    assert not callbacks and not context._external_operations
    totals = await context.budget.snapshot()
    assert totals['spent']['tool_calls'] == 0 and totals['activeJobs'] == 0
