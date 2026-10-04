"""Qualified stateless MCP HTTP: physical admission before SDK parsing.

The supported profile has finite UTF-8 JSON or POST-SSE replies, no remote
session lease, no reconnect or unsolicited GET, and no automatic HTTP retry.
Credentials and complete effective requests are private host policy inputs.
"""
import asyncio
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from uuid import uuid4

import anyio
import httpx2

from magic_agents.coordination.service import CoordinationError


PROFILE = 'http-stateless-bounded-v1'
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_REDIRECTS = 3
LOCAL_UNWIND_SECONDS = 1.0


def _strict_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Non-finite JSON value')
    try:
        value = json.loads(content, object_pairs_hook=pairs, parse_constant=constant)
        from magic_agents.coordination.dispatch import _encoded
        _encoded(value, max_bytes=MAX_RESPONSE_BYTES)
        return value
    except (ValueError, TypeError, UnicodeError, CoordinationError) as error:
        raise CoordinationError('mcp_response_invalid', 'MCP reply is not finite unambiguous bounded JSON') from error


@dataclass
class _OperationWindow:
    identity: str
    deadline: float
    request_remaining: int = MAX_REQUEST_BYTES
    response_remaining: int = MAX_RESPONSE_BYTES
    task: object = None
    closed: bool = False
    pending: dict = field(default_factory=dict)
    owner: object = None
    initialization_ready: bool = False
    initialization_notification_claimed: bool = False


_WINDOW = ContextVar('mcp_http_operation_window', default=None)
_INITIALIZATION_ACK = ContextVar('mcp_initialization_acknowledgement', default=None)


class _AcknowledgedWriteStream:
    """Keep the mandatory initialization acknowledgement in its operation."""
    def __init__(self, stream, transport):
        self.stream, self.transport = stream, transport

    async def send(self, item):
        if getattr(item.message, 'method', None) != 'notifications/initialized':
            return await self.stream.send(item)
        window = _WINDOW.get()
        if window is None or window.owner is not self.transport or window.closed:
            error = CoordinationError('mcp_operation_closed', 'MCP notification outlived its initiating operation')
            self.transport.fail(error)
            raise error
        if not window.initialization_ready or window.initialization_notification_claimed:
            error = CoordinationError('mcp_notification_unexpected', 'MCP initialization notification has no unmatched initialize result')
            self.transport.fail(error)
            raise error
        window.initialization_notification_claimed = True
        completed = asyncio.get_running_loop().create_future()
        try:
            token = _INITIALIZATION_ACK.set((self.transport, window, completed))
            try:
                await self.stream.send(item)
            finally:
                _INITIALIZATION_ACK.reset(token)
            await completed
        finally:
            if not completed.done(): completed.cancel()
            elif not completed.cancelled(): completed.exception()

    def close(self): return self.stream.close()
    async def aclose(self): return await self.stream.aclose()
    def clone(self): return type(self)(self.stream.clone(), self.transport)
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return await self.stream.__aexit__(*args)


class BoundedMCPTransport(httpx2.AsyncBaseTransport):
    def __init__(self, dispatch, *, timeout, transport=None):
        self.dispatch, self.timeout = dispatch, min(float(timeout), 300.0)
        self.inner = transport if transport is not None else httpx2.AsyncHTTPTransport(
            retries=0, trust_env=False, http2=False,
            limits=httpx2.Limits(max_connections=4, max_keepalive_connections=0))
        self.failure = None
        self.protocol_version = None

    def check(self):
        if self.failure is not None:
            raise self.failure
        self.dispatch.guard()

    def fail(self, error):
        if self.failure is None:
            self.failure = error

    async def run(self, operation, *, timeout=None):
        """A swallowed SDK transport failure still terminates its logical call."""
        self.check()
        window = _OperationWindow(uuid4().hex, asyncio.get_running_loop().time() +
                                  min(self.timeout, timeout if timeout is not None else self.timeout))
        window.owner = self
        token = _WINDOW.set(window)
        try:
            request = asyncio.create_task(operation())
            window.task = request
        finally:
            _WINDOW.reset(token)
        try:
            # Wire work stops at window.deadline. Allow a bounded local-only
            # unwind for the SDK to deliver the synthetic denied response before
            # its client read stream closes; no late physical request can pass.
            done, _ = await asyncio.wait({request}, timeout=max(0, window.deadline + .25 - asyncio.get_running_loop().time()))
            if not done:
                raise TimeoutError('MCP SDK response did not finish within its operation window')
            result = request.result()
            # Initialization awaits its explicit acknowledgement. Any remaining
            # registered physical handlers also settle under this same bound.
            while window.pending:
                _, unfinished = await asyncio.wait(set(window.pending), timeout=max(
                    0, window.deadline + .25 - asyncio.get_running_loop().time()))
                if unfinished:
                    raise TimeoutError('MCP physical work outlived its operation deadline')
            self.check()
            window.closed = True
            return result
        except TimeoutError as error:
            denied = CoordinationError('mcp_operation_deadline', 'MCP operation exhausted its shared physical-request deadline')
            self.fail(denied)
            raise denied from error
        except BaseException:
            if self.failure is not None:
                raise self.failure
            raise
        finally:
            window.closed = True
            if not request.done(): request.cancel()
            for task in set(window.pending.values()): task.cancel()
            with anyio.CancelScope(shield=True):
                # Never shield an unbounded gather. A broken SDK task cannot
                # extend the admitted operation indefinitely; closing its owned
                # session/transport below also closes its input streams.
                _, unfinished = await asyncio.wait({request, *window.pending}, timeout=LOCAL_UNWIND_SECONDS)
                if unfinished:
                    request.cancel()
                    self.fail(CoordinationError('mcp_cleanup_failed', 'MCP request did not unwind within its local teardown bound'))
                    request.add_done_callback(lambda task: None if task.cancelled() else task.exception())
                    raise self.failure
                if not request.cancelled(): request.exception()

    async def handle_async_request(self, request):
        wire = None
        registered = None
        window = None
        try:
            self.check()
            # The SDK follows eligible same-origin redirects itself. Every
            # resulting POST comes through this transport and is admitted anew.
            if request.method != 'POST':
                raise CoordinationError('mcp_transport_unsupported', 'This MCP profile permits finite stateless POST exchanges only')
            if 'mcp-session-id' in request.headers or 'last-event-id' in request.headers:
                raise CoordinationError('mcp_transport_unsupported', 'Stateless MCP cannot adopt a session or resume another request')
            window = _WINDOW.get()
            if window is None or window.owner is not self:
                raise CoordinationError('mcp_operation_required', 'MCP HTTP dispatch has no initiating operation window')
            if window.task is not None and (window.task.cancelled() or window.task.cancelling()):
                # SDK cancellation notifications and queued tasks cannot obtain
                # a new physical dispatch after their initiating operation dies.
                raise asyncio.CancelledError('Initiating MCP operation was cancelled')
            if window.closed or window.task is not None and window.task.done():
                raise CoordinationError('mcp_operation_closed', 'MCP request outlived its initiating operation')
            registered = asyncio.get_running_loop().create_future()
            window.pending[registered] = asyncio.current_task()
            remaining = window.deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise CoordinationError('mcp_operation_deadline', 'MCP operation deadline expired before physical dispatch')
            body = await request.aread()
            if len(body) > window.request_remaining:
                raise CoordinationError('external_payload_limit', 'MCP request exceeds its physical byte ceiling')
            window.request_remaining -= len(body)
            headers = list(request.headers.multi_items())
            if sum(len(k.encode()) + len(v.encode()) for k, v in headers) > MAX_HEADER_BYTES:
                raise CoordinationError('external_payload_limit', 'MCP request headers exceed their byte ceiling')
            # Disable library decompression before the raw byte cap. Compressed
            # responses are unsupported; identity bytes bound both wire/decoded.
            request.headers['accept-encoding'] = 'identity'
            wire = json.loads(body) if body else None
            private = {'profile': PROFILE, 'method': request.method, 'url': str(request.url),
                       'headers': [list(pair) for pair in request.headers.multi_items()], 'body': wire,
                       'operation_id': window.identity,
                       'timeout_seconds': remaining, 'max_request_bytes': MAX_REQUEST_BYTES,
                       'max_response_bytes': MAX_RESPONSE_BYTES}

            async def physical():
                response = None
                try:
                    self.check()
                    if window.closed or (window.task is not None and (window.task.done() or window.task.cancelling())):
                        raise CoordinationError('mcp_operation_closed', 'MCP request lost its operation before physical dispatch')
                    async with asyncio.timeout_at(window.deadline):
                        response = await self.inner.handle_async_request(request)
                        if response.status_code >= 400:
                            raise CoordinationError('mcp_http_outcome_unknown', 'MCP HTTP request failed; remote effects require reconciliation')
                        if 'mcp-session-id' in response.headers:
                            # The remote side may already have created a lease.
                            # Raise inside the reservation: retain exposure and
                            # never advertise successful stateless completion.
                            raise CoordinationError('mcp_stateful_session_unsupported', 'Remote MCP session creation requires a separately qualified lease adapter')
                        encoding = response.headers.get('content-encoding', 'identity').strip().lower()
                        if encoding not in ('', 'identity'):
                            raise CoordinationError('external_response_encoding', 'Coordinated MCP requires identity content encoding')
                        response_headers = [list(pair) for pair in response.headers.multi_items()]
                        if sum(len(k.encode()) + len(v.encode()) for k, v in response_headers) > MAX_HEADER_BYTES:
                            raise CoordinationError('external_response_limit', 'MCP response headers exceed their byte ceiling')
                        length = response.headers.get('content-length')
                        if length is not None and (not length.isascii() or not length.isdecimal()
                                or len(length.lstrip('0')) > 7 or int(length) > MAX_RESPONSE_BYTES):
                            raise CoordinationError('external_response_limit', 'MCP response length exceeds its byte ceiling')
                        chunks = []
                        async for chunk in response.stream:
                            if len(chunk) > window.response_remaining:
                                raise CoordinationError('external_response_limit', 'MCP response exceeds its physical byte ceiling')
                            window.response_remaining -= len(chunk)
                            if chunk: chunks.append(chunk)
                        content = b''.join(chunks)
                        text = content.decode('utf-8')
                        content_type = response.headers.get('content-type', '').lower()
                        if 300 <= response.status_code < 400:
                            pass  # The SDK admits each eligible redirect anew.
                        elif isinstance(wire, dict) and 'id' not in wire:
                            if response.status_code != 202 or content:
                                raise CoordinationError('mcp_response_invalid', 'MCP notification requires an empty accepted response')
                        elif response.status_code == 202:
                            raise CoordinationError('mcp_response_incomplete', 'An accepted MCP request has no completed outcome')
                        elif content_type.startswith('text/event-stream'):
                            # Require a complete response before SDK SSE code can
                            # attempt its automatic Last-Event-ID reconnection.
                            await self._complete_sse(response.status_code, response_headers, content, wire, request)
                        elif content_type.startswith('application/json'):
                            self._complete_message(_strict_json(content), wire)
                        else:
                            raise CoordinationError('mcp_response_invalid', 'MCP reply has an unsupported content type')
                        return {'status': response.status_code, 'headers': response_headers, 'body': text}
                finally:
                    if response is not None:
                        with anyio.CancelScope(shield=True):
                            async with asyncio.timeout(1):
                                await response.aclose()

            def admission_guard():
                if window.closed or window.task is not None and (window.task.done() or window.task.cancelling()):
                    raise CoordinationError('mcp_operation_closed', 'MCP request lost its operation before physical admission')
            result = await self.dispatch.call('mcp.http.request', private, physical, admission_guard=admission_guard)
            acknowledgement = _INITIALIZATION_ACK.get()
            if (acknowledgement is not None and acknowledgement[0] is self and acknowledgement[1] is window
                    and result['status'] == 202 and not acknowledgement[2].done()):
                acknowledgement[2].set_result(None)
            return httpx2.Response(result['status'], headers=result['headers'], content=result['body'].encode('utf-8'))
        except BaseException as error:
            # SDK post_writer and terminate_session intentionally catch broad
            # exceptions. The owner must still observe the original denial.
            if isinstance(error, Exception):
                from magic_agents.coordination.dispatch import is_protected
                if not is_protected(error):
                    failure = CoordinationError('mcp_transport_failure', 'Bounded MCP transport failed')
                    failure.__cause__ = error
                    error = failure
            self.fail(error)
            acknowledgement = _INITIALIZATION_ACK.get()
            if (acknowledgement is not None and acknowledgement[0] is self
                    and acknowledgement[1] is window and not acknowledgement[2].done()):
                acknowledgement[2].set_exception(error)
            if not isinstance(error, Exception):
                raise
            # A task-group exception would cancel the SDK session's parent
            # before it can surface the protected error. Resolve the wire
            # waiter locally while run() raises the latched original failure.
            return httpx2.Response(400, json={'jsonrpc': '2.0',
                'id': wire.get('id') if isinstance(wire, dict) else None,
                'error': {'code': -32603, 'message': 'Coordinated MCP transport operation denied'}})
        finally:
            if registered is not None:
                window.pending.pop(registered, None)
                if not registered.done(): registered.set_result(None)

    def _complete_message(self, message, request):
        from mcp_types import JSONRPCError, JSONRPCResponse, jsonrpc_message_adapter, methods
        try:
            if (not isinstance(message, dict) or not isinstance(request, dict)
                    or 'id' not in request or type(message.get('id')) is not type(request['id'])
                    or message.get('id') != request['id'] or ('result' in message) == ('error' in message)
                    or 'method' in message):
                raise ValueError('Uncorrelated or ambiguous MCP reply')
            parsed = jsonrpc_message_adapter.validate_python(message, by_name=False)
            if not isinstance(parsed, (JSONRPCResponse, JSONRPCError)):
                raise ValueError('Expected a JSON-RPC outcome')
            if isinstance(parsed, JSONRPCResponse):
                version = self.protocol_version or request.get('params', {}).get('protocolVersion') or '2025-11-25'
                methods.validate_server_result(request['method'], version, message['result'])
                if request['method'] == 'initialize':
                    self.protocol_version = message['result']['protocolVersion']
                    _WINDOW.get().initialization_ready = True
        except (ValueError, TypeError, KeyError) as error:
            raise CoordinationError('mcp_response_invalid', 'MCP reply is invalid or does not complete its originating request') from error

    async def _complete_sse(self, status, headers, content, request, http_request):
        if not isinstance(request, dict) or 'id' not in request:
            raise CoordinationError('mcp_response_incomplete', 'Unexpected SSE reply to an MCP notification')
        response = httpx2.Response(status, headers=headers, content=content, request=http_request)
        found = False
        async for event in httpx2.EventSource(response):
            if event.event != 'message' or not event.data: continue
            message = _strict_json(event.data)
            if isinstance(message, dict) and ('result' in message or 'error' in message):
                self._complete_message(message, request)
                found = True
                break
        if not found:
            raise CoordinationError('mcp_response_incomplete', 'MCP SSE ended without its response; automatic resumption is unsupported')

    async def aclose(self):
        with anyio.CancelScope(shield=True):
            async with asyncio.timeout(1):
                await self.inner.aclose()


@asynccontextmanager
async def controlled_http_client(config, dispatch):
    """Use the installed SDK with a qualified, per-physical-request transport."""
    from mcp.client.streamable_http import streamable_http_client
    from importlib.metadata import PackageNotFoundError, version
    from magic_agents.node_system.NodeMcp import NodeMcp
    try:
        qualified = all(version(package) == expected for package, expected in NodeMcp.coordination_external_transport_versions.items())
    except PackageNotFoundError:
        qualified = False
    if not qualified:
        raise CoordinationError('mcp_transport_version_unsupported', 'Installed MCP transport dependencies have not been qualified')
    transport = BoundedMCPTransport(dispatch,
        timeout=max(config.init_timeout, config.tool_timeout, config.discovery_timeout))
    async with httpx2.AsyncClient(transport=transport, headers=config.headers or {},
            trust_env=False, follow_redirects=False, max_redirects=MAX_REDIRECTS,
            timeout=transport.timeout) as client:
        async with streamable_http_client(config.url, http_client=client, terminate_on_close=False) as streams:
            # A session ID is rejected before its response reaches the SDK;
            # terminate_on_close=False cannot suppress an accepted remote lease.
            yield (streams[0], _AcknowledgedWriteStream(streams[1], transport), *streams[2:]), transport
