"""Request rendering, execution and typed failures shared by every Fetch mode.

Plain step mode, Hook-controlled step mode and tool mode (an LLM tool) all
render their HTTP request here, with one sandboxed Jinja environment.

Trust model:

* Only the Fetch node's own static configuration (node data and the
  ``tool_parameters`` request template) is compiled as a template, and only
  there are ``{{env.NAME}}`` placeholders resolved.
* Values that arrive at runtime (input handles, Hook content, model tool
  arguments, upstream node output, web content) are data. They are inserted
  into the rendered request as values and are never compiled or env-resolved.

Body fields (``json_data``, ``data``, ``params``) are rendered per string leaf
in a JSON string context: the result of every expression is JSON-escaped, the
literal template text is escaped too, and the rendered leaf is JSON-decoded
back. ``{{ value }}`` and the legacy ``{{ (value | tojson)[1:-1] }}`` both
produce the exact original text, and an inserted value can never add keys.

Failures raise :class:`FetchError` subclasses whose class names
(``HTTPError``, ``NetworkError``, ``TemplateError``, ``UnexpectedError``) are
what traces, debug SSE events and observers report. Their message and
``context`` never contain an env-resolved value, a query string, URL user
info or response headers.
"""
from __future__ import annotations

import asyncio
import copy
import math
import http
import itertools
import json
import logging
import re
import uuid
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import quote, quote_plus, unquote, urlsplit

import aiohttp
import jinja2
import yarl
from jinja2 import nodes, pass_context
from jinja2.sandbox import SandboxedEnvironment
from markupsafe import Markup

from magic_agents.util import template_parser
from magic_agents.util.env_resolver import resolve_env_string

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed failures
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """A Fetch request that could not be built, sent or read.

    ``error_type`` is the class name. ``context`` is a sanitized dict (method,
    URL without query/fragment/user info, status code, field names). The
    response body of an HTTP error is kept on ``response_body`` for Hook
    recovery only; it is never part of the message or the context.
    """

    def __init__(self, message: str, *, context: Optional[dict] = None,
                 status_code: Optional[int] = None, response_body: Optional[str] = None,
                 reason: Optional[str] = None, exception_type: Optional[str] = None):
        super().__init__(message)
        self.context = dict(context or {})
        self.status_code = status_code
        self.response_body = response_body
        self.reason = reason
        self.exception_type = exception_type
        if status_code is not None:
            self.context.setdefault("status_code", status_code)

    @property
    def error_type(self) -> str:
        return type(self).__name__


class HTTPError(FetchError):
    """The server answered with a 4xx/5xx status."""


class NetworkError(FetchError):
    """The request never got an HTTP answer (connection, DNS, TLS, timeout)."""


class TemplateError(FetchError):
    """The request could not be built from the node configuration or inputs."""


class UnexpectedError(FetchError):
    """Any other failure, including a malformed JSON response body."""


def operation_failure_for(error: FetchError):
    """Map a step-mode failure to the Hook ``OperationFailure`` contract.

    HTTP and network failures keep their typed codes. Template and unexpected
    failures are returned unchanged: the invocation adapter reports them as
    ``NODE_EXCEPTION`` with ``details.exception_type`` set to the class name.
    """
    from magic_agents.hooks.invocation_control import OperationFailure

    if isinstance(error, HTTPError):
        status = error.status_code
        return OperationFailure(
            "HTTP_ERROR", f"HTTP {status}: {error.reason}",
            retryable=status == 429 or status >= 500,
            details={"http_status": status, "status_code": status,
                     "response_body": error.response_body, "context": dict(error.context)})
    if isinstance(error, NetworkError):
        return OperationFailure("NETWORK_ERROR", str(error), retryable=True,
                                details={"exception_type": error.exception_type or "NetworkError",
                                         "context": dict(error.context)})
    return error


def error_context(error: BaseException) -> Optional[dict]:
    """The sanitized Fetch diagnostic context carried by a node failure.

    A step-mode ``FetchError`` carries it directly. Under Hook control the
    node fails with ``OperationFailure``; its outcome ``details.context``
    holds the same sanitized dict.
    """
    if isinstance(error, FetchError):
        return dict(error.context) if error.context else None
    outcome = getattr(error, "outcome", None)
    if isinstance(outcome, dict):
        details = (outcome.get("error") or {}).get("details")
        if isinstance(details, dict) and isinstance(details.get("context"), dict) and details["context"]:
            return dict(details["context"])
    return None


# ---------------------------------------------------------------------------
# Secrets: env values resolved for one request are scrubbed from diagnostics
# ---------------------------------------------------------------------------


class FetchSecrets:
    """Env values resolved while rendering one request.

    Every message, log line and diagnostic context built for that request is
    passed through :meth:`scrub`. Values shorter than ``MIN_LENGTH`` are not
    tracked: they cannot be told apart from ordinary text.
    """

    MIN_LENGTH = 4
    REPLACEMENT = "[REDACTED]"

    def __init__(self) -> None:
        self._values: set[str] = set()

    def add(self, value: Any) -> None:
        if not isinstance(value, str) or len(value) < self.MIN_LENGTH:
            return
        # A secret can reach a diagnostic in an encoded form: percent-encoded
        # by aiohttp/yarl (query params, path, normalized URLs), form-encoded,
        # or JSON-escaped in an echoed body.
        for variant in _encoded_variants(value):
            if len(variant) >= self.MIN_LENGTH:
                self._values.add(variant)

    def scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            for secret in sorted(self._values, key=len, reverse=True):
                value = value.replace(secret, self.REPLACEMENT)
            return value
        if isinstance(value, dict):
            return {key: self.scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.scrub(item) for item in value]
        return value


def _encoded_variants(value: str) -> set[str]:
    variants = {value, quote(value, safe=""), quote(value), quote_plus(value), quote_plus(value, safe="/"),
                unquote(value), json.dumps(value)[1:-1], json.dumps(value, ensure_ascii=False)[1:-1]}
    builders = (
        lambda: yarl.URL.build(query={"k": value}).raw_query_string[2:],
        lambda: yarl.URL.build(path="/" + value).raw_path[1:],
        lambda: yarl.URL("http://h/?k=" + value).raw_query_string[2:],
        lambda: yarl.URL("http://h/" + value).raw_path[1:],
    )
    for build in builders:
        try:
            variants.add(build())
        except Exception:  # not representable in that URL part
            pass
    return variants


def safe_url(url: Any, secrets: Optional[FetchSecrets] = None) -> str:
    """``scheme://host[:port]/path``: no query, fragment or user info."""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        try:
            port = parts.port
        except ValueError:
            port = None
    except ValueError:
        return "[invalid url]"
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{port}" if port else "")
    result = f"{parts.scheme}://{netloc}{parts.path}" if parts.scheme or netloc else parts.path
    return secrets.scrub(result) if secrets is not None else result


# ---------------------------------------------------------------------------
# The sandboxed environment
# ---------------------------------------------------------------------------


class _JsonText(Markup):
    """``tojson`` output. Markup, so the JSON string context keeps its escapes.

    A whole ``tojson`` document carries ``document = True``; slicing it (the
    legacy ``(x | tojson)[1:-1]`` idiom) yields a fragment without the flag.
    """


def _tojson(value, indent=None):
    text = _JsonText(template_parser.tojson(value, indent=indent))
    text.document = True
    return text


def _json_fragment(text: str) -> str:
    """JSON-escape text for the inside of a JSON string literal."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


_ESCAPE_FILTER = "_fetch_json_escape"


_SLOT_FILTER = "_fetch_json_slot"
_ENV_REF = "_fetch_env"
_SLOTS_REF = "_fetch_slots"
_NONCE_REF = "_fetch_nonce"


def _json_escape(value):
    """Escape one expression result for the inside of a JSON string.

    Only a sliced ``tojson`` fragment (the legacy ``(x | tojson)[1:-1]``
    idiom) is already JSON-escaped and passes through. Every other value,
    other Markup (``| e``, ``| safe``) and a whole ``tojson`` document
    included, is escaped: a body leaf holding ``{{ x | tojson }}`` holds that
    JSON text as a string.
    """
    if isinstance(value, _JsonText) and not getattr(value, "document", False):
        return str(value)
    return _json_fragment(str(value))


@pass_context
def _json_slot(context, value):
    """JSON-text mode: record the value; :func:`_fill_json_slots` places it."""
    slots = context.get(_SLOTS_REF)
    slots.append(value)
    return "\x00%s:%d\x00" % (context.get(_NONCE_REF), len(slots) - 1)


def _slot_text(value, in_string: bool) -> str:
    if in_string:
        return _json_escape(value)
    if isinstance(value, _JsonText):
        return str(value)
    text = str(value)
    try:
        json.loads(text)
    except ValueError:
        # Not one JSON value: escaped, so it can never add keys or items.
        return _json_fragment(text)
    # Exactly one JSON value (legacy raw insertion of JSON text): it fills a
    # value position and cannot add a key or a sibling item.
    return text


def _fill_json_slots(rendered: str, nonce: str, slots: list) -> str:
    """Place each expression result by where it sits in the rendered JSON.

    Inside a string literal it is JSON-escaped; outside one it is inserted
    raw only when it is exactly one JSON value (or ``tojson`` output).
    """
    marker = re.compile("\x00%s:(\\d+)\x00" % re.escape(nonce))
    state = {"in_string": False, "escaped": False}

    def scan(text):
        for char in text:
            if state["escaped"]:
                state["escaped"] = False
            elif state["in_string"] and char == "\\":
                state["escaped"] = True
            elif char == '"':
                state["in_string"] = not state["in_string"]

    parts, position = [], 0
    for match in marker.finditer(rendered):
        literal = rendered[position:match.start()]
        scan(literal)
        piece = _slot_text(slots[int(match.group(1))], state["in_string"])
        scan(piece)
        parts += [literal, piece]
        position = match.end()
    parts.append(rendered[position:])
    return "".join(parts)


def _build_environment() -> SandboxedEnvironment:
    # SandboxedEnvironment blocks attribute escapes (``__globals__``,
    # ``__class__``, ``str.format``) while keeping common authored idioms such
    # as ``{% set _ = items.append(x) %}``. Inputs are copied per request
    # (see FetchRenderer), so a template cannot mutate another node's data.
    environment = SandboxedEnvironment(keep_trailing_newline=True)
    environment.filters["fromjson"] = template_parser.fromjson
    environment.filters["regex_replace"] = template_parser.regex_replace
    environment.filters["regex_findall"] = template_parser.regex_findall
    environment.filters["tojson"] = _tojson
    environment.filters[_ESCAPE_FILTER] = _json_escape
    environment.filters[_SLOT_FILTER] = _json_slot
    return environment


FETCH_TEMPLATE_ENV = _build_environment()

_PLAIN, _LEAF, _JSON_TEXT = "plain", "leaf", "json_text"
# Bodies of these statements are captured into a value (macro result, set
# block); escaping them as well would escape that value twice. Filter and
# call blocks write inline: they are rewritten into a set block plus an
# output first (see _wrap_outputs).
_CAPTURING = (nodes.Macro, nodes.CallBlock, nodes.AssignBlock, nodes.FilterBlock)
_INLINE_BLOCKS = (nodes.FilterBlock, nodes.CallBlock)


def has_template_syntax(value: Any) -> bool:
    return isinstance(value, str) and any(marker in value for marker in ("{{", "{%", "{#"))


def _replace_env_placeholders(node: nodes.Node, names: list[str]) -> None:
    """Turn ``{{ env.NAME }}`` outputs into a lookup of the resolved value.

    The resolved value never becomes template source; it is passed to the
    render as a variable, like any other inserted value.
    """
    if isinstance(node, nodes.Output):
        for index, child in enumerate(node.nodes):
            if (isinstance(child, nodes.Getattr) and isinstance(child.node, nodes.Name)
                    and child.node.name == "env" and child.node.ctx == "load"):
                names.append(child.attr)
                node.nodes[index] = nodes.Getitem(
                    nodes.Name(_ENV_REF, "load", lineno=child.lineno),
                    nodes.Const(child.attr, lineno=child.lineno), "load", lineno=child.lineno)
        return
    for child in node.iter_child_nodes():
        _replace_env_placeholders(child, names)


def _capture_inline_block(block: nodes.Node, name: str) -> list[nodes.Node]:
    """``{% filter f %}b{% endfilter %}`` -> ``{% set n | f %}b{% endset %}{{ n }}``.

    A call block becomes ``{% set n %}{% call ... %}{% endset %}{{ n }}``.
    The block's text then reaches the output through an expression, which is
    escaped like every other inserted value.
    """
    lineno = block.lineno
    if isinstance(block, nodes.FilterBlock):
        assign = nodes.AssignBlock(nodes.Name(name, "store", lineno=lineno), block.filter, block.body, lineno=lineno)
    else:
        assign = nodes.AssignBlock(nodes.Name(name, "store", lineno=lineno), None, [block], lineno=lineno)
    return [assign, nodes.Output([nodes.Name(name, "load", lineno=lineno)], lineno=lineno)]


def _wrap_outputs(node: nodes.Node, filter_name: str, *, escape_literals: bool, counter) -> None:
    """Pass every expression result through ``filter_name``.

    With ``escape_literals`` the literal template text is JSON-escaped too
    (body leaves are rendered inside a JSON string).
    """
    if isinstance(node, _CAPTURING):
        return
    if isinstance(node, nodes.Output):
        children = []
        for child in node.nodes:
            if isinstance(child, nodes.TemplateData):
                if escape_literals:
                    child.data = _json_fragment(child.data)
                children.append(child)
            else:
                children.append(nodes.Filter(child, filter_name, [], [], None, None, lineno=child.lineno))
        node.nodes = children
        return
    for field in node.fields:
        value = getattr(node, field, None)
        if isinstance(value, list) and any(isinstance(item, _INLINE_BLOCKS) for item in value):
            rewritten = []
            for item in value:
                if isinstance(item, _INLINE_BLOCKS):
                    rewritten += _capture_inline_block(item, "_fetch_block_%d" % next(counter))
                else:
                    rewritten.append(item)
            setattr(node, field, rewritten)
    for child in node.iter_child_nodes():
        _wrap_outputs(child, filter_name, escape_literals=escape_literals, counter=counter)


@lru_cache(maxsize=512)
def _compile(source: str, mode: str):
    tree = FETCH_TEMPLATE_ENV.parse(source)
    names: list[str] = []
    _replace_env_placeholders(tree, names)
    if mode == _LEAF:
        _wrap_outputs(tree, _ESCAPE_FILTER, escape_literals=True, counter=itertools.count())
    elif mode == _JSON_TEXT:
        _wrap_outputs(tree, _SLOT_FILTER, escape_literals=False, counter=itertools.count())
    tree.set_environment(FETCH_TEMPLATE_ENV)
    return FETCH_TEMPLATE_ENV.from_string(tree), tuple(dict.fromkeys(names))


def _copy_data(value: Any) -> Any:
    """Copy JSON-like containers so a template cannot mutate shared inputs."""
    if isinstance(value, dict):
        return {key: _copy_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_data(item) for item in value]
    return value


def _template_message(exc: BaseException) -> str:
    if isinstance(exc, jinja2.TemplateSyntaxError):
        # str() of a syntax error quotes the template source line; the message
        # alone does not.
        return f"invalid template syntax (line {exc.lineno}): {exc.message}"
    if isinstance(exc, jinja2.TemplateError):
        return exc.message or type(exc).__name__
    return str(exc) or type(exc).__name__


class FetchRenderer:
    """Render static Fetch configuration for one request.

    ``context`` holds the runtime values (node inputs or tool arguments) that
    templates may insert. Env placeholders resolved here are recorded in
    ``secrets``.
    """

    def __init__(self, context: dict, secrets: FetchSecrets, *, base_context: Optional[dict] = None):
        self.context = _copy_data(dict(context or {}))
        self.secrets = secrets
        self.base_context = base_context or {}

    def _fail(self, field: str, exc: BaseException, prefix: Optional[str] = None) -> TemplateError:
        prefix = prefix or ("URL templating failed" if field == "url" else f"Request {field} rendering failed")
        return TemplateError(self.secrets.scrub(f"{prefix}: {_template_message(exc)}"), context={
            **self.base_context, "field": field,
            "available_inputs": sorted(str(key) for key in self.context),
            "exception_type": type(exc).__name__,
        })

    def _render(self, source: str, mode: str, field: str, extra: Optional[dict] = None) -> str:
        try:
            template, names = _compile(source, mode)
            values = {}
            for name in names:
                value = resolve_env_string("{{env.%s}}" % name)
                self.secrets.add(value)
                values[name] = value
            return template.render({**self.context, **(extra or {}), _ENV_REF: values})
        except Exception as exc:  # jinja2 errors, sandbox SecurityError, filter errors
            raise self._fail(field, exc) from None

    def text(self, source: Any, field: str) -> Any:
        """Plain text (URL, header value, method): values inserted as text."""
        if not has_template_syntax(source):
            return source
        return self._render(source, _PLAIN, field)

    def leaf(self, source: Any, field: str) -> Any:
        """One string in a body: rendered in a JSON string context."""
        if not has_template_syntax(source):
            return source
        rendered = self._render(source, _LEAF, field)
        try:
            return json.loads('"' + rendered + '"', strict=False)
        except json.JSONDecodeError as exc:
            raise self._fail(field, ValueError(
                f"the template output is not a plain string value ({exc.msg}); "
                "use {{ value }} for text")) from None

    def tree(self, value: Any, field: str) -> Any:
        """Render every string leaf of nested dicts and lists."""
        if isinstance(value, str):
            return self.leaf(value, field)
        if isinstance(value, list):
            return [self.tree(item, field) for item in value]
        if isinstance(value, dict):
            return {key: self.tree(item, field) for key, item in value.items()}
        return value

    def json_source(self, source: str, field: str) -> Any:
        """A JSON-text field (``json_data``/``params`` given as a string).

        Text that is valid JSON before rendering is parsed and rendered per
        leaf. Other text (for example ``{"n": {{ count }}}``) is rendered as a
        whole and then parsed. Each expression result is placed by position:
        inside a JSON string it is escaped; outside one it is inserted raw
        only when it is exactly one JSON value (``tojson`` output, a number,
        JSON text such as ``["a", "b"]``), otherwise escaped. An inserted value
        therefore never adds a key or a sibling item.
        """
        try:
            parsed = json.loads(source)
        except json.JSONDecodeError:
            pass
        else:
            return self.tree(parsed, field)
        nonce, slots = uuid.uuid4().hex, []
        rendered = self._render(source, _JSON_TEXT, field, {_SLOTS_REF: slots, _NONCE_REF: nonce})
        rendered = _fill_json_slots(rendered, nonce, slots)
        try:
            return json.loads(rendered, strict=False)
        except json.JSONDecodeError as exc:
            raise self._fail(field, ValueError(
                f"rendered text is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})")) from None

    def body(self, value: Any, field: str) -> Any:
        """``json_data``/``params`` (JSON) or ``data`` (raw text or form)."""
        if isinstance(value, str) and field != "data":
            return self.json_source(value, field)
        return self.tree(value, field)


def parse_json_input(value: Any, field: str, base_context: Optional[dict] = None) -> Any:
    """JSON arriving on an input handle: parsed as data, never rendered."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise TemplateError(
            f"Request {field} input is not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})",
            context={**(base_context or {}), "field": field, "exception_type": "JSONDecodeError"}) from None


def parse_mapping(value: Any, field: str, base_context: Optional[dict] = None) -> dict:
    """A headers/params mapping given as a dict or as JSON object text."""
    if value is None:
        return {}
    if isinstance(value, str):
        if not value.strip():
            return {}
        value = parse_json_input(value, field, base_context)
    if not isinstance(value, dict):
        raise TemplateError(f"Request {field} must be a JSON object, got {type(value).__name__}",
                            context={**(base_context or {}), "field": field})
    return dict(value)


def check_header_values(headers: dict, base_context: Optional[dict] = None) -> dict:
    """Reject line breaks in header names and values (header injection)."""
    for name, value in headers.items():
        for part, text in (("name", name), ("value", value)):
            if isinstance(text, str) and ("\r" in text or "\n" in text):
                label = name if part == "value" and isinstance(name, str) and "\r" not in name and "\n" not in name else "[header]"
                raise TemplateError(f"Request header {part} for '{label}' contains a line break",
                                    context={**(base_context or {}), "field": "headers"})
    return headers


# RFC 9110 method token.
_METHOD_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


def check_method(method: Any) -> str:
    """Reject a method that is not an HTTP token (line breaks, spaces)."""
    if not isinstance(method, str) or not _METHOD_TOKEN.fullmatch(method):
        raise TemplateError("Request method is not a valid HTTP method token",
                            context={"field": "method"})
    return method


# ---------------------------------------------------------------------------
# The prepared request and the step-mode HTTP core
# ---------------------------------------------------------------------------


class FetchRequest:
    """A fully rendered request plus the secrets resolved while rendering it."""

    def __init__(self, method: str, url: str, headers: dict, *, params: Any = None,
                 json_body: Any = None, data_body: Any = None,
                 secrets: Optional[FetchSecrets] = None):
        self.method = method
        self.url = url
        self.headers = headers
        self.params = params
        self.json_body = json_body
        self.data_body = data_body
        self.secrets = secrets or FetchSecrets()

    @property
    def has_body(self) -> bool:
        return self.json_body is not None or self.data_body is not None

    @property
    def safe_url(self) -> str:
        return safe_url(self.url, self.secrets)

    def context(self, **extra) -> dict:
        return {"method": self.method, "url": self.safe_url, **extra}

    def aiohttp_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {"method": self.method, "url": self.url, "headers": self.headers}
        if self.params is not None:
            kwargs["params"] = self.params
        if self.json_body is not None:
            kwargs["json"] = self.json_body
        elif self.data_body is not None:
            kwargs["data"] = self.data_body
        return kwargs


async def _read_text(response) -> Optional[str]:
    try:
        text = await response.text()
    except Exception:  # body unreadable or already released
        return None
    return text if isinstance(text, str) else None


def _reason(status: int, reason: Any) -> str:
    if isinstance(reason, str) and reason:
        return reason
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return "HTTP error"


def http_error(request: FetchRequest, status: int, reason: Any, body: Optional[str] = None) -> HTTPError:
    reason = request.secrets.scrub(_reason(status, reason))
    return HTTPError(f"HTTP request failed with status {status}: {reason}",
                     context=request.context(status_code=status), status_code=status,
                     response_body=request.secrets.scrub(body), reason=reason)


def network_message(request: FetchRequest, exc: BaseException) -> str:
    """Tool-mode text of a client error without query, user info or secrets.

    Invalid-URL errors quote the whole URL: only its safe form is kept. Other
    client errors name at most host and port; any full URL is replaced too.
    """
    if isinstance(exc, aiohttp.InvalidURL):
        description = getattr(exc, "description", None)
        safe = safe_url(exc.url, request.secrets)
        return request.secrets.scrub(f"{safe} - {description}") if description else safe
    text = str(exc)
    if isinstance(request.url, str) and request.url:
        text = text.replace(request.url, request.safe_url)
        try:
            text = text.replace(str(yarl.URL(request.url)), request.safe_url)
        except Exception:  # not a parseable URL
            pass
    return request.secrets.scrub(text)


def network_error(request: FetchRequest, exc: BaseException) -> NetworkError:
    kind = type(exc).__name__
    detail = "timed out" if isinstance(exc, asyncio.TimeoutError) else "no HTTP response"
    return NetworkError(f"Network request failed ({kind}, {detail}): {request.method} {request.safe_url}",
                        context=request.context(exception_type=kind), exception_type=kind)


async def send_step_request(request: FetchRequest) -> Any:
    """Send one request with step-mode semantics and return the parsed body.

    JSON content types are decoded (a malformed JSON body raises
    ``UnexpectedError``); every other content type is returned as text.
    Status 4xx/5xx raises ``HTTPError``.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(**request.aiohttp_kwargs()) as response:
                status = response.status
                if isinstance(status, int) and status >= 400:
                    raise http_error(request, status, response.reason, await _read_text(response))
                response.raise_for_status()
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    # Documentation endpoints such as llms.txt return text/plain.
                    return await response.text()
                except ValueError as exc:
                    raise UnexpectedError(
                        f"Response body is not valid JSON: {getattr(exc, 'msg', None) or type(exc).__name__}",
                        context=request.context(status_code=status, exception_type=type(exc).__name__),
                        exception_type=type(exc).__name__) from None
    except FetchError:
        raise
    except aiohttp.ClientResponseError as exc:
        raise http_error(request, exc.status, exc.message) from None
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise network_error(request, exc) from None
    except Exception as exc:
        kind = type(exc).__name__
        # Exception text may quote the full URL; keep only its safe form.
        detail = str(exc).replace(request.url, request.safe_url) if isinstance(request.url, str) else str(exc)
        raise UnexpectedError(
            request.secrets.scrub(f"Unexpected error during fetch ({kind}): {detail}"),
            context=request.context(exception_type=kind), exception_type=kind) from None


def admitted_http_options(request: FetchRequest):
    """Freeze and bound the actual request before host admission or transport."""
    from magic_agents.coordination.service import CoordinationError
    kwargs = copy.deepcopy(request.aiohttp_kwargs())
    try:
        # aiohttp's default JSON serializer uses ASCII escaping. Quote its
        # physical expansion, not the smaller canonical UTF-8 representation.
        if 'json' in kwargs:
            body = json.dumps(kwargs.pop('json'), allow_nan=False)
            kwargs['data'] = body
            headers = kwargs.setdefault('headers', {})
            if not any(key.lower() == 'content-type' for key in headers):
                headers['Content-Type'] = 'application/json'
        if 'data' in kwargs and not isinstance(kwargs['data'], str):
            # No unbounded multipart/file streams in the qualified adapter.
            raise ValueError('Controlled HTTP supports text or JSON bodies')
        encoded = json.dumps(kwargs, ensure_ascii=True, allow_nan=False).encode('utf-8')
        if len(encoded) > 4 * 1024 * 1024:
            raise ValueError('Physical HTTP request byte ceiling exceeded')
    except (TypeError, ValueError, UnicodeError) as error:
        raise CoordinationError('external_request_limit', 'HTTP request is not bounded supported JSON/text') from error
    kwargs['allow_redirects'] = False
    kwargs['auto_decompress'] = False
    return kwargs


async def send_admitted_request(request: FetchRequest, *, timeout: float, max_response_bytes: int = 2 * 1024 * 1024,
                                strict_json: bool = False, prepared: dict | None = None):
    """One controlled HTTP dispatch; no redirects or unbounded body buffering."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        from magic_agents.coordination.service import CoordinationError
        raise CoordinationError('external_timeout_limit', 'Controlled HTTP requires a finite timeout <= 300 seconds')
    kwargs = copy.deepcopy(prepared) if prepared is not None else admitted_http_options(request)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout, connect=min(30, timeout),
                                                                  sock_read=min(30, timeout)), trust_env=False) as session:
        from magic_llm.util.http import disable_automatic_retries
        disable_automatic_retries(session)
        async with session.request(**kwargs) as response:
            from magic_llm.util.http import read_bounded_response, HttpError
            try:
                body = (await read_bounded_response(response, max_response_bytes)).decode('utf-8')
            except HttpError as error:
                from magic_agents.coordination.service import CoordinationError
                raise CoordinationError('external_response_limit', 'HTTP response violated its admitted transport bounds') from error
            if not 200 <= response.status < 300:
                raise http_error(request, response.status, response.reason, body)
            if strict_json:
                from magic_agents.execution.condition_evaluator_llm import _object_without_duplicates
                return json.loads(body, object_pairs_hook=_object_without_duplicates,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            try: return json.loads(body)
            except json.JSONDecodeError: return body


async def coordinated_fetch(scope, node_id, request: FetchRequest):
    """Quote the rendered request, retaining semantic fields and private headers."""
    session = scope.dispatch_session(node_id)
    timeout = min(300, max(0, scope.budget.deadline - scope.budget.clock()))
    prepared = admitted_http_options(request)
    effective = {**copy.deepcopy(prepared), 'timeout_seconds': timeout,
                 'max_request_bytes': 4 * 1024 * 1024, 'max_response_bytes': 2 * 1024 * 1024}
    async def operation():
        return await send_admitted_request(request, timeout=timeout,
                                           max_response_bytes=effective['max_response_bytes'], prepared=prepared)
    return await session.call('http.fetch', effective, operation)
