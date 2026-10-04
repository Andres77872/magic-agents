import json
import logging
import re
from typing import Any, Iterable, Optional

import aiohttp

from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.node_system.fetch_request import (
    FetchRenderer,
    FetchRequest,
    FetchSecrets,
    TemplateError as FetchTemplateError,
    check_header_values,
    check_method,
    network_message,
    parse_json_input,
    parse_mapping,
    send_step_request,
)
from magic_agents.util.primitive_coercion import coerce_primitive_by_type, input_has_value

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redaction helper for diagnostic log output
# ---------------------------------------------------------------------------

_SENSITIVE_KEY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r'api.?key', r'secret', r'token', r'authorization',
        r'auth', r'password', r'passwd', r'credential',
        r'access.?key', r'private.?key',
    ]
]

_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key>(?P<quote>[\"'])(?P<quoted_key>[^\"']+)(?P=quote)|"
    r"(?P<bare_key>[A-Za-z_][A-Za-z0-9_.-]*))(?P<separator>\s*[:=]\s*)"
)


def _is_sensitive_key(key: Any) -> bool:
    return isinstance(key, str) and any(
        pattern.search(key) for pattern in _SENSITIVE_KEY_PATTERNS
    )


def _redact_json_value(value: Any) -> Any:
    """Recursively redact sensitive mapping values in parsed JSON."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive_key(key) else _redact_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    return value


def _redact_sensitive_assignments(text: str) -> str:
    """Best-effort redaction for malformed JSON and key/value text.

    Error bodies are not guaranteed to be valid JSON. This scanner recognizes
    quoted JSON keys as well as bare ``key=value``/``Header: value`` forms. If
    a sensitive quoted value is truncated before its closing quote, the rest of
    the input is discarded as part of that value instead of being logged.
    """
    pieces: list[str] = []
    cursor = 0

    while match := _ASSIGNMENT_PATTERN.search(text, cursor):
        pieces.append(text[cursor:match.end()])
        cursor = match.end()
        key = match.group("quoted_key") or match.group("bare_key") or ""
        if not _is_sensitive_key(key):
            continue

        if cursor >= len(text):
            pieces.append('"[REDACTED]"')
            break

        first = text[cursor]
        if first in {'"', "'"}:
            quote = first
            value_end = cursor + 1
            escaped = False
            while value_end < len(text):
                char = text[value_end]
                if char == quote and not escaped:
                    value_end += 1
                    break
                if char == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
                value_end += 1
            pieces.append(f'{quote}[REDACTED]{quote}')
            cursor = value_end
            continue

        if first in "[{":
            # Valid containers are handled by the recursive JSON path. For a
            # malformed sensitive container, suppress the remaining fragment;
            # trying to recover its boundary risks exposing nested credentials.
            pieces.append('"[REDACTED]"')
            cursor = len(text)
            break

        value_end = cursor
        while value_end < len(text) and text[value_end] not in ",}]\r\n&":
            value_end += 1
        pieces.append('"[REDACTED]"')
        cursor = value_end

    pieces.append(text[cursor:])
    return "".join(pieces)


def _redact_body_preview(body_text: str, max_length: int = 500) -> str:
    """Truncate and redact sensitive fields from a response body for logging.

    Valid JSON is parsed before truncation so nested secrets can be redacted
    without a long value turning the preview into invalid JSON. Malformed JSON
    and key/value text use a conservative scanner that also handles truncated
    quoted values. The safe result is truncated only after redaction.

    Args:
        body_text: Raw response body text.
        max_length: Maximum character length before truncation (default 500).

    Returns:
        Redacted body preview string safe for diagnostic log output.
    """
    if max_length <= 0:
        return ""

    try:
        parsed = json.loads(body_text)
    except json.JSONDecodeError:
        return _redact_sensitive_assignments(body_text)[:max_length]

    return json.dumps(_redact_json_value(parsed))[:max_length]


def _scrub_exception(error: BaseException, secrets: FetchSecrets) -> str:
    """Exception text without any env value resolved for the request."""
    return secrets.scrub(str(error))


_TEMPLATE_VARIABLE = re.compile(r'\{\{(\w+)\}\}')


def _template_variables(value: Any) -> list[str]:
    """``{{name}}`` placeholders in a string or in any nested string leaf.

    The renderer fills nested dict values and list items too, so the tool
    schema must expose their variables as well.
    """
    if isinstance(value, str):
        return _TEMPLATE_VARIABLE.findall(value)
    if isinstance(value, dict):
        return [name for item in value.values() for name in _template_variables(item)]
    if isinstance(value, (list, tuple)):
        return [name for item in value for name in _template_variables(item)]
    return []


def _json_parse_mapping(value: Any, field_name: str, tool_name: str, allow_none: bool = False) -> dict:
    """Safely normalize a mapping field that may be a dict, JSON string, None, or invalid."""
    if value is None:
        return None if allow_none else {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {} if not allow_none else None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            logger.warning(
                "Tool '%s' %s parsed as JSON but not a dict (type=%s) — ignoring",
                tool_name, field_name, type(parsed).__name__,
            )
        except json.JSONDecodeError:
            logger.warning(
                "Tool '%s' %s is invalid JSON — ignoring", tool_name, field_name,
            )
    return {} if not allow_none else None


class FetchToolCallable:
    """Callable tool that executes HTTP fetches with Jinja2 templating.

    Encapsulates NodeFetch's HTTP configuration. When invoked by the
    agentic loop, it executes the fetch and returns the response body.
    The configuration is rendered with the shared sandboxed Fetch renderer;
    tool arguments are inserted as data. ``literal_fields`` names request
    fields that arrived on input handles: they are sent verbatim.
    """

    def __init__(
        self,
        url_template: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        data: Optional[Any] = None,
        json_data: Optional[Any] = None,
        params: Optional[Any] = None,
        tool_name: str = "fetch",
        tool_description: Optional[str] = None,
        tool_parameters: Optional[dict] = None,
        debug: bool = False,
        literal_fields: Optional[Iterable[str]] = None,
    ):
        self._url = url_template
        self._method = (method or "GET").upper().strip()
        self._headers = headers or {}
        self._data = data
        self._json_data = json_data
        self._params = params
        self._tool_name = tool_name
        self._tool_description = tool_description or self._build_description()
        self._tool_parameters = tool_parameters  # Explicit schema params (optional)
        self._constants: dict = {}  # Literal values from tool_parameters (constant mode)
        self._debug = debug
        self._literal_fields = frozenset(literal_fields or ())

    @property
    def __name__(self) -> str:
        """Return the tool name so _collect_tools can register it in tool_functions."""
        return self._tool_name

    def _build_description(self) -> str:
        """Auto-generate description from HTTP config."""
        return f"HTTP {self._method} request to {self._url}"

    def _extract_template_variables(self) -> list[str]:
        """Extract Jinja2 template variable names from URL, headers, data, params.

        Parses all string values for {{variable}} patterns and returns
        a deduplicated sorted list of variable names.
        """
        variables: set[str] = set()
        fields = {'url': self._url, 'headers': self._headers, 'data': self._data,
                  'json_data': self._json_data, 'params': self._params}
        # Fields from input handles are data, never rendered: no variables.
        for name, value in fields.items():
            if name in self._literal_fields:
                continue
            variables.update(_template_variables(value))
        return sorted(variables)

    @property
    def tool_schema(self) -> dict:
        """Build explicit OpenAI-compatible tool schema.

        If _tool_parameters is provided (explicit schema), use it directly.
        Otherwise, auto-generate from Jinja2 template variable extraction.
        """
        if self._tool_parameters:
            # Three-mode auto-detection: explicit schema, Jinja2, or constant
            properties = {}
            required = []
            constants = {}

            for param_name, param_def in self._tool_parameters.items():
                if isinstance(param_def, dict) and "type" in param_def:
                    # Mode 3: Explicit schema (backward compat)
                    properties[param_name] = {
                        "type": param_def.get("type", "string"),
                        "description": param_def.get("description", f"Parameter '{param_name}'"),
                    }
                    if param_def.get("required", False):
                        required.append(param_name)
                elif (isinstance(param_def, str) and "{{" in param_def) or _template_variables(param_def):
                    # Mode 1: Jinja2 variable extraction (nested leaves included)
                    for var in _template_variables(param_def):
                        properties[var] = {
                            "type": "string",
                            "description": f"Template variable '{var}'",
                        }
                        if var not in required:
                            required.append(var)
                else:
                    # Mode 2: Constant — store literal, do NOT expose as tool parameter
                    constants[param_name] = param_def

            # Persist constants for execution-time merge in __call__
            self._constants = constants

            # If no explicit required list and we have properties, default to all being required
            if not required and properties:
                required = list(properties.keys())
        else:
            # Auto-generate from Jinja2 template variables
            variables = self._extract_template_variables()
            properties = {}
            required = []
            for var in variables:
                properties[var] = {"type": "string", "description": f"Template variable '{var}'"}
                required.append(var)

            # If no template variables found, provide a generic 'parameters' field
            if not properties:
                properties["parameters"] = {
                    "type": "string",
                    "description": "Parameters for the HTTP request"
                }
                required = ["parameters"]

        return {
            "type": "function",
            "function": {
                "name": self._tool_name,
                "description": self._tool_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    @property
    def tool_callable(self):
        return self

    def _build_request(self, kwargs: dict, secrets: FetchSecrets) -> FetchRequest:
        """Render the effective request once with the tool call arguments.

        Static configuration (the node fields and every non-schema
        ``tool_parameters`` entry) is the template; it is rendered exactly
        once. Model arguments are inserted as data and are never templated or
        env-resolved. Fields in ``_literal_fields`` came from input handles and
        are sent verbatim.
        """
        renderer = FetchRenderer(kwargs, secrets)
        literal = self._literal_fields
        overrides: dict[str, Any] = {}
        if self._tool_parameters:
            # When tool_parameters is set, its non-schema entries (Jinja2
            # strings and literal constants) override the request fields.
            # Explicit-schema dicts ({"type": ...}) only describe arguments.
            overrides = {key: value for key, value in self._tool_parameters.items()
                         if not (isinstance(value, dict) and "type" in value)}

        url = overrides.get('url', self._url)
        if 'url' in overrides or 'url' not in literal:
            url = renderer.text(url, 'url')
        method = self._method
        if 'method' in overrides:
            method = str(renderer.text(overrides['method'], 'method')).upper().strip()
        check_method(method)

        headers = _json_parse_mapping(self._headers, "headers", self._tool_name)
        if 'headers' not in literal:
            headers = {key: renderer.text(value, 'headers') for key, value in headers.items()}
        if 'headers' in overrides:
            override = _json_parse_mapping(overrides['headers'], "headers", self._tool_name)
            headers.update({key: renderer.text(value, 'headers') for key, value in override.items()})
        check_header_values(headers)

        params = overrides['params'] if 'params' in overrides else _json_parse_mapping(
            self._params, "params", self._tool_name, allow_none=True)
        params = renderer.body(params, 'params') if params is not None else None

        data = overrides.get('data', self._data)
        if data is not None and ('data' in overrides or 'data' not in literal):
            data = renderer.body(data, 'data')

        json_data = overrides.get('json_data', self._json_data)
        if json_data is not None:
            if 'json_data' in overrides or 'json_data' not in literal:
                json_data = renderer.body(json_data, 'json_data')
            else:
                json_data = parse_json_input(json_data, 'json_data')

        # Unknown entries are extra JSON body fields, rendered as body leaves.
        known_keys = {'url', 'method', 'headers', 'data', 'json_data', 'params'}
        extras = {key: renderer.tree(value, 'json_data') for key, value in overrides.items()
                  if key not in known_keys}
        if extras:
            if json_data is None:
                json_data = {}
            if isinstance(json_data, dict):
                json_data = {**json_data, **extras}

        return FetchRequest(method, url, headers, params=params, json_body=json_data,
                            data_body=data, secrets=secrets)

    async def __call__(self, **kwargs: str) -> str:
        """Execute HTTP fetch with provided parameters as template context.

        When ``_tool_parameters`` is set, it IS the request template — its
        non‑schema entries (Jinja2 variables and literal constants) define
        the effective request config (URL, method, headers, data, params,
        and any extra body fields).  Explicit‑schema dicts (``{"type": …}``)
        are only used for tool schema generation and are NOT applied here.

        Args:
            **kwargs: Values inserted into the URL, headers and body templates.
                They are data: never compiled as templates or env-resolved.

        Returns:
            Response body as JSON string, or error string on non‑2xx.
        """
        strict = getattr(self, '_strict_errors', False)
        secrets = FetchSecrets()
        safe = "[unrendered url]"
        request = None
        try:
            request = self._build_request(kwargs, secrets)
            safe = request.safe_url
            if not request.has_body and request.method != 'GET':
                return json.dumps({"error": f"No body provided for {request.method} request"})

            scope = getattr(self, '_dispatch_scope', None)
            if scope is not None:
                from magic_agents.node_system.fetch_request import coordinated_fetch
                result = await coordinated_fetch(scope, self._source_node_id, request)
                return result if strict or isinstance(result, str) else json.dumps(result)

            async with aiohttp.ClientSession() as session:
                async with session.request(**request.aiohttp_kwargs()) as response:
                    if response.status < 200 or response.status >= 300:
                        # Read the response body — it is available even on
                        # non‑2xx responses and must be passed to the agent.
                        # Env values used by this request are scrubbed.
                        body_text = secrets.scrub(await response.text())
                        redacted_preview = _redact_body_preview(body_text)
                        reason = secrets.scrub(response.reason)
                        logger.warning(
                            "Tool '%s' %s %s returned HTTP %d %s. Response body: %s",
                            self._tool_name, request.method, safe,
                            response.status, reason,
                            redacted_preview,
                        )
                        if strict:
                            from magic_agents.hooks.invocation_control import OperationFailure
                            raise OperationFailure("HTTP_ERROR", f"HTTP {response.status}: {reason}",
                                retryable=response.status == 429 or response.status >= 500,
                                details={"http_status": response.status, "status_code": response.status,
                                         "response_body": body_text})
                        # Return the FULL body to the agent (secrets scrubbed).
                        # The redacted preview is ONLY for the log.
                        return (
                            f"HTTP {response.status}: {reason}"
                            f"\n\nResponse body:\n{body_text}"
                        )
                    body = await response.text()
                    # Try to parse as JSON for cleaner output
                    try:
                        parsed = json.loads(body)
                        return parsed if strict else json.dumps(parsed)
                    except (json.JSONDecodeError, ValueError):
                        return body

        except FetchTemplateError as e:
            if strict:
                raise
            logger.warning("Tool '%s' could not build its request: %s", self._tool_name, e)
            return json.dumps({"error": f"Unexpected error: {e}"})
        except aiohttp.ClientResponseError as e:
            message = secrets.scrub(e.message)
            if strict:
                from magic_agents.hooks.invocation_control import OperationFailure
                raise OperationFailure("HTTP_ERROR", f"HTTP {e.status}: {message}",
                    retryable=e.status == 429 or e.status >= 500,
                    details={"http_status": e.status, "status_code": e.status}) from None
            logger.warning(
                "Tool '%s' caught aiohttp.ClientResponseError: HTTP %d %s",
                self._tool_name, e.status, message,
            )
            return f"HTTP {e.status}: {message}"
        except aiohttp.ClientError as e:
            # Never the full URL (query, user info, encoded secrets): an
            # invalid-URL error keeps only scheme://host[:port]/path.
            message = network_message(request, e) if request is not None else _scrub_exception(e, secrets)
            if strict:
                from magic_agents.hooks.invocation_control import OperationFailure
                raise OperationFailure("NETWORK_ERROR", message or type(e).__name__, retryable=True,
                    details={"exception_type": type(e).__name__}) from None
            logger.warning(
                "Tool '%s' caught %s: %s",
                self._tool_name, type(e).__name__, message,
            )
            return json.dumps({"error": f"Network error: {message}"})
        except Exception as e:
            if getattr(self, '_dispatch_scope', None) is not None:
                from magic_agents.coordination.dispatch import is_protected
                if is_protected(e): raise
            if strict:
                raise
            message = _scrub_exception(e, secrets)
            if request is not None and isinstance(request.url, str) and request.url:
                message = message.replace(request.url, request.safe_url)
            logger.warning(
                "Tool '%s' caught %s: %s",
                self._tool_name, type(e).__name__, message,
            )
            return json.dumps({"error": f"Unexpected error: {message}"})


class NodeFetch(Node):
    coordination_external_dispatch_version = 1

    """
    Fetch node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default output handle name - can be overridden by JSON data.handles
    DEFAULT_OUTPUT_HANDLE = 'handle_fetch_output'
    DEFAULT_INPUT_URL = 'handle-url'
    DEFAULT_INPUT_METHOD = 'handle-fetch-method'
    DEFAULT_INPUT_DATA = 'handle-fetch-data'
    DEFAULT_INPUT_JSON_DATA = 'handle-fetch-json_data'
    DEFAULT_INPUT_HEADERS = 'handle-fetch-headers'
    DEFAULT_INPUT_TEMPLATE_CONTEXT = 'handle_fetch_input'

    def __init__(self,
                 data: FetchNodeModel,
                 handles: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._default_method = (data.method or 'GET').upper().strip()
        self._default_headers = data.headers or {}
        self._default_params = data.params or None
        self._default_url = data.url
        self._default_data = data.data or None
        # Add jsondata attribute if it exists in the model
        self._default_jsondata = getattr(data, 'json_data', None)
        if not self._default_jsondata:
            self._default_jsondata = None
        self.method = self._default_method
        self.headers = self._default_headers
        self.params = self._default_params
        self.url = self._default_url
        self.data = self._default_data
        self.jsondata = self._default_jsondata
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLE_URL = handles.get('url', self.DEFAULT_INPUT_URL)
        self.INPUT_HANDLE_METHOD = handles.get('method', self.DEFAULT_INPUT_METHOD)
        self.INPUT_HANDLE_DATA = handles.get('data', self.DEFAULT_INPUT_DATA)
        self.INPUT_HANDLE_JSON_DATA = handles.get('json_data', self.DEFAULT_INPUT_JSON_DATA)
        self.INPUT_HANDLE_HEADERS = handles.get('headers', self.DEFAULT_INPUT_HEADERS)
        # Backward-compatible template-context input used by existing browsing examples.
        # It does not override request fields; it is exposed to Jinja templates via self.inputs.
        self.INPUT_HANDLE_TEMPLATE_CONTEXT = handles.get('input', self.DEFAULT_INPUT_TEMPLATE_CONTEXT)
        self.OUTPUT_HANDLE = handles.get('output', handles.get('response', self.DEFAULT_OUTPUT_HANDLE))
        # Tool mode configuration
        self.tool_mode = getattr(data, 'tool_mode', False)
        self.tool_name = getattr(data, 'tool_name', None) or 'fetch'
        self.tool_description = getattr(data, 'tool_description', None)
        self.tool_parameters = getattr(data, 'tool_parameters', None)
        self.debug = getattr(data, 'debug', False)

    def _runtime_literal_fields(self) -> frozenset[str]:
        """Request fields whose value arrived on an input handle.

        These values are data: they are sent verbatim, never compiled as
        templates and never env-resolved. Templates and ``{{env.NAME}}``
        placeholders apply only to the node's own static fields.
        """
        handles = {
            'url': self.INPUT_HANDLE_URL,
            'headers': self.INPUT_HANDLE_HEADERS,
            'data': self.INPUT_HANDLE_DATA,
            'json_data': self.INPUT_HANDLE_JSON_DATA,
        }
        return frozenset(field for field, handle in handles.items() if input_has_value(self.inputs, handle))

    def _resolve_runtime_request_config(self) -> tuple[str, str, Any, Any, Any]:
        url = self._default_url
        method = self._default_method
        headers = self._default_headers
        data = self._default_data
        jsondata = self._default_jsondata

        if input_has_value(self.inputs, self.INPUT_HANDLE_URL):
            url = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_URL], 'str', field_name=self.INPUT_HANDLE_URL)
        if input_has_value(self.inputs, self.INPUT_HANDLE_METHOD):
            method = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_METHOD], 'str', field_name=self.INPUT_HANDLE_METHOD).upper().strip()
        if input_has_value(self.inputs, self.INPUT_HANDLE_HEADERS):
            headers = self.inputs[self.INPUT_HANDLE_HEADERS]
        if input_has_value(self.inputs, self.INPUT_HANDLE_DATA):
            data = self.inputs[self.INPUT_HANDLE_DATA]
        if input_has_value(self.inputs, self.INPUT_HANDLE_JSON_DATA):
            jsondata = self.inputs[self.INPUT_HANDLE_JSON_DATA]

        return url, method, headers, data, jsondata

    def _build_tool_callable(self) -> FetchToolCallable:
        """The LLM tool for this node's current configuration and inputs."""
        url, method, headers, data, json_data = self._resolve_runtime_request_config()
        return FetchToolCallable(
            url_template=url,
            method=method,
            headers=headers,
            data=data,
            json_data=json_data,
            params=self.params,
            tool_name=self.tool_name,
            tool_description=getattr(self, 'tool_description', None),
            tool_parameters=getattr(self, 'tool_parameters', None),
            debug=self.debug,
            literal_fields=self._runtime_literal_fields(),
        )

    def _build_step_request(self) -> FetchRequest:
        """Render the step-mode request; raises ``TemplateError``.

        Static fields are templates rendered with ``self.inputs`` (env
        placeholders resolved); fields from input handles are sent verbatim.
        Body fields are rendered per string leaf (see ``fetch_request``).
        """
        # A method from an input handle is data: it must be an HTTP token
        # before it reaches a log line, a context or aiohttp.
        check_method(self.method)
        secrets = FetchSecrets()
        renderer = FetchRenderer(self.inputs, secrets, base_context={"method": self.method})
        literal = self._runtime_literal_fields()

        url = self.url if 'url' in literal else renderer.text(self.url, 'url')
        if not isinstance(url, str) or not url:
            raise FetchTemplateError("Fetch node has no URL", context={"field": "url", "method": self.method})

        headers = parse_mapping(self.headers, 'headers', {"method": self.method})
        if 'headers' not in literal:
            headers = {key: renderer.text(value, 'headers') for key, value in headers.items()}
        check_header_values(headers, {"method": self.method})

        json_data = data = params = None
        if self.jsondata is not None:
            json_data = (parse_json_input(self.jsondata, 'json_data', {"method": self.method})
                         if 'json_data' in literal else renderer.body(self.jsondata, 'json_data'))
        elif self.data:
            data = self.data if 'data' in literal else renderer.body(self.data, 'data')
        if self.params is not None:
            params = renderer.body(self.params, 'params')

        return FetchRequest(self.method, url, headers, params=params, json_body=json_data,
                            data_body=data, secrets=secrets)

    async def process(self, chat_log):
        self.url, self.method, self.headers, self.data, self.jsondata = self._resolve_runtime_request_config()

        # Tool mode: yield callable with explicit schema, do NOT execute fetch
        if self.tool_mode:
            callable_tool = self._build_tool_callable()
            callable_tool._source_node_id = self.node_id
            scope = getattr(chat_log, 'coordination', None)
            if scope is not None:
                from magic_agents.coordination.dispatch import register_dispatch_callable
                callable_tool._dispatch_scope = scope
                register_dispatch_callable(callable_tool)
            yield self.yield_static(callable_tool, content_type=self.OUTPUT_HANDLE)
            return

        # Step mode. A failure raises a typed FetchError (HTTPError,
        # NetworkError, TemplateError, UnexpectedError): the node fails, its
        # downstream is bypassed at once and onError Hooks can recover it.
        # An explicit Hook call (child call or redirect) always sends its
        # request, also with empty content into a static-URL Fetch.
        run = (any(value is not None for value in self.inputs.values())
               or getattr(self, '_explicit_invocation', False))
        if not run:
            if self.debug:
                logger.debug("NodeFetch:%s no inputs set; skipping request", self.node_id)
            yield self.yield_static({}, content_type=self.OUTPUT_HANDLE)
            return

        try:
            request = self._build_step_request()
        except FetchTemplateError as error:
            logger.error("NodeFetch:%s %s", self.node_id, error)
            raise

        if not request.has_body and request.method != 'GET':
            yield self.yield_static({}, content_type=self.OUTPUT_HANDLE)
            return

        logger.info("NodeFetch:%s %s %s", self.node_id, request.method, request.safe_url)
        if self.debug:
            payload_type = 'json' if request.json_body is not None else ('data' if request.data_body is not None else 'none')
            logger.debug("NodeFetch:%s request payload type=%s headers_keys=%s",
                         self.node_id, payload_type, list(request.headers.keys()))
        try:
            scope = getattr(chat_log, 'coordination', None)
            if scope is not None:
                from magic_agents.node_system.fetch_request import coordinated_fetch
                response = await coordinated_fetch(scope, self.node_id, request)
            else:
                response = await send_step_request(request)
        except Exception as error:
            logger.error("NodeFetch:%s %s: %s", self.node_id, type(error).__name__, error)
            raise
        logger.info("NodeFetch:%s request completed", self.node_id)
        yield self.yield_static(response, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Fetch-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Fetch-specific variables as documented
        state['url'] = self.url
        state['method'] = self.method
        state['headers'] = self._safe_copy_dict(self.headers) if isinstance(self.headers, dict) else self.headers
        
        # Capture body data if available
        if self.data:
            state['body'] = self.data
        if self.jsondata:
            state['json_data'] = self.jsondata
        
        return state
