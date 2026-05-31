import json
import logging
import re
from typing import Any, Optional

import aiohttp
from jinja2 import Template
from urllib.parse import urlsplit

from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.env_resolver import resolve_env_placeholders
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


def _redact_body_preview(body_text: str, max_length: int = 500) -> str:
    """Truncate and redact sensitive fields from a response body for logging.

    For valid JSON objects at top level, key-based redaction is applied:
    any key matching a sensitive pattern (api_key, secret, token, etc.)
    has its value replaced with ``"[REDACTED]"``.  Non‑JSON bodies and
    JSON arrays/lists receive truncation‑only treatment.

    Args:
        body_text: Raw response body text.
        max_length: Maximum character length before truncation (default 500).

    Returns:
        Redacted body preview string safe for diagnostic log output.
    """
    truncated = body_text[:max_length]
    try:
        parsed = json.loads(truncated)
    except json.JSONDecodeError:
        # Non-JSON body: truncation only, no key-based redaction.
        return truncated

    if isinstance(parsed, dict):
        # Depth-1 only: iterate top-level keys.
        for key in list(parsed.keys()):
            if any(pattern.search(key) for pattern in _SENSITIVE_KEY_PATTERNS):
                parsed[key] = "[REDACTED]"

    return json.dumps(parsed)


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
    ):
        self._url = url_template
        self._method = method
        self._headers = headers or {}
        self._data = data
        self._json_data = json_data
        self._params = params
        self._tool_name = tool_name
        self._tool_description = tool_description or self._build_description()
        self._tool_parameters = tool_parameters  # Explicit schema params (optional)
        self._constants: dict = {}  # Literal values from tool_parameters (constant mode)
        self._debug = debug

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
        pattern = re.compile(r'\{\{(\w+)\}\}')
        variables: set[str] = set()
        for value in [self._url, self._headers, self._data, self._json_data, self._params]:
            if isinstance(value, str):
                variables.update(pattern.findall(value))
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        variables.update(pattern.findall(v))
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
            jinja2_pattern = re.compile(r'\{\{(\w+)\}\}')

            for param_name, param_def in self._tool_parameters.items():
                if isinstance(param_def, dict) and "type" in param_def:
                    # Mode 3: Explicit schema (backward compat)
                    properties[param_name] = {
                        "type": param_def.get("type", "string"),
                        "description": param_def.get("description", f"Parameter '{param_name}'"),
                    }
                    if param_def.get("required", False):
                        required.append(param_name)
                elif isinstance(param_def, str) and "{{" in param_def:
                    # Mode 1: Jinja2 variable extraction
                    vars_in_value = jinja2_pattern.findall(param_def)
                    for var in vars_in_value:
                        properties[var] = {
                            "type": "string",
                            "description": f"Template variable '{var}'",
                        }
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

    async def __call__(self, **kwargs: str) -> str:
        """Execute HTTP fetch with provided parameters as template context.

        When ``_tool_parameters`` is set, it IS the request template — its
        non‑schema entries (Jinja2 variables and literal constants) define
        the effective request config (URL, method, headers, data, params,
        and any extra body fields).  Explicit‑schema dicts (``{"type": …}``)
        are only used for tool schema generation and are NOT applied here.

        Args:
            **kwargs: Template variables for URL, headers, body parameters.

        Returns:
            Response body as JSON string, or error string on non‑2xx.
        """
        try:
            # --- Phase 3: build effective request config from _tool_parameters ---
            # When tool_parameters is set, it IS the request template.
            # Render all non‑explicit‑schema entries, then use the result as
            # overrides for the default request fields.
            effective_url = self._url
            effective_method = self._method
            effective_headers = _json_parse_mapping(self._headers, "headers", self._tool_name)
            effective_data = self._data
            effective_json_data = self._json_data
            effective_params = _json_parse_mapping(self._params, "params", self._tool_name, allow_none=True)

            if self._tool_parameters:
                rendered_config: dict[str, Any] = {}
                jinja2_pattern = re.compile(r'\{\{(\w+)\}\}')
                for key, value in self._tool_parameters.items():
                    if isinstance(value, dict) and "type" in value:
                        continue  # Mode 3: explicit schema only — skip for request execution
                    if isinstance(value, str) and "{{" in value:
                        # Mode 1: Jinja2 template — render with tool call args
                        rendered_config[key] = Template(
                            resolve_env_placeholders(value)
                        ).render(kwargs)
                    else:
                        # Mode 2: literal constant — use as‑is
                        rendered_config[key] = value

                # Apply known request‑field overrides
                if 'url' in rendered_config:
                    effective_url = rendered_config['url']
                if 'method' in rendered_config:
                    effective_method = rendered_config['method']
                if 'headers' in rendered_config:
                    hv = rendered_config['headers']
                    if isinstance(hv, dict):
                        effective_headers.update(hv)
                if 'data' in rendered_config:
                    effective_data = rendered_config['data']
                if 'json_data' in rendered_config:
                    effective_json_data = rendered_config['json_data']
                if 'params' in rendered_config:
                    effective_params = rendered_config['params']

                # Unknown entries → merge into JSON body as extra fields
                known_keys = {'url', 'method', 'headers', 'data', 'json_data', 'params'}
                extras = {k: v for k, v in rendered_config.items() if k not in known_keys}
                if extras:
                    if effective_json_data is None:
                        effective_json_data = {}
                    if isinstance(effective_json_data, dict):
                        effective_json_data.update(extras)

            # --- Render effective config with Jinja2 ---
            url_template = Template(resolve_env_placeholders(effective_url))
            rendered_url = url_template.render(kwargs)

            # Render headers
            rendered_headers = {}
            for k, v in effective_headers.items():
                if isinstance(v, str):
                    rendered_headers[k] = Template(resolve_env_placeholders(v)).render(kwargs)
                else:
                    rendered_headers[k] = v

            # Render body/params
            def _render_template_value(value):
                if value is None:
                    return None
                resolved = resolve_env_placeholders(value)
                if isinstance(resolved, str):
                    tpl = Template(resolved)
                    return tpl.render(kwargs)
                if isinstance(resolved, dict):
                    return {
                        k: Template(resolve_env_placeholders(v)).render(kwargs) if isinstance(v, str) else v
                        for k, v in resolved.items()
                    }
                return resolved

            rendered_data = _render_template_value(effective_data)
            rendered_json_data = _render_template_value(effective_json_data)
            rendered_params = _render_template_value(effective_params)

            async with aiohttp.ClientSession() as session:
                method_fn = session.request
                fetch_kwargs: dict[str, Any] = {
                    'method': effective_method,
                    'url': rendered_url,
                    'headers': rendered_headers if isinstance(rendered_headers, dict) else json.loads(rendered_headers),
                }

                if rendered_params is not None:
                    fetch_kwargs['params'] = rendered_params if isinstance(rendered_params, dict) else json.loads(rendered_params)

                if rendered_json_data is not None:
                    fetch_kwargs['json'] = rendered_json_data if isinstance(rendered_json_data, dict) else json.loads(rendered_json_data)
                elif rendered_data is not None:
                    fetch_kwargs['data'] = rendered_data if isinstance(rendered_data, dict) else json.loads(rendered_data)

                if 'json' not in fetch_kwargs and 'data' not in fetch_kwargs:
                    if effective_method != 'GET':
                        return json.dumps({"error": f"No body provided for {effective_method} request"})

                async with method_fn(**fetch_kwargs) as response:
                    if response.status < 200 or response.status >= 300:
                        # Read the response body — it is available even on
                        # non‑2xx responses and must be passed to the agent.
                        body_text = await response.text()
                        redacted_preview = _redact_body_preview(body_text)

                        # Safe URL: scheme + netloc + path only (no query/fragment)
                        parts = urlsplit(rendered_url)
                        safe_url = f"{parts.scheme}://{parts.netloc}{parts.path}"
                        logger.warning(
                            "Tool '%s' %s %s returned HTTP %d %s. Response body: %s",
                            self._tool_name, effective_method, safe_url,
                            response.status, response.reason,
                            redacted_preview,
                        )
                        # Return the FULL unredacted body to the agent.
                        # The redacted preview is ONLY for the log.
                        return (
                            f"HTTP {response.status}: {response.reason}"
                            f"\n\nResponse body:\n{body_text}"
                        )
                    body = await response.text()
                    # Try to parse as JSON for cleaner output
                    try:
                        return json.dumps(json.loads(body))
                    except (json.JSONDecodeError, ValueError):
                        return body

        except aiohttp.ClientResponseError as e:
            logger.warning(
                "Tool '%s' caught aiohttp.ClientResponseError: HTTP %d %s",
                self._tool_name, e.status, e.message,
            )
            return f"HTTP {e.status}: {e.message}"
        except aiohttp.ClientError as e:
            logger.warning(
                "Tool '%s' caught %s: %s",
                self._tool_name, type(e).__name__, str(e),
            )
            return json.dumps({"error": f"Network error: {str(e)}"})
        except Exception as e:
            logger.warning(
                "Tool '%s' caught %s: %s",
                self._tool_name, type(e).__name__, str(e),
            )
            return json.dumps({"error": f"Unexpected error: {str(e)}"})


class NodeFetch(Node):
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
        self.OUTPUT_HANDLE = handles.get('output', handles.get('response', self.DEFAULT_OUTPUT_HANDLE))
        # Tool mode configuration
        self.tool_mode = getattr(data, 'tool_mode', False)
        self.tool_name = getattr(data, 'tool_name', None) or 'fetch'
        self.tool_parameters = getattr(data, 'tool_parameters', None)
        self.debug = getattr(data, 'debug', False)

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

    async def fetch(self, session, url, headers, data=None, json_data=None, params=None):
        # Use the appropriate method (GET, POST, PUT, etc.)
        method = session.request
        kwargs = {
            'method': self.method,
            'url': url,
            'headers': headers if type(headers) is dict else json.loads(headers)
        }

        if params is not None:
            params = params if type(params) is dict else json.loads(params)
            kwargs['params'] = params

        # Add data based on what's available
        if json_data is not None:
            json_data = json_data if type(json_data) is dict else json.loads(json_data)
            kwargs['json'] = json_data
        elif data is not None:
            data = data if type(data) is dict else json.loads(data)
            kwargs['data'] = data

        if 'json' not in kwargs and 'data' not in kwargs:
            if self.method != 'GET':
                return {}

        parts = urlsplit(url)
        safe_url = f"{parts.scheme}://{parts.netloc}{parts.path}"

        logger.info("NodeFetch:%s %s %s", self.node_id, self.method, safe_url)
        if self.debug:
            payload_type = 'json' if 'json' in kwargs else ('data' if 'data' in kwargs else 'none')
            logger.debug("NodeFetch:%s request payload type=%s headers_keys=%s", self.node_id, payload_type, list(kwargs['headers'].keys()))

        async with method(**kwargs) as response:
            if self.debug:
                logger.debug("NodeFetch:%s response status=%s", self.node_id, response.status)
            response.raise_for_status()
            return await response.json()

    def _render_request_value(self, value):
        resolved_value = resolve_env_placeholders(value)
        template = Template(json.dumps(resolved_value))
        return json.loads(template.render(self.inputs).replace('\n', ''))

    async def process(self, chat_log):
        self.url, self.method, self.headers, self.data, self.jsondata = self._resolve_runtime_request_config()

        # Tool mode: yield callable with explicit schema, do NOT execute fetch
        if self.tool_mode:
            tool_parameters = getattr(self, 'tool_parameters', None)
            callable_tool = FetchToolCallable(
                url_template=self.url,
                method=self.method,
                headers=self.headers,
                data=self.data,
                json_data=self.jsondata,
                params=self.params,
                tool_name=self.tool_name,
                tool_parameters=tool_parameters,
                debug=self.debug,
            )
            yield self.yield_static(callable_tool, content_type=self.OUTPUT_HANDLE)
            return

        # Normal mode: existing fetch execution logic (unchanged)
        # Prepare the data to send
        data_to_send = None
        json_data_to_send = None
        params_to_send = None
        run = any(value is not None for value in self.inputs.values())
        if not run:
            if self.debug:
                logger.debug("NodeFetch:%s no inputs set; skipping request", self.node_id)
            yield self.yield_static({}, content_type=self.OUTPUT_HANDLE)
            return
        
        # Template the URL with Jinja2 to support dynamic query parameters and path segments
        try:
            resolved_url = resolve_env_placeholders(self.url)
            url_template = Template(resolved_url)
            rendered_url = url_template.render(self.inputs)
            if self.debug:
                logger.debug("NodeFetch:%s templated URL: %s", self.node_id, rendered_url)
        except Exception as e:
            logger.error("NodeFetch:%s URL templating failed: %s", self.node_id, e)
            yield self.yield_debug_error(
                error_type="TemplateError",
                error_message=f"URL templating failed: {str(e)}",
                context={
                    "url_template": self.url,
                    "available_inputs": list(self.inputs.keys()),
                    "exception_type": type(e).__name__
                }
            )
            return

        resolved_headers = resolve_env_placeholders(self.headers)
        
        if self.jsondata is not None:
            json_data_to_send = self._render_request_value(self.jsondata)
        elif self.data:
            data_to_send = self._render_request_value(self.data)

        if self.params is not None:
            params_to_send = self._render_request_value(self.params)

        try:
            async with aiohttp.ClientSession() as session:
                logger.debug("NodeFetch:%s executing fetch", self.node_id)
                response_json = await self.fetch(
                    session,
                    rendered_url,  # Use templated URL instead of static self.url
                    headers=resolved_headers,
                    data=data_to_send,
                    json_data=json_data_to_send,
                    params=params_to_send
                )
            logger.info("NodeFetch:%s request completed", self.node_id)
            yield self.yield_static(response_json, content_type=self.OUTPUT_HANDLE)
        except aiohttp.ClientResponseError as e:
            logger.error("NodeFetch:%s HTTP error %s: %s", self.node_id, e.status, e.message)
            yield self.yield_debug_error(
                error_type="HTTPError",
                error_message=f"HTTP request failed with status {e.status}: {e.message}",
                context={
                    "url": rendered_url,
                    "method": self.method,
                    "status_code": e.status,
                    "headers": dict(e.headers) if hasattr(e, 'headers') else None
                }
            )
        except aiohttp.ClientError as e:
            logger.error("NodeFetch:%s client error: %s", self.node_id, e)
            yield self.yield_debug_error(
                error_type="NetworkError",
                error_message=f"Network request failed: {str(e)}",
                context={
                    "url": rendered_url,
                    "method": self.method,
                    "exception_type": type(e).__name__
                }
            )
        except Exception as e:
            logger.error("NodeFetch:%s unexpected error: %s", self.node_id, e)
            yield self.yield_debug_error(
                error_type="UnexpectedError",
                error_message=f"Unexpected error during fetch: {str(e)}",
                context={
                    "url": rendered_url,
                    "method": self.method,
                    "exception_type": type(e).__name__
                }
            )

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
