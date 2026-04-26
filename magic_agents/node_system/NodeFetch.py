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
            # Use explicit tool_parameters from config
            properties = {}
            required = []
            for param_name, param_def in self._tool_parameters.items():
                properties[param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", f"Parameter '{param_name}'"),
                }
                if param_def.get("required", False):
                    required.append(param_name)

            # If no explicit required list, default to all params being required
            if not required:
                required = list(self._tool_parameters.keys())
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

        Args:
            **kwargs: Template variables for URL, headers, body parameters.

        Returns:
            Response body as JSON string, or error string on non-2xx.
        """
        try:
            # Render URL template
            url_template = Template(resolve_env_placeholders(self._url))
            rendered_url = url_template.render(kwargs)

            # Render headers
            rendered_headers = {}
            for k, v in self._headers.items():
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

            rendered_data = _render_template_value(self._data)
            rendered_json_data = _render_template_value(self._json_data)
            rendered_params = _render_template_value(self._params)

            async with aiohttp.ClientSession() as session:
                method_fn = session.request
                fetch_kwargs: dict[str, Any] = {
                    'method': self._method,
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
                    if self._method != 'GET':
                        return json.dumps({"error": f"No body provided for {self._method} request"})

                async with method_fn(**fetch_kwargs) as response:
                    if response.status < 200 or response.status >= 300:
                        return f"HTTP {response.status}: {response.reason}"
                    body = await response.text()
                    # Try to parse as JSON for cleaner output
                    try:
                        return json.dumps(json.loads(body))
                    except (json.JSONDecodeError, ValueError):
                        return body

        except aiohttp.ClientResponseError as e:
            return f"HTTP {e.status}: {e.message}"
        except aiohttp.ClientError as e:
            return json.dumps({"error": f"Network error: {str(e)}"})
        except Exception as e:
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
