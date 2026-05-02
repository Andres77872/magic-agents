"""
NodePythonExec — Python execution node with dual-mode support.

Dual-mode behavior:
- **Node mode** (when `data.code` is set): Executes user-provided Python code
  via the `run(handler)` contract. Receives edge inputs as handler dict keys
  (excluding config handles). Yields results via `handle-python_exec-result`.

- **Tool mode** (when `data.code` is absent): Yields a wrapped `PythonExecutor`
  callable as an LLM tool. The executor exposes a dual-param schema (`code` + `handler`)
  for backward compatibility. This is the unchanged existing behavior.

Output handles:
- `DEFAULT_OUTPUT_HANDLE` = 'handle-tool-definition' (tool mode)
- `DEFAULT_OUTPUT_HANDLE_CODE_RESULT` = 'handle-python_exec-result' (node mode)

Handler dict construction (_build_handler_dict):
Excludes config handles: safety_mode, timeout, max_output_chars, code.
All remaining self.inputs keys become handler dict entries.
"""

import asyncio
import logging
from typing import Optional, AsyncGenerator, Dict, Any

from magic_agents.models.factory.Nodes import PythonExecNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.node_system.python_code_runner import CodeRunner
from magic_agents.util.primitive_coercion import coerce_primitive_by_type, input_has_value

logger = logging.getLogger(__name__)

# Set of config handle logical names to exclude from handler dict
_CONFIG_HANDLE_KEYS = frozenset({'safety_mode', 'timeout', 'max_output_chars', 'code'})


class PythonExecToolWrapper:
    """Wraps a PythonExecutor with dual-param (code + handler) tool schema.

    Provides backward-compatible tool callable for LLM tool mode.
    When the LLM provides only `code`, delegates to PythonExecutor (legacy path).
    When the LLM provides only `handler`, wraps into run(handler) execution.
    When both are provided, `handler` takes precedence (warning logged).
    """

    def __init__(
        self,
        executor: Any,
        code_runner: Optional[CodeRunner] = None,
        node_code: Optional[str] = None,
        node_id: Optional[str] = None,
    ):
        self._executor = executor
        self._code_runner = code_runner
        self._node_code = node_code
        self._node_id = node_id

    @property
    def __name__(self) -> str:
        """Return the tool name for tool_functions registration."""
        return "execute_python"

    @property
    def tool_schema(self) -> dict:
        """Dual-param tool schema exposing both code and handler.

        Both parameters are optional — the LLM can choose which to provide.
        """
        return {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute Python code via run(handler) contract",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute (legacy mode). "
                                           "Provide raw Python statements or define a run(handler) function."
                        },
                        "handler": {
                            "type": "object",
                            "description": "Structured handler dict for run(handler) execution. "
                                           "When provided, executes the configured code with this handler."
                        }
                    }
                }
            }
        }

    @property
    def tool_callable(self):
        return self

    async def __call__(self, **kwargs: str) -> str:
        """Execute Python code via the appropriate execution path.

        Resolution order:
        1. Both `code` and `handler` provided → handler takes precedence (warning logged)
        2. Only `handler` provided → use data.code or default wrapper with handler
        3. Only `code` provided → legacy path: delegate to PythonExecutor

        Args:
            **kwargs: May contain 'code' (str) and/or 'handler' (dict/str).

        Returns:
            Execution result as string.
        """
        code = kwargs.get('code')
        handler = kwargs.get('handler')

        # Resolve handler: accept dict or JSON string
        handler_dict = None
        if handler is not None:
            if isinstance(handler, dict):
                handler_dict = handler
            elif isinstance(handler, str):
                import json
                try:
                    handler_dict = json.loads(handler)
                except json.JSONDecodeError:
                    handler_dict = {"value": handler}

        # Case 1: Both code and handler provided — handler takes precedence
        if code is not None and handler_dict is not None:
            logger.warning(
                "NodePythonExec:%s both code and handler provided; preferring handler",
                self._node_id,
            )
            return await self._execute_with_handler(code, handler_dict)

        # Case 2: Only handler provided
        if handler_dict is not None:
            effective_code = self._node_code or "def run(handler): return handler"
            return await self._execute_with_handler(effective_code, handler_dict)

        # Case 3: Only code provided — legacy path
        return await self._executor(code=code or "")

    async def _execute_with_handler(self, code: str, handler: dict) -> str:
        """Execute code with handler dict via CodeRunner."""
        if self._code_runner is None:
            self._code_runner = CodeRunner()

        result = await self._code_runner.execute(code, handler)

        import json
        return json.dumps(result)


class NodePythonExec(Node):
    """Python execution node with dual-mode (tool/node) support.

    **Node mode** (when `data.code` is set):
    Executes user-provided Python code via the `run(handler)` contract.
    Edge inputs (excluding config handles: safety_mode, timeout, max_output_chars, code)
    populate the handler dict. Results are yielded via `handle-python_exec-result`.

    **Tool mode** (when `data.code` is absent):
    Yields a `PythonExecToolWrapper` callable as an LLM tool. The wrapper exposes
    a dual-param schema (`code` + `handler`) for backward compatibility with
    existing agent graphs that use `execute_python(code)`.

    When safety_mode='in_process' is active, a warning is logged:
    "arbitrary code execution is enabled. Do not use with untrusted code."
    """
    DEFAULT_OUTPUT_HANDLE = 'handle-tool-definition'
    DEFAULT_OUTPUT_HANDLE_CODE_RESULT = 'handle-python_exec-result'
    DEFAULT_INPUT_SAFETY_MODE = 'handle-python_exec-safety_mode'
    DEFAULT_INPUT_TIMEOUT = 'handle-python_exec-timeout'
    DEFAULT_INPUT_MAX_OUTPUT_CHARS = 'handle-python_exec-max_output_chars'

    def __init__(self, data: PythonExecNodeModel, handles: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self._code = data.code  # NEW: store code for mode detection
        self._data = data
        self._default_safety_mode = getattr(data, 'safety_mode', 'subprocess')
        self._default_timeout = getattr(data, 'timeout', 30.0)
        self._default_max_output_chars = getattr(data, 'max_output_chars', 8000)
        handles = handles or {}
        self.INPUT_HANDLE_SAFETY_MODE = handles.get('safety_mode', self.DEFAULT_INPUT_SAFETY_MODE)
        self.INPUT_HANDLE_TIMEOUT = handles.get('timeout', self.DEFAULT_INPUT_TIMEOUT)
        self.INPUT_HANDLE_MAX_OUTPUT_CHARS = handles.get('max_output_chars', self.DEFAULT_INPUT_MAX_OUTPUT_CHARS)
        self.OUTPUT_HANDLE = handles.get('output', self.DEFAULT_OUTPUT_HANDLE)

        # Node mode output handle: use configured handles.output if provided,
        # otherwise the default code result handle
        self.OUTPUT_HANDLE_CODE_RESULT = handles.get(
            'output',
            self.DEFAULT_OUTPUT_HANDLE_CODE_RESULT,
        )

        self.executor = None
        self._code_runner: Optional[CodeRunner] = None
        self._resolved_safety_mode = None
        self._resolved_timeout = None
        self._resolved_max_output_chars = None

        # Only refresh executor for tool mode
        if not self._has_code():
            self._refresh_executor(
                self._default_safety_mode,
                self._default_timeout,
                self._default_max_output_chars,
            )

    def _has_code(self) -> bool:
        """Check if the node has user code configured for node mode.

        Returns:
            True if data.code is a non-empty string (node mode).
            False if data.code is None or empty (tool mode).
        """
        return bool(self._code)

    def _build_handler_dict(self) -> dict:
        """Build handler dict from self.inputs, excluding config handles.

        Config handles (safety_mode, timeout, max_output_chars, code) are
        runtime configuration, NOT data inputs. They are excluded from the
        handler dict that gets passed to user code's run(handler).

        Returns:
            Dict of input key-value pairs appropriate for run(handler).
            May be empty if no non-config inputs are present.
        """
        handler = {}
        for target_handle, value in self.inputs.items():
            # Skip config handles (these are runtime configuration, not data)
            if target_handle in (
                self.INPUT_HANDLE_SAFETY_MODE,
                self.INPUT_HANDLE_TIMEOUT,
                self.INPUT_HANDLE_MAX_OUTPUT_CHARS,
                'code',  # Also exclude 'code' if it somehow appears in inputs
            ):
                continue
            # All remaining keys become handler dict entries
            handler[target_handle] = value
        return handler

    def _refresh_executor(self, safety_mode: str, timeout: float, max_output_chars: int) -> None:
        from magic_llm.util.python_executor import PythonExecutor

        self.executor = PythonExecutor(
            safety_mode=safety_mode,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        self._resolved_safety_mode = safety_mode
        self._resolved_timeout = timeout
        self._resolved_max_output_chars = max_output_chars

        if self.executor.safety_mode == 'in_process':
            logger.warning(
                "NodePythonExec:%s running with safety_mode='in_process' — "
                "arbitrary code execution is enabled. Do not use with untrusted code.",
                self.node_id
            )

    def _resolve_runtime_config(self) -> tuple[str, float, int]:
        safety_mode = self._default_safety_mode
        timeout = self._default_timeout
        max_output_chars = self._default_max_output_chars

        if input_has_value(self.inputs, self.INPUT_HANDLE_SAFETY_MODE):
            safety_mode = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_SAFETY_MODE], 'str', field_name=self.INPUT_HANDLE_SAFETY_MODE)
        if input_has_value(self.inputs, self.INPUT_HANDLE_TIMEOUT):
            timeout = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_TIMEOUT], 'float', field_name=self.INPUT_HANDLE_TIMEOUT)
        if input_has_value(self.inputs, self.INPUT_HANDLE_MAX_OUTPUT_CHARS):
            max_output_chars = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_MAX_OUTPUT_CHARS], 'int', field_name=self.INPUT_HANDLE_MAX_OUTPUT_CHARS)

        return safety_mode, timeout, max_output_chars

    def _log_in_process_warning(self):
        """Log warning when safety_mode='in_process' in node mode."""
        logger.warning(
            "NodePythonExec:%s running with safety_mode='in_process' — "
            "arbitrary code execution is enabled. Do not use with untrusted code.",
            self.node_id,
        )

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute the node: dual-mode branching.

        **Node mode** (self._has_code() is True):
        1. Resolve runtime config from inputs/edge overrides
        2. Log warning if safety_mode='in_process'
        3. Build handler dict from inputs (excluding config handles)
        4. Execute via CodeRunner with timeout enforcement
        5. Yield result via self.OUTPUT_HANDLE_CODE_RESULT

        **Tool mode** (self._has_code() is False):
        Existing behavior unchanged — resolve runtime config, refresh executor,
        yield wrapped executor via self.OUTPUT_HANDLE.
        """
        # ─── Node Mode ───────────────────────────────────────────────
        if self._has_code():
            runtime_safety_mode, runtime_timeout, runtime_max_output_chars = self._resolve_runtime_config()

            # Log warning for in_process mode
            if runtime_safety_mode == 'in_process':
                self._log_in_process_warning()

            # Build handler dict from edge inputs (excluding config handles)
            handler = self._build_handler_dict()

            # Create or reuse CodeRunner
            if self._code_runner is None or (
                runtime_safety_mode != self._resolved_safety_mode
                or runtime_timeout != self._resolved_timeout
                or runtime_max_output_chars != self._resolved_max_output_chars
            ):
                self._code_runner = CodeRunner(
                    safety_mode=runtime_safety_mode,
                    timeout=runtime_timeout,
                    max_output_chars=runtime_max_output_chars,
                )
                self._resolved_safety_mode = runtime_safety_mode
                self._resolved_timeout = runtime_timeout
                self._resolved_max_output_chars = runtime_max_output_chars

            # Execute user code with timeout enforcement
            result = await self._code_runner.execute(self._code, handler)

            if "error" in result:
                # Error case: yield error content via code result handle
                yield self.yield_static(
                    self._safe_value(result),
                    content_type=self.OUTPUT_HANDLE_CODE_RESULT,
                )
            else:
                # Success case: yield result value via code result handle
                yield self.yield_static(
                    self._safe_value(result["result"]),
                    content_type=self.OUTPUT_HANDLE_CODE_RESULT,
                )
            return

        # ─── Tool Mode (unchanged) ───────────────────────────────────
        runtime_safety_mode, runtime_timeout, runtime_max_output_chars = self._resolve_runtime_config()

        if (
            self.executor is None
            or runtime_safety_mode != self._resolved_safety_mode
            or runtime_timeout != self._resolved_timeout
            or runtime_max_output_chars != self._resolved_max_output_chars
        ):
            self._refresh_executor(runtime_safety_mode, runtime_timeout, runtime_max_output_chars)

        # Wrap executor with dual-param tool schema for backward compat
        wrapped_tool = PythonExecToolWrapper(
            executor=self.executor,
            code_runner=self._code_runner,
            node_code=self._code,
            node_id=self.node_id,
        )

        yield self.yield_static(wrapped_tool, content_type=self.OUTPUT_HANDLE)
