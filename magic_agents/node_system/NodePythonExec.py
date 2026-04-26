import logging
from typing import Optional

from magic_agents.models.factory.Nodes import PythonExecNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.primitive_coercion import coerce_primitive_by_type, input_has_value

logger = logging.getLogger(__name__)


class NodePythonExec(Node):
    """Python callable tool node.

    Yields a PythonExecutor callable from magic-llm.
    The executor accepts `code: str` at runtime via the agentic loop.
    """
    DEFAULT_OUTPUT_HANDLE = 'handle-tool-definition'
    DEFAULT_INPUT_SAFETY_MODE = 'handle-python_exec-safety_mode'
    DEFAULT_INPUT_TIMEOUT = 'handle-python_exec-timeout'
    DEFAULT_INPUT_MAX_OUTPUT_CHARS = 'handle-python_exec-max_output_chars'

    def __init__(self, data: PythonExecNodeModel, handles: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self._default_safety_mode = getattr(data, 'safety_mode', 'subprocess')
        self._default_timeout = getattr(data, 'timeout', 30.0)
        self._default_max_output_chars = getattr(data, 'max_output_chars', 8000)
        handles = handles or {}
        self.INPUT_HANDLE_SAFETY_MODE = handles.get('safety_mode', self.DEFAULT_INPUT_SAFETY_MODE)
        self.INPUT_HANDLE_TIMEOUT = handles.get('timeout', self.DEFAULT_INPUT_TIMEOUT)
        self.INPUT_HANDLE_MAX_OUTPUT_CHARS = handles.get('max_output_chars', self.DEFAULT_INPUT_MAX_OUTPUT_CHARS)
        self.OUTPUT_HANDLE = handles.get('output', self.DEFAULT_OUTPUT_HANDLE)
        self.executor = None
        self._resolved_safety_mode = None
        self._resolved_timeout = None
        self._resolved_max_output_chars = None
        self._refresh_executor(
            self._default_safety_mode,
            self._default_timeout,
            self._default_max_output_chars,
        )

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

    async def process(self, chat_log):
        runtime_safety_mode, runtime_timeout, runtime_max_output_chars = self._resolve_runtime_config()
        if (
            self.executor is None
            or runtime_safety_mode != self._resolved_safety_mode
            or runtime_timeout != self._resolved_timeout
            or runtime_max_output_chars != self._resolved_max_output_chars
        ):
            self._refresh_executor(runtime_safety_mode, runtime_timeout, runtime_max_output_chars)

        yield self.yield_static(self.executor, content_type=self.OUTPUT_HANDLE)
