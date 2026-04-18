import logging
from typing import Optional

from magic_agents.models.factory.Nodes import PythonExecNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodePythonExec(Node):
    """Python callable tool node.

    Yields a PythonExecutor callable from magic-llm.
    The executor accepts `code: str` at runtime via the agentic loop.
    """
    DEFAULT_OUTPUT_HANDLE = 'handle-tool-definition'

    def __init__(self, data: PythonExecNodeModel, handles: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        from magic_llm.util.python_executor import PythonExecutor
        self.executor = PythonExecutor(
            safety_mode=getattr(data, 'safety_mode', 'subprocess'),
            timeout=getattr(data, 'timeout', 30.0),
            max_output_chars=getattr(data, 'max_output_chars', 8000),
        )
        if self.executor.safety_mode == 'in_process':
            logger.warning(
                "NodePythonExec:%s running with safety_mode='in_process' — "
                "arbitrary code execution is enabled. Do not use with untrusted code.",
                self.node_id
            )
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', self.DEFAULT_OUTPUT_HANDLE)

    async def process(self, chat_log):
        yield self.yield_static(self.executor, content_type=self.OUTPUT_HANDLE)
