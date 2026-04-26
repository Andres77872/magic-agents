import logging
from typing import Optional

from magic_agents.models.factory.Nodes import ConstantNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeConstant(Node):
    """Constant node that yields a typed primitive value."""

    DEFAULT_OUTPUT_HANDLE = 'handle_constant_output'

    def __init__(self, data: ConstantNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value = data.value
        self._value_type = data.value_type
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', handles.get('value', self.DEFAULT_OUTPUT_HANDLE))

    async def process(self, chat_log):
        logger.info("NodeConstant:%s yielding %s constant", self.node_id, self._value_type)
        yield self.yield_static(self._value, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        state = super()._capture_internal_state()
        state['value'] = self._value
        state['value_type'] = self._value_type
        return state
