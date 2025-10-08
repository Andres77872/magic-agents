import logging

from magic_agents.models.factory.Nodes import TextNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeText(Node):
    def __init__(self, data: TextNodeModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = data.text

    async def process(self, chat_log):
        logger.info("NodeText:%s yielding static text (len=%d)", self.node_id, len(self._text) if self._text else 0)
        yield self.yield_static(self._text)
