"""Silent terminal sink used for intentionally discarded graph outputs."""

import logging

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeVoid(Node):
    """Consume routed inputs without emitting a user-facing output handle."""

    async def process(self, chat_log):
        logger.debug("NodeVoid:%s consumed terminal input", self.node_id)
        if False:  # Keep the abstract process contract as an async generator.
            yield None

    def _capture_internal_state(self):
        state = super()._capture_internal_state()
        state["is_silent_sink"] = True
        return state
