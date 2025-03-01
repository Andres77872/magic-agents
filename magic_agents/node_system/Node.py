from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, AsyncGenerator

from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.util.telemetry import magic_telemetry

logger = logging.getLogger(__name__)


class Node(abc.ABC):
    """
    Abstract base class for a Node in the LLM orchestration system.
    Each subclass must implement the `process` method.
    """

    def __init__(
        self,
        cost: float = 0.0,
        node_id: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Initialize a Node instance.

        Parameters:
        - cost (float): Cost associated with the node.
        - node_id (Optional[str]): Unique identifier for the node.
        - debug (bool): Whether to enable debugging logs.
        """
        self.cost = cost
        self.parents: Dict[str, Any] = {}
        self.debug = debug
        self._response: Optional[Any] = None
        self.node_id = node_id
        self.extra_params = kwargs

        if self.debug:
            logger.debug(f"Node ({self.node_id}) initialized with params: {kwargs}")

    def prep(self, content: Any) -> Dict[str, Any]:
        """
        Prepare the node's content into a standardized response structure.
        """
        self._response = content
        return {
            "node": self.__class__.__name__,
            "content": content,
        }

    def add_parent(self, parent: Dict[str, Any], source: str, target: str):
        """
        Add a parent node's output to the current node context.
        """
        content = parent.get("content")
        if content is not None:
            self.parents[target] = content
            if self.debug:
                logger.debug(
                    f"Node ({self.node_id}): Parent added - source={source}, target={target}"
                )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically decorate the `process` method of the subclass with telemetry
        process_method = getattr(cls, "process", None)
        if process_method and callable(process_method):
            cls.process = magic_telemetry(process_method)

    async def __call__(
        self, chat_log: ModelAgentRunLog
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Invoke the node execution. Return precomputed result if already computed.
        """
        if self._response is not None:
            # Response is precomputed; yield immediately.
            yield {"type": "end", "content": self.prep(self._response)}
            return

        # Execute subclass-specific logic.
        async for result in self.process(chat_log):
            yield result

    @abc.abstractmethod
    async def process(
        self, chat_log: ModelAgentRunLog
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Abstract method. Subclasses implement their specific processing logic here.
        """
        raise NotImplementedError("Subclasses must implement the 'process' method.")

    @property
    def response(self) -> Optional[Any]:
        """
        Get the current response/content of this node after execution.
        """
        return self._response

    def get_debug(self) -> bool:
        """
        Check if debug mode is enabled for this node.
        """
        return self.debug