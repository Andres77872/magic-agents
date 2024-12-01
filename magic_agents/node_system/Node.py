import abc
import functools
from typing import Callable
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


class Node(abc.ABC):
    def __init__(self,
                 cost=0,
                 debug: bool = False,
                 **kwargs):
        self.cost = cost
        self.parents = {}
        self.debug = debug

    def prep(self, content):
        return {
            self.__class__.__name__: content
        }

    def add_parent(self, parent: dict, target: str):
        # Define a mapping of possible node types to their keys
        node_types = [
            "NodeText", "NodeUserInput", "NodeParser",
            "NodeChat", "NodeLLM", "NodeFetch"
        ]

        for node_type in node_types:
            if content := parent.get(node_type):
                self.parents.update({target: content})
                return

        raise Exception("Node target not supported", parent)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically decorate the `__call__` method of the subclass
        if '__call__' in cls.__dict__:
            cls.__call__ = Node.magic_telemetry(cls.__call__)

    async def __call__(self, chat_log: ModelAgentRunLog):
        pass

    def get_debug(self):
        return self.debug

    @staticmethod
    def magic_telemetry(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print('Executing', func.__qualname__.split('.')[0])
            response = func(*args, **kwargs)
            return response

        return wrapper
