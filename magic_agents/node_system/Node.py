import abc
import functools
import inspect
import time
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
            "NodeChat", "NodeLLM", "NodeFetch", "NodeClientLLM"
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
    def magic_telemetry(func):
        qualname = func.__qualname__.split('.')[0]
        if not inspect.isasyncgenfunction(func):
            raise TypeError(f"Function {qualname} is not an async generator. "
                            f"magic_telemetry can only be applied to async generator functions.")

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            debug = args[0].get_debug() if args and hasattr(args[0], "get_debug") else False

            start_time = time.monotonic()
            if debug:
                print(f"Executing {qualname}...")
            async for i in func(*args, **kwargs):
                yield i
            end_time = time.monotonic()
            execution_time = end_time - start_time
            if debug:
                print(f"{qualname} execution time: {execution_time:.4f} seconds")

        return wrapper
