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
        print(self.__class__.__name__, parent)

        if c := parent.get('NodeText'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeUserInput'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeMerger'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeParser'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeChat'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeLLM'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeFindit'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeBrowsing'):
            trg = {
                target: c
            }
        elif c := parent.get('NodeFetch'):
            trg = {
                target: c
            }
        else:
            raise Exception('Node target not supported', parent)

        self.parents.update(trg)

    async def __call__(self, chat_log: ModelAgentRunLog):
        pass

    def get_debug(self):
        return self.debug

    @staticmethod
    def magic_telemetry(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # print("magic_telemetry:kwargs", kwargs)
            # print("magic_telemetry:args", args)
            response = func(*args, **kwargs)
            # print("magic_telemetry:response", response)
            return response

        return wrapper
