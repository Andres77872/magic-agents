import abc
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.util.telemetry import magic_telemetry


class Node(abc.ABC):
    def __init__(self,
                 cost=0,
                 node_id=None,
                 debug: bool = False,
                 **kwargs):
        self.cost = cost
        self.parents = {}
        self.debug = debug
        self.response = None
        self.node_id = node_id

    def prep(self, content):
        self.response = content
        return {
            'node': self.__class__.__name__,
            'content': content
        }

    def add_parent(self, parent: dict, source: str, target: str):
        self.parents.update({target: parent['content']})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Automatically decorate the `process` method of the subclass
        if 'process' in cls.__dict__:
            cls.process = magic_telemetry(cls.process)

    async def __call__(self, chat_log: ModelAgentRunLog):
        # Check if the response is already computed
        if self.response is not None:
            yield {
                'type': 'end',
                'content': self.prep(self.response)
            }
        else:
            # Execute the subclass-specific logic
            async for result in self.process(chat_log):
                yield result

    @abc.abstractmethod
    async def process(self, chat_log: ModelAgentRunLog):
        pass

    def get_debug(self):
        return self.debug
