"""An embedded skill source. Full definitions are private configuration."""
from magic_agents.models.factory.Nodes.SkillsNodeModel import SkillsNodeModel
from magic_agents.skills import SkillPromptBundle
from .Node import Node


class NodeSkills(Node):
    DEFAULT_OUTPUT_HANDLE = 'handle-skills'

    def __init__(self, data: SkillsNodeModel, handles=None, **kwargs):
        super().__init__(**kwargs)
        self.OUTPUT_HANDLE = (handles or {}).get('output', self.DEFAULT_OUTPUT_HANDLE)
        if not isinstance(self.OUTPUT_HANDLE, str) or not self.OUTPUT_HANDLE:
            raise ValueError('SKILLS_CONNECTION_INVALID: invalid output alias')
        self._bundle = SkillPromptBundle.from_model(self.node_id, data)

    async def process(self, chat_log):
        yield self.yield_static(self._bundle, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        return {**super()._capture_internal_state(), 'skills': self._bundle.safe_summary()}
