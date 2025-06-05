import json
import re
import uuid

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.Node import Node


class NodeLLM(Node):
    INPUT_HANDLER_CLIENT_PROVIDER = 'handle-client-provider'
    INPUT_HANDLER_CHAT = 'handle-chat'
    INPUT_HANDLER_SYSTEM_CONTEXT = 'handle-system-context'
    INPUT_HANDLER_USER_MESSAGE = 'handle_user_message'

    def __init__(self,
                 data: LlmNodeModel,
                 node_id: str,
                 debug: bool = False,
                 **kwargs):
        super().__init__(
            debug=debug,
            node_id=node_id,
            **kwargs)
        # allow re-execution inside Loop when requested
        self.iterate = getattr(data, 'iterate', False)
        self.stream = data.stream
        self.json_output = data.json_output
        self.extra_data = data.extra_data
        self.generated = ''
        if 'temperature' not in self.extra_data and data.temperature is not None:
            self.extra_data['temperature'] = data.temperature
        if 'top_p' not in self.extra_data and data.top_p is not None:
            self.extra_data['top_p'] = data.top_p

    async def process(self, chat_log):
        params = self.inputs
        if not params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT) and not params.get(self.INPUT_HANDLER_USER_MESSAGE):
            yield self.yield_static('')
            return

        client: MagicLLM = self.get_input(self.INPUT_HANDLER_CLIENT_PROVIDER, required=True)
        if c := params.get(self.INPUT_HANDLER_CHAT):
            chat = c
        else:
            chat = ModelChat(params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT))
            if k := params.get(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(k)
            else:
                raise ValueError(f"NodeLLM '{self.INPUT_HANDLER_USER_MESSAGE}' requires either a user message.")
        if not self.stream:
            intention = await client.llm.async_generate(chat, **self.extra_data)
            self.generated = intention.content
            yield self.yield_static(ChatCompletionModel(
                id=uuid.uuid4().hex,
                model=client.llm.model,
                choices=[ChoiceModel()],
                usage=intention.usage),
                content_type='content')
        else:
            async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                self.generated += i.choices[0].delta.content or ''
                yield self.yield_static(i, content_type='content')
        # if self.json_output:
        #     print(self.generated)
        #     self.generated = json.loads(self.generated)

        if self.json_output:

            # Extract JSON from markdown code blocks or plain text
            json_content = None

            # Try to find JSON in markdown code blocks (```json or ```)
            json_pattern = r'```(?:json)?\s*(.*?)\s*```'
            matches = re.findall(json_pattern, self.generated, re.DOTALL)

            if matches:
                # Use the first JSON block found
                json_content = matches[0].strip()
            else:
                # If no code blocks found, try to extract JSON from the entire string
                # Look for content that starts with { and ends with }
                brace_pattern = r'\{.*\}'
                match = re.search(brace_pattern, self.generated, re.DOTALL)
                if match:
                    json_content = match.group().strip()
                else:
                    json_content = self.generated.strip()
            if json_content:
                self.generated = json.loads(json_content)
            else:
                raise ValueError('No JSON content found', self.generated)
        yield self.yield_static(self.generated)

    async def __call__(self, chat_log):
        # if configured to iterate inside a Loop, always re-run instead of using cached response
        if getattr(self, 'iterate', False):
            self._response = None
        async for result in super().__call__(chat_log):
            yield result
