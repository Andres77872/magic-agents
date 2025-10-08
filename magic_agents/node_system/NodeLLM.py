import json
import re
import uuid
import logging

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


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
        # Avoid logging full params to prevent leaking content; log keys only
        logger.debug("NodeLLM:%s inputs keys: %s", self.node_id, list(params.keys()))
        no_inputs = False
        if not params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT) and not params.get(self.INPUT_HANDLER_USER_MESSAGE):
            no_inputs = True

        client: MagicLLM = self.get_input(self.INPUT_HANDLER_CLIENT_PROVIDER, required=True)
        if c := params.get(self.INPUT_HANDLER_CHAT):
            chat = c
            if sys_prompt := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                chat.set_system(sys_prompt)
            if user_prompt := self.get_input(self.INPUT_HANDLER_USER_MESSAGE):
                # Convert list to string for loop aggregation results
                if isinstance(user_prompt, list):
                    user_prompt = json.dumps(user_prompt)
                chat.add_user_message(user_prompt)
        else:
            if no_inputs:
                logger.debug("NodeLLM:%s no inputs provided; yielding empty content", self.node_id)
                yield self.yield_static('')
                return
            chat = ModelChat(params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT))
            if k := params.get(self.INPUT_HANDLER_USER_MESSAGE):
                # Convert list to string for loop aggregation results
                if isinstance(k, list):
                    k = json.dumps(k)
                chat.add_user_message(k)
            else:
                logger.error("NodeLLM:%s missing required input '%s'", self.node_id, self.INPUT_HANDLER_USER_MESSAGE)
                raise ValueError(f"NodeLLM '{self.INPUT_HANDLER_USER_MESSAGE}' requires either a user message.")

        if not self.stream:
            logger.info("NodeLLM:%s generating (non-stream) with model=%s", self.node_id, client.llm.model)
            intention = await client.llm.async_generate(chat, **self.extra_data)
            self.generated = intention.content
            yield self.yield_static(ChatCompletionModel(
                id=uuid.uuid4().hex,
                model=client.llm.model,
                choices=[ChoiceModel()],
                usage=intention.usage),
                content_type='content')
        else:
            logger.info("NodeLLM:%s streaming generation with model=%s", self.node_id, client.llm.model)
            async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                self.generated += i.choices[0].delta.content or ''
                yield self.yield_static(i, content_type='content')
        # if self.json_output:
        #     print(self.generated)
        #     self.generated = json.loads(self.generated)

        if self.json_output:
            logger.debug("NodeLLM:%s parsing JSON output", self.node_id)

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
                logger.debug("NodeLLM:%s JSON parsed successfully", self.node_id)
            else:
                logger.error("NodeLLM:%s no JSON content found in generated output (%d chars)", self.node_id, len(self.generated))
                raise ValueError('No JSON content found', self.generated)
        yield self.yield_static(self.generated)

    async def __call__(self, chat_log):
        # if configured to iterate inside a Loop, always re-run instead of using cached response
        if getattr(self, 'iterate', False):
            self._response = None
        async for result in super().__call__(chat_log):
            yield result
