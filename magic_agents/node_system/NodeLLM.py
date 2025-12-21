import json
import re
import uuid
import logging
from typing import Optional

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeLLM(Node):
    """
    LLM node - handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_CLIENT_PROVIDER = 'handle-client-provider'
    DEFAULT_INPUT_CHAT = 'handle-chat'
    DEFAULT_INPUT_SYSTEM_CONTEXT = 'handle-system-context'
    DEFAULT_INPUT_USER_MESSAGE = 'handle_user_message'
    # Output handles
    DEFAULT_OUTPUT_CONTENT = 'handle_streaming_content'
    DEFAULT_OUTPUT_GENERATED = 'handle_generated_content'

    def __init__(self,
                 data: LlmNodeModel,
                 node_id: str,
                 debug: bool = False,
                 handles: Optional[dict] = None,
                 **kwargs):
        super().__init__(
            debug=debug,
            node_id=node_id,
            **kwargs)
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLER_CLIENT_PROVIDER = handles.get('client_provider', handles.get('client', self.DEFAULT_INPUT_CLIENT_PROVIDER))
        self.INPUT_HANDLER_CHAT = handles.get('chat', self.DEFAULT_INPUT_CHAT)
        self.INPUT_HANDLER_SYSTEM_CONTEXT = handles.get('system_context', handles.get('system', self.DEFAULT_INPUT_SYSTEM_CONTEXT))
        self.INPUT_HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_INPUT_USER_MESSAGE))
        # Output handles
        self.OUTPUT_HANDLE_CONTENT = handles.get('output_content', handles.get('streaming', self.DEFAULT_OUTPUT_CONTENT))
        self.OUTPUT_HANDLE_GENERATED = handles.get('output_generated', handles.get('generated', self.DEFAULT_OUTPUT_GENERATED))
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
        if 'max_tokens' not in self.extra_data and data.max_tokens is not None:
            self.extra_data['max_tokens'] = data.max_tokens

    async def process(self, chat_log):
        params = self.inputs
        # Avoid logging full params to prevent leaking content; log keys only
        logger.debug("NodeLLM:%s inputs keys: %s", self.node_id, list(params.keys()))
        no_inputs = False
        if not params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT) and not params.get(self.INPUT_HANDLER_USER_MESSAGE):
            no_inputs = True

        def extract_message(msg):
            """Extract string message from various input types."""
            if isinstance(msg, str):
                return msg
            if isinstance(msg, list):
                return json.dumps(msg)
            if isinstance(msg, dict):
                # From conditional: try 'value' key first, then 'content', then serialize
                if 'value' in msg:
                    return extract_message(msg['value'])
                if 'content' in msg:
                    return extract_message(msg['content'])
                return json.dumps(msg)
            return str(msg) if msg is not None else ''

        client: MagicLLM = self.get_input(self.INPUT_HANDLER_CLIENT_PROVIDER, required=True)
        if c := params.get(self.INPUT_HANDLER_CHAT):
            chat = c
            if sys_prompt := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                chat.set_system(extract_message(sys_prompt))
            if user_prompt := self.get_input(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(extract_message(user_prompt))
        else:
            if no_inputs:
                logger.debug("NodeLLM:%s no inputs provided; yielding empty content", self.node_id)
                yield self.yield_static('', content_type=self.OUTPUT_HANDLE_GENERATED)
                return
            sys_context = params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT)
            chat = ModelChat(extract_message(sys_context) if sys_context else None)
            if k := params.get(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(extract_message(k))
            else:
                logger.error("NodeLLM:%s missing required input '%s'", self.node_id, self.INPUT_HANDLER_USER_MESSAGE)
                yield self.yield_debug_error(
                    error_type="InputError",
                    error_message=f"NodeLLM requires input '{self.INPUT_HANDLER_USER_MESSAGE}' with a user message.",
                    context={
                        "available_inputs": list(params.keys()),
                        "required_input": self.INPUT_HANDLER_USER_MESSAGE,
                        "node_config": {"stream": self.stream, "json_output": self.json_output}
                    }
                )
                return

        if not self.stream:
            logger.info("NodeLLM:%s generating (non-stream) with model=%s", self.node_id, client.llm.model)
            intention = await client.llm.async_generate(chat, **self.extra_data)
            self.generated = intention.content
            yield self.yield_static(ChatCompletionModel(
                id=uuid.uuid4().hex,
                model=client.llm.model,
                choices=[ChoiceModel()],
                usage=intention.usage),
                content_type=self.OUTPUT_HANDLE_CONTENT)
        else:
            logger.info("NodeLLM:%s streaming generation with model=%s", self.node_id, client.llm.model)
            async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                self.generated += i.choices[0].delta.content or ''
                yield self.yield_static(i, content_type=self.OUTPUT_HANDLE_CONTENT)
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
                try:
                    self.generated = json.loads(json_content)
                    logger.debug("NodeLLM:%s JSON parsed successfully", self.node_id)
                except json.JSONDecodeError as e:
                    logger.error("NodeLLM:%s JSON parsing failed: %s", self.node_id, e)
                    yield self.yield_debug_error(
                        error_type="JSONParseError",
                        error_message=f"Failed to parse JSON content: {str(e)}",
                        context={
                            "json_content_preview": json_content[:200] if len(json_content) > 200 else json_content,
                            "full_generated_length": len(self.generated),
                            "error_position": getattr(e, 'pos', None)
                        }
                    )
                    return
            else:
                logger.error("NodeLLM:%s no JSON content found in generated output (%d chars)", self.node_id, len(self.generated))
                yield self.yield_debug_error(
                    error_type="JSONExtractionError",
                    error_message="No JSON content found in generated output. The LLM did not produce valid JSON.",
                    context={
                        "generated_preview": self.generated[:500] if len(self.generated) > 500 else self.generated,
                        "generated_length": len(self.generated),
                        "json_output_required": self.json_output,
                        "model": getattr(client.llm, 'model', 'unknown') if 'client' in locals() else 'unknown'
                    }
                )
                return
        yield self.yield_static(self.generated, content_type=self.OUTPUT_HANDLE_GENERATED)

    def _capture_internal_state(self):
        """Capture LLM-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add LLM-specific variables
        state['stream'] = self.stream
        state['json_output'] = self.json_output
        state['iterate'] = self.iterate
        state['generated'] = self.generated[:500] if len(self.generated) > 500 else self.generated  # Truncate long outputs
        state['extra_data'] = self.extra_data
        
        return state
    
    async def __call__(self, chat_log):
        # if configured to iterate inside a Loop, always re-run instead of using cached response
        if getattr(self, 'iterate', False):
            self._response = None
        async for result in super().__call__(chat_log):
            yield result
