import os
import sys
import json
import pytest
import asyncio
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from magic_agents import run_agent
from magic_agents.agt_flow import build

# Load API keys if needed for downstream nodes
var_env = json.load(open('/home/andres/Documents/agents_key.json')) if os.path.exists('/home/andres/Documents/agents_key.json') else {}


# Shared agent definition for all tests
AGT_FLOW = {
    "type": "chat",
    "debug": True,
    "edges": [
        {"id": "e1", "source": "user-input", "target": "cond", "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
        {"id": "e2", "source": "cond", "target": "send-msg", "sourceHandle": "empty", "targetHandle": "handle_send_extra"},
        {"id": "e3", "source": "cond", "target": "llm-node", "sourceHandle": "not_empty", "targetHandle": "handle_user_message"},
        {"id": "e4", "source": "client-node", "target": "llm-node", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
        {"id": "e5", "source": "send-msg", "target": "end-node", "sourceHandle": "handle_message_output", "targetHandle": "handle-5"},
        {"id": "e6", "source": "llm-node", "target": "end-node", "sourceHandle": "handle_generated_content", "targetHandle": "handle-6"}
    ],
    "nodes": [
        {"id": "user-input", "type": "user_input"},
        {"id": "cond", "type": "conditional", "data": {"condition": "{{ 'not_empty' if value|default('')|trim else 'empty' }}"}},
        {"id": "send-msg", "type": "send_message", "data": {"message": "", "json_extras": "Input is empty"}},
        {"id": "client-node", "type": "client", "data": {"engine": "openai", "model": "gpt-4o-mini", "api_info": {"api_key": var_env.get('openai_key', ''), "base_url": "https://api.openai.com/v1"}}},
        {"id": "llm-node", "type": "llm", "data": {"top_p": 1, "stream": False, "max_tokens": 5, "temperature": 0.2}},
        {"id": "end-node", "type": "end"}
    ]
}


class TestConditionalFlows:
    """Tests for flows that use the new NodeConditional branching node."""

    def setup_method(self):
        self.load_chat = lambda **kwargs: None  # stub

    @pytest.mark.asyncio
    async def test_empty_input_handling_with_conditional(self):
        """If user input is empty, flow should follow 'empty' branch and bypass 'not_empty'."""
        agt = {
            "type": "chat",
            "debug": True,
            "edges": [
                {
                    "id": "user-to-cond",
                    "source": "user-input",
                    "target": "cond-check",
                    "sourceHandle": "handle_user_message",
                    "targetHandle": "handle_input"
                },
                {
                    "id": "cond-empty-to-text",
                    "source": "cond-check",
                    "target": "text-empty",
                    "sourceHandle": "empty",
                    "targetHandle": "handle_text_input"
                },
                {
                    "id": "cond-not-empty-to-text",
                    "source": "cond-check",
                    "target": "text-not-empty",
                    "sourceHandle": "not_empty",
                    "targetHandle": "handle_text_input"
                },
                {
                    "id": "text-empty-to-end",
                    "source": "text-empty",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-5"
                },
                {
                    "id": "text-not-empty-to-end",
                    "source": "text-not-empty",
                    "target": "end-node",
                    "sourceHandle": "handle_generated_end",
                    "targetHandle": "handle-6"
                }
            ],
            "nodes": [
                {"id": "user-input", "type": "user_input"},
                {
                    "id": "cond-check",
                    "type": "conditional",
                    "data": {
                        # Jinja2 template: branch based on trimmed input length
                        "condition": "{{ 'not_empty' if value|default('')|trim else 'empty' }}"
                    }
                },
                {"id": "text-empty", "type": "text", "data": {"text": "Please provide a valid input. Empty messages are not allowed."}},
                {"id": "text-not-empty", "type": "text", "data": {"text": "Thanks for your message."}},
                {"id": "end-node", "type": "end"}
            ]
        }

        # Build graph with empty user message
        graph = build(agt_data=agt, message='', load_chat=self.load_chat)
        response = ""
        async for item in run_agent(graph=graph):
            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                # text node returns string directly
                response += content if isinstance(content, str) else str(content)

        assert "empty" in response.lower() or "valid input" in response.lower()

    @pytest.mark.asyncio
    async def test_empty_input_sendmessage(self):
        """Empty input should route to NodeSendMessage and bypass LLM using shared flow."""
        agt = deepcopy(AGT_FLOW)
        graph = build(agt_data=agt, message="", load_chat=self.load_chat)
        content_collected = ""
        async for item in run_agent(graph=graph):
            if isinstance(item, dict) and 'content' in item:
                msg = item['content']
                if hasattr(msg, 'choices'):
                    delta = msg.choices[0].delta.content
                    content_collected += delta or ""
        assert "input is empty" in content_collected.lower()




    @pytest.mark.asyncio
    async def test_not_empty_input_llm(self):
        """Non-empty input should return an LLM-generated response (streamed) and bypass SendMessage."""
        if 'openai_key' not in var_env:
            pytest.skip("OpenAI key not configured")

        agt = deepcopy(AGT_FLOW)
        # Ensure the LLM node streams tokens so we can incrementally collect them
        for node in agt["nodes"]:
            if node["id"] == "llm-node":
                node.setdefault("data", {})["stream"] = True

        graph = build(agt_data=agt, message="Hello!", load_chat=self.load_chat)
        content_collected = ""

        async for item in run_agent(graph=graph):
            if isinstance(item, dict) and "content" in item:
                msg = item["content"]
                # When streaming, content is a ChatCompletionModel with delta tokens
                if hasattr(msg, "choices") and msg.choices:
                    delta = msg.choices[0].delta.content
                    content_collected += delta or ""
        print(f"\nLLM content: {content_collected}")
        # Validate that we obtained some non-empty content from the LLM
        assert content_collected.strip() != "", "LLM did not return any content"
        # The SendMessage branch should not have fired
        assert "input is empty" not in content_collected.lower()
