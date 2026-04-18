"""
Slice 18 — Real API integration tests (slow, non-deterministic).

Full end-to-end flows with real OpenAI + Serper API calls.
These tests are skipped if required API keys are not available.

Requirements:
- OPENAI_API_KEY environment variable, OR
- A JSON file at a path specified by MAGIC_AGENTS_API_KEY_FILE env var

DO NOT hardcode personal paths. Use environment variables only.
"""
import json
import os
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build

from conftest import skip_if_no_api_keys, _is_placeholder_value


def extract_streamed_content(item):
    """Extract streamed content from LLM output."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "content":
        return ""
    content = item.get("content")
    if content is None:
        return ""
    if hasattr(content, "choices") and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
    return ""


@pytest.mark.needs_api
@pytest.mark.slow
class TestBrowsingFlowNoSearch:
    """Browsing flow with a greeting — should NOT trigger search."""

    @pytest.mark.asyncio
    async def test_browsing_flow_no_search(self, api_keys):
        """browsing.json with greeting → OpenAI (no Serper call needed)."""
        keys = skip_if_no_api_keys(api_keys)

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "user_input", "type": "user_input"},
                {"id": "finish", "type": "end"},
                {"id": "llm-final", "type": "llm", "data": {
                    "top_p": 0.95,
                    "temperature": 0.65,
                    "max_tokens": 256,
                    "stream": True,
                }},
                {"id": "client", "type": "client", "data": {
                    "model": "gpt-4o-mini",
                    "engine": "openai",
                    "api_info": {
                        "api_key": keys["openai_key"],
                        "base_url": "https://api.openai.com/v1",
                    },
                }},
            ],
            "edges": [
                {"id": "e1", "source": "user_input", "target": "llm-final",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "client", "target": "llm-final",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e3", "source": "llm-final", "target": "finish",
                 "sourceHandle": "handle_generated_end", "targetHandle": "handle_generated_end"},
            ],
        }

        graph = build(agt, message="Hello, how are you?", load_chat=None)
        response = ""
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                response += text

        assert len(response) > 0


@pytest.mark.needs_api
@pytest.mark.slow
class TestLoopWithRealLLM:
    """Loop execution with real LLM calls."""

    @pytest.mark.asyncio
    async def test_loop_with_real_llm(self, api_keys):
        """loop.json with real LLM → OpenAI processes each item."""
        keys = skip_if_no_api_keys(api_keys)

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["apple", "banana"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "client", "type": "client", "data": {
                    "model": "gpt-4o-mini",
                    "engine": "openai",
                    "api_info": {
                        "api_key": keys["openai_key"],
                        "base_url": "https://api.openai.com/v1",
                    },
                }},
                {"id": "processor", "type": "llm", "data": {
                    "stream": True,
                    "iterate": True,
                    "max_tokens": 50,
                    "temperature": 0.7,
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "processor",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle-system-context"},
                {"id": "e2", "source": "client", "target": "processor",
                 "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
                {"id": "e3", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e4", "source": "loop", "target": "processor",
                 "sourceHandle": "handle_item", "targetHandle": "handle_user_message"},
                {"id": "e5", "source": "processor", "target": "loop",
                 "sourceHandle": "handle_generated_content", "targetHandle": "handle_loop"},
                {"id": "e6", "source": "loop", "target": "end",
                 "sourceHandle": "handle_end", "targetHandle": "h1"},
            ],
        }

        graph = build(
            agt,
            message="Describe each fruit in one word",
            load_chat=None,
        )
        response = ""
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                response += text

        # Should contain references to fruits being processed
        assert len(response) > 0


@pytest.mark.needs_api
@pytest.mark.slow
class TestImageJsonFlow:
    """image.json flow with real LLM."""

    @pytest.mark.asyncio
    async def test_image_json_french_response(self, api_keys):
        """image.json → OpenAI (response should reference French system prompt)."""
        keys = skip_if_no_api_keys(api_keys)

        # Load image.json
        examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "json")
        image_json_path = os.path.join(examples_dir, "image.json")
        if not os.path.exists(image_json_path):
            pytest.skip("image.json not found in examples/json/")

        with open(image_json_path, "r") as f:
            agt = json.load(f)

        # Replace any hardcoded API keys with our loaded key
        for node in agt.get("content", {}).get("nodes", []):
            if node.get("type") == "client" and "data" in node:
                api_info = node["data"].get("api_info", {})
                if "api_key" in api_info:
                    api_info["api_key"] = keys["openai_key"]

        graph = build(agt, message="What is the capital of France?", load_chat=None)
        response = ""
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                response += text

        # Should get a response (the system prompt says "Always respond in French")
        assert len(response) > 0


@pytest.mark.needs_api
@pytest.mark.slow
class TestBrowsingFlowWithSearch:
    """Browsing flow with a factual query — triggers search + LLM."""

    @pytest.mark.asyncio
    async def test_browsing_flow_with_search(self, api_keys):
        """browsing.json with factual query → OpenAI + Serper (if available)."""
        keys = skip_if_no_api_keys(api_keys)

        # Check if Serper key is also available
        serper_key = os.environ.get("SERPER_API_KEY") or keys.get("serper_key")
        if not serper_key:
            pytest.skip("SERPER_API_KEY not set — browsing flow with search requires Serper")

        # Check if Jina API key is available (fetch nodes require it)
        jina_key = os.environ.get("JINA_API_KEY", "")
        if not jina_key or _is_placeholder_value(jina_key):
            pytest.skip("JINA_API_KEY not set or invalid — browsing flow with search requires Jina for fetch nodes")

        # Load browsing.json
        examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "json")
        browsing_json_path = os.path.join(examples_dir, "browsing.json")
        if not os.path.exists(browsing_json_path):
            pytest.skip("browsing.json not found in examples/json/")

        with open(browsing_json_path, "r") as f:
            agt = json.load(f)

        # Replace hardcoded API keys
        for node in agt.get("nodes", []):
            if node.get("type") == "client" and "data" in node:
                api_info = node["data"].get("api_info", {})
                if "api_key" in api_info:
                    api_info["api_key"] = keys["openai_key"]
            # Replace Jina API key in fetch nodes if present
            if node.get("type") == "fetch" and "data" in node:
                headers = node["data"].get("headers", {})
                if "Authorization" in headers:
                    # Keep the Jina key as-is (it's not the OpenAI key)
                    pass

        graph = build(agt, message="What is the current population of Tokyo?", load_chat=None)
        response = ""
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                response += text

        assert len(response) > 0
