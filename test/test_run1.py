"""Opt-in live smoke tests plus credential-free graph contract checks.

Set ``MAGIC_AGENTS_RUN_LIVE_TESTS=1`` and provide the required provider
credentials to execute the two network-backed smoke tests.  Merely exposing
credentials to the test process must not make the ordinary test suite perform
billable, non-deterministic network calls.
"""

from copy import deepcopy
import json
import os

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from test_support import _is_placeholder_value

# Load API keys from environment or configured file path
_api_keys_file = os.environ.get("MAGIC_AGENTS_API_KEY_FILE", "")
_api_keys_env = os.environ.get("OPENAI_API_KEY", "")
_api_keys_serper = os.environ.get("SERPER_API_KEY", "")

if _api_keys_file and os.path.exists(_api_keys_file):
    with open(_api_keys_file, encoding="utf-8") as api_keys_file:
        var_env = json.load(api_keys_file)
elif _api_keys_env:
    var_env = {"openai_key": _api_keys_env, "serper_key": _api_keys_serper}
else:
    var_env = {}

_live_tests_enabled = os.environ.get("MAGIC_AGENTS_RUN_LIVE_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
_live_opt_in = pytest.mark.skipif(
    not _live_tests_enabled,
    reason="live provider smoke test; set MAGIC_AGENTS_RUN_LIVE_TESTS=1 to opt in",
)
_needs_openai = pytest.mark.skipif(
    _is_placeholder_value(var_env.get("openai_key", "")),
    reason="a real OpenAI API key is required",
)
_needs_openai_and_serper = pytest.mark.skipif(
    _is_placeholder_value(var_env.get("openai_key", ""))
    or _is_placeholder_value(var_env.get("serper_key", "")),
    reason="real OpenAI and Serper API keys are required",
)

template_str = """
You are a query rewrite assistant for a search engine. Your task is to analyze user queries and determine if they require browsing for information. If so, you will rewrite the query to optimize it for search. If not, you will return an empty query.

Here is the user query:
<user_query>
{{ handle_parser_input }}
</user_query>

Analyze the query to determine if it is related to obtaining information through browsing. Ignore queries that are simple greetings, farewells, or other non-informational statements (e.g., "hi", "bye", "thanks").

If the query requires browsing for information, rewrite it to optimize for search. Your rewrite should aim to clarify and focus the query while maintaining its original intent.

Your output must be a JSON object in one of these two formats:

1. For queries requiring browsing:
{
"query":"[rewritten query]"
}

2. For queries not requiring browsing or non-informational queries:
{
"query":""
}

Examples:
Input: "What's the weather like in New York?"
Output: {"query":"current weather forecast New York"}

Input: "hi there"
Output: {"query":""}

Input: "Who won the last World Cup?"
Output: {"query":"winner most recent FIFA World Cup"}

Provide only the JSON object as your response, without any additional text or explanation.
    """
agt = {
    "type": "chat",
    "debug": True,
    "edges": [
        {
            "id": "user_input__llm-final",
            "source": "user_input",
            "target": "llm-final",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_user_message"
        },
        {
            "id": "user_input__system-prompt-rewrite",
            "source": "user_input",
            "target": "system-prompt-rewrite",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_parser_input"
        },
        {
            "id": "llm-client__llm-rewrite",
            "source": "llm-client",
            "target": "llm-rewrite",
            "sourceHandle": "handle-client-provider",
            "targetHandle": "handle-client-provider"
        },
        {
            "id": "system-prompt-rewrite__llm-rewrite",
            "source": "system-prompt-rewrite",
            "target": "llm-rewrite",
            "sourceHandle": "handle_parser_output",
            "targetHandle": "handle_user_message"
        },
        {
            "id": "llm-rewrite__parser-browsing-rewrite",
            "source": "llm-rewrite",
            "target": "parser-browsing-rewrite",
            "sourceHandle": "handle_generated_content",
            "targetHandle": "handle_parser_input"
        },
        {
            "id": "parser-browsing-rewrite__fetch",
            "source": "parser-browsing-rewrite",
            "target": "fetch",
            "sourceHandle": "handle_parser_output",
            "targetHandle": "handle_fetch_input"
        },
        {
            "id": "fetch__parser",
            "source": "fetch",
            "target": "parser-browsing-response",
            "sourceHandle": "handle_fetch_output",
            "targetHandle": "handle_parser_input"
        },
        {
            "id": "fetch__parser-browsing-references",
            "source": "fetch",
            "target": "parser-browsing-references",
            "sourceHandle": "handle_fetch_output",
            "targetHandle": "handle_parser_input"
        },
        {
            "id": "parser-browsing-references__send-message",
            "source": "parser-browsing-references",
            "target": "send-message",
            "sourceHandle": "handle_parser_output",
            "targetHandle": "handle_send_extra"
        },
        {
            "id": "send-message__finish",
            "source": "send-message",
            "target": "finish",
            "sourceHandle": "handle_message_output",
            "targetHandle": "handle_flow_input"
        },
        {
            "id": "parser__system-prompt",
            "source": "parser-browsing-response",
            "target": "system-prompt",
            "sourceHandle": "handle_parser_output",
            "targetHandle": "handle_parser_input"
        },
        {
            "id": "system-prompt__llm-final",
            "source": "system-prompt",
            "target": "llm-final",
            "sourceHandle": "handle_parser_output",
            "targetHandle": "handle-system-context"
        },
        {
            "id": "llm-client__llm-final",
            "source": "llm-client",
            "target": "llm-final",
            "sourceHandle": "handle-client-provider",
            "targetHandle": "handle-client-provider"
        },
        {
            "id": "llm-final__finish",
            "source": "llm-final",
            "target": "finish",
            "sourceHandle": "handle_generated_content",
            "targetHandle": "handle_flow_input"
        }
    ],
    "nodes": [
        {
            "id": "user_input",
            "type": "user_input"
        },
        {
            "id": "finish",
            "type": "end"
        },
        {
            "id": "system-prompt-rewrite",
            "data": {
                "text": template_str
            },
            "type": "parser"
        },
        {
            "id": "llm-client",
            "data": {
                "engine": "openai",
                "api_info": {
                    "api_key": var_env.get('openai_key', ''),
                    "base_url": "https://api.openai.com/v1"
                },
                "model": "gpt-4.1-mini-2025-04-14"
            },
            "type": "client"
        },
        {
            "id": "system-prompt",
            "data": {
                "text": "using the next XML information {{handle_parser_input}} respond the user question"
            },
            "type": "parser"
        },
        {
            "id": "llm-rewrite",
            "data": {
                "top_p": 1,
                "stream": False,
                "max_tokens": 512,
                "temperature": 0.7,
                "json_output": True
            },
            "type": "llm"
        },
        {
            "id": "fetch",
            "data": {
                "url": "https://google.serper.dev/search",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-KEY": var_env.get('serper_key', ''),
                },
                "json_data": {
                    "q": '{{handle_fetch_input}}'
                }
            },
            "type": "fetch"
        },
        {
            "id": "send-message",
            "data": {
                "json_extras": '{{handle_user_message}}',
            },
            "type": "send_message"
        },
        {
            "id": "parser-browsing-response",
            "data": {
                "text": "<search_results>{% for item in handle_parser_input.organic %}<result><title>{{ item.title }}</title><link>{{ item.link }}</link><snippet>{{ item.snippet }}</snippet>{% if item.date is defined %}<date>{{ item.date }}</date>{% endif %}</result>{% endfor %}</search_results>",
            },
            "type": "parser"
        },
        {
            "id": "parser-browsing-references",
            "data": {
                "text": '{"results_ref": [{% for x in handle_parser_input.organic %}{"title": {{ x.title | tojson }},"snippet": {{ x.snippet | tojson }},"link": "{{ x.link }}","imageUrl": "{{ x.imageUrl }}","position": {{ loop.index0 }}}{% if not loop.last %},{% endif %}{% endfor %}]}',
            },
            "type": "parser"
        },
        {
            "id": "parser-browsing-rewrite",
            "data": {
                "text": '{{ handle_parser_input.query }}',
            },
            "type": "parser"
        },
        {
            "id": "llm-final",
            "data": {
                "top_p": 1,
                "stream": True,
                "max_tokens": 512,
                "temperature": 0.7
            },
            "type": "llm"
        }
    ],
    "master": "1"
}


def _build_smoke_graph(agent_definition, message):
    """Build without mutating the module-level serialized fixture."""
    return build(
        agt_data=deepcopy(agent_definition),
        message=message,
        load_chat=lambda **_: None,
    )


def _assert_clean_contract(graph):
    assert graph._validation_errors is None
    assert graph.get_contract_errors() == []
    assert graph.get_contract_warnings() == []


def test_browsing_graph_uses_current_handles():
    graph = _build_smoke_graph(
        agt,
        message="que es la entropia?, dame las referencias",
    )

    _assert_clean_contract(graph)


@pytest.mark.asyncio
@pytest.mark.needs_api
@pytest.mark.slow
@pytest.mark.credential_gated
@_live_opt_in
@_needs_openai_and_serper
async def test_run_agent_live_browsing():
    graph = _build_smoke_graph(
        agt,
        message="que es la entropia?, dame las referencias",
    )

    events = [event async for event in run_agent(graph=graph)]

    assert any(event.get("type") == "content" for event in events)


agt_2 = {
    "type": "chat",
    "debug": True,
    "edges": [
        {
            "id": "xy-edge__0handle_user_message-bb3c62ae-396c-4743-9f61-71064f19b65fhandle-system-context",
            "source": "0",
            "target": "bb3c62ae-396c-4743-9f61-71064f19b65f",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle-system-context"
        },
        {
            "id": "xy-edge__a7c43855-5b27-4a56-9f81-84a3ad1ae662handle-client-provider-bb3c62ae-396c-4743-9f61-71064f19b65fhandle-client-provider",
            "source": "a7c43855-5b27-4a56-9f81-84a3ad1ae662",
            "target": "bb3c62ae-396c-4743-9f61-71064f19b65f",
            "sourceHandle": "handle-client-provider",
            "targetHandle": "handle-client-provider"
        },
        {
            "id": "xy-edge__a7c43855-5b27-4a56-9f81-84a3ad1ae662handle-client-provider-77984a46-f85c-4fb4-917a-787058bdafaehandle-client-provider",
            "source": "a7c43855-5b27-4a56-9f81-84a3ad1ae662",
            "target": "77984a46-f85c-4fb4-917a-787058bdafae",
            "sourceHandle": "handle-client-provider",
            "targetHandle": "handle-client-provider"
        },
        {
            "id": "xy-edge__7a8ee9ee-adc3-4472-a53a-921f7ced36a0handle_text_output-3cd55aef-e446-4614-8fe4-9310dbe77ccahandle_list",
            "source": "7a8ee9ee-adc3-4472-a53a-921f7ced36a0",
            "target": "3cd55aef-e446-4614-8fe4-9310dbe77cca",
            "sourceHandle": "handle_text_output",
            "targetHandle": "handle_list"
        },
        {
            "id": "xy-edge__bb3c62ae-396c-4743-9f61-71064f19b65fhandle_generated_content-3cd55aef-e446-4614-8fe4-9310dbe77ccahandle_loop",
            "source": "bb3c62ae-396c-4743-9f61-71064f19b65f",
            "target": "3cd55aef-e446-4614-8fe4-9310dbe77cca",
            "sourceHandle": "handle_generated_content",
            "targetHandle": "handle_loop"
        },
        {
            "id": "xy-edge__feb1223d-8f98-43cb-b428-2a87714fce41handle_text_output-77984a46-f85c-4fb4-917a-787058bdafaehandle-system-context",
            "source": "feb1223d-8f98-43cb-b428-2a87714fce41",
            "target": "77984a46-f85c-4fb4-917a-787058bdafae",
            "sourceHandle": "handle_text_output",
            "targetHandle": "handle-system-context"
        },
        {
            "id": "xy-edge__3cd55aef-e446-4614-8fe4-9310dbe77ccahandle_item-bb3c62ae-396c-4743-9f61-71064f19b65fhandle_user_message",
            "source": "3cd55aef-e446-4614-8fe4-9310dbe77cca",
            "target": "bb3c62ae-396c-4743-9f61-71064f19b65f",
            "sourceHandle": "handle_item",
            "targetHandle": "handle_user_message"
        },
        {
            "id": "xy-edge__3cd55aef-e446-4614-8fe4-9310dbe77ccahandle_end-77984a46-f85c-4fb4-917a-787058bdafaehandle_user_message",
            "source": "3cd55aef-e446-4614-8fe4-9310dbe77cca",
            "target": "77984a46-f85c-4fb4-917a-787058bdafae",
            "sourceHandle": "handle_end",
            "targetHandle": "handle_user_message"
        },
        {
            "id": "final-llm__finish",
            "source": "77984a46-f85c-4fb4-917a-787058bdafae",
            "target": "5",
            "sourceHandle": "handle_generated_content",
            "targetHandle": "handle_flow_input"
        }
    ],
    "nodes": [
        {
            "id": "0",
            "type": "user_input",
            "measured": {
                "width": 506,
                "height": 179
            },
            "position": {
                "x": -1666.7350815992695,
                "y": 369.6754907927339
            }
        },
        {
            "id": "5",
            "type": "end",
            "dragging": False,
            "measured": {
                "width": 370,
                "height": 179
            },
            "position": {
                "x": 1827.6361753428084,
                "y": 1327.8314985233858
            },
            "selected": False
        },
        {
            "id": "3cd55aef-e446-4614-8fe4-9310dbe77cca",
            "data": {},
            "type": "loop",
            "measured": {
                "width": 355,
                "height": 324
            },
            "position": {
                "x": 305.0178762547595,
                "y": 1549.5650238720527
            }
        },
        {
            "id": "a7c43855-5b27-4a56-9f81-84a3ad1ae662",
            "data": {
                "model": "gpt-4.1-2025-04-14",
                "engine": "openai",
                "api_info": {
                    "api_key": var_env.get('openai_key', ''),
                    "base_url": "https://api.openai.com/v1"
                }
            },
            "type": "client",
            "measured": {
                "width": 329,
                "height": 739
            },
            "position": {
                "x": -986.4856640928764,
                "y": 762.5258604478856
            }
        },
        {
            "id": "bb3c62ae-396c-4743-9f61-71064f19b65f",
            "data": {
                "model": "",
                "top_p": 1,
                "stream": True,
                "iterate": True,
                "max_tokens": 512,
                "json_output": False,
                "temperature": 0.7
            },
            "type": "llm",
            "measured": {
                "width": 280,
                "height": 1323
            },
            "position": {
                "x": 386.89511202650374,
                "y": -11.550584590174504
            }
        },
        {
            "id": "7a8ee9ee-adc3-4472-a53a-921f7ced36a0",
            "data": {
                "text": "[1,2,3]"
            },
            "type": "text",
            "measured": {
                "width": 294,
                "height": 325
            },
            "position": {
                "x": -462.64093948861154,
                "y": 1592.8627621547648
            }
        },
        {
            "id": "77984a46-f85c-4fb4-917a-787058bdafae",
            "data": {
                "model": "",
                "top_p": 1,
                "stream": True,
                "iterate": False,
                "max_tokens": 512,
                "json_output": False,
                "temperature": 0.7
            },
            "type": "llm",
            "measured": {
                "width": 280,
                "height": 1323
            },
            "position": {
                "x": 1203.0720591976494,
                "y": 516.8306555430545
            }
        },
        {
            "id": "feb1223d-8f98-43cb-b428-2a87714fce41",
            "data": {
                "text": "Describe the content"
            },
            "type": "text",
            "measured": {
                "width": 294,
                "height": 325
            },
            "position": {
                "x": 739.8972164714987,
                "y": 1090.8467279103088
            }
        }
    ],
    "master": "1"
}


def test_loop_graph_uses_current_handles():
    graph = _build_smoke_graph(agt_2, message="Write N times the user input")

    _assert_clean_contract(graph)


@pytest.mark.asyncio
@pytest.mark.needs_api
@pytest.mark.slow
@pytest.mark.credential_gated
@_live_opt_in
@_needs_openai
async def test_run_agent_live_loop():
    graph = _build_smoke_graph(agt_2, message="Write N times the user input")

    events = [event async for event in run_agent(graph=graph)]

    assert any(event.get("type") == "content" for event in events)
