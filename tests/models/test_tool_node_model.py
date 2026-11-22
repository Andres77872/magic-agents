import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes.ToolNodeModel import ToolNodeModel


def valid_tool(name: str = "client_lookup-order_1") -> dict:
    return {
        "tool": {
            "type": "function",
            "function": {
                "name": name,
                "description": "Ask the client to look up an order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    }


def test_valid_raw_openai_function_tool_passes_and_is_preserved():
    payload = valid_tool()
    model = ToolNodeModel(**payload)
    assert model.tool == payload["tool"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda t: t.update({"type": "web_search"}), "tool.type"),
        (lambda t: t.pop("function"), "tool.function"),
        (lambda t: t.update({"function": []}), "tool.function"),
        (lambda t: t["function"].update({"name": "invalid name!"}), "tool.function.name"),
        (lambda t: t["function"].update({"name": "x" * 65}), "tool.function.name"),
        (lambda t: t["function"].pop("description"), "tool.function.description"),
        (lambda t: t["function"].update({"description": "  "}), "tool.function.description"),
        (lambda t: t["function"].pop("parameters"), "tool.function.parameters"),
        (lambda t: t["function"].update({"parameters": []}), "tool.function.parameters"),
        (lambda t: t["function"].update({"parameters": {"type": "array"}}), "tool.function.parameters"),
    ],
)
def test_invalid_tool_schemas_fail_with_actionable_path(mutate, expected):
    payload = valid_tool()
    mutate(payload["tool"])
    with pytest.raises(ValidationError) as exc_info:
        ToolNodeModel(**payload)
    assert expected in str(exc_info.value)
