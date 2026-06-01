import re
from typing import Any

from pydantic import field_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


_FUNCTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolNodeModel(BaseNodeModel):
    """Schema-only tool node model.

    The ``tool`` payload is a raw OpenAI-compatible function tool object. This
    model validates the minimum backend contract but preserves the object shape
    unchanged for provider ``tools=`` passthrough.
    """

    tool: dict[str, Any]

    @field_validator("tool")
    @classmethod
    def validate_tool_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "function":
            raise ValueError('tool.type must be "function"')

        function = value.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool.function must be an object")

        name = function.get("name")
        if not isinstance(name, str) or not _FUNCTION_NAME_PATTERN.fullmatch(name):
            raise ValueError("tool.function.name must match ^[a-zA-Z0-9_-]{1,64}$")

        description = function.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("tool.function.description is required")

        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError('tool.function.parameters must be an object JSON Schema with type "object"')

        return value
