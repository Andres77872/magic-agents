"""Shared JSON Schema contract for authored callable tools."""
from __future__ import annotations

import copy
import re
from typing import Any

from jsonschema import validators


def function_tool_schema(name: str, description: str, parameters: dict | None) -> dict:
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
        raise ValueError("tool_name must contain 1–64 letters, numbers, underscores or dashes")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("tool_description must not be empty")
    schema = copy.deepcopy(parameters if parameters is not None else {
        "type": "object", "properties": {}, "additionalProperties": True,
    })
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("tool_parameters must be a JSON Schema with type object")

    def check_refs(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = value.get(key)
                if ref is not None and (not isinstance(ref, str) or not ref.startswith("#")):
                    raise ValueError("tool_parameters references must be local to the schema")
            for item in value.values():
                check_refs(item)
        elif isinstance(value, list):
            for item in value:
                check_refs(item)

    check_refs(schema)
    try:
        validators.validator_for(schema).check_schema(schema)
    except Exception as exc:
        raise ValueError(f"Invalid tool_parameters JSON Schema: {exc}") from exc
    return {"type": "function", "function": {
        "name": name, "description": description, "parameters": schema,
    }}


def validate_tool_arguments(schema: dict, arguments: dict) -> None:
    validator = validators.validator_for(schema)(schema)
    error = next(validator.iter_errors(arguments), None)
    if error is not None:
        path = ".".join(str(part) for part in error.absolute_path) or "arguments"
        raise ValueError(f"Invalid tool arguments at {path}: {error.message}")
