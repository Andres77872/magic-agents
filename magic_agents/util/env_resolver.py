import os
import re
from typing import Any


ENV_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def resolve_env_string(value: str) -> str:
    """Resolve {{env.VAR_NAME}} placeholders inside a string."""
    return ENV_PLACEHOLDER_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)


def resolve_env_placeholders(value: Any) -> Any:
    """Recursively resolve env placeholders in nested JSON-like values.

    Only {{env.VAR_NAME}} placeholders are resolved. All other Jinja-style
    runtime placeholders (for example {{handle_fetch_input}}) are preserved.
    """
    if isinstance(value, str):
        return resolve_env_string(value)
    if isinstance(value, list):
        return [resolve_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {key: resolve_env_placeholders(item) for key, item in value.items()}
    return value
