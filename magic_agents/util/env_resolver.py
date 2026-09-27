import logging
import os
import re
from typing import Any


logger = logging.getLogger(__name__)


ENV_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*env\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def resolve_env_string(value: str) -> str:
    """Resolve {{env.VAR_NAME}} placeholders inside a string."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        val = os.environ.get(var_name, "")
        if not val:
            logger.warning(
                "Environment variable '%s' is not set; '{{env.%s}}' resolved to empty string",
                var_name,
                var_name,
            )
        return val

    return ENV_PLACEHOLDER_PATTERN.sub(_replace, value)


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


# MCP server settings that carry connection credentials. Client and Fetch nodes
# resolve their own credential fields when they run.
_MCP_CONNECTION_FIELDS = ("command", "args", "env", "cwd", "url", "headers")


def resolve_connection_placeholders(nodes: Any) -> Any:
    """Resolve {{env.VAR_NAME}} only in node connection settings.

    Placeholders are secrets for the platform's own connections (LLM clients,
    HTTP fetches, MCP servers). They are never resolved in text, prompts,
    templates or constants, where the resolved value would be echoed back to the
    graph author and leak the server's environment.

    Returns a new list; the caller's node dicts are not modified.
    """
    if not isinstance(nodes, list):
        return nodes

    resolved_nodes = []
    for node in nodes:
        data = node.get("data") if isinstance(node, dict) else None
        servers = data.get("servers") if isinstance(data, dict) and node.get("type") == "mcp" else None
        if not isinstance(servers, list):
            resolved_nodes.append(node)
            continue

        resolved_servers = [
            {
                key: resolve_env_placeholders(value) if key in _MCP_CONNECTION_FIELDS else value
                for key, value in server.items()
            } if isinstance(server, dict) else server
            for server in servers
        ]
        resolved_nodes.append({**node, "data": {**data, "servers": resolved_servers}})
    return resolved_nodes
