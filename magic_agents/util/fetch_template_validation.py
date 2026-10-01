"""Parse configured Fetch request templates before executable nodes are built.

Validation never renders templates, resolves environment variables, or quotes
configuration values. Dynamic arguments and valid env placeholders stay intact.
"""
import re

from jinja2 import Environment, TemplateSyntaxError


_FIELDS = {
    "url": "endpoint",
    "headers": None,
    "params": "query",
    "data": "body",
    "json_data": "json_body",
}
_SAFE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}\Z")


def validate_fetch_templates(nodes):
    """Return safe structured diagnostics for invalid effective Fetch syntax."""
    errors = []
    parser = Environment()

    def check(value, path, node_id):
        if isinstance(value, str):
            # Ordinary request strings need no template parsing. Parse syntax
            # only: missing credentials/runtime arguments are not evaluated.
            if not any(marker in value for marker in ("{{", "{%", "{#")):
                return
            try:
                parser.parse(value)
            except TemplateSyntaxError as exc:
                errors.append({
                    "error_type": "FetchTemplateValidationError",
                    "error_message": f"Fetch node '{node_id}' has invalid template syntax in '{path}' (line {exc.lineno}).",
                    "context": {"node_id": node_id, "field": path, "lineno": exc.lineno},
                })
        elif isinstance(value, dict):
            for key, item in value.items():
                label = f".{key}" if isinstance(key, str) and _SAFE_KEY.fullmatch(key) else "[item]"
                check(item, path + label, node_id)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check(item, f"{path}[{index}]", node_id)

    for node in nodes:
        if node.get("type") != "fetch":
            continue
        data = node.get("data")
        data = data if isinstance(data, dict) else node
        for field, alias in _FIELDS.items():
            selected = field
            value = data.get(field)
            if value is None and alias is not None:
                selected, value = alias, data.get(alias)
            check(value, selected, node.get("id"))
        parameters = data.get("tool_parameters")
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                # Explicit argument schema entries are documentation for the
                # model, not HTTP request templates. Descriptions can contain
                # literal template notation without ever being executed.
                if isinstance(value, dict) and "type" in value:
                    continue
                label = f".{key}" if isinstance(key, str) and _SAFE_KEY.fullmatch(key) else "[item]"
                check(value, "tool_parameters" + label, node.get("id"))
    return errors
