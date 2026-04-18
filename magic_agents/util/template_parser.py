from jinja2 import Environment
import json
import re

env = Environment()

def regex_replace(s, pattern, repl, ignorecase=False, dotall=False):
    flags = 0
    if ignorecase:
        flags |= re.IGNORECASE
    if dotall:
        flags |= re.DOTALL
    return re.sub(pattern, repl, s, flags=flags)

def regex_findall(s, pattern, ignorecase=False, dotall=False):
    flags = 0
    if ignorecase:
        flags |= re.IGNORECASE
    if dotall:
        flags |= re.DOTALL
    return re.findall(pattern, s, flags=flags)

def fromjson(s):
    """Parse a JSON string into a Python object.

    Idempotent: if the input is already a dict, list, or other non-string
    type, it is returned as-is. This handles the case where the parser
    node's safe_json_parse has already converted a JSON string to a dict
    before template rendering.

    Enables Jinja2 templates to decode JSON strings passed as inputs,
    which is essential for parser-to-parser JSON handoff patterns
    (e.g. dynamic flow routing, state management across nodes).
    """
    if isinstance(s, (dict, list)):
        return s
    if s is None:
        return None
    return json.loads(s)

def tojson(value, indent=None):
    """Serialize a Python object to a JSON string.

    Enables Jinja2 templates to emit properly-escaped JSON output,
    used by parser nodes that produce JSON for downstream consumption.
    """
    if indent is not None:
        return json.dumps(value, indent=indent)
    return json.dumps(value)

env.filters['regex_replace'] = regex_replace
env.filters['regex_findall'] = regex_findall
env.filters['fromjson'] = fromjson
env.filters['tojson'] = tojson

def template_parse(template, params):
    t = env.from_string(template)  # Use custom env with filters
    o = t.render(params)
    return o