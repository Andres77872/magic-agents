from jinja2 import Environment
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

env.filters['regex_replace'] = regex_replace
env.filters['regex_findall'] = regex_findall

def template_parse(template, params):
    t = env.from_string(template)  # Use custom env with filters
    o = t.render(params)
    return o