"""Authored Jinja templates render in a sandbox and cannot read the server environment."""
import pytest
from jinja2.exceptions import SecurityError

from magic_agents.execution.condition_evaluator_jinja2 import Jinja2Evaluator, Jinja2StrictEvaluator
from magic_agents.util.template_parser import template_parse

ESCAPE = "{{ lipsum.__globals__.os.environ.get('MAGIC_AGENTS_SANDBOX_PROBE') }}"


@pytest.mark.parametrize('render', [
    lambda template: template_parse(template, {}),
    lambda template: Jinja2Evaluator().evaluate(template, {}),
    lambda template: Jinja2StrictEvaluator().evaluate(template, {}),
])
def test_attribute_escapes_are_blocked(monkeypatch, render):
    monkeypatch.setenv('MAGIC_AGENTS_SANDBOX_PROBE', 'server-secret')
    with pytest.raises(SecurityError):
        render(ESCAPE)


def test_parser_filters_still_work():
    rendered = template_parse('{{ x | tojson }} {{ d.items() | list }} {{ "a-b" | regex_replace("-", "+") }}',
                              {'x': {'k': 1}, 'd': {'a': 1}})
    assert rendered == '{"k": 1} [(\'a\', 1)] a+b'
