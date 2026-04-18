"""
Tests for template parsing helpers and env placeholder resolution.

These protect the fromjson/tojson filters and the env placeholder handling
used by JSON flow configs.
"""
import os
import pytest
from magic_agents.util.template_parser import template_parse, env
from magic_agents.util.env_resolver import resolve_env_placeholders, resolve_env_string


class TestFromjsonFilter:
    """Tests for the fromjson Jinja2 filter."""

    def test_fromjson_simple_object(self):
        """Parse a simple JSON object."""
        template = "{{ data | fromjson }}"
        result = template_parse(template, {"data": '{"key": "value"}'})
        assert result == "{'key': 'value'}"

    def test_fromjson_list(self):
        """Parse a JSON array."""
        template = "{{ data | fromjson }}"
        result = template_parse(template, {"data": '[1, 2, 3]'})
        assert result == "[1, 2, 3]"

    def test_fromjson_nested_object(self):
        """Parse nested JSON and access a key."""
        template = "{% set config = data | fromjson %}{{ config.system }}"
        result = template_parse(template, {
            "data": '{"system": "You are helpful", "temperature": 0.7}'
        })
        assert result == "You are helpful"

    def test_fromjson_access_list_element(self):
        """Parse JSON array and access an element."""
        template = "{% set items = data | fromjson %}{{ items[1] }}"
        result = template_parse(template, {"data": '["first", "second", "third"]'})
        assert result == "second"

    def test_fromjson_invalid_json_raises(self):
        """Invalid JSON should raise an error during template render."""
        template = "{{ data | fromjson }}"
        with pytest.raises(Exception):
            template_parse(template, {"data": "not valid json"})

    def test_fromjson_idempotent_dict(self):
        """fromjson returns dict as-is when input is already parsed.

        This handles the case where the parser node's safe_json_parse
        has already converted a JSON string to a dict before template
        rendering — the | fromjson filter should not double-parse.
        """
        template = "{% set data = payload | fromjson %}{{ data.user.name }}"
        result = template_parse(template, {
            "payload": {"user": {"name": "John"}}
        })
        assert result == "John"

    def test_fromjson_idempotent_list(self):
        """fromjson returns list as-is when input is already parsed."""
        template = "{% set items = payload | fromjson %}{{ items[1] }}"
        result = template_parse(template, {
            "payload": ["first", "second", "third"]
        })
        assert result == "second"

    def test_fromjson_none_passthrough(self):
        """fromjson returns None as-is when input is None."""
        template = "{% set data = payload | fromjson %}{{ data is none }}"
        result = template_parse(template, {"payload": None})
        assert result == "True"


class TestTojsonFilter:
    """Tests for the tojson Jinja2 filter."""

    def test_tojson_simple_string(self):
        """Serialize a string to JSON."""
        template = "{{ data | tojson }}"
        result = template_parse(template, {"data": "hello"})
        assert result == '"hello"'

    def test_tojson_dict(self):
        """Serialize a dict to JSON."""
        template = "{{ data | tojson }}"
        result = template_parse(template, {"data": {"key": "value"}})
        assert result == '{"key": "value"}'

    def test_tojson_list(self):
        """Serialize a list to JSON."""
        template = "{{ data | tojson }}"
        result = template_parse(template, {"data": [1, 2, 3]})
        assert result == '[1, 2, 3]'

    def test_tojson_with_indent(self):
        """Serialize with indentation."""
        template = "{{ data | tojson(2) }}"
        result = template_parse(template, {"data": {"key": "value"}})
        assert "{\n" in result
        assert '"key": "value"' in result

    def test_tojson_escapes_special_chars(self):
        """tojson should properly escape quotes and special chars."""
        template = "{{ data | tojson }}"
        result = template_parse(template, {"data": 'say "hello"'})
        assert result == '"say \\"hello\\""'


class TestFromjsonTojsonRoundTrip:
    """Tests for fromjson/tojson round-trip patterns used in state management."""

    def test_roundtrip_state_transformation(self):
        """Simulate the state management pattern from test_advanced_flows.py."""
        template = """{% set state = data | fromjson %}
{% set _ = state.update({"step": 2}) %}
{{ state | tojson }}"""
        result = template_parse(template, {
            "data": '{"original": "hello", "step": 1}'
        })
        assert '"step": 2' in result
        assert '"original": "hello"' in result

    def test_roundtrip_list_append(self):
        """Simulate appending to a list via fromjson/tojson."""
        template = """{% set items = data | fromjson %}
{% set _ = items.append("new_item") %}
{{ items | tojson }}"""
        result = template_parse(template, {"data": '["a", "b"]'})
        assert '"a"' in result
        assert '"b"' in result
        assert '"new_item"' in result


class TestEnvResolver:
    """Tests for config-time env placeholder resolution."""

    def test_resolve_env_string_replaces_only_env_placeholders(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

        result = resolve_env_string("Bearer {{env.OPENAI_API_KEY}} :: {{ handle_fetch_input }}")

        assert result == "Bearer test-openai-key :: {{ handle_fetch_input }}"

    def test_resolve_env_placeholders_walks_nested_values(self, monkeypatch):
        monkeypatch.setenv("TOKEN", "nested-token")

        payload = {
            "headers": {"Authorization": "Bearer {{env.TOKEN}}"},
            "json": [{"token": "{{env.TOKEN}}"}, "{{ handle_fetch_input }}"],
        }

        result = resolve_env_placeholders(payload)

        assert result == {
            "headers": {"Authorization": "Bearer nested-token"},
            "json": [{"token": "nested-token"}, "{{ handle_fetch_input }}"],
        }

    def test_resolve_env_string_uses_empty_string_for_missing_env(self):
        os.environ.pop("MISSING_ENV_FOR_TEST", None)

        result = resolve_env_string("x{{env.MISSING_ENV_FOR_TEST}}y")

        assert result == "xy"
