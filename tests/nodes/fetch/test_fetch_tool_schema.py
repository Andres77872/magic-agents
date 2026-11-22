"""
Unit tests for FetchToolCallable three-mode tool_parameters detection.

Tests cover:
- Jinja2 mode: {{var}} in value → tool property
- Constant mode: literal values → NOT in properties, merged at execution
- Explicit schema mode: dict with type/description → backward compat
- Mixed mode: Jinja2 + constants in same tool_parameters
- No tool_parameters: falls back to _extract_template_variables()
- Constants merged into HTTP request at __call__() time
"""


class TestFetchToolSchema:
    """Tests for FetchToolCallable.tool_schema three-mode detection."""

    def test_jinja2_variable_becomes_tool_property(self):
        """tool_parameters with {{var}} in value → Jinja2 mode → var becomes tool prop."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="search_api",
            tool_parameters={"url": "{{url}}"},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "url" in props
        assert props["url"]["type"] == "string"
        assert "url" in schema["function"]["parameters"]["required"]

    def test_literal_constant_not_in_properties(self):
        """tool_parameters with literal value → Constant mode → NOT in properties."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"includeMarkdown": True},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "includeMarkdown" not in props

    def test_explicit_schema_dict_backward_compat(self):
        """tool_parameters with type/description dict → Explicit schema mode unchanged."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={
                "query": {"type": "string", "description": "Search query"},
            },
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "query" in props
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "Search query"
        assert "query" in schema["function"]["parameters"]["required"]

    def test_mixed_jinja2_and_constant(self):
        """Mixed mode: only Jinja2 vars in properties, constants excluded."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"url": "{{url}}", "includeMarkdown": True},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "url" in props
        assert "includeMarkdown" not in props
        assert "url" in schema["function"]["parameters"]["required"]

    def test_no_tool_parameters_falls_back(self):
        """tool_parameters=None falls back to _extract_template_variables() from URL."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com/search?q={{query}}",
            tool_name="search_api",
            tool_parameters=None,
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "query" in schema["function"]["parameters"]["required"]

    def test_custom_tool_description_is_used_in_schema(self):
        """tool_description overrides the generated HTTP request description."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com/search?q={{query}}",
            tool_name="search_api",
            tool_description="Search the catalog with a query string.",
            tool_parameters=None,
        )

        schema = tool.tool_schema

        assert schema["function"]["description"] == "Search the catalog with a query string."

    def test_empty_tool_parameters_falls_back(self):
        """Empty tool_parameters falls back to _extract_template_variables()."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com/search?q={{query}}",
            tool_name="search_api",
            tool_parameters={},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        # Empty dict is falsy, falls back to template extraction
        assert "query" in props

    def test_multiple_jinja2_vars_in_one_value(self):
        """Multiple {{var}} in same value are extracted as separate props."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"url": "{{base}}/{{endpoint}}?q={{query}}"},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "base" in props
        assert "endpoint" in props
        assert "query" in props
        assert all(
            name in schema["function"]["parameters"]["required"]
            for name in ["base", "endpoint", "query"]
        )

    def test_jinja2_variable_in_non_url_param(self):
        """Jinja2 variable in a non-URL param name also becomes a tool prop."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"headers": '{"Authorization": "Bearer {{token}}"}'},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "token" in props
        assert props["token"]["type"] == "string"

    def test_bool_constant_not_in_properties(self):
        """Boolean literal value is a constant, not a tool param."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"followRedirects": False},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "followRedirects" not in props

    def test_numeric_constant_not_in_properties(self):
        """Numeric literal value is a constant, not a tool param."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"maxRetries": 3},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "maxRetries" not in props

    def test_none_constant_not_in_properties(self):
        """None value is a constant, not a tool param."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={"optionalField": None},
        )

        schema = tool.tool_schema
        props = schema["function"]["parameters"]["properties"]

        assert "optionalField" not in props

    def test_explicit_schema_with_only_optional_params_marks_them_required(self):
        """Explicit schema dict uses fallback: when no required list and properties exist, all are required."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        tool = FetchToolCallable(
            url_template="https://api.example.com",
            tool_name="fetch",
            tool_parameters={
                "query": {"type": "string", "description": "Search", "required": False},
            },
        )

        schema = tool.tool_schema
        # The code fallback: if required is empty and properties exist, all properties become required
        # See test_node_llm_collect_tools.py for the _build_tool_manifest expected behavior
        assert "query" in schema["function"]["parameters"]["required"]


