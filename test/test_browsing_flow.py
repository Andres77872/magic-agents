"""
Test suite for the browsing agent flow.

This test module validates the browsing.json agent flow which:
1. Analyzes user queries to determine if search is needed
2. Routes via conditional node (has_queries/no_queries branches)
3. Expands queries and performs parallel web searches
4. Consolidates results and generates final responses

Tests cover:
- Conditional routing (search vs no-search queries)
- Query expansion behavior
- Edge cases (empty input, greetings, commands)
- Flow structure validation
"""

import os
import sys
import json
import pytest
import asyncio
from pathlib import Path
from copy import deepcopy
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from magic_agents import run_agent
from magic_agents.agt_flow import build, validate_graph


# Path to the browsing.json flow
BROWSING_FLOW_PATH = Path(__file__).parent.parent / "examples" / "json" / "browsing.json"

# Load API keys from environment or configured file path
VAR_ENV = {}
_api_keys_file = os.environ.get("MAGIC_AGENTS_API_KEY_FILE", "")
_api_keys_env = os.environ.get("OPENAI_API_KEY", "")
_api_keys_serper = os.environ.get("SERPER_API_KEY", "")

if _api_keys_file and os.path.exists(_api_keys_file):
    VAR_ENV = json.load(open(_api_keys_file))
elif _api_keys_env:
    VAR_ENV = {"openai_key": _api_keys_env, "serper_key": _api_keys_serper}


def load_browsing_flow() -> dict:
    """Load the browsing.json flow definition."""
    with open(BROWSING_FLOW_PATH, 'r') as f:
        return json.load(f)


class TestBrowsingFlowStructure:
    """Tests for the browsing flow structure and validation."""

    def test_flow_file_exists(self):
        """Verify the browsing.json file exists."""
        assert BROWSING_FLOW_PATH.exists(), f"Flow file not found: {BROWSING_FLOW_PATH}"

    def test_flow_has_required_nodes(self):
        """Verify the flow contains all required node types."""
        flow = load_browsing_flow()
        node_types = {node['type'] for node in flow['nodes']}
        node_ids = {node['id'] for node in flow['nodes']}
        
        # Required node types for the flow
        assert 'user_input' in node_types, "Flow must have user_input node"
        assert 'conditional' in node_types, "Flow must have conditional node"
        assert 'end' in node_types, "Flow must have end node"
        assert 'parser' in node_types, "Flow must have parser nodes"
        assert 'llm' in node_types, "Flow must have LLM nodes"
        assert 'fetch' in node_types, "Flow must have fetch nodes"
        assert 'send_message' in node_types, "Flow must have send_message nodes"
        
        # Key node IDs
        assert 'user_input' in node_ids, "Must have 'user_input' node"
        assert 'conditional-queries-check' in node_ids, "Must have conditional routing node"

    def test_flow_graph_validation(self):
        """Validate the flow graph structure."""
        flow = load_browsing_flow()
        result = validate_graph(flow['nodes'], flow['edges'])
        
        # Should have exactly one user_input node (valid)
        assert result['valid'], f"Graph validation failed: {result['errors']}"

    def test_conditional_node_configuration(self):
        """Verify the conditional node is properly configured."""
        flow = load_browsing_flow()
        
        conditional_node = None
        for node in flow['nodes']:
            if node['id'] == 'conditional-queries-check':
                conditional_node = node
                break
        
        assert conditional_node is not None, "Conditional node not found"
        assert conditional_node['type'] == 'conditional'
        assert 'data' in conditional_node
        assert 'condition' in conditional_node['data']
        
        # Verify output handles are defined
        data = conditional_node['data']
        assert 'output_handles' in data
        assert 'has_queries' in data['output_handles']
        assert 'no_queries' in data['output_handles']
        assert data.get('default_handle') == 'no_queries'

    def test_flow_has_parallel_fetch_nodes(self):
        """Verify multiple fetch nodes exist for parallel query execution."""
        flow = load_browsing_flow()
        
        fetch_nodes = [node for node in flow['nodes'] if node['type'] == 'fetch']
        
        # Should have 3 fetch nodes for parallel query execution
        assert len(fetch_nodes) == 3, f"Expected 3 fetch nodes, found {len(fetch_nodes)}"

    def test_conditional_routing_edges(self):
        """Verify conditional node has correct outgoing edges."""
        flow = load_browsing_flow()
        
        conditional_edges = [
            edge for edge in flow['edges']
            if edge['source'] == 'conditional-queries-check'
        ]
        
        # Check has_queries edge exists
        has_queries_edges = [
            e for e in conditional_edges 
            if e.get('sourceHandle') == 'has_queries'
        ]
        assert len(has_queries_edges) > 0, "Missing 'has_queries' output edge"
        
        # Check no_queries edge exists
        no_queries_edges = [
            e for e in conditional_edges 
            if e.get('sourceHandle') == 'no_queries'
        ]
        assert len(no_queries_edges) > 0, "Missing 'no_queries' output edge"


class TestConditionalRouting:
    """Tests for the conditional routing behavior based on query type."""

    @pytest.fixture
    def mock_llm_response_no_search(self):
        """Mock LLM response for non-search queries (greetings, etc.)."""
        return {"queries": []}

    @pytest.fixture
    def mock_llm_response_with_search(self):
        """Mock LLM response for search queries."""
        return {
            "queries": [
                "test query 1",
                "test query 2",
                "test query 3"
            ]
        }

    def test_condition_evaluates_correctly_for_empty_queries(self):
        """Test Jinja2 condition evaluates to 'no_queries' when queries array is empty."""
        import jinja2
        
        condition = "{{ 'has_queries' if queries is defined and queries is iterable and queries|list|length > 0 else 'no_queries' }}"
        env = jinja2.Environment()
        template = env.from_string(condition)
        
        # Empty queries should route to no_queries
        result = template.render(queries=[])
        assert result.strip() == 'no_queries'
        
        # Undefined queries should route to no_queries
        result = template.render()
        assert result.strip() == 'no_queries'

    def test_condition_evaluates_correctly_for_populated_queries(self):
        """Test Jinja2 condition evaluates to 'has_queries' when queries exist."""
        import jinja2
        
        condition = "{{ 'has_queries' if queries is defined and queries is iterable and queries|list|length > 0 else 'no_queries' }}"
        env = jinja2.Environment()
        template = env.from_string(condition)
        
        # Non-empty queries should route to has_queries
        result = template.render(queries=["query1", "query2"])
        assert result.strip() == 'has_queries'
        
        # Single query should also route to has_queries
        result = template.render(queries=["query1"])
        assert result.strip() == 'has_queries'


class TestQueryExpansionParsers:
    """Tests for the parser nodes that extract queries from LLM output."""

    def test_query_extraction_parser_0(self):
        """Test first query extraction parser."""
        import jinja2
        
        template_str = """{%- if handle_parser_input_0.queries is defined and handle_parser_input_0.queries|length > 0 -%}
  {{ handle_parser_input_0.queries[0] }}
{%- endif -%}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        input_data = {"queries": ["first query", "second query", "third query"]}
        result = template.render(handle_parser_input_0=input_data)
        
        assert "first query" in result

    def test_query_extraction_parser_1(self):
        """Test second query extraction parser."""
        import jinja2
        
        template_str = """{%- if handle_parser_input_0.queries is defined and handle_parser_input_0.queries|length > 1 -%}
  {{ handle_parser_input_0.queries[1] }}
{%- endif -%}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        input_data = {"queries": ["first query", "second query", "third query"]}
        result = template.render(handle_parser_input_0=input_data)
        
        assert "second query" in result

    def test_query_extraction_parser_2(self):
        """Test third query extraction parser."""
        import jinja2
        
        template_str = """{%- if handle_parser_input_0.queries is defined and handle_parser_input_0.queries|length > 2 -%}
  {{ handle_parser_input_0.queries[2] }}
{%- endif -%}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        input_data = {"queries": ["first query", "second query", "third query"]}
        result = template.render(handle_parser_input_0=input_data)
        
        assert "third query" in result

    def test_query_extraction_with_insufficient_queries(self):
        """Test parser handles case when fewer queries are returned."""
        import jinja2
        
        template_str = """{%- if handle_parser_input_0.queries is defined and handle_parser_input_0.queries|length > 2 -%}
  {{ handle_parser_input_0.queries[2] }}
{%- endif -%}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        # Only 2 queries - third parser should output nothing
        input_data = {"queries": ["first query", "second query"]}
        result = template.render(handle_parser_input_0=input_data)
        
        assert result.strip() == ""


class TestSearchResultsParsing:
    """Tests for the parser that consolidates search results."""

    def test_browsing_response_parser_template(self):
        """Test the search results consolidation parser."""
        import jinja2
        
        # Simplified version of the template
        template_str = """{%- set all_items = [] -%}
{%- for input_source in [handle_parser_input_0, handle_parser_input_1] -%}
  {%- if input_source.data is defined and input_source.data is not none -%}
    {%- for item in input_source.data -%}
      {%- set _ = all_items.append(item) -%}
    {%- endfor -%}
  {%- endif -%}
{%- endfor -%}
{%- if all_items|length > 0 -%}
<search_results>
{%- for item in all_items -%}
<result>
<title>{{ item.title }}</title>
<link>{{ item.url }}</link>
</result>
{%- endfor -%}
</search_results>
{%- endif -%}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        input_0 = {
            "data": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2"}
            ]
        }
        input_1 = {
            "data": [
                {"title": "Result 3", "url": "https://example.com/3", "content": "Content 3"}
            ]
        }
        
        result = template.render(
            handle_parser_input_0=input_0,
            handle_parser_input_1=input_1
        )
        
        assert "<search_results>" in result
        assert "Result 1" in result
        assert "Result 2" in result
        assert "Result 3" in result


class TestBrowsingFlowExecution:
    """Integration tests for the browsing flow execution.
    
    Note: These tests require API keys and make real API calls.
    Skip if keys are not available.
    """

    @pytest.fixture
    def flow_with_mock_client(self):
        """Create flow with modified client for testing."""
        flow = load_browsing_flow()
        
        # Replace API key with test key if available
        if 'openai_key' in VAR_ENV:
            for node in flow['nodes']:
                if node['type'] == 'client' and 'data' in node:
                    if 'api_info' in node['data']:
                        node['data']['api_info']['api_key'] = VAR_ENV['openai_key']
        
        return flow

    @pytest.mark.asyncio
    @pytest.mark.skipif('openai_key' not in VAR_ENV, reason="OpenAI API key not available")
    async def test_flow_builds_successfully(self, flow_with_mock_client):
        """Test that the flow can be built without errors."""
        graph = build(
            agt_data=flow_with_mock_client,
            message="Hello",
            load_chat=None
        )
        
        assert graph is not None
        assert hasattr(graph, 'nodes')
        assert len(graph.nodes) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif('openai_key' not in VAR_ENV, reason="OpenAI API key not available")
    async def test_greeting_routes_to_no_search(self, flow_with_mock_client):
        """Test that a greeting message bypasses search."""
        # This is a conceptual test - in real execution, the LLM decides
        # Here we verify the flow can handle such routing
        graph = build(
            agt_data=flow_with_mock_client,
            message="Hi there!",
            load_chat=None
        )
        
        # Flow should build without validation errors for valid structure
        validation_errors = getattr(graph, '_validation_errors', None)
        if validation_errors:
            # Filter out warnings
            errors_only = [e for e in validation_errors if e.get('severity') != 'warning']
            assert len(errors_only) == 0, f"Unexpected validation errors: {errors_only}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_user_input(self):
        """Test flow handling of empty user input."""
        flow = load_browsing_flow()
        
        # Should build without error even with empty message
        graph = build(
            agt_data=flow,
            message="",
            load_chat=None
        )
        
        assert graph is not None

    def test_very_long_user_input(self):
        """Test flow handling of very long user input."""
        flow = load_browsing_flow()
        
        long_message = "What is the meaning of life? " * 100
        
        graph = build(
            agt_data=flow,
            message=long_message,
            load_chat=None
        )
        
        assert graph is not None

    def test_special_characters_in_input(self):
        """Test flow handling of special characters."""
        flow = load_browsing_flow()
        
        special_message = "What does <script>alert('test')</script> mean? {{template}}"
        
        graph = build(
            agt_data=flow,
            message=special_message,
            load_chat=None
        )
        
        assert graph is not None

    def test_unicode_input(self):
        """Test flow handling of unicode characters."""
        flow = load_browsing_flow()
        
        unicode_message = "¿Qué significa 你好 в мире? 🌍"
        
        graph = build(
            agt_data=flow,
            message=unicode_message,
            load_chat=None
        )
        
        assert graph is not None


class TestQueryClassification:
    """Tests to validate query classification logic."""

    @pytest.mark.parametrize("query,should_search", [
        # Non-search queries (greetings, commands, acknowledgments)
        ("hi", False),
        ("hello there", False),
        ("thanks", False),
        ("goodbye", False),
        ("ok", False),
        ("I'm tired", False),
        
        # Search queries (information-seeking)
        ("What is Python?", True),
        ("How to make pasta?", True),
        ("Who won the World Cup 2022?", True),
        ("What's the weather in New York?", True),
        ("Tell me about quantum computing", True),
        ("Explain machine learning", True),
    ])
    def test_query_classification_expectations(self, query: str, should_search: bool):
        """
        Verify that keyword-based query classification aligns with expectations.

        The browsing.json flow uses an LLM-based conditional to decide whether
        to search, but this test validates that a simple keyword heuristic
        produces the same classification — serving as a sanity check that
        our test data is consistent.
        """
        search_indicators = [
            "what", "who", "where", "when", "why", "how",
            "tell me", "explain", "describe", "define",
            "search", "find", "look up"
        ]

        query_lower = query.lower()
        has_search_indicator = any(ind in query_lower for ind in search_indicators)

        if should_search:
            assert has_search_indicator, (
                f"Query '{query}' is expected to trigger search but has no "
                f"search indicators. Add an indicator or adjust expectation."
            )
        else:
            assert not has_search_indicator, (
                f"Query '{query}' is expected NOT to trigger search but contains "
                f"search indicators. Remove indicator or adjust expectation."
            )


class TestReferencesParser:
    """Tests for the references/citations parser."""

    def test_references_parser_generates_valid_json(self):
        """Test that the references parser outputs valid JSON."""
        import jinja2
        
        template_str = """{%- set seen_urls = [] -%}
{%- set unique_items = [] -%}
{%- for parser_input in [handle_parser_input_0, handle_parser_input_1] -%}
  {%- if parser_input.data is defined -%}
  {%- for x in parser_input.data -%}
    {%- if x.url not in seen_urls -%}
      {%- set _ = seen_urls.append(x.url) -%}
      {%- set _ = unique_items.append(x) -%}
    {%- endif -%}
  {%- endfor -%}
  {%- endif -%}
{%- endfor -%}
{"results_ref": [{% for x in unique_items %}{"title": {{ x.title | tojson }},"snippet": {{ x.description | default('') | tojson }},"link": "{{ x.url }}","position": {{ loop.index0 }}}{% if not loop.last %},{% endif %}{% endfor %}]}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        input_0 = {
            "data": [
                {"title": "Result 1", "url": "https://example.com/1", "description": "Desc 1"},
                {"title": "Result 2", "url": "https://example.com/2", "description": "Desc 2"}
            ]
        }
        input_1 = {
            "data": [
                {"title": "Result 3", "url": "https://example.com/3", "description": "Desc 3"}
            ]
        }
        
        result = template.render(
            handle_parser_input_0=input_0,
            handle_parser_input_1=input_1
        )
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert "results_ref" in parsed
        assert len(parsed["results_ref"]) == 3

    def test_references_parser_deduplicates_urls(self):
        """Test that duplicate URLs are removed."""
        import jinja2
        
        template_str = """{%- set seen_urls = [] -%}
{%- set unique_items = [] -%}
{%- for parser_input in [handle_parser_input_0, handle_parser_input_1] -%}
  {%- if parser_input.data is defined -%}
  {%- for x in parser_input.data -%}
    {%- if x.url not in seen_urls -%}
      {%- set _ = seen_urls.append(x.url) -%}
      {%- set _ = unique_items.append(x) -%}
    {%- endif -%}
  {%- endfor -%}
  {%- endif -%}
{%- endfor -%}
{{ unique_items | length }}"""
        
        env = jinja2.Environment()
        template = env.from_string(template_str)
        
        # Same URL in both inputs
        input_0 = {
            "data": [
                {"title": "Result 1", "url": "https://example.com/same"},
            ]
        }
        input_1 = {
            "data": [
                {"title": "Result 1 Duplicate", "url": "https://example.com/same"},
            ]
        }
        
        result = template.render(
            handle_parser_input_0=input_0,
            handle_parser_input_1=input_1
        )
        
        # Should only have 1 unique item
        assert result.strip() == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
