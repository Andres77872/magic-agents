"""
Approval tests for JSON example file integrity.

Tests that scan all JSON files under examples/ to ensure:
- No hardcoded API key patterns exist (sk-proj-, jina_, etc.)
- Files that should use env placeholders have {{env. patterns
- All JSON files are valid and parseable
"""
import json
import os
import re
from pathlib import Path

import pytest


# Patterns that indicate hardcoded API keys
HARDCODED_KEY_PATTERNS = [
    "sk-proj-",    # OpenAI project keys
    "sk-",         # OpenAI keys (general)
    "jina_",       # Jina API keys
    "sk-or-",      # OpenRouter keys
]
PROVIDER_KEY_PATTERNS = {
    "OPENAI_API_KEY": re.compile(r"(?:sk-proj-|sk-)[A-Za-z0-9_-]{16,}"),
    "OPENROUTER_API_KEY": re.compile(r"sk-or-[A-Za-z0-9_-]{16,}"),
    "SERPER_API_KEY": re.compile(r"[A-Fa-f0-9]{32,}"),
    "JINA_API_KEY": re.compile(r"jina_[A-Za-z0-9_-]{16,}"),
}

# The expected env placeholder pattern used by build()
ENV_PLACEHOLDER_PATTERN = "{{env."


def _find_example_json_files() -> list[Path]:
    """Find all JSON files under examples/."""
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"
    if not examples_dir.exists():
        return []
    return sorted(examples_dir.rglob("*.json"))


def _project_root() -> Path:
    return Path(__file__).parent.parent


def _nodes_of(data: dict) -> list[dict]:
    if isinstance(data.get("content"), dict):
        return data["content"].get("nodes", [])
    return data.get("nodes", [])


def _edges_of(data: dict) -> list[dict]:
    if isinstance(data.get("content"), dict):
        return data["content"].get("edges", [])
    return data.get("edges", [])


def _tracked_env_fixture_files() -> list[Path]:
    root = _project_root()
    candidates = [root / ".env.test"]
    candidates.extend(root.glob(".env*.example"))
    candidates.extend(root.glob(".env*.template"))
    return sorted({path for path in candidates if path.exists()})


class TestNoHardcodedApiKeys:
    """Ensure no hardcoded API keys exist in example JSON files."""

    def test_no_hardcoded_api_keys_in_examples(self):
        """Scan all JSON files under examples/ for hardcoded key patterns."""
        json_files = _find_example_json_files()
        if not json_files:
            pytest.skip("No JSON files found in examples/")

        violations = []
        for json_file in json_files:
            content = json_file.read_text()
            for pattern in HARDCODED_KEY_PATTERNS:
                if pattern in content:
                    # Find which line(s) contain the pattern
                    for line_num, line in enumerate(content.splitlines(), 1):
                        if pattern in line:
                            violations.append(
                                f"{json_file.relative_to(Path(__file__).parent.parent)}:"
                                f"{line_num}: contains '{pattern}'"
                            )

        if violations:
            pytest.fail(
                f"Found hardcoded API key patterns in {len(violations)} location(s):\n"
                + "\n".join(violations)
            )

    def test_no_real_looking_provider_keys_in_env_fixtures(self):
        """Tracked env/test fixtures may name variables, but must not contain live-looking values."""
        env_files = _tracked_env_fixture_files()
        if not env_files:
            pytest.skip("No tracked env fixture files found")

        violations = []
        for env_file in env_files:
            for line in env_file.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                name, value = stripped.split("=", 1)
                name = name.strip()
                value = value.strip().strip('"').strip("'")
                pattern = PROVIDER_KEY_PATTERNS.get(name)
                if pattern and pattern.search(value):
                    violations.append(f"{env_file.relative_to(_project_root())}: {name} contains a real-looking credential")

        assert not violations, "Env credential hygiene violations (values redacted):\n" + "\n".join(violations)

    def test_all_example_jsons_are_valid_json(self):
        """Every JSON file under examples/ must be valid JSON."""
        json_files = _find_example_json_files()
        if not json_files:
            pytest.skip("No JSON files found in examples/")

        invalid_files = []
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                invalid_files.append(f"{json_file.name}: {e}")

        if invalid_files:
            pytest.fail(
                f"Found {len(invalid_files)} invalid JSON file(s):\n"
                + "\n".join(invalid_files)
            )


class TestEnvPlaceholders:
    """Ensure env placeholder patterns are used where expected."""

    def test_json_files_with_client_nodes_use_env_placeholders(self):
        """JSON files that have client nodes should use {{env.OPENAI_API_KEY}}."""
        json_files = _find_example_json_files()
        if not json_files:
            pytest.skip("No JSON files found in examples/")

        files_with_client_no_placeholder = []
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue  # Already caught by the valid JSON test

            nodes = _nodes_of(data)
            has_client = any(n.get("type") == "client" for n in nodes)
            has_placeholder = ENV_PLACEHOLDER_PATTERN in json_file.read_text()

            if has_client and not has_placeholder:
                files_with_client_no_placeholder.append(
                    json_file.relative_to(Path(__file__).parent.parent)
                )

        if files_with_client_no_placeholder:
            pytest.fail(
                f"JSON files with client nodes but no env placeholders:\n"
                + "\n".join(str(f) for f in files_with_client_no_placeholder)
            )


class TestDeepResearchExample:
    """Validate deep_research.json example structure and buildability."""

    DEEP_RESEARCH_PATH = Path(__file__).parent.parent / "examples" / "json" / "deep_research.json"

    def test_deep_research_exists(self):
        """deep_research.json must exist in canonical location."""
        assert self.DEEP_RESEARCH_PATH.exists(), "examples/json/deep_research.json not found"

    def test_deep_research_user_input_has_data_field(self):
        """user_input node must have explicit data field per JSON_CONTRACT.md."""
        with open(self.DEEP_RESEARCH_PATH) as f:
            data = json.load(f)
        
        user_input_nodes = [n for n in _nodes_of(data) if n["type"] == "user_input"]
        assert len(user_input_nodes) == 1, "Expected exactly one user_input node"
        
        user_input = user_input_nodes[0]
        assert "data" in user_input, "user_input node missing 'data' field (required per JSON_CONTRACT.md)"
        assert user_input["data"] == {}, "user_input data should be empty {} for basic input"

    def test_deep_research_has_tool_nodes(self):
        """deep_research.json must have tool-providing nodes for agent loop."""
        with open(self.DEEP_RESEARCH_PATH) as f:
            data = json.load(f)
        
        node_types = {n["type"] for n in _nodes_of(data)}
        
        assert "fetch" in node_types, "Expected fetch node(s) for search/scrape tools"
        assert "llm" in node_types, "Expected llm node(s) for agent execution"

    def test_deep_research_fetch_nodes_have_tool_mode(self):
        """fetch nodes must have tool_mode=true for agent loop integration."""
        with open(self.DEEP_RESEARCH_PATH) as f:
            data = json.load(f)
        
        fetch_nodes = [n for n in _nodes_of(data) if n["type"] == "fetch"]
        assert len(fetch_nodes) >= 2, "Expected at least 2 fetch nodes (search + scrape)"
        
        for fetch_node in fetch_nodes:
            node_data = fetch_node.get("data", {})
            assert node_data.get("tool_mode") == True, f"fetch node '{fetch_node['id']}' must have tool_mode=true"
            assert "tool_name" in node_data, f"fetch node '{fetch_node['id']}' must have tool_name"

    def test_deep_research_tool_edges_connect_to_llm(self):
        """Tool nodes must connect to LLM via handle-tool-definition-* handles.
        
        sourceHandle is optional for tool edges - backend normalizes via _assign_tool_handles().
        targetHandle must use handle-tool-definition-* pattern for LLM tool input slots.
        """
        with open(self.DEEP_RESEARCH_PATH) as f:
            data = json.load(f)
        
        nodes = _nodes_of(data)
        edges = _edges_of(data)
        tool_nodes = [n["id"] for n in nodes if n["type"] in ("fetch", "python_exec")]
        llm_nodes = [n["id"] for n in nodes if n["type"] == "llm"]
        
        tool_to_llm_edges = [
            e for e in edges
            if e["source"] in tool_nodes and e["target"] in llm_nodes
        ]
        
        assert len(tool_to_llm_edges) >= 2, "Expected search and scrape tool edges"
        
        for edge in tool_to_llm_edges:
            # sourceHandle is optional - backend normalizes it via _assign_tool_handles()
            # If present, it's validated during build; if absent, backend assigns correct value
            assert edge["targetHandle"].startswith("handle-tool-definition-"), f"Edge {edge['id']} must use handle-tool-definition-* target"

    def test_deep_research_buildable(self):
        """deep_research.json must successfully build with magic_agents.agt_flow.build()."""
        from magic_agents.agt_flow import build
        
        with open(self.DEEP_RESEARCH_PATH) as f:
            data = json.load(f)
        
        os.environ["OPENAI_API_KEY"] = "test-key-placeholder"
        os.environ["SERPER_API_KEY"] = "test-key-placeholder"
        
        graph = build(data, message="test query")
        
        assert graph is not None, "build() returned None"
        assert len(graph.nodes) > 0, "build() produced no nodes"
        assert len(graph.edges) > 0, "build() produced no edges"


class TestGraphFormatHelpers:
    def test_nodes_and_edges_support_flat_graphs(self):
        data = {"nodes": [{"id": "n1"}], "edges": [{"id": "e1"}]}
        assert _nodes_of(data) == [{"id": "n1"}]
        assert _edges_of(data) == [{"id": "e1"}]

    def test_nodes_and_edges_support_nested_graphs(self):
        data = {"type": "graph", "content": {"nodes": [{"id": "n1"}], "edges": [{"id": "e1"}]}}
        assert _nodes_of(data) == [{"id": "n1"}]
        assert _edges_of(data) == [{"id": "e1"}]
