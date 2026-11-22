"""Contract checks for every tracked graph fixture in ``tests/fixtures/graphs``.

These graphs are linked from the docs as complete authored examples, so they
must stay credential-free and build without error diagnostics.
"""

import json
import re

import pytest

from magic_agents.agt_flow import build
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from tests.support import FIXTURE_GRAPHS, load_fixture_graph

FIXTURES = sorted(path.name for path in FIXTURE_GRAPHS.glob("*.json"))

# Real provider credentials have recognizable shapes; placeholders such as
# ``{{env.JINA_API_KEY}}`` are the only allowed way to reference a secret.
CREDENTIAL_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-or-v1-[0-9a-f]{20,}"),
    re.compile(r"\bjina_[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+(?!\{\{env\.)[A-Za-z0-9_-]{16,}"),
]


ENV_PLACEHOLDER = re.compile(r"\{\{\s*env\.[A-Z0-9_]+\s*\}\}")


def _content(graph):
    return graph.get("content", graph)


def test_fixture_set_is_the_documented_one():
    assert FIXTURES == ["browsing.json", "browsing_provider_fallback.json", "deep_research.json", "image.json"]


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_contains_no_embedded_credentials(name):
    text = (FIXTURE_GRAPHS / name).read_text(encoding="utf-8")
    graph = json.loads(text)

    assert [pattern.pattern for pattern in CREDENTIAL_PATTERNS if pattern.search(text)] == []
    for node in _content(graph)["nodes"]:
        if node["type"] == "client":
            api_key = (node.get("data", {}).get("api_info") or {}).get("api_key")
            assert api_key is None or ENV_PLACEHOLDER.fullmatch(api_key), node["id"]


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_builds_with_clean_contract_and_preserved_edge_identity(name):
    authored = _content(load_fixture_graph(name))
    graph = build(load_fixture_graph(name), message="hello", load_chat=None)

    assert graph.contract_report.mode == "warn"
    assert graph.get_contract_errors() == []
    assert [e for e in (graph._validation_errors or []) if e.get("severity") != "warning"] == []
    built_ids = [edge.id for edge in graph.edges]
    assert len(built_ids) == len(set(built_ids))
    assert {edge["id"] for edge in authored["edges"]} <= set(built_ids)

    dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)
    control_edges = set(dispatcher.invocation_control.connections)
    for node_id in graph.nodes:
        # Lifecycle-Hook control connections (child calls) are invoked on
        # demand, never awaited; edge-hook triggers are extra virtual inputs.
        incoming = {edge.id for edge in graph.edges if edge.target == node_id} - control_edges
        expected = dispatcher.get_tracker(node_id).expected_edges
        assert incoming <= expected, node_id
        assert all(edge_id.startswith("__edge_hook__:") for edge_id in expected - incoming), node_id


@pytest.mark.parametrize("name", FIXTURES)
def test_off_mode_skips_contract_diagnostics(name):
    graph_def = load_fixture_graph(name)
    graph_def["contract_config"] = {"mode": "off"}

    graph = build(graph_def, message="hello", load_chat=None)

    assert graph.contract_report.mode == "off"
    assert graph.contract_report.diagnostics == []


def test_deep_research_exposes_its_fetch_nodes_as_llm_tools():
    authored = _content(load_fixture_graph("deep_research.json"))
    fetch_nodes = {n["id"]: n["data"] for n in authored["nodes"] if n["type"] == "fetch"}
    llm_ids = {n["id"] for n in authored["nodes"] if n["type"] == "llm"}

    graph = build(load_fixture_graph("deep_research.json"), message="research", load_chat=None)

    assert len(fetch_nodes) >= 2
    assert all(data.get("tool_mode") is True and data.get("tool_name") for data in fetch_nodes.values())
    tool_edges = [e for e in graph.edges if e.source in fetch_nodes and e.target in llm_ids]
    assert {e.source for e in tool_edges} == set(fetch_nodes)
    assert all(e.targetHandle.startswith("handle-tool-definition-") for e in tool_edges)
