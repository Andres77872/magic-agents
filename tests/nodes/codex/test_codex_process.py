"""
Unit tests for NodeCodex.process() standalone behavior.

Note: magic_telemetry wraps process() with start/end telemetry events,
so iterating process() yields 3 raw events per actual yield. Tests must
filter by type == node.OUTPUT_HANDLE to find content yields.

Each content yield has structure:
    {'type': '<handle>', 'content': {'node': 'NodeCodex', 'content': '<actual_value>'}}
"""
import asyncio

from magic_agents.models.factory.Nodes.CodexNodeModel import CodexNodeModel
from magic_agents.node_system.NodeCodex import NodeCodex
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


def make_codex_node(entries=None, handles=None, **kwargs):
    """Create a NodeCodex with configurable entries, handles, and kwargs."""
    data = CodexNodeModel(codex_entries=entries or [])
    kwargs.setdefault("node_id", "codex-test")
    kwargs.setdefault("node_type", "codex")
    node = NodeCodex(data=data, handles=handles, **kwargs)
    return node


def _run_process(node, input_value=None):
    """Run node.process(chat_log) with injected input, return all raw results."""
    if input_value is not None:
        node.inputs[node.INPUT_HANDLE] = input_value
    chat_log = ModelAgentRunLog()

    async def _test():
        results = []
        async for item in node.process(chat_log):
            results.append(item)
        return results

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_test())
    finally:
        loop.close()


def _content_results(results, node):
    """Filter results to only actual content yields (skip telemetry events)."""
    return [r for r in results if r.get("type") == node.OUTPUT_HANDLE]


def _single_content(results, node):
    """Get the single content result or raise if not exactly one.

    Returns the FULL content dict: {'node': 'NodeCodex', 'content': '<value>'}
    """
    items = _content_results(results, node)
    if len(items) != 1:
        raise AssertionError(
            f"Expected exactly 1 content result, got {len(items)}: {items}"
        )
    return items[0]


# --- helpers to extract the actual string value from the prep() wrapper ---
def _content_value(item):
    """Extract actual string content from yield_static/prep wrapper dict."""
    return item["content"]["content"]


def _content_value_str(item):
    """Extract content as str (safe for 'in' checks against nested dicts)."""
    return str(item["content"]["content"])


# ============================================================================
# Test 1: Empty entries passthrough
# ============================================================================
class TestNodeCodexProcessEmptyEntries:
    """Empty codex_entries — pure passthrough."""

    def test_empty_entries_passthrough(self):
        """Node with codex_entries=[], input 'hello world' → unchanged output."""
        node = make_codex_node(entries=[])
        results = _run_process(node, "hello world")
        item = _single_content(results, node)
        assert _content_value(item) == "hello world"
        assert "<codex_content>" not in _content_value_str(item)


# ============================================================================
# Test 2: No trigger match passthrough
# ============================================================================
class TestNodeCodexProcessNoMatch:
    """No trigger match — passthrough unchanged."""

    def test_no_trigger_match_passthrough(self):
        """Entry triggers=['pricing'], input 'hello world' → passthrough."""
        node = make_codex_node(entries=[{"triggers": ["pricing"], "content": "Pricing info"}])
        results = _run_process(node, "hello world")
        item = _single_content(results, node)
        assert _content_value(item) == "hello world"
        assert "<codex_content>" not in _content_value_str(item)

    def test_word_boundary_no_false_positive_he(self):
        """Trigger 'he', input 'the system' → passthrough (no substring match)."""
        node = make_codex_node(entries=[{"triggers": ["he"], "content": "Should not match"}])
        results = _run_process(node, "the system is up")
        item = _single_content(results, node)
        assert _content_value(item) == "the system is up"
        assert "<codex_content>" not in _content_value_str(item)

    def test_word_boundary_no_false_positive_email_ai(self):
        """Trigger 'ai', input 'EMAIL' → passthrough (substring/no boundary)."""
        node = make_codex_node(entries=[{"triggers": ["ai"], "content": "AI info"}])
        results = _run_process(node, "The EMAIL system is down")
        item = _single_content(results, node)
        assert _content_value(item) == "The EMAIL system is down"
        assert "<codex_content>" not in _content_value_str(item)

    def test_word_boundary_no_false_positive_plain_train_said(self):
        """Trigger 'ai', input with ai-inside-words → passthrough."""
        node = make_codex_node(entries=[{"triggers": ["ai"], "content": "AI info"}])
        results = _run_process(node, "The plain train said")
        item = _single_content(results, node)
        assert _content_value(item) == "The plain train said"
        assert "<codex_content>" not in _content_value_str(item)

    def test_word_boundary_hyphen_matches(self):
        """Trigger 'ai' matches hyphen-delimited 'state-of-the-ai-system' (Python \\b)."""
        node = make_codex_node(entries=[{"triggers": ["ai"], "content": "AI info"}])
        results = _run_process(node, "state-of-the-ai-system")
        item = _single_content(results, node)
        assert "<codex_content>AI info</codex_content>" in _content_value_str(item)

    def test_word_boundary_helping_no_match(self):
        """Trigger 'help', input 'helping' → passthrough (no boundary after help)."""
        node = make_codex_node(entries=[{"triggers": ["help"], "content": "Help docs"}])
        results = _run_process(node, "helping others")
        item = _single_content(results, node)
        assert _content_value(item) == "helping others"
        assert "<codex_content>" not in _content_value_str(item)


# ============================================================================
# Test 3 & 5: Single trigger match / Case-insensitive matching
# ============================================================================
class TestNodeCodexProcessSingleMatch:
    """Single trigger match — codex content prepended."""

    def test_single_trigger_match(self):
        """Entry triggers=['help'], content='Help resources', input 'I need help' → prepended."""
        node = make_codex_node(entries=[{"triggers": ["help"], "content": "Help resources"}])
        assert isinstance(node, NodeCodex)
        results = _run_process(node, "I need help")
        item = _single_content(results, node)
        content = _content_value(item)
        assert content.startswith("<codex_content>Help resources</codex_content>\n")
        assert content.endswith("I need help")

    def test_case_insensitive_match(self):
        """Trigger 'HELP', input 'I need help' → match (case-insensitive)."""
        node = make_codex_node(entries=[{"triggers": ["HELP"], "content": "Help resources"}])
        results = _run_process(node, "I need help")
        item = _single_content(results, node)
        assert "<codex_content>Help resources</codex_content>" in _content_value_str(item)

    def test_match_punctuation_boundary(self):
        """Trigger 'help', input 'help!' → match (boundary before !)."""
        node = make_codex_node(entries=[{"triggers": ["help"], "content": "Help docs"}])
        results = _run_process(node, "help!")
        item = _single_content(results, node)
        assert "<codex_content>Help docs</codex_content>" in _content_value_str(item)

    def test_match_react_js(self):
        """Trigger 'react', input 'React.js' → match (. is word-char boundary)."""
        node = make_codex_node(entries=[{"triggers": ["react"], "content": "React info"}])
        results = _run_process(node, "I use React.js")
        item = _single_content(results, node)
        assert "<codex_content>React info</codex_content>" in _content_value_str(item)


# ============================================================================
# Test 4: Multiple trigger matches / order
# ============================================================================
class TestNodeCodexProcessMultipleMatches:
    """Multiple entries matching — all prepended in order."""

    def test_two_entries_both_match(self):
        """Two entries with different triggers, input contains both → both prepended in order."""
        node = make_codex_node(entries=[
            {"triggers": ["help"], "content": "Help"},
            {"triggers": ["pricing"], "content": "Pricing"},
        ])
        results = _run_process(node, "I need help and pricing")
        item = _single_content(results, node)
        content = _content_value(item)
        # Both blocks present, in entry order
        assert "<codex_content>Help</codex_content>" in content
        assert "<codex_content>Pricing</codex_content>" in content
        # First block appears before second
        help_pos = content.index("<codex_content>Help</codex_content>")
        pricing_pos = content.index("<codex_content>Pricing</codex_content>")
        assert help_pos < pricing_pos, "Entries must appear in array order"
        # Original message preserved at the end
        assert content.endswith("I need help and pricing")

    def test_trigger_dedup_single_entry(self):
        """Entry triggers=['help', 'ayuda'], input contains both → entry included once."""
        node = make_codex_node(entries=[
            {"triggers": ["help", "ayuda"], "content": "Help resources"},
        ])
        results = _run_process(node, "I need help or ayuda")
        item = _single_content(results, node)
        content = _content_value(item)
        # Entry appears once
        assert content.count("<codex_content>Help resources</codex_content>") == 1


# ============================================================================
# Test 8 & 9: None/empty input → empty output
# ============================================================================
class TestNodeCodexProcessEdgeInput:
    """Edge cases: None, empty string input."""

    def test_none_input_yields_empty(self):
        """get_input() returns None → yields empty string."""
        node = make_codex_node(entries=[{"triggers": ["help"], "content": "Help"}])
        # Do NOT set any input → get_input returns None
        results = _run_process(node, input_value=None)
        item = _single_content(results, node)
        assert _content_value(item) == ""

    def test_empty_string_input_yields_empty(self):
        """Input '' → yields empty string."""
        node = make_codex_node(entries=[{"triggers": ["help"], "content": "Help"}])
        results = _run_process(node, "")
        item = _single_content(results, node)
        assert _content_value(item) == ""


# ============================================================================
# Test 10 & 11: No XML escaping applied
# ============================================================================
class TestNodeCodexProcessNoEscaping:
    """No XML escaping — content and user message pass verbatim."""

    def test_no_escaping_content_contains_delimiter(self):
        """Codex content contains '</codex_content>' → verbatim in output."""
        node = make_codex_node(entries=[
            {"triggers": ["test"], "content": "Some </codex_content> text"},
        ])
        results = _run_process(node, "test me")
        item = _single_content(results, node)
        content = _content_value(item)
        # Content appears verbatim between wrapper tags
        assert "<codex_content>Some </codex_content> text</codex_content>" in content

    def test_user_message_contains_delimiter(self):
        """User message contains '</codex_content>' → verbatim preserved."""
        node = make_codex_node(entries=[
            {"triggers": ["hello"], "content": "Greeting"},
        ])
        results = _run_process(node, "hello </codex_content> world")
        item = _single_content(results, node)
        content = _content_value(item)
        # User message appears at end without escaping
        assert content.rstrip().endswith("hello </codex_content> world")


# ============================================================================
# Test 13: Custom handles
# ============================================================================
class TestNodeCodexProcessCustomHandles:
    """Custom handle names are respected."""

    def test_custom_handles(self):
        """Node with custom handles → uses custom input/output handles."""
        node = make_codex_node(
            entries=[{"triggers": ["help"], "content": "Help"}],
            handles={"input": "custom_in", "output": "custom_out"},
        )
        assert node.INPUT_HANDLE == "custom_in"
        assert node.OUTPUT_HANDLE == "custom_out"
        node.inputs["custom_in"] = "I need help"
        chat_log = ModelAgentRunLog()

        async def _test():
            results = []
            async for item in node.process(chat_log):
                results.append(item)
            return results

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_test())
        finally:
            loop.close()

        content_items = [r for r in results if r.get("type") == "custom_out"]
        assert len(content_items) == 1
        assert "<codex_content>Help</codex_content>" in _content_value_str(content_items[0])

    def test_custom_handles_partial(self):
        """Only output handle specified, input uses default."""
        node = make_codex_node(
            entries=[{"triggers": ["help"], "content": "Help"}],
            handles={"output": "custom_out"},
        )
        assert node.INPUT_HANDLE == "handle_codex_input"
        assert node.OUTPUT_HANDLE == "custom_out"

# ============================================================================
# Test 14 & 15: _capture_internal_state
# ============================================================================
class TestNodeCodexInternalState:
    """_capture_internal_state() return value."""

    def test_no_match_state(self):
        """No trigger match → matched_triggers empty, prepended_count 0."""
        node = make_codex_node(entries=[{"triggers": ["pricing"], "content": "Pricing"}])
        results = _run_process(node, "hello world")
        # Consume generator first
        _single_content(results, node)
        state = node._capture_internal_state()
        assert state["matched_triggers"] == []
        assert state["prepended_count"] == 0
        assert state["message_length"] == len("hello world")
        assert state["codex_entry_count"] == 1
        assert state["trigger_count"] == 1

    def test_with_match_state(self):
        """Two entries match → prepended_count 2, matched_triggers populated."""
        node = make_codex_node(entries=[
            {"triggers": ["help"], "content": "Help"},
            {"triggers": ["pricing"], "content": "Pricing"},
        ])
        results = _run_process(node, "I need help and pricing")
        _single_content(results, node)
        state = node._capture_internal_state()
        assert state["prepended_count"] == 2
        assert len(state["matched_triggers"]) == 2
        assert state["message_length"] == len("I need help and pricing")

    def test_none_input_state(self):
        """None input → matched_triggers empty, prepended_count 0, message_length 0."""
        node = make_codex_node(entries=[{"triggers": ["help"], "content": "Help"}])
        results = _run_process(node, input_value=None)
        _single_content(results, node)
        state = node._capture_internal_state()
        assert state["matched_triggers"] == []
        assert state["prepended_count"] == 0
        assert state["message_length"] == 0

    def test_static_config_fields(self):
        """Static config fields always available regardless of execution."""
        node = make_codex_node(entries=[
            {"triggers": ["help", "ayuda"], "content": "Help"},
            {"triggers": ["pricing"], "content": "Pricing"},
        ])
        state = node._capture_internal_state()
        assert state["codex_entry_count"] == 2
        assert state["trigger_count"] == 3
        assert state["input_handle"] == "handle_codex_input"
        assert state["output_handle"] == "handle_user_message"


# ============================================================================
# Additional edge patterns from spec/design
# ============================================================================
class TestNodeCodexProcessEdgePatterns:
    """Additional edge patterns from spec/design."""

    def test_multiple_entries_first_only_matches(self):
        """First entry matches, second doesn't → only first is prepended."""
        node = make_codex_node(entries=[
            {"triggers": ["help"], "content": "Help"},
            {"triggers": ["pricing"], "content": "Pricing"},
        ])
        results = _run_process(node, "help me")
        item = _single_content(results, node)
        content = _content_value(item)
        assert "<codex_content>Help</codex_content>" in content
        assert "<codex_content>Pricing</codex_content>" not in content

    def test_second_entry_only_matches(self):
        """Second entry matches, first doesn't → only second is prepended."""
        node = make_codex_node(entries=[
            {"triggers": ["help"], "content": "Help"},
            {"triggers": ["pricing"], "content": "Pricing"},
        ])
        results = _run_process(node, "tell me about pricing")
        item = _single_content(results, node)
        content = _content_value(item)
        assert "<codex_content>Help</codex_content>" not in content
        assert "<codex_content>Pricing</codex_content>" in content

