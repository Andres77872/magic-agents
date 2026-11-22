"""Unit tests for STM windowing pure functions (Batch D, tasks 5.1-5.3).

Tests cover:
  - 5.1: apply_windowing() — core truncation logic
  - 5.2: estimate_tokens() — hardcoded GPT-5 token estimation
  - 5.3: tag_tool_chains() / find_safe_tail_cut() — tool-chain helpers

All tests are pure unit tests against module-level functions in
``magic_agents.node_system.utils``.  No NodeChat/NodeLLM fixtures required.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from magic_agents.node_system.utils import (
    TOKENS_PER_MESSAGE,
    TOKENS_PER_NAME,
    ASSISTANT_PRIME_TOKENS,
    apply_windowing,
    estimate_tokens,
    find_safe_tail_cut,
    tag_tool_chains,
)

# ===================================================================
# Helpers
# ===================================================================


def _msg(role: str, content: str = "", **kw) -> dict:
    """Build a minimal message dict with optional extra keys."""
    d: dict = {"role": role, "content": content}
    d.update(kw)
    return d


def _user(content: str = "", **kw):
    return _msg("user", content, **kw)


def _assistant(content: str = "", **kw):
    return _msg("assistant", content, **kw)


def _system(content: str = "", **kw):
    return _msg("system", content, **kw)


def _assistant_with_tool_calls(tool_calls: list | None = None, **kw):
    return _msg("assistant", "", tool_calls=tool_calls or [{"id": "1"}], **kw)


def _tool(content: str = "result", **kw):
    return _msg("tool", content, **kw)


import magic_agents.node_system.utils as windowing_utils

# Captured before the autouse fixture below swaps it for an offline encoder.
_REAL_GET_ENCODING = windowing_utils._get_encoding


class LocalEncoding:
    def encode(self, value):
        return str(value).split() if value else []


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "magic_agents.node_system.utils._get_encoding",
        lambda: LocalEncoding(),
    )


# ===================================================================
# 5.1 — apply_windowing() core tests
# ===================================================================


class TestApplyWindowing:
    """Core ``apply_windowing()`` truncation scenarios."""

    # ── Identity / passthrough ──────────────────────────────────────

    def test_passthrough_when_no_limits(self):
        """W8: No limits → messages unchanged (identity)."""
        msgs = [_user("a"), _assistant("b"), _user("c")]
        result = apply_windowing(msgs)
        assert result == msgs

    def test_empty_messages(self):
        """W10: Empty list → empty list."""
        assert apply_windowing([]) == []
        assert apply_windowing([], max_messages=10) == []
        assert apply_windowing([], max_input_tokens=1000) == []

    def test_all_system_messages(self):
        """W2: Only system messages → returned as-is."""
        msgs = [_system("a"), _system("b"), _system("c")]
        result = apply_windowing(msgs, max_messages=1)
        assert result == msgs

    def test_system_only_messages(self):
        """W2: System msgs only, no non-system → returned as-is."""
        msgs = [_system("a"), _system("b"), _system("c")]
        result = apply_windowing(msgs, max_messages=5)
        assert result == msgs

    def test_single_user_message(self):
        """W3: Single user message → returned as-is."""
        msg = _user("hello")
        result = apply_windowing([msg], max_messages=1)
        assert result == [msg]

    # ── max_messages tail truncation ────────────────────────────────

    def test_max_messages_tail_truncation(self):
        """W4: 20 non-system → max_messages=10, expect 10."""
        msgs = [_user(f"msg{i}") for i in range(20)]
        result = apply_windowing(msgs, max_messages=10)
        # 10 non-system messages: last 9 from truncatable + last user
        assert len(result) == 10
        # The last message should be the original last user message
        assert result[-1]["content"] == "msg19"

    def test_max_messages_at_boundary(self):
        """W4: 10 non-system → max_messages=10 → no-op."""
        msgs = [_user(f"msg{i}") for i in range(10)]
        result = apply_windowing(msgs, max_messages=10)
        assert len(result) == 10
        assert result == msgs  # exact match (new list, same content)

    def test_max_messages_exact_one(self):
        """W4: max_messages=1 with 5 non-system → last user only."""
        msgs = [_user(f"msg{i}") for i in range(5)]
        result = apply_windowing(msgs, max_messages=1)
        assert len(result) == 1
        assert result[0]["content"] == "msg4"

    # ── System message preservation ─────────────────────────────────

    def test_system_messages_preserved(self):
        """W2: 3 system + 50 non-system → max_messages=10 → 3 sys + 10 non-sys."""
        msgs = (
            [_system("sys1"), _system("sys2"), _system("sys3")]
            + [_user(f"msg{i}") for i in range(50)]
        )
        result = apply_windowing(msgs, max_messages=10)
        # 3 system messages should all be present
        sys_count = sum(1 for m in result if m["role"] == "system")
        assert sys_count == 3
        # At most 10 non-system messages
        non_sys_count = sum(1 for m in result if m["role"] != "system")
        assert non_sys_count == 10

    # ── Last user message preservation ──────────────────────────────

    def test_last_user_preserved(self):
        """W3: 20 non-system → max_messages=5 → last user preserved."""
        msgs = [_user(f"msg{i}") for i in range(20)]
        result = apply_windowing(msgs, max_messages=5)
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "msg19"
        # 5 non-system messages total
        non_sys_count = sum(1 for m in result if m["role"] != "system")
        assert non_sys_count == 5

    def test_last_user_preserved_with_even_count(self):
        """W3: Len(exactly matches boundary), last user correct."""
        msgs = [_user(f"msg{i}") for i in range(5)]
        result = apply_windowing(msgs, max_messages=2)
        # preserves last user (msg4) + 1 from truncatable
        assert len(result) == 2
        assert result[-1]["content"] == "msg4"

    # ── Tool-call chain atomicity ───────────────────────────────────

    def test_tool_chain_preserved_at_boundary(self):
        """W6: Chain at boundary kept atomically.

        Messages: [user(1), assistant(tool_calls), tool, tool, user(2)]
        max_messages=3 → 4 kept (last user + entire chain because chain
        straddles the cut boundary).
        """
        msgs = [
            _user("user1"),
            _assistant_with_tool_calls(),
            _tool("result1"),
            _tool("result2"),
            _user("user2"),
        ]
        result = apply_windowing(msgs, max_messages=3)
        # All 4 non-system messages: chain (3 msgs) + last user (1 msg)
        assert len(result) == 4
        assert result[-1]["content"] == "user2"
        # The chain is intact
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[2]["role"] == "tool"

    def test_tool_chain_dropped_entirely(self):
        """W6: Entire chain dropped atomically.

        Messages: [assistant(tool_calls), tool, user(1), user(2)]
        max_messages=2 → [user(1), user(2)] (chain entirely dropped because
        keeping it would exceed budget).
        """
        msgs = [
            _assistant_with_tool_calls(),
            _tool("result"),
            _user("user1"),
            _user("user2"),
        ]
        result = apply_windowing(msgs, max_messages=2)
        assert len(result) == 2
        assert result[0]["content"] == "user1"
        assert result[1]["content"] == "user2"

    def test_tool_chain_fits_entirely(self):
        """W6: Chain fits entirely within budget → no splitting."""
        msgs = [
            _user("user1"),
            _assistant_with_tool_calls(),
            _tool("result1"),
            _user("user2"),
        ]
        result = apply_windowing(msgs, max_messages=4)
        # All messages kept — within limit
        assert len(result) == 4
        # Chain is intact
        assert result[1]["role"] == "assistant" and "tool_calls" in result[1]
        assert result[2]["role"] == "tool"

    def test_no_tool_calls(self):
        """W6: No tool_calls in any message → standard truncation."""
        msgs = [_user(f"msg{i}") for i in range(10)]
        result = apply_windowing(msgs, max_messages=5)
        assert len(result) == 5
        assert result[-1]["content"] == "msg9"

    def test_orphan_tool_message(self):
        """W6: Tool without preceding assistant → droppable individually.

        Orphan tool messages get chain_group_id=None and are treated
        as standalone droppable messages.
        """
        msgs = [
            _tool("orphan1"),
            _tool("orphan2"),
            _user("user1"),
            _user("user2"),
        ]
        result = apply_windowing(msgs, max_messages=2)
        # 2 messages: last user preserved + one orphan dropped
        assert len(result) == 2
        assert result[0]["content"] == "user1"
        assert result[1]["content"] == "user2"

    # ── Input list immutability ─────────────────────────────────────

    def test_input_list_not_mutated(self):
        """W9: Original list unchanged after call."""
        original = [_user("a"), _user("b"), _user("c")]
        copy = list(original)
        apply_windowing(original, max_messages=2)
        assert len(original) == len(copy)
        for orig, cpy in zip(original, copy):
            assert orig["role"] == cpy["role"]
            assert orig.get("content") == cpy.get("content")

    # ── Unknown keys preserved ──────────────────────────────────────

    def test_unknown_keys_preserved(self):
        """E2: Unknown message keys pass through windowing."""
        msgs = [
            {"role": "user", "content": "hi", "custom_key": "should_survive"},
            {"role": "assistant", "content": "hello", "extra_data": {"x": 1}},
        ]
        result = apply_windowing(msgs, max_messages=1)
        assert len(result) == 1
        assert "custom_key" in result[0] or "extra_data" in result[0]

    # ── token_budget strategy ───────────────────────────────────────

    def test_token_budget_within_limit(self):
        """W5: Messages under max_input_tokens → no truncation.

        Retained messages preserve their chronological order and internal
        chain bookkeeping is not exposed in provider messages.
        """
        msgs = [_user("hello"), _assistant("world")]
        result = apply_windowing(
            msgs, max_input_tokens=100_000, truncation_strategy="token_budget"
        )
        assert result == msgs
        assert [message["role"] for message in result] == ["user", "assistant"]

    def test_token_budget_truncation(self):
        """W5: Many messages exceeding max_input_tokens → oldest dropped.

        The last user is retained in its original position. The assistant
        "final" follows it because it is the newest retained message.
        """
        # Build enough messages to exceed a small budget
        long_text = " ".join(["word"] * 200)  # ~200 tokens each
        msgs = [_user(long_text) for _ in range(50)] + [_assistant("final")]
        result = apply_windowing(
            msgs, max_input_tokens=500, truncation_strategy="token_budget"
        )
        # Should have been truncated — fewer messages than original
        assert len(result) < len(msgs)
        # Keep the last user without moving it after the assistant response.
        assert result[-2]["role"] == "user"
        assert result[-1]["role"] == "assistant"
        # Truncation happened (not all 51 messages)
        assert len(result) < 10
        # The newest truncatable message (assistant "final") is kept
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert any("final" in m.get("content", "") for m in assistant_msgs)

    def test_tail_strategy_caps_non_system_messages_without_mutating_input(self):
        msgs = [{"role": "system", "content": "Owned instructions."}] + [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"Turn {index}"}
            for index in range(21)]
        before = deepcopy(msgs)
        result = apply_windowing(msgs, max_messages=5, max_input_tokens=100_000, truncation_strategy="tail")
        assert result == [msgs[0], *msgs[-5:]]
        assert msgs == before

    def test_token_budget_strategy_ignores_max_messages(self):
        """Documented contract: ``token_budget`` skips the message-count step."""
        msgs = [{"role": "system", "content": "Owned instructions."}] + [
            {"role": "user" if index % 2 == 0 else "assistant", "content": f"Turn {index}"}
            for index in range(21)]
        result = apply_windowing(msgs, max_messages=5, max_input_tokens=100_000,
                                 truncation_strategy="token_budget")
        assert result == msgs

    def test_token_budget_basic(self):
        """W7: Basic token budget with hardcoded GPT-5."""
        msgs = [_user("hello"), _assistant("world")]
        result = apply_windowing(msgs, max_messages=1)
        assert len(result) == 1

    # ── Edge: max_messages=0 treated as None ────────────────────────

    def test_max_messages_zero_treated_as_none(self):
        """E1: max_messages=0 treated as None (no message limit)."""
        msgs = [_user(f"msg{i}") for i in range(10)]
        result = apply_windowing(msgs, max_messages=0)
        assert len(result) == 10  # unchanged

    # ── Mixed role order ────────────────────────────────────────────

    def test_system_messages_position_preserved(self):
        """W2: System messages re-inserted at original relative positions.

        System msg at index 0 should remain at index 0 in output.
        """
        msgs = [
            _system("sys"),
            _user("a"),
            _user("b"),
            _user("c"),
        ]
        result = apply_windowing(msgs, max_messages=2)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "sys"
        assert len([m for m in result if m["role"] != "system"]) == 2

    # ── assistant-only truncatable (no user messages besides last) ──

    def test_assistant_messages_in_truncatable(self):
        """Assistant messages in truncatable area are handled."""
        msgs = [_assistant("a"), _assistant("b"), _user("final")]
        result = apply_windowing(msgs, max_messages=2)
        assert len(result) == 2
        assert result[-1]["content"] == "final"


# ===================================================================
# 5.2 — estimate_tokens() tests
# ===================================================================


class TestEstimateTokens:
    """``estimate_tokens()`` hardcoded GPT-5 token estimation."""

        # Without model kwarg, the function uses hardcoded GPT-5 internally.
        # If GPT-5 is unknown to tiktoken, it falls back to cl100k_base.
        # Either way, token estimation succeeds.

    def test_estimate_string_content(self):
        """T1: Exact token count for known string."""
        msgs = [_user("hello world")]
        tokens = estimate_tokens(msgs)
        assert tokens == TOKENS_PER_MESSAGE + 2  # 3 + 2 tokens for "hello world"

    def test_estimate_empty_content(self):
        """T1: Empty string content → only overhead."""
        msgs = [_user("")]
        tokens = estimate_tokens(msgs)
        # 3 overhead tokens + 0 content tokens = 3
        assert tokens == TOKENS_PER_MESSAGE

    @pytest.mark.parametrize("text_key", ["text", "content"])
    def test_estimate_multimodal_counts_text_parts(self, text_key):
        """Regression: ModelChat text parts use "text"; they were counted as 0 tokens."""
        image = {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        with_text = [{"role": "user", "content": [{"type": "text", text_key: "describe this"}, image]}]
        image_only = [{"role": "user", "content": [image]}]

        # LocalEncoding counts words, so the text part adds exactly 2 tokens.
        assert estimate_tokens(with_text) - estimate_tokens(image_only) == 2

    def test_estimate_multimodal_base64_image(self):
        """T1: Base64 data URI image estimate via len//4."""
        b64_payload = "A" * 400  # ~100 estimated tokens
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "content": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_payload}"},
                    },
                ],
            }
        ]
        tokens = estimate_tokens(msgs)
        # 3 overhead + ~3 text tokens + ~100 image tokens
        assert tokens > 100

    def test_estimate_name_tokens(self):
        """T1: name key adds TOKENS_PER_NAME."""
        msgs = [_user("hello", name="Andres")]
        tokens_with_name = estimate_tokens(msgs)
        tokens_without = estimate_tokens([_user("hello")])
        assert tokens_with_name == tokens_without + TOKENS_PER_NAME

    def test_estimate_assistant_prime_tokens(self):
        """T1: assistant role adds ASSISTANT_PRIME_TOKENS."""
        msgs_user = [_user("hello")]
        msgs_assistant = [_assistant("hello")]
        tokens_user = estimate_tokens(msgs_user)
        tokens_asst = estimate_tokens(msgs_assistant)
        assert tokens_asst == tokens_user + ASSISTANT_PRIME_TOKENS

    def test_estimate_multiple_messages(self):
        """T1: Multiple messages sum correctly."""
        msgs = [_user("a"), _assistant("b"), _user("c")]
        tokens = estimate_tokens(msgs)
        # Each message has TOKENS_PER_MESSAGE (3)
        # user "a" = 1 token, assistant "b" = 1 token + 3 prime, user "c" = 1 token
        assert tokens == (3 * TOKENS_PER_MESSAGE) + 1 + (1 + ASSISTANT_PRIME_TOKENS) + 1

# ===================================================================
# 5.3 — tag_tool_chains() / find_safe_tail_cut() tests
# ===================================================================


def test_encoding_falls_back_to_cl100k_when_tiktoken_does_not_know_the_model(monkeypatch):
    requested = []

    def unknown_model(name):
        requested.append(("model", name))
        raise KeyError(name)

    def get_encoding(name):
        requested.append(("encoding", name))
        return LocalEncoding()

    monkeypatch.setattr(windowing_utils.tiktoken, "encoding_for_model", unknown_model)
    monkeypatch.setattr(windowing_utils.tiktoken, "get_encoding", get_encoding)

    assert isinstance(_REAL_GET_ENCODING(), LocalEncoding)
    assert requested == [("model", windowing_utils.HARDCODED_ENCODING_MODEL), ("encoding", "cl100k_base")]


class TestTagToolChains:
    """``tag_tool_chains()`` chain detection and tagging."""

    def test_tag_single_chain(self):
        """W6: One chain detected and tagged with same chain_group_id."""
        msgs = [
            _user("u1"),
            _assistant_with_tool_calls(),
            _tool("r1"),
            _tool("r2"),
            _user("u2"),
        ]
        tagged = tag_tool_chains(msgs)
        # user messages have None
        assert tagged[0]["_chain_group_id"] is None
        assert tagged[4]["_chain_group_id"] is None
        # chain messages share the same ID
        assert tagged[1]["_chain_group_id"] is not None
        assert tagged[1]["_chain_group_id"] == tagged[2]["_chain_group_id"]
        assert tagged[2]["_chain_group_id"] == tagged[3]["_chain_group_id"]

    def test_tag_multiple_chains(self):
        """W6: Multiple chains get unique IDs."""
        msgs = [
            _assistant_with_tool_calls([{"id": "a"}]),
            _tool("ra"),
            _user("u1"),
            _assistant_with_tool_calls([{"id": "b"}]),
            _tool("rb"),
        ]
        tagged = tag_tool_chains(msgs)
        # Two chains — different IDs
        id_a = tagged[0]["_chain_group_id"]
        id_b = tagged[3]["_chain_group_id"]
        assert id_a is not None
        assert id_b is not None
        assert id_a != id_b
        # Chain A: msg[0], msg[1]
        assert tagged[1]["_chain_group_id"] == id_a
        # Chain B: msg[3], msg[4]
        assert tagged[4]["_chain_group_id"] == id_b
        # Non-chain msg between them
        assert tagged[2]["_chain_group_id"] is None

    def test_tag_orphan_tool(self):
        """W6: Standalone tool message NOT chain-tagged."""
        msgs = [
            _user("u1"),
            _tool("orphan"),  # no preceding assistant(tool_calls)
        ]
        tagged = tag_tool_chains(msgs)
        assert tagged[0]["_chain_group_id"] is None
        assert tagged[1]["_chain_group_id"] is None  # orphan, no chain

    def test_tag_no_chains(self):
        """W6: No chains → all chain_group_id are None."""
        msgs = [_user("a"), _assistant("b"), _user("c")]
        tagged = tag_tool_chains(msgs)
        for msg in tagged:
            assert msg["_chain_group_id"] is None

    def test_tag_empty_list(self):
        """W6: Empty list returns empty list."""
        assert tag_tool_chains([]) == []

    def test_tag_consecutive_chains_no_interleaving(self):
        """W6: Two consecutive chains (no user in-between)."""
        msgs = [
            _assistant_with_tool_calls([{"id": "a"}]),
            _tool("ra"),
            _assistant_with_tool_calls([{"id": "b"}]),
            _tool("rb"),
        ]
        tagged = tag_tool_chains(msgs)
        assert tagged[0]["_chain_group_id"] is not None
        assert tagged[1]["_chain_group_id"] == tagged[0]["_chain_group_id"]
        # Second chain starts at index 2
        assert tagged[2]["_chain_group_id"] is not None
        assert tagged[2]["_chain_group_id"] != tagged[0]["_chain_group_id"]
        assert tagged[3]["_chain_group_id"] == tagged[2]["_chain_group_id"]


class TestFindSafeTailCut:
    """``find_safe_tail_cut()`` chain-aware boundary detection."""

    def _tagged(self, msgs: list[dict]) -> list[dict]:
        """Convenience: tag messages then return them."""
        return tag_tool_chains(msgs)

    def test_find_safe_cut_exact(self):
        """Cut exactly at a non-chain boundary — no adjustment."""
        msgs = self._tagged([
            _user("oldest"),
            _user("middle"),
            _user("newest"),
        ])
        cut = find_safe_tail_cut(msgs, target_count=2)
        # Keep 2 newest → cut at index 1 (drop oldest)
        assert cut == 1

    def test_find_safe_cut_chain_boundary(self):
        """Chain straddles naive cut → move boundary to chain start.

        Messages: [chain_start, tool, tool, user]
        Target_count=2 → naive cut at index 2 (keep last 2: tool, user).
        But the tool is inside the chain → cut moves to chain start (index 0).
        """
        msgs = self._tagged([
            _assistant_with_tool_calls(),
            _tool("r1"),
            _tool("r2"),
            _user("last"),
        ])
        cut = find_safe_tail_cut(msgs, target_count=2)
        # Naive cut = 4 - 2 = 2. Chain (index 0-2) straddles cut.
        # Chain start = 0, so cut = 0.
        assert cut == 0, f"Expected 0, got {cut}"

    def test_find_safe_cut_no_adjustment_needed(self):
        """Chain entirely within kept region — no adjustment."""
        msgs = self._tagged([
            _user("drop me"),
            _assistant_with_tool_calls(),
            _tool("r1"),
            _tool("r2"),
            _user("keep me"),
        ])
        cut = find_safe_tail_cut(msgs, target_count=4)
        # Keep last 4 → naive cut = 1. Chain at indices 1-3.
        # Chain start = 1 == naive cut → no adjustment needed.
        assert cut == 1

    def test_find_safe_cut_target_larger_than_list(self):
        """Target >= len(list) → cut = 0 (no truncation)."""
        msgs = self._tagged([
            _user("a"),
            _user("b"),
        ])
        cut = find_safe_tail_cut(msgs, target_count=10)
        assert cut == 0

    def test_find_safe_cut_target_zero(self):
        """Target = 0 → cut = len(truncatable) (drop everything)."""
        msgs = self._tagged([_user("a")])
        cut = find_safe_tail_cut(msgs, target_count=0)
        assert cut == len(msgs)  # drop all truncatable messages

    def test_find_safe_cut_multiple_chains_first_dropped(self):
        """First chain entirely in dropped region — cut at chain end.

        Messages: [chain_a_start, tool_a, chain_b_start, tool_b, user]
        Target_count=2 → naive cut at index 3 (keep last 2: tool_b, user).
        Chain_b (idx 2-3) straddles but chain start = 2 > naive cut 3? No,
        chain start < naive cut won't match. Let's think...
        Actually: naive_cut = 5 - 2 = 3. Check index 3: chain_b = not None.
        chain_start at 2 < naive_cut 3 → move cut to 2.
        """
        msgs = self._tagged([
            _assistant_with_tool_calls([{"id": "a"}]),
            _tool("ra"),
            _assistant_with_tool_calls([{"id": "b"}]),
            _tool("rb"),
            _user("last"),
        ])
        cut = find_safe_tail_cut(msgs, target_count=2)
        # Keep last 2: tool_b, user. Naive cut = 3.
        # Chain B at indices 2-3, chain_start=2 < 3 → move to 2.
        assert cut == 2, f"Expected 2, got {cut}"

    def test_find_safe_cut_chain_entirely_in_dropped(self):
        """Chain entirely before the cut → no straddle, no adjustment."""
        msgs = self._tagged([
            _assistant_with_tool_calls(),
            _tool("r1"),
            _user("mid"),
            _user("last"),
        ])
        # Keep last 2 → naive cut = 2 (drop first 2: chain). No straddle.
        cut = find_safe_tail_cut(msgs, target_count=2)
        assert cut == 2, f"Expected 2, got {cut}"
