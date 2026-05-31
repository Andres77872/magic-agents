"""
Utility functions for NodeChat/NodeLLM STM windowing and context-budget controls.

Provides pure, testable functions for message truncation, token estimation,
and tool-call chain atomicity enforcement.

All functions are stateless and have no side effects.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import tiktoken


# ---------------------------------------------------------------------------
# Token estimation constants (mirrors ModelChat.num_tokens_from_messages)
# ---------------------------------------------------------------------------
TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
ASSISTANT_PRIME_TOKENS = 3
IMAGE_TOKEN_HTTP_ESTIMATE = 85
HARDCODED_ENCODING_MODEL = "gpt-5"


# ===================================================================
# estimate_tokens
# ===================================================================

def _get_encoding() -> tiktoken.Encoding:
    """Resolve tiktoken encoding for hardcoded GPT-5 tokenizer.

    Attempts ``tiktoken.encoding_for_model(HARDCODED_ENCODING_MODEL)``
    (``"gpt-5"``).  If GPT-5 is not yet known to tiktoken (``KeyError``),
    silently falls back to ``tiktoken.get_encoding("cl100k_base")``.

    .. note::
       This is a **silent fallback by design** (spec E1).  The
       ``cl100k_base`` fallback matches the previous approach and will
       automatically resolve to the correct GPT-5 encoding once tiktoken
       adds native support — no code change needed.
    """
    try:
        return tiktoken.encoding_for_model(HARDCODED_ENCODING_MODEL)
    except KeyError:
        # GPT-5 not yet in tiktoken → silent fallback
        return tiktoken.get_encoding("cl100k_base")


def _estimate_image_tokens(content_part: dict[str, Any]) -> int:
    """Estimate token cost for a single image content part.

    * Data URI (base64): ``len(payload) // 4``
    * HTTP/HTTPS URL: fixed ``IMAGE_TOKEN_HTTP_ESTIMATE``
    """
    image_url = content_part.get("image_url", {}) or {}
    url: str = image_url.get("url", "")
    if url.startswith("data:"):
        # base64 data URI — rough estimate
        payload = url.split(",", 1)[-1] if "," in url else url
        return max(1, len(payload) // 4)
    # Remote HTTP(S) URL — fixed estimate
    return IMAGE_TOKEN_HTTP_ESTIMATE


def estimate_tokens(
    messages: list[dict[str, Any]],
) -> int:
    """Estimate the total token count for a list of messages.

    Uses ``tiktoken`` with a **hardcoded GPT-5 tokenizer policy** (spec
    T1-POLICY).  The ``model`` parameter has been removed — token
    estimation no longer accepts a configurable model name.

    Formula (per message):
        ``TOKENS_PER_MESSAGE`` (3)
        + content tokens  (``len(encoder.encode(content))`` for strings,
                           or sum of text parts for multimodal content)
        + ``TOKENS_PER_NAME`` (1)  if ``name`` key is present
        + ``ASSISTANT_PRIME_TOKENS`` (3)  if ``role == 'assistant'``

    Args:
        messages: List of message dicts (each with ``role``, ``content``,
                  and optionally ``name``, ``tool_calls``).

    Returns:
        Estimated total token count (int).
    """
    encoding = _get_encoding()
    total = 0

    for msg in messages:
        total += TOKENS_PER_MESSAGE

        # Content tokens
        content = msg.get("content")
        if isinstance(content, str):
            total += len(encoding.encode(content))
        elif isinstance(content, list):
            # Multimodal content — list of text / image_url parts
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += len(encoding.encode(part.get("content", "")))
                    elif part.get("type") == "image_url":
                        total += _estimate_image_tokens(part)

        # Name key
        if msg.get("name"):
            total += TOKENS_PER_NAME

        # Assistant prime tokens
        if msg.get("role") == "assistant":
            total += ASSISTANT_PRIME_TOKENS

    return total


# ===================================================================
# tag_tool_chains
# ===================================================================

def tag_tool_chains(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pre-scan messages and tag tool-call chains with ``_chain_group_id``.

    A *tool-call chain* is:
        1. An ``assistant`` message whose ``tool_calls`` key is truthy.
        2. Zero or more *consecutive* ``tool`` role messages immediately
           following it.

    Each message in the same chain receives a shared UUID string.
    Standalone (non-chain) messages get ``_chain_group_id = None``.

    .. note::
       This function mutates message dicts **in place** but is intended
       to be called on a copy made by ``apply_windowing``.

    Args:
        messages: List of message dicts.

    Returns:
        The same list (mutated) with ``_chain_group_id`` injected.
    """
    current_chain_id: Optional[str] = None
    in_chain = False

    for msg in messages:
        role = msg.get("role", "")
        has_tool_calls = bool(msg.get("tool_calls"))

        if role == "assistant" and has_tool_calls:
            # Start a new chain
            current_chain_id = str(uuid.uuid4())
            msg["_chain_group_id"] = current_chain_id
            in_chain = True
        elif role == "tool" and in_chain:
            # Continuation of the current chain
            msg["_chain_group_id"] = current_chain_id
        else:
            # Not in a chain — reset
            msg["_chain_group_id"] = None
            in_chain = False
            current_chain_id = None

    return messages


# ===================================================================
# find_safe_tail_cut
# ===================================================================

def _chain_end(
    messages: list[dict[str, Any]],
    start: int,
    chain_id: str,
) -> int:
    """Return the index of the LAST message belonging to *chain_id*,
    starting the search from *start* onward.

    Returns *start* if *messages[start]* is the last message of its chain.
    """
    idx = start
    while idx < len(messages) and messages[idx].get("_chain_group_id") == chain_id:
        idx += 1
    return idx - 1  # last index that belongs to the chain


def _chain_start(
    messages: list[dict[str, Any]],
    start: int,
    chain_id: str,
) -> int:
    """Return the index of the FIRST message belonging to *chain_id*,
    searching backward from *start*.

    Returns *start* if *messages[start]* is the first message of its chain.
    """
    idx = start
    while idx >= 0 and messages[idx].get("_chain_group_id") == chain_id:
        idx -= 1
    return idx + 1  # first index that belongs to the chain


def find_safe_tail_cut(
    truncatable: list[dict[str, Any]],
    target_count: int,
) -> int:
    """Find the cut index for tail truncation respecting tool-chain atomicity.

    Scans from the **newest** (end) toward the **oldest** (start), counting
    messages.  When the accumulated count reaches *target_count*, checks
    whether the current position falls *inside* a tool-call chain.  If it
    does, the cut is moved to the **start** of that chain so the entire
    chain is kept (atomic).

    Returns the index in *truncatable* from which to **keep** (i.e. drop
    everything before this index).  Returns ``0`` when nothing can / should
    be dropped.
    """
    if target_count <= 0:
        return len(truncatable)  # drop everything from truncatable
    if target_count >= len(truncatable):
        return 0  # no truncation needed

    # Naive cut: keep the last target_count messages
    naive_cut = len(truncatable) - target_count

    # Check whether the message just BEFORE the naive cut is part of a chain
    # that straddles the boundary.  We scan from the cut point backward.
    check_idx = naive_cut
    while check_idx < len(truncatable):
        chain_id = truncatable[check_idx].get("_chain_group_id")
        if chain_id is not None:
            # This message is part of a chain — find where it starts
            chain_start_idx = _chain_start(truncatable, check_idx, chain_id)
            if chain_start_idx < naive_cut:
                # The chain straddles the naive cut; move cut to chain start
                return chain_start_idx
            # Chain is entirely within the kept region — no adjustment needed
            # Skip past the whole chain
            chain_end_idx = _chain_end(truncatable, check_idx, chain_id)
            check_idx = chain_end_idx + 1
        else:
            check_idx += 1

    return naive_cut


# ===================================================================
# apply_windowing  (PUBLIC API)
# ===================================================================

def apply_windowing(
    messages: list[dict[str, Any]],
    max_messages: Optional[int] = None,
    max_input_tokens: Optional[int] = None,
    truncation_strategy: str = "tail",
) -> list[dict[str, Any]]:
    """Apply STM windowing to a message list.

    Token estimation uses a **hardcoded GPT-5 tokenizer policy** (spec
    T1-POLICY).  The ``model`` parameter has been removed — windowing
    no longer accepts a configurable model name.

    Rules
    -----
    1. System messages (``role='system'``) are **always** preserved and
       are **not** counted toward ``max_messages``.
    2. The **last user message** (current turn) is **always** preserved.
    3. Tool-call chains (``assistant`` with ``tool_calls`` + consecutive
       ``tool`` messages) are **atomic** — they are kept or dropped as a
       whole.
    4. ``max_messages`` is applied first (message count truncation) and
       then ``max_input_tokens`` (token budget).
    5. ``truncation_strategy='token_budget'`` **skips** the message-count
       truncation step.

    The function is **pure**: it does **not** mutate the input list.

    Args:
        messages:           Full message list to window.
        max_messages:       Max non-system messages to keep (``None`` = no
                            message limit).
        max_input_tokens:   Max estimated tokens (``None`` = no token
                            budget).
        truncation_strategy: ``'tail'`` or ``'token_budget'``.

    Returns:
        Windowed message list.
    """
    # --- Guard: no-op when no limits are set ---------------------------
    if max_messages is None and max_input_tokens is None:
        return list(messages)  # return a copy per W9 (immutability)

    # Normalise edge cases
    if max_messages is not None and max_messages <= 0:
        max_messages = None

    # Work on a copy — never mutate the input (W9)
    working = list(messages)

    # --- Step 1: Extract & preserve system messages --------------------
    system_msgs = [m for m in working if m.get("role") == "system"]
    non_system = [m for m in working if m.get("role") != "system"]

    # --- Step 2: Extract & preserve last user message ------------------
    last_user: list[dict[str, Any]] = []
    for m in reversed(non_system):
        if m.get("role") == "user":
            last_user = [m]
            break

    # Truncatable = non-system minus the last user message
    # Use identity comparison to correctly handle when the last user
    # message is NOT the final element of non_system (e.g. last msg
    # is an assistant message with tool_calls).
    if last_user:
        last_msg = last_user[0]
        truncatable = [m for m in non_system if m is not last_msg]
    else:
        truncatable = list(non_system)

    # --- Step 3: Pre-scan for tool-call chains -------------------------
    if truncatable:
        truncatable = tag_tool_chains(truncatable)

    # --- Step 4: Apply max_messages truncation -------------------------
    if (
        max_messages is not None
        and truncation_strategy != "token_budget"
        and truncatable
    ):
        # target_count excludes the last user message (preserved in step 2)
        target = max_messages - 1 if last_user else max_messages
        if target < 0:
            target = 0

        if len(truncatable) > target:
            cut = find_safe_tail_cut(truncatable, target)
            if cut > 0:
                truncatable = truncatable[cut:]
        # else: within limit — no truncation needed

    # --- Step 5: Apply max_input_tokens truncation ---------------------
    if max_input_tokens is not None and max_input_tokens > 0:
        total = estimate_tokens(system_msgs + truncatable + last_user)
        while total > max_input_tokens and truncatable:
            # Drop the oldest message (or entire oldest chain) one at a time
            first_cid = truncatable[0].get("_chain_group_id")
            if first_cid is not None:
                # Find the end of this chain
                chain_end_idx = _chain_end(truncatable, 0, first_cid)
                truncatable = truncatable[chain_end_idx + 1 :]
            else:
                truncatable = truncatable[1:]

            total = estimate_tokens(system_msgs + truncatable + last_user)

    # --- Step 6: Re-assemble and return --------------------------------
    preserved = list(system_msgs) + truncatable + list(last_user)
    return preserved
