import logging
import re
from typing import Optional

from magic_agents.models.factory.Nodes import CodexNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)

CODEX_DELIMITER_OPEN = '<codex_content>'
CODEX_DELIMITER_CLOSE = '</codex_content>'


class NodeCodex(Node):
    """
    Codex knowledge hub node.

    Sits between UserInput and Chat/LLM consumers. Receives user message
    on `handle_codex_input`, matches triggers against codex entries,
    prepends matched content wrapped in `<codex_content>...</codex_content>`,
    and emits on `handle_user_message`.

    Safe to emit `handle_user_message` because runtime routing is
    source-node-ID scoped, not globally handle-name scoped.

    IMPORTANT: <codex_content> is prompt annotation text — the runtime
    does NOT parse or validate XML. Content containing tag-like sequences
    passes through unmodified. This is NOT a security boundary.

    Competing edge warning: Graphs with a Codex node should NOT retain
    a direct UserInput → Chat edge on handle_user_message. Two competing
    writes to Chat.inputs['handle_user_message'] cause race behavior.
    Detection/enforcement is out of scope for this change.
    """
    DEFAULT_INPUT_HANDLE = 'handle_codex_input'
    DEFAULT_OUTPUT_HANDLE = 'handle_user_message'

    def __init__(self, data: CodexNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._codex_entries = data.codex_entries

        # Pre-compute casefolded trigger frozenset for O(1) early-exit
        self._trigger_set = frozenset(
            t.casefold()
            for entry in self._codex_entries
            for t in entry.triggers
        )

        # Allow JSON handles override
        handles = handles or {}
        self.INPUT_HANDLE = handles.get('input', self.DEFAULT_INPUT_HANDLE)
        self.OUTPUT_HANDLE = handles.get('output', self.DEFAULT_OUTPUT_HANDLE)

        # Per-process dynamic state (captured by _capture_internal_state)
        self._last_matched_triggers: list[str] = []
        self._last_prepended_count: int = 0
        self._last_message_length: int = 0

    @staticmethod
    def _word_boundary_match(trigger: str, text_folded: str) -> bool:
        """Case-insensitive word-boundary match using \\b anchors."""
        return bool(re.search(
            rf'\b{re.escape(trigger.casefold())}\b',
            text_folded
        ))

    async def process(self, chat_log):
        msg = self.get_input(self.INPUT_HANDLE)
        if not msg:
            # None or empty input → yield empty passthrough
            self._last_matched_triggers = []
            self._last_prepended_count = 0
            self._last_message_length = 0
            yield self.yield_static("", content_type=self.OUTPUT_HANDLE)
            return

        msg_str = str(msg)
        msg_folded = msg_str.casefold()
        self._last_message_length = len(msg_str)

        # Early-exit: no codex entries or no trigger present in message
        if not self._codex_entries or not any(
            t in msg_folded for t in self._trigger_set
        ):
            self._last_matched_triggers = []
            self._last_prepended_count = 0
            yield self.yield_static(msg_str, content_type=self.OUTPUT_HANDLE)
            return

        # Full scan: match each entry against triggers with word-boundary semantics
        matched = []
        matched_triggers = []
        for entry in self._codex_entries:
            for trigger in entry.triggers:
                if self._word_boundary_match(trigger, msg_folded):
                    # Content is raw text — no XML escaping.
                    # <codex_content> is prompt annotation, not parsed XML.
                    matched.append(
                        f"{CODEX_DELIMITER_OPEN}{entry.content}{CODEX_DELIMITER_CLOSE}"
                    )
                    matched_triggers.append(trigger)
                    break  # each entry at most once

        self._last_matched_triggers = matched_triggers
        self._last_prepended_count = len(matched)

        if matched:
            result = "\n".join(matched) + "\n" + msg_str
        else:
            result = msg_str  # fallback passthrough

        yield self.yield_static(result, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Codex-specific internal state for debugging."""
        state = super()._capture_internal_state()
        # Static config fields
        state['codex_entry_count'] = len(self._codex_entries)
        state['trigger_count'] = len(self._trigger_set)
        state['input_handle'] = self.INPUT_HANDLE
        state['output_handle'] = self.OUTPUT_HANDLE
        # Dynamic execution fields
        state['matched_triggers'] = self._last_matched_triggers
        state['prepended_count'] = self._last_prepended_count
        state['message_length'] = self._last_message_length
        return state
