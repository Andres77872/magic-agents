import asyncio
import hashlib
import json
import logging
import uuid
from typing import Any, Optional, AsyncGenerator

from magic_llm.model import ModelChat

from magic_agents.models.factory.Nodes.MemoryNodeModel import (
    MemoryEntry,
    ExtractedMemory,
    ExtractionResult,
    MemoryNodeModel,
)
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeMemory(Node):
    """Memory vector insights node — multi-phase async process.

    Receives a user message, optionally extracts structured memories via LLM,
    embeds and upserts them to a vector DB, searches for relevant past memories,
    and injects matching memories as <memory_content> prompt context.

    Follows a fail-open policy: any phase error logs a warning and continues
    to the next phase or yields the original message unchanged.

    <memory_content> is prompt annotation text — the runtime does NOT parse
    or validate XML. Content containing tag-like sequences passes through
    unmodified. This is NOT a security boundary.
    """

    DEFAULT_INPUT_HANDLE = 'handle_memory_input'
    DEFAULT_OUTPUT_HANDLE = 'handle_memory_output'
    DEFAULT_CLIENT_HANDLE = 'handle-client-provider'

    MEMORY_DELIMITER_OPEN = '<memory_content>'
    MEMORY_DELIMITER_CLOSE = '</memory_content>'

    def __init__(
        self,
        data: MemoryNodeModel,
        node_id: str,
        debug: bool = False,
        handles: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(debug=debug, node_id=node_id, **kwargs)

        # Extract constructor-injected dependencies
        self._vector_db = kwargs.pop('vector_db', None)
        self._embedding_client = kwargs.pop('embedding_client', None)
        self._client = kwargs.pop('client', None)
        if self._vector_db is None:
            from magic_agents.vector_storage import InMemoryVectorDB

            self._vector_db = InMemoryVectorDB()
            logger.info(
                "NodeMemory '%s': No vector_db provided. "
                "Auto-created ephemeral InMemoryVectorDB. "
                "Memory data is LOST on process restart.",
                self.node_id,
            )

        # Model data
        self._instructions = data.instructions
        self._seed_entries = data.memory_entries
        self._top_k = data.top_k
        self._context_messages_count = data.context_messages_count
        self._memory_entries: list[MemoryEntry] = []

        # NEW: history_messages injected from api.magic_llm deps
        self._history_messages: list[dict] = kwargs.pop('history_messages', [])
        # Optional API-side tracker lets request finalization await background
        # MEMORY extraction/upsert before computing client-visible usage totals.
        self._background_task_tracker = kwargs.pop('background_task_tracker', None)

        # Handle overrides
        handles = handles or {}
        self.INPUT_HANDLE = handles.get('input', self.DEFAULT_INPUT_HANDLE)
        self.OUTPUT_HANDLE = handles.get('output', self.DEFAULT_OUTPUT_HANDLE)
        self.CLIENT_HANDLE = handles.get('client', self.DEFAULT_CLIENT_HANDLE)

        # Per-process counters for _capture_internal_state
        self._extracted_count: int = 0
        self._injected_count: int = 0

    async def process(self, chat_log) -> AsyncGenerator[dict, None]:
        """Execute the 6-phase memory process with fail-open error handling.

        Phase 1 — Input: Get user message and MagicLLM client from input handles.
        Phase 2 — LLM Extraction (optional): Extract structured memories using LLM.
        Phase 3 — Embedding + Upsert: Embed new memories, upsert to vector DB.
        Phase 4 — Vector Search: Search for similar past memories.
        Phase 5 — Injection: Wrap matches in <memory_content>, prepend to message.
        Phase 6 — Yield: Emit the result on OUTPUT_HANDLE.
        """
        # Reset per-process counters
        self._extracted_count = 0
        self._injected_count = 0

        # =====================================================================
        # PHASE 1: Input
        # =====================================================================
        msg = self.get_input(self.INPUT_HANDLE)
        if not msg:
            # None or empty input -> yield empty passthrough
            yield self.yield_static("", content_type=self.OUTPUT_HANDLE)
            return

        msg_str = str(msg)

        client = self._client
        try:
            input_client = self.get_input(self.CLIENT_HANDLE)
            if input_client is not None:
                client = input_client
        except Exception as exc:
            logger.warning(
                "NodeMemory '%s': Failed to get client input: %s",
                self.node_id, exc,
            )

        # =====================================================================
        # FIRE-AND-FORGET: Background extraction + upsert (Phases 2-3)
        # =====================================================================
        if self._instructions and (client or self._embedding_client):
            background_coro = self._extract_and_upsert_background(msg_str, client, chat_log)
            if callable(self._background_task_tracker):
                try:
                    self._background_task_tracker(background_coro)
                except Exception as exc:
                    logger.warning(
                        "NodeMemory '%s': background_task_tracker failed, falling back to create_task: %s",
                        self.node_id,
                        exc,
                    )
                    asyncio.create_task(background_coro, name=f"mem-extract-{self.node_id}")
            else:
                asyncio.create_task(background_coro, name=f"mem-extract-{self.node_id}")

        # =====================================================================
        # SEQUENTIAL: Vector search + injection (Phases 4-6)
        # =====================================================================
        # PHASE 4: Vector Search
        # =====================================================================
        matches: list[MemoryEntry] = []
        if client or self._embedding_client:
            try:
                if self._embedding_client is None:
                    logger.warning(
                        "NodeMemory '%s': No embedding_client provided. Skipping embedding Phase 4.",
                        self.node_id,
                    )
                else:
                    query_embedding_resp = await self._embedding_client.llm.async_embedding(msg_str)

                    if query_embedding_resp is not None and query_embedding_resp.data:
                        matches = await self._vector_db.search(
                            query_embedding=query_embedding_resp.data[0].embedding,
                            top_k=self._top_k,
                            filter_scope={
                                "session_id": str(chat_log.id_chat) if chat_log.id_chat is not None else "",
                                "node_id": self.node_id or "",
                            },
                        )
                    else:
                        logger.warning(
                            "NodeMemory '%s': async_embedding returned None "
                            "for query — skipping vector search. "
                            "Only OpenAI engines support embedding. "
                            "Memory enrichment disabled.",
                            self.node_id,
                        )

            except Exception as exc:
                logger.warning(
                    "NodeMemory '%s': Vector search failed: %s",
                    self.node_id, exc,
                )

        # =====================================================================
        # PHASE 5: Injection
        # =====================================================================
        if matches:
            sorted_matches = sorted(
                matches, key=lambda m: m.created_at,
            )
            memory_blocks = "\n".join(
                f"<memory_content>{m.content}</memory_content>"
                for m in sorted_matches
            )
            self._injected_count = len(sorted_matches)
            result = f"{memory_blocks}\n{msg_str}"
        else:
            result = msg_str

        # =====================================================================
        # PHASE 6: Yield
        # =====================================================================
        yield self.yield_static(result, content_type=self.OUTPUT_HANDLE)

    async def _extract_and_upsert_background(
        self,
        msg_str: str,
        client: Any,
        chat_log: Any,
    ) -> None:
        """Fire-and-forget: extract memories from context + current msg, then embed+upsert.

        Runs asyncio.create_task from process() — errors are logged, never propagated.
        Memories become available to future process() calls (not current turn).

        Args:
            msg_str: Current user message string.
            client: MagicLLM client for LLM extraction.
            chat_log: ModelAgentRunLog for session context.

        Returns:
            None (fire-and-forget).
        """
        memory_entries: list[MemoryEntry] = []

        # ── Phase 2: LLM Extraction ──
        if self._instructions and client:
            try:
                json_instr = (
                    "\n\nYou MUST respond with a JSON object matching this schema: "
                    '{"memories": [{"content": "memory text", "trigger": "category"}]}'
                )
                extraction_chat = ModelChat(system=self._instructions + json_instr)

                # Include context messages if configured
                if self._context_messages_count > 0 and self._history_messages:
                    context_msgs = self._history_messages[-self._context_messages_count:]
                    for hist_msg in context_msgs:
                        role = hist_msg.get('role', 'user')
                        content = hist_msg.get('content', '')
                        if role == 'user':
                            extraction_chat.add_user_message(content)
                        elif role == 'assistant':
                            extraction_chat.add_assistant_message(content)
                        elif role == 'system':
                            extraction_chat.add_system_message(content)

                extraction_chat.add_user_message(msg_str)

                result = await client.llm.async_generate(
                    extraction_chat, json_output=True,
                )
                parsed = ExtractionResult(**json.loads(result.content))
                for mem in parsed.memories:
                    entry = MemoryEntry(
                        content=mem.content,
                        trigger=mem.trigger,
                    )
                    memory_entries.append(entry)
                self._extracted_count = len(memory_entries)
                self._memory_entries.extend(memory_entries)

            except Exception as exc:
                logger.warning(
                    "NodeMemory '%s': Background extraction failed: %s",
                    self.node_id, exc,
                )
                return  # Nothing to embed/upsert

        # ── Phase 3: Embedding + Upsert ──
        if not memory_entries:
            return  # Nothing to upsert

        if self._embedding_client is None:
            logger.warning(
                "NodeMemory '%s': No embedding_client for background upsert.",
                self.node_id,
            )
            return

        try:
            for entry in memory_entries:
                try:
                    embedding_resp = await self._embedding_client.llm.async_embedding(entry.content)
                    if embedding_resp is not None and embedding_resp.data:
                        doc_id = str(uuid.uuid4())  # UUID4 ID

                        await self._vector_db.upsert(
                            id=doc_id,
                            embedding=embedding_resp.data[0].embedding,
                            metadata={
                                "doc_id": doc_id,
                                "memory_entry_id": entry.id,
                                "content": entry.content,
                                "trigger": entry.trigger,
                                "created_at": entry.created_at.isoformat(),
                                "session_id": str(chat_log.id_chat) if chat_log.id_chat is not None else "",
                                "node_id": self.node_id or "",
                            },
                        )
                except Exception as exc:
                    logger.warning(
                        "NodeMemory '%s': Background upsert failed for entry: %s",
                        self.node_id, exc,
                    )
        except Exception as exc:
            logger.warning(
                "NodeMemory '%s': Background upsert batch failed: %s",
                self.node_id, exc,
            )

    def _detect_vector_db_backend(self) -> str:
        """Detect vector DB backend type for diagnostic purposes.

        Returns:
            "ephemeral" — InMemoryVectorDB (auto-created or manually passed)
            "qdrant" — QdrantVectorDB
            "external" — any other VectorDB-compatible implementation
        """
        class_name = type(self._vector_db).__qualname__
        if class_name == "InMemoryVectorDB":
            return "ephemeral"
        if class_name == "QdrantVectorDB":
            return "qdrant"
        return "external"

    def _capture_internal_state(self) -> dict[str, Any]:
        """Capture NodeMemory-specific internal state for debugging."""
        state = super()._capture_internal_state()
        state["memory_entry_count"] = len(self._memory_entries)
        state["extracted_count"] = self._extracted_count
        state["injected_count"] = self._injected_count
        state["vector_db_available"] = self._vector_db is not None
        state["vector_db_backend"] = self._detect_vector_db_backend()
        state["top_k"] = self._top_k
        state["_context_messages_count"] = self._context_messages_count
        state["_history_messages_count"] = len(self._history_messages)
        return state
