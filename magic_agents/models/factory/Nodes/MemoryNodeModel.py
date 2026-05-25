import uuid
from datetime import datetime, UTC
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class MemoryEntry(BaseModel):
    """A single memory entry with auto-generated ID and timestamp.

    CRITICAL: MemoryEntry.id is STRICTLY auto-generated. Users MUST NOT provide
    an 'id' field in JSON. If provided, a ValidationError is raised.
    This is enforced by the reject_manual_id field_validator.
    """

    model_config = ConfigDict(extra='forbid')

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Auto-generated 32-char UUID4 hex. "
        "DO NOT provide in JSON — rejected with ValidationError.",
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Populated by VectorDB search() — traceable "
        "vector DB document ID (scope-aware composite hash). "
        "Not writable by user input. None if no metadata available.",
    )
    content: str = Field(
        min_length=1,
        description="Memory content text",
    )
    trigger: str = Field(
        default="",
        description="Memory tag/category (singular, unlike CodexEntry triggers list)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of memory creation",
    )

    @field_validator('id', mode='before')
    @classmethod
    def reject_manual_id(cls, v: Optional[str]) -> Optional[str]:
        """Reject any user-provided 'id' value.

        Runs BEFORE default_factory. If the user explicitly sets 'id' in JSON,
        this validator fires and raises ValueError. Returns v (which is None
        when field is omitted) to pass control to default_factory.
        """
        if v is not None:
            raise ValueError(
                "MemoryEntry.id is auto-generated. "
                "Do not provide a value for 'id'. "
                "Remove the 'id' field from the memory entry JSON."
            )
        return v  # None -> let default_factory generate the ID


class ExtractedMemory(BaseModel):
    """Raw extraction output from LLM — before MemoryEntry wrapping."""

    model_config = ConfigDict(extra='forbid')

    content: str = Field(min_length=1, description="Memory content text")
    trigger: str = Field(default="", description="Memory tag/category")


class ExtractionResult(BaseModel):
    """LLM extraction output container — owned by NodeMemory, not user-defined."""

    model_config = ConfigDict(extra='forbid')

    memories: list[ExtractedMemory] = Field(default_factory=list)


class MemoryNodeModel(BaseNodeModel):
    """Node configuration for memory vector insights.

    The instructions field is an optional LLM extraction template.
    When None, only vector search is performed (no extraction).
    memory_entries serves as seed data for rebuild() and export snapshot.
    """

    instructions: Optional[str] = Field(
        default=None,
        description="LLM extraction prompt template. "
        "When None, only vector search is performed.",
    )
    memory_entries: list[MemoryEntry] = Field(
        default_factory=list,
        description="Seed memory entries. Vector DB is the source of truth for search. "
        "This field is a seed/export snapshot, not authoritative runtime state.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top memory matches to inject. "
        "Default 5. Range 1-50. "
        "Configures the vector DB search limit.",
    )
    context_messages_count: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of previous chat messages to include "
        "in LLM extraction context. "
        "Range 0-100. Intentional behavior change — recent context is the primary use case.",
    )
    # NOTE: 'handles' field is NOT defined here — factory pops it before model validation
