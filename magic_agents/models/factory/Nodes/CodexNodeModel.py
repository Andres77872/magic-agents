import uuid
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


def is_valid_uuid4(s: str) -> bool:
    """Validate that a string is a proper UUID4.

    Accepts both hyphenated (550e8400-e29b-41d4-a716-446655440000)
    and non-hyphenated (550e8400e29b41d4a716446655440000) formats.
    """
    try:
        val = uuid.UUID(s, version=4)
        # Strip hyphens from input and compare to hex (no-hyphen) canonical form
        return val.hex == s.replace('-', '')
    except (ValueError, AttributeError):
        return False


class CodexEntry(BaseModel):
    """A single knowledge entry containing trigger keywords and associated content."""

    model_config = ConfigDict(extra='forbid')  # Backend-authoritative: reject unknown fields

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Entry ID. Auto-generated as 32-char UUID4 hex if omitted. "
        "If provided, must be a valid UUID4 (hyphenated or non-hyphenated).",
    )
    triggers: list[str] = Field(
        min_length=1,
        description="Trigger keywords (case-insensitive word-boundary match)",
    )
    content: str = Field(
        min_length=1,
        description="Content to prepend when triggers match. "
        "Wrapped in <codex_content>...</codex_content> — prompt text, not parsed XML.",
    )

    @field_validator('id', mode='before')
    @classmethod
    def validate_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate id if explicitly provided; if None/omitted, default_factory handles it.

        Accepts both hyphenated and non-hyphenated UUID4 strings.
        Returns the value as-is — the validator only accepts valid UUID4;
        the stored form preserves whatever format the user provided
        or the default_factory generated.
        """
        if v is None:
            return v  # Let default_factory generate
        if not isinstance(v, str) or not is_valid_uuid4(v):
            raise ValueError(
                f"Invalid UUID4 id: '{v}'. "
                f"Must be a valid UUID4 string (e.g., "
                f"'550e8400-e29b-41d4-a716-446655440000' or "
                f"'550e8400e29b41d4a716446655440000')."
            )
        return v


class CodexNodeModel(BaseNodeModel):
    """Node configuration for codex knowledge hub.

    The codex_entries list contains knowledge entries. When a user message
    triggers any keyword in an entry's triggers list, that entry's content
    is prepended to the message wrapped in <codex_content>...</codex_content>.

    NOTE: <codex_content> is prompt annotation text visible to the LLM.
    The runtime does not parse or validate XML. Content containing tag-like
    sequences passes through unmodified. This is NOT a security boundary.
    """

    codex_entries: list[CodexEntry] = Field(default_factory=list)
    # NOTE: 'handles' field is NOT defined here — factory pops it before model validation
