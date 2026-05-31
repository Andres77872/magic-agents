from typing import Literal, Optional, Any

from pydantic import ConfigDict, Field

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class ChatNodeModel(BaseNodeModel):
    """
    Chat node model - replaces None model in node_map.
    Backend-authoritative validation for session configuration.
    
    The JSON definition is the source of truth.
    Session management aligns with backend thread persistence contract.
    
    BACKEND-AUTHORITATIVE ARCHITECTURE:
    - Backend injects persisted + runtime history via `history_messages` field
    - NodeChat does NOT load persisted history from DB - backend prepares it
    - NodeChat composes additional layers on top of backend-provided base
    """
    
    # Session configuration
    session_id: Optional[str] = None  # Thread/conversation ID for persistence
    session_required: bool = False  # If True, enforce session presence (auto-create fallback)
    
    # History configuration - BACKEND-AUTHORITATIVE
    # history_messages: Backend-injected persisted + runtime messages (Slot 1)
    # This field is populated by backend via build() call, NOT from JSON config
    history_messages: Optional[list[dict[str, Any]]] = None  # Backend-authoritative history base
    custom_messages: Optional[list[dict[str, Any]]] = None  # Pre-user context injection (Slot 3)
    messages_append_mode: bool = False  # False=legacy REPLACE, True=APPEND
    
    # Existing fields (legacy compatibility)
    message: Optional[str] = None  # Kept for backward compat; not primary input
    memory: Optional[dict[str, Any]] = None  # {stm, ltm, max_input_tokens} — DEPRECATED for max_input_tokens
    handles: Optional[dict[str, str]] = None  # Handle name overrides

    # STM windowing fields
    max_messages: Optional[int] = Field(
        default=None, ge=1,
        description="Maximum non-system conversation messages to keep. "
                    "System messages are always preserved and NOT counted. "
                    "Applied after 5-slot merge, before yield."
    )
    max_input_tokens: Optional[int] = Field(
        default=None, ge=1,
        description="Maximum estimated tokens for the assembled chat. "
                    "Applied AFTER max_messages. When None and legacy "
                    "memory.max_input_tokens is set, falls back to legacy value."
    )
    truncation_strategy: Literal['tail', 'token_budget'] = Field(
        default='tail',
        description="Windowing strategy: 'tail' (keep newest N), "
                    "'token_budget' (token-aware selection)."
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name for usage/logging tracking. "
                    "Does NOT affect STM windowing token estimation, "
                    "which uses a hardcoded GPT-5 tokenizer. "
                    "Optional — purely informational."
    )