from typing import Optional, Any

from pydantic import ConfigDict

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
    memory: Optional[dict[str, Any]] = None  # {stm, ltm, max_input_tokens}
    handles: Optional[dict[str, str]] = None  # Handle name overrides