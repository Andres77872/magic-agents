"""
EmitInterface for hook function output capabilities.

Provides emit.user() (concrete), emit.debug() (structure-only),
and emit.feedback() (placeholder) helpers for hook functions.

Contracts:
- emit.user(): Routes to SYSTEM_EVENT_STREAMING ('content')
- emit.debug(): Returns structure for SYSTEM_EVENT_DEBUG (no runtime integration)
- emit.feedback(): Placeholder dict (extras transport TBD)
"""

from __future__ import annotations

from datetime import datetime, UTC
from typing import TYPE_CHECKING, Any, Dict, Optional

from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

from magic_agents.util.const import SYSTEM_EVENT_STREAMING, SYSTEM_EVENT_DEBUG

if TYPE_CHECKING:
    from magic_agents.node_system.Node import Node


class EmitInterface:
    """Context object passed to hook functions providing emit capabilities.
    
    Hook functions receive this via context.emit field and can:
    - emit.user("message") → Send assistant message to client
    - emit.debug("metric", payload) → Return debug event structure
    - emit.feedback({"rating": 5}) → Return placeholder feedback dict
    
    Design Note: Node is injected only at runtime (not at import time)
    to avoid circular dependencies between hooks and node_system.
    """
    
    def __init__(self, node: Node, node_id: str):
        """Initialize EmitInterface with node reference and identity.
        
        Args:
            node: Node instance for prep() wrapping (injected at runtime).
            node_id: Node ID for context in debug/feedback events.
        """
        self._node = node
        self._node_id = node_id
    
    def user(self, message: str, extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Emit user-facing message to client (SYSTEM_EVENT_STREAMING channel).
        
        Creates a ChatCompletionModel with the message and routes it to
        the 'content' output channel for streaming to the client.
        
        This is the CONCRETE implementation for hook-to-client messaging.
        
        Args:
            message: Content string to send to the user/client.
            extras: Optional extras dict for ChatCompletionModel.extras field.
        
        Returns:
            Dict with 'type': 'content' (SYSTEM_EVENT_STREAMING) and
            'content': prep-wrapped ChatCompletionModel.
        
        Example:
            # In a NodeHook function:
            def my_hook(context, emit):
                result = emit.user("Processing step 1 complete")
                # result is routed to output_queue → SYSTEM_EVENT_STREAMING
        
        Integration:
            - Executor routes dict['type'] == 'content' to streaming output
            - Message appears in client's streaming response
        """
        # Create streaming-style ChatCompletionModel
        chat_message = ChatCompletionModel(
            id='',  # Empty ID for hook-emitted messages (not from LLM)
            model='',  # Empty model (not from LLM provider)
            choices=[
                ChoiceModel(
                    delta=DeltaModel(
                        content=message,
                        role='assistant'
                    )
                )
            ],
            extras=extras or {}
        )
        
        # Use node's prep() method for wrapping (consistent with other nodes)
        wrapped_content = self._node.prep(chat_message)
        
        return {
            'type': SYSTEM_EVENT_STREAMING,  # 'content'
            'content': wrapped_content
        }
    
    def debug(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Emit debug event structure (SYSTEM_EVENT_DEBUG format).
        
        Returns a structured dict for debug events WITHOUT runtime integration.
        The integration with debug system (EmitterRegistry) is TBD.
        
        This is a STRUCTURE-ONLY contract per design decision.
        
        Args:
            event_type: Custom event type identifier (e.g., 'CUSTOM_METRIC').
            payload: Event data dict with custom fields.
        
        Returns:
            Dict with 'type': 'debug' and 'content' containing:
            - event_type: Custom type identifier
            - node_id: Source node ID
            - timestamp: ISO format timestamp
            - payload fields merged into content
        
        Example:
            # In a NodeHook function:
            def my_hook(context, emit):
                result = emit.debug("PROGRESS", {"step": 2, "percent": 50})
                # Returns structure, executor routes to debug channel (TBD)
        
        Integration TBD:
            - Current: Returns dict structure only
            - Future: Executor routes to debug panel via EmitterRegistry
        """
        return {
            'type': SYSTEM_EVENT_DEBUG,  # 'debug'
            'content': {
                'event_type': event_type,
                'node_id': self._node_id,
                'timestamp': datetime.now(UTC).isoformat(),
                **payload
            }
        }
    
    def feedback(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for future feedback mechanism.
        
        Returns a placeholder dict structure without extras propagation.
        The transport/path for extras remains an extension point.
        
        This is a PLACEHOLDER CONTRACT per design decision.
        
        Args:
            event_data: Feedback event data (e.g., {"rating": 5, "comment": "great"}).
        
        Returns:
            Dict with 'type': 'feedback' and 'content' containing:
            - event_type: 'FEEDBACK'
            - node_id: Source node ID
            - event_data fields merged into content
        
        Example:
            # In a NodeHook function:
            def my_hook(context, emit):
                result = emit.feedback({"rating": 5, "approved": True})
                # Returns placeholder, extras propagation TBD
        
        Integration TBD:
            - extras propagation mechanism to be designed in future
            - placeholder provides stable interface for hook functions
        """
        return {
            'type': 'feedback',  # Custom handle (not a system event)
            'content': {
                'event_type': 'FEEDBACK',
                'node_id': self._node_id,
                **event_data
            }
        }