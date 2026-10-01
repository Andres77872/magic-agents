import json
import logging
from typing import Any, Optional

from magic_llm.model import ModelChat

from magic_agents.models.factory.Nodes.ChatNodeModel import ChatNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.node_system.chat_attachments import ChatAttachmentProcessor, MAX_CHAT_FILES
from magic_agents.node_system.utils import apply_windowing

logger = logging.getLogger(__name__)


class NodeChat(Node):
    """
    Chat node - handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    
    Merge order (per backend contract):
    1. Persisted session history (from backend via chat_log.id_chat)
    2. Runtime messages input (handle_messages) - APPEND or REPLACE based on mode
    3. custom_messages from config - APPEND
    4. System context - INSERT at index 0
    5. User message - APPEND (final slot)
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_SYSTEM_CONTEXT = 'handle-system-context'
    DEFAULT_INPUT_USER_MESSAGE = 'handle_user_message'
    DEFAULT_INPUT_MESSAGES = 'handle_messages'
    DEFAULT_INPUT_USER_FILES = 'handle_user_files'
    DEFAULT_INPUT_USER_IMAGES = 'handle_user_images'
    # Output handle
    DEFAULT_OUTPUT_HANDLE = 'handle_chat_output'

    def __init__(self, data: ChatNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        """
        Initialize Chat node with validated ChatNodeModel.
        
        Args:
            data: ChatNodeModel instance with validated session configuration
            
        BACKEND-AUTHORITATIVE ARCHITECTURE:
        - Backend injects persisted + runtime history via `data.history_messages`
        - NodeChat does NOT load persisted history from DB - backend prepares it
        - NodeChat composes additional layers on top of backend-provided base
        """
        file_cache = kwargs.pop('file_cache', None)
        self._attachment_processor = ChatAttachmentProcessor(
            redis_config=data.file_cache_redis, cache=file_cache,
            ttl_seconds=data.file_cache_ttl_seconds,
        )
        super().__init__(**kwargs)
        
        # Session configuration from validated model
        self._session_id = data.session_id
        self._session_required = data.session_required
        self._messages_append_mode = data.messages_append_mode
        self._custom_messages = data.custom_messages or []
        
        # BACKEND-AUTHORITATIVE: History messages from backend (Slot 1)
        # Backend prepares persisted + runtime history and passes via build()
        self._history_messages = data.history_messages or []
        
        # Legacy fields (backward compatibility)
        self._memory = data.memory or {}
        if 'stm' in self._memory:
            logger.warning(
                "NodeChat:%s: 'memory.stm' is deprecated and ignored. "
                "Use 'max_messages' instead.", self.node_id
            )

        # STM windowing fields
        self._max_messages = data.max_messages
        self._truncation_strategy = data.truncation_strategy

        # Resolve max_input_tokens with precedence (C2):
        # 1. New first-class field
        # 2. Legacy memory dict fallback + deprecation warning
        # 3. None
        if data.max_input_tokens is not None:
            self._max_input_tokens = data.max_input_tokens
        elif data.memory and 'max_input_tokens' in data.memory:
            self._max_input_tokens = data.memory['max_input_tokens']
            logger.warning(
                "NodeChat:%s: 'memory.max_input_tokens' is deprecated. "
                "Use 'max_input_tokens' directly.", self.node_id
            )
        else:
            self._max_input_tokens = None

        # Windowing diagnostics counters
        self._messages_before_windowing = 0
        self._messages_discarded = 0
        self._total_tokens_estimated = None

        # Handle name overrides from validated model
        handles = handles if handles is not None else (data.handles or {})
        self.INPUT_HANDLER_SYSTEM_CONTEXT = handles.get('system_context', handles.get('system', self.DEFAULT_INPUT_SYSTEM_CONTEXT))
        self.INPUT_HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_INPUT_USER_MESSAGE))
        self.INPUT_HANDLER_MESSAGES = handles.get('messages', self.DEFAULT_INPUT_MESSAGES)
        self.INPUT_HANDLER_USER_FILES = handles.get('user_files', handles.get('files', self.DEFAULT_INPUT_USER_FILES))
        self.INPUT_HANDLER_USER_IMAGES = handles.get('user_images', handles.get('images', self.DEFAULT_INPUT_USER_IMAGES))
        # Output handle
        self.OUTPUT_HANDLE = handles.get('output', handles.get('chat', self.DEFAULT_OUTPUT_HANDLE))
        
        # Initialize empty ModelChat - backend loads history via chat_log.id_chat
        # Uses resolved self._max_input_tokens (Layer 2 safety net)
        self.chat = ModelChat(max_input_tokens=self._max_input_tokens)

    @staticmethod
    def _decode_attachment_input(value: Any) -> list[Any]:
        """Normalize scalar, list, or JSON-encoded attachment input."""
        if value is None:
            return []
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            if isinstance(decoded, list):
                return decoded
            return [decoded]
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _normalize_image_reference(value: Any) -> str | bytes | list[str | bytes]:
        """Return a provider-compatible image reference from common envelopes."""
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, list) and all(isinstance(item, (str, bytes)) for item in value):
            return value
        if isinstance(value, dict):
            image_url = value.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            reference = value.get("url") or value.get("image") or image_url
            if isinstance(reference, (str, bytes)):
                return reference
        raise ValueError(
            "Image entries must be strings, bytes, lists of image references, "
            "or mappings containing url/image/image_url."
        )

    @classmethod
    def _normalize_file_descriptors(
        cls,
        values: list[Any],
    ) -> list[tuple[str, str | bytes | list[str | bytes] | None]]:
        """Normalize file inputs already converted to prompt text and optional images.

        Keep established ``[text, image]`` pairs and equivalent mappings
        (``text``/``content`` with optional image/url) working alongside real
        raw-file descriptors processed by ``ChatAttachmentProcessor``.
        """
        descriptors: list[tuple[str, str | bytes | list[str | bytes] | None]] = []
        for index, value in enumerate(values):
            text: Any
            image: Any = None
            if isinstance(value, (list, tuple)) and len(value) == 2:
                text, image = value
            elif isinstance(value, dict):
                text = value.get("content", value.get("text", value.get("message")))
                image = value.get("image", value.get("url", value.get("image_url")))
            else:
                raise ValueError(
                    f"File entry {index} must be a [text, image] pair or a mapping "
                    "with text/content and an optional image/url."
                )

            if not isinstance(text, str) or not text:
                raise ValueError(f"File entry {index} requires non-empty text/content.")
            normalized_image = None if image is None else cls._normalize_image_reference(image)
            descriptors.append((text, normalized_image))
        return descriptors

    async def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject file inputs into provider messages; keep caller history unmodified.

        API clients can reuse this NodeChat boundary for chats without a graph.
        Files stay descriptors in persisted history and are parsed here when the
        provider messages are assembled. Owned Redis clients are always closed.
        """
        prepared: list[dict[str, Any]] = []
        try:
            for original in messages:
                message = dict(original)
                files = self._decode_attachment_input(message.pop('files', None))
                images = self._decode_attachment_input(message.pop('images', None))
                if len(files) > MAX_CHAT_FILES:
                    raise ValueError(f'Chat supports at most {MAX_CHAT_FILES} file attachments per message.')
                if not files and not images:
                    prepared.append(message)
                    continue
                if message.get('role') != 'user':
                    raise ValueError('File and image attachments must belong to a user message.')
                for value in files:
                    if isinstance(value, dict) and (any(key in value for key in ('data', 'mime_type'))
                            or ('name' in value and not any(key in value for key in ('text', 'content', 'message')))):
                        file_text = await self._attachment_processor.parse_file(value)
                        prepared.append({'role': 'user', 'content': file_text})
                    else:
                        # Preserve previously extracted [text, image] inputs.
                        file_text, file_image = self._normalize_file_descriptors([value])[0]
                        temporary = ModelChat()
                        temporary.add_user_message(file_text, file_image)
                        prepared.extend(temporary.messages)
                content = message.get('content', '')
                if images:
                    pair_like = [
                        isinstance(item, (list, tuple, dict))
                        and not (isinstance(item, dict)
                                 and any(key in item for key in ('url', 'image', 'image_url'))
                                 and not any(key in item for key in ('text', 'content', 'message')))
                        for item in images
                    ]
                    if any(pair_like) and not all(pair_like):
                        raise ValueError('Image entries must not mix image references with legacy file descriptors.')
                    if all(pair_like):
                        for file_text, file_image in self._normalize_file_descriptors(images):
                            temporary = ModelChat()
                            temporary.add_user_message(file_text, file_image)
                            prepared.extend(temporary.messages)
                        if content:
                            prepared.append(message)
                    else:
                        normalized_images = [self._normalize_image_reference(image) for image in images]
                        if isinstance(content, list):
                            # Preserve already normalized image/text content from history.
                            prepared.append({**message, 'content': [*content, *[
                                {'type': 'image_url', 'image_url': {'url': image}}
                                for image in normalized_images
                            ]]})
                        elif isinstance(content, str) or content is None:
                            temporary = ModelChat()
                            temporary.add_user_message(content or ' ', normalized_images)
                            multimodal = temporary.messages[0]['content']
                            if not content:
                                multimodal = [part for part in multimodal if part.get('type') != 'text']
                            prepared.append({**message, 'content': multimodal})
                        else:
                            raise ValueError('User message content must be text or multimodal content.')
                elif content:
                    prepared.append(message)
            return prepared
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Close only connections owned by this node; injected cache stays caller-owned."""
        await self._attachment_processor.aclose()

    async def process(self, chat_log):
        """
        Process chat node with merge logic aligned with backend contract.
        
        Merge order (BACKEND-AUTHORITATIVE ARCHITECTURE):
        1. history_messages from backend (persisted + runtime) - injected via build()
        2. Runtime messages input (APPEND or REPLACE based on mode) - from input handles
        3. custom_messages from config (APPEND)
        4. System context (INSERT at index 0)
        5. User message (APPEND, final slot)
        
        Args:
            chat_log: ModelAgentRunLog with id_chat for session context
        """
        # Slot 1: Base messages from backend-authoritative history
        # Backend loads persisted history + runtime messages and passes via build()
        # NodeChat does NOT load from DB - backend is authoritative source
        base_messages = list(self._history_messages)  # Copy to avoid mutation
        
        if base_messages:
            logger.debug("NodeChat:%s starting with %d backend-injected history messages", 
                        self.node_id, len(base_messages))
        
        # Log session context for debugging
        session_id = self._session_id or (chat_log.id_chat if chat_log else None)
        if session_id:
            logger.debug("NodeChat:%s using session_id=%s for thread context", self.node_id, session_id)
        
        # Slot 2: Handle runtime messages input
        if c := self.get_input(self.INPUT_HANDLER_MESSAGES):
            if self._messages_append_mode:
                # APPEND mode: extend base messages (preserves history)
                logger.debug("NodeChat:%s appending runtime messages (append_mode=True)", self.node_id)
                base_messages.extend(c)
            else:
                # REPLACE mode: legacy semantics (overwrite history)
                logger.debug("NodeChat:%s replacing messages with runtime input (append_mode=False)", self.node_id)
                base_messages = list(c)  # Create new list to avoid mutation
        
        # Slot 3: Append custom_messages from config
        if self._custom_messages:
            logger.debug("NodeChat:%s appending %d custom_messages", self.node_id, len(self._custom_messages))
            base_messages.extend(self._custom_messages)
        
        # Build ModelChat with merged messages
        self.chat.messages = base_messages
        
        # Slot 4: System context (INSERT at index 0)
        if c := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
            logger.debug("NodeChat:%s setting system context", self.node_id)
            self.chat.set_system(c)
        
        # Slot 5: Files are real inputs; NodeChat injects their parsed content.
        current_text = self.get_input(self.INPUT_HANDLER_USER_MESSAGE)
        raw_images = self.get_input(self.INPUT_HANDLER_USER_IMAGES)
        raw_files = self.get_input(self.INPUT_HANDLER_USER_FILES)
        images: list[Any] = []
        files: list[Any] = []
        self._attachment_processor.warnings.clear()
        try:
            images = self._decode_attachment_input(raw_images)
            files = self._decode_attachment_input(raw_files)
            if current_text or files or images:
                self.chat.messages.append({'role': 'user', 'content': current_text or '',
                                           'files': files, 'images': images})
            self.chat.messages = await self.prepare_messages(self.chat.messages)
        except (TypeError, ValueError) as exc:
            logger.error("NodeChat:%s invalid attachment input: %s", self.node_id, exc)
            yield self.yield_debug_error(
                error_type="ValidationError", error_message=str(exc),
                context={"images_handle": self.INPUT_HANDLER_USER_IMAGES,
                         "files_handle": self.INPUT_HANDLER_USER_FILES,
                         "images_count": len(images), "files_count": len(files)},
            )
            return
        for warning in self._attachment_processor.warnings:
            diagnostic = self.yield_debug_error('FileCacheWarning', warning)
            diagnostic['content']['severity'] = 'warning'
            yield diagnostic

        # Post-merge windowing (Layer 1 - PRIMARY)
        if self._max_messages is not None or self._max_input_tokens is not None:
            self._messages_before_windowing = len(self.chat.messages)
            self.chat.messages = apply_windowing(
                messages=self.chat.messages,
                max_messages=self._max_messages,
                max_input_tokens=self._max_input_tokens,
                truncation_strategy=self._truncation_strategy,
            )
            self._messages_discarded = self._messages_before_windowing - len(self.chat.messages)

        logger.info("NodeChat:%s chat prepared with %d messages (session=%s, append_mode=%s)", 
                   self.node_id, len(self.chat.messages), session_id, self._messages_append_mode)
        yield self.yield_static(self.chat, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Chat-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Chat-specific variables
        if hasattr(self, 'chat') and self.chat:
            messages = getattr(self.chat, 'messages', [])
            state['messages_count'] = len(messages)
            # Check if system message exists in messages list
            state['has_system_message'] = any(
                msg.get('role') == 'system' for msg in messages if isinstance(msg, dict)
            )
        
        # Capture session configuration
        state['session_id'] = self._session_id
        state['session_required'] = self._session_required
        state['messages_append_mode'] = self._messages_append_mode
        state['custom_messages_count'] = len(self._custom_messages)
        state['file_cache_configured'] = bool(self._attachment_processor.redis_config or self._attachment_processor.cache is not None)
        state['file_cache_warnings'] = list(self._attachment_processor.warnings)
        
        # Capture memory configuration
        state['memory'] = self._memory
        
        # STM windowing diagnostics (N3)
        state['max_messages'] = self._max_messages
        state['max_input_tokens'] = self._max_input_tokens
        state['truncation_strategy'] = self._truncation_strategy
        state['messages_before_windowing'] = self._messages_before_windowing
        state['messages_after_windowing'] = len(self.chat.messages) if self.chat else 0
        state['messages_discarded'] = self._messages_discarded
        state['total_tokens_estimated'] = self._total_tokens_estimated
        state['windowing_applied'] = (
            self._max_messages is not None or self._max_input_tokens is not None
        )
        
        return state
