HANDLE_SYSTEM_CONTEXT = 'handle-system-context'
HANDLE_USER_MESSAGE = 'handle_user_message'
HANDLE_VOID = 'handle-void'
HANDLE_USER_MESSAGE_CONTEXT = 'handle-user-message-context'

# System event types - these are special types for execution flow control
# JSON can override these on a per-node basis via data.handles
SYSTEM_EVENT_STREAMING = 'content'  # Default type for streaming content to user
SYSTEM_EVENT_DEBUG = 'debug'
SYSTEM_EVENT_DEBUG_SUMMARY = 'debug_summary'

# Set of system event types that should not be treated as output handles
SYSTEM_EVENT_TYPES = frozenset({
    SYSTEM_EVENT_STREAMING,
    SYSTEM_EVENT_DEBUG,
    SYSTEM_EVENT_DEBUG_SUMMARY,
})
