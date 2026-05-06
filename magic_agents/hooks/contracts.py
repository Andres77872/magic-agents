"""
Hook Runtime Contracts — TypedDict schemas per event type.

Provides per-event-type TypedDict schemas for all 11 FlowHooks methods'
input payloads. These schemas document the required and optional fields
for each event type's HookContext.inputs dict.

Usage:
    from magic_agents.hooks.contracts import GraphStartInputs, LLMStartInputs

    # Type-check: inputs conform to GraphStartInputs
    ctx: GraphStartInputs = {
        "graph_id": "abc",
        "node_count": 3,
        "edge_count": 2,
    }

Schemas are runtime-checkable (TypedDict with total=False) but NOT enforced
at the Protocol level for backward compatibility. They serve as documentation
and optional validation via HookContextFactory.

Design decision: total=False means all fields are optional at the type level,
but the spec defines which are required vs optional. Required fields MUST be
present when the factory constructs a context. Optional fields may be absent.
"""

from typing import TypedDict, Optional, Literal, Dict, Any, List


# ─── Canonical Types ────────────────────────────────────────────────────────

BypassReason = Literal["upstream_error", "condition", "not_ready"]
"""Canonical 3-value bypass reason type (M3).

- "upstream_error": Downstream node skipped because an upstream node failed.
- "condition": Node's condition evaluated to false (static or iteration).
- "not_ready": Node inputs were not ready for execution (single-node bypass).
"""


# ─── Graph Lifecycle ────────────────────────────────────────────────────────

class GraphStartInputs(TypedDict, total=False):
    """Required: graph_id, node_count, edge_count."""
    graph_id: str
    node_count: int
    edge_count: int


class GraphEndInputs(TypedDict, total=False):
    """Required: graph_id, execution_time_ms."""
    graph_id: str
    execution_time_ms: float


class GraphErrorInputs(TypedDict, total=False):
    """Required: graph_id, error_message, error_type."""
    graph_id: str
    error_message: str
    error_type: str


# ─── Node Lifecycle ─────────────────────────────────────────────────────────

class NodeStartInputs(TypedDict, total=False):
    """Required: node_id, node_type, node_class, input_keys."""
    node_id: str
    node_type: str
    node_class: str
    input_keys: List[str]


class NodeEndInputs(TypedDict, total=False):
    """Required: node_id, duration_ms, output_keys."""
    node_id: str
    duration_ms: float
    output_keys: List[str]


class NodeBypassInputs(TypedDict, total=False):
    """Required: node_id, reason (BypassReason).

    Optional: upstream_error_node (str) — present when reason="upstream_error".
    Fine-grained path info (phase, bypass_path) is carried in metadata, not inputs.
    """
    node_id: str
    reason: BypassReason
    upstream_error_node: Optional[str]


# ─── LLM Lifecycle ──────────────────────────────────────────────────────────

class LLMStartInputs(TypedDict, total=False):
    """Required: model, streaming.  Optional: iteration, llm_call_count.

    Added: provider, tools, tool_choice, deduplicate — populated from
    HookRelay path when config data is available (may be None).
    """
    model: str
    streaming: bool
    iteration: Optional[int]
    llm_call_count: Optional[int]
    provider: Optional[str]            # NEW — provider name (e.g., "openai")
    tools: Optional[list]              # NEW — tool specifications
    tool_choice: Optional[str | dict | None]  # NEW — tool choice strategy
    deduplicate: Optional[bool]        # NEW — deduplication flag


class LLMEndInputs(TypedDict, total=False):
    """Required: model, content, content_preview, finish_reason.

    Added: provider_request_id, prompt_tokens, completion_tokens,
    total_tokens, iteration (0-indexed).
    Fires per-provider-request (N times for N-iteration loop).
    loop_complete discriminator REMOVED — use on_llm_loop_end instead.
    """
    model: str
    content: str
    content_preview: str
    finish_reason: Optional[str]
    provider_request_id: Optional[str]  # NEW — from response.id
    prompt_tokens: Optional[int]        # NEW — from response.usage
    completion_tokens: Optional[int]    # NEW — from response.usage
    total_tokens: Optional[int]         # NEW — from response.usage
    iteration: int                      # NEW — 0-indexed iteration


class LLMLoopEndInputs(TypedDict, total=False):
    """Aggregated loop completion context for on_llm_loop_end.

    Carries accumulated content from ALL iterations/generations.
    Fires exactly once per loop completion (streaming AND non-streaming).

    iteration: 0-indexed final iteration number.
    total_iterations: 1-indexed count (total iterations executed).
    """
    model: str
    content: str
    content_preview: str
    finish_reason: Optional[str]
    iteration: int                          # 0-indexed
    total_iterations: int                   # 1-indexed count
    provider_request_id: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


# ─── Tool Lifecycle ─────────────────────────────────────────────────────────

class ToolStartInputs(TypedDict, total=False):
    """Required: tool_name, tool_call_id, arguments."""
    tool_name: str
    tool_call_id: str
    arguments: Dict[str, Any]


class ToolEndInputs(TypedDict, total=False):
    """Required: tool_name, success.

    Optional: execution_time_ms (float) — populated from result.duration_ms (S3).
    """
    tool_name: str
    success: bool
    execution_time_ms: Optional[float]


# ─── Edge Hook ──────────────────────────────────────────────────────────────

class EdgeHookInputs(TypedDict, total=False):
    """Edge-level routing data for NodeHook dispatch via EdgeHookConfig.

    Carries unique handle-level routing data NOT available from graph/node hooks.
    """
    content: Any
    source: str
    target: str
    source_handle: Optional[str]
    target_handle: Optional[str]


# ─── FlowHooks Method → Schema Mapping ──────────────────────────────────────
#
# | FlowHooks Method     | Input Schema        | Notes                          |
# |----------------------|---------------------|--------------------------------|
# | on_graph_start       | GraphStartInputs    | graph-level metadata           |
# | on_graph_end         | GraphEndInputs      | duration + graph_id            |
# | on_graph_error       | GraphErrorInputs    | error info + graph_id          |
# | on_node_start        | NodeStartInputs     | node identity + input_keys     |
# | on_node_end          | NodeEndInputs       | node identity + duration       |
# | on_node_bypass       | NodeBypassInputs    | reason + upstream_error        |
# | on_llm_start         | LLMStartInputs      | model config + iteration,      |
# |                      |                     | llm_config populated with      |
# |                      |                     | real data when available       |
# | on_llm_end           | LLMEndInputs        | model output + finish,         |
# |                      |                     | + provider_request_id, tokens, |
# |                      |                     | iteration (0-indexed)          |
# | on_llm_loop_end      | LLMLoopEndInputs    | NEW — aggregated loop          |
# |                      |                     | completion (once per loop)     |
# | on_tool_start        | ToolStartInputs     | tool call + arguments          |
# | on_tool_end          | ToolEndInputs       | tool result + duration         |
# | (edge via NodeHook)  | EdgeHookInputs      | routing data only              |
