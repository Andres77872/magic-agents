"""Library-owned graph FlowHooks persistence implementation.

This module owns generic graph lifecycle persistence for magic-agents. Concrete
database/repository behavior stays behind an injected sink/port so this package
does not import api.magic_llm runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from magic_agents.hooks.flow_hooks import HookContext


class ExecutionPersistencePort(Protocol):
    async def begin_run(
        self,
        *,
        id_conversation: str,
        run_type: str,
        id_agent: str | None,
        run_meta: dict[str, Any] | None,
    ) -> str: ...

    async def begin_execution(
        self,
        *,
        id_run: str,
        parent_execution_id: str | None,
        execution_kind: str,
        execution_name: str | None = None,
        execution_meta: dict[str, Any] | None = None,
        nested_depth: int = 0,
        nested_request_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> str: ...

    async def record_event(
        self,
        *,
        id_execution: str,
        event_type: str,
        event_payload: dict[str, Any] | None = None,
    ) -> str: ...

    async def record_usage_fact(
        self,
        *,
        id_execution: str,
        id_run: str,
        provider: str,
        model_alias: str,
        usage_data: dict[str, Any],
        price_snapshot: dict[str, Any],
        call_source: str,
        id_message: str | None,
        provider_request_id: str | None = None,
    ) -> tuple[str, float]: ...

    async def complete_execution(self, *, id_execution: str, status: str) -> None: ...

    async def complete_run(self, *, id_run: str, status: str) -> None: ...


@dataclass(frozen=True)
class AssistantMessageContext:
    """Persisted assistant message identity for usage/message correlation."""

    id_message: str
    id_conversation: str
    id_user_message: str
    id_agent: str


class GraphPersistenceHook:
    """FlowHooks implementation backed by an injected execution persistence port."""

    def __init__(
        self,
        *,
        sink: ExecutionPersistencePort,
        id_chat: str,
        id_thread: str,
        id_user: str,
        id_agent: str | None = None,
        assistant_message: AssistantMessageContext | None = None,
        nested_depth: int = 0,
        nested_request_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self._sink = sink
        self._id_chat = id_chat
        self._id_thread = id_thread
        self._id_user = id_user
        self._id_agent = id_agent
        self._assistant_message = assistant_message
        self._nested_depth = nested_depth
        self._nested_request_id = nested_request_id
        self._parent_run_id = parent_run_id

        self._run_id = ""
        self._root_execution_id = ""
        self._node_execution_ids: dict[str, str] = {}
        self._llm_execution_id = ""
        self._tool_execution_ids: list[str] = []
        self._tool_execution_id_by_tool_call: dict[str, str] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def root_execution_id(self) -> str:
        return self._root_execution_id

    def _validate_identity(self) -> None:
        missing = [
            name
            for name, value in (
                ("id_chat", self._id_chat),
                ("id_thread", self._id_thread),
                ("id_user", self._id_user),
            )
            if not str(value or "").strip()
        ]
        if missing:
            raise ValueError(f"Blank persistence identity: {', '.join(missing)}")

    def _reset(self) -> None:
        self._run_id = ""
        self._root_execution_id = ""
        self._node_execution_ids.clear()
        self._llm_execution_id = ""
        self._tool_execution_ids.clear()
        self._tool_execution_id_by_tool_call.clear()

    def consume_tool_execution_id_mapping(self) -> dict[str, str]:
        """Return tool_call_id→execution_id mapping and clear internal storage.

        One-shot consumption: returns a copy, then clears internal dict.
        Safe for multiple calls (second call returns empty dict).
        Never raises.
        """
        result = dict(self._tool_execution_id_by_tool_call)
        self._tool_execution_id_by_tool_call.clear()
        return result

    @staticmethod
    def _error_payload(error: Exception) -> dict[str, str]:
        return {"error_type": type(error).__name__, "error_message": str(error)}

    async def on_graph_start(self, context: HookContext) -> None:
        self._validate_identity()
        self._reset()
        run_meta = {
            "id_chat": self._id_chat,
            "id_thread": self._id_thread,
            "id_user": self._id_user,
            "id_agent": self._id_agent,
        }
        self._run_id = await self._sink.begin_run(
            id_conversation=self._id_chat,
            run_type="agent",
            id_agent=self._id_agent,
            run_meta=run_meta,
        )
        self._root_execution_id = await self._sink.begin_execution(
            id_run=self._run_id,
            parent_execution_id=None,
            execution_kind="run_root",
            execution_name=f"agent_{self._id_agent}" if self._id_agent else "agent",
            execution_meta={**run_meta, **(context.inputs or {})},
            nested_depth=self._nested_depth,
            nested_request_id=self._nested_request_id,
            parent_run_id=self._parent_run_id,
        )
        await self._sink.record_event(
            id_execution=self._root_execution_id,
            event_type="graph_start",
            event_payload=run_meta,
        )

    async def on_graph_end(self, context: HookContext) -> None:
        if not self._run_id or not self._root_execution_id:
            return
        await self._sink.record_event(
            id_execution=self._root_execution_id,
            event_type="graph_end",
            event_payload={"duration_ms": context.duration_ms} if context.duration_ms is not None else None,
        )
        await self._sink.complete_execution(id_execution=self._root_execution_id, status="completed")
        await self._sink.complete_run(id_run=self._run_id, status="completed")

    async def on_graph_error(self, context: HookContext, error: Exception) -> None:
        if not self._run_id or not self._root_execution_id:
            return
        await self._sink.record_event(
            id_execution=self._root_execution_id,
            event_type="error",
            event_payload=self._error_payload(error),
        )
        await self._sink.complete_execution(id_execution=self._root_execution_id, status="failed")
        await self._sink.complete_run(id_run=self._run_id, status="failed")

    async def on_node_start(self, context: HookContext) -> None:
        if not self._run_id:
            return
        node_id = context.node_id or context.inputs.get("node_id") or "unknown"
        execution_id = await self._sink.begin_execution(
            id_run=self._run_id,
            parent_execution_id=self._root_execution_id or None,
            execution_kind="node",
            execution_name=node_id,
            execution_meta={"node_type": context.node_type, "node_class": context.node_class},
            nested_depth=self._nested_depth,
        )
        self._node_execution_ids[node_id] = execution_id
        await self._sink.record_event(
            id_execution=execution_id,
            event_type="node_start",
            event_payload={"node_id": node_id, "node_type": context.node_type, "node_class": context.node_class},
        )

    async def on_node_end(self, context: HookContext) -> None:
        node_id = context.node_id or "unknown"
        execution_id = self._node_execution_ids.pop(node_id, None)
        if not execution_id:
            return
        await self._sink.record_event(
            id_execution=execution_id,
            event_type="node_end",
            event_payload={"duration_ms": context.duration_ms} if context.duration_ms is not None else None,
        )
        await self._sink.complete_execution(id_execution=execution_id, status="completed")

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        node_id = context.node_id or "unknown"
        execution_id = self._node_execution_ids.pop(node_id, None)
        if not execution_id:
            return
        await self._sink.record_event(
            id_execution=execution_id,
            event_type="error",
            event_payload=self._error_payload(error),
        )
        await self._sink.complete_execution(id_execution=execution_id, status="failed")

    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        if not self._root_execution_id:
            return
        await self._sink.record_event(
            id_execution=self._root_execution_id,
            event_type="node_bypass",
            event_payload={"node_id": context.node_id, "reason": reason},
        )

    async def on_llm_start(self, context: HookContext, llm_config: dict[str, Any] | None = None) -> None:
        if not self._run_id:
            return
        node_id = context.node_id or ""
        parent_execution_id = self._node_execution_ids.get(node_id) or self._root_execution_id or None
        model = (llm_config or {}).get("model") or context.inputs.get("model") or "unknown"
        provider = (llm_config or {}).get("provider") or context.inputs.get("provider") or "unknown"
        self._llm_execution_id = await self._sink.begin_execution(
            id_run=self._run_id,
            parent_execution_id=parent_execution_id,
            execution_kind="llm_request",
            execution_name=model,
            execution_meta={"model": model, "provider": provider},
            nested_depth=self._nested_depth + 1,
        )
        await self._sink.record_event(
            id_execution=self._llm_execution_id,
            event_type="llm_start",
            event_payload={"model": model, "provider": provider},
        )

    async def on_llm_end(self, context: HookContext, response: dict[str, Any] | None = None) -> None:
        if not self._run_id or not self._llm_execution_id:
            return
        data = response or context.outputs or {}
        usage_data = {
            "prompt_tokens": data.get("prompt_tokens", 0) or 0,
            "completion_tokens": data.get("completion_tokens", 0) or 0,
            "total_tokens": data.get("total_tokens", 0) or 0,
            "call_sequence": context.sequence_number,
        }
        await self._sink.record_usage_fact(
            id_execution=self._llm_execution_id,
            id_run=self._run_id,
            provider=data.get("provider") or context.inputs.get("provider") or "unknown",
            model_alias=data.get("model") or context.inputs.get("model") or "unknown",
            usage_data=usage_data,
            price_snapshot={
                "input_price_mtok": context.inputs.get("input_price_mtok", 0),
                "output_price_mtok": context.inputs.get("output_price_mtok", 0),
            },
            call_source="hook",
            id_message=self._assistant_message.id_message if self._assistant_message else None,
            provider_request_id=data.get("provider_request_id"),
        )
        await self._sink.record_event(
            id_execution=self._llm_execution_id,
            event_type="llm_end",
            event_payload={"finish_reason": data.get("finish_reason"), "total_tokens": usage_data["total_tokens"]},
        )
        await self._sink.complete_execution(id_execution=self._llm_execution_id, status="completed")
        self._llm_execution_id = ""

    async def on_llm_loop_end(self, context: HookContext) -> None:
        if self._root_execution_id:
            await self._sink.record_event(
                id_execution=self._root_execution_id,
                event_type="iteration_end",
                event_payload=context.outputs or None,
            )

    async def on_tool_start(self, context: HookContext) -> None:
        if not self._run_id:
            return
        tool_name = context.inputs.get("tool_name") or "unknown"
        tool_call_id = context.inputs.get("tool_call_id")
        execution_id = await self._sink.begin_execution(
            id_run=self._run_id,
            parent_execution_id=self._llm_execution_id or self._root_execution_id or None,
            execution_kind="tool_call",
            execution_name=tool_name,
            execution_meta={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": context.inputs.get("arguments"),
            },
            nested_depth=self._nested_depth + 1,
        )
        self._tool_execution_ids.append(execution_id)
        if tool_call_id:
            self._tool_execution_id_by_tool_call[tool_call_id] = execution_id
        await self._sink.record_event(
            id_execution=execution_id,
            event_type="tool_start",
            event_payload={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": context.inputs.get("arguments"),
            },
        )

    async def on_tool_end(self, context: HookContext) -> None:
        if not self._tool_execution_ids:
            return
        execution_id = self._tool_execution_ids.pop()
        failed = bool(context.error_message)
        await self._sink.record_event(
            id_execution=execution_id,
            event_type="tool_end",
            event_payload={**(context.outputs or {}), "error": context.error_message},
        )
        await self._sink.complete_execution(id_execution=execution_id, status="failed" if failed else "completed")


# Compatibility alias for consumers migrating from api.magic_llm naming.
PersistenceHook = GraphPersistenceHook
