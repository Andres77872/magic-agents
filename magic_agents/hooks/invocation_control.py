"""Invocation controls for real graph Nodes and registered tool operations.

The reactive dispatcher owns this adapter. It invokes the existing NodeHook
templates, Node.process generators, and FetchToolCallable HTTP implementation;
it does not build or schedule a second graph runtime.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import time
import uuid
from datetime import datetime, UTC
from types import MappingProxyType

from magic_agents.hooks.context_factory import HookContextFactory


def snapshot(value):
    def check(item, ancestors):
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (list, dict) or id(item) in ancestors:
            raise ValueError("Invocation content must be acyclic portable JSON")
        if type(item) is dict and any(type(key) is not str for key in item):
            raise ValueError("Invocation object keys must be strings")
        for child in item.values() if type(item) is dict else item:
            check(child, ancestors | {id(item)})
    check(value, set())
    return json.loads(json.dumps(value, allow_nan=False))


def success(content):
    return {"status": "success", "content": snapshot(content)}


def failure(code, message, *, retryable=False, details=None):
    return {"status": "error", "error": {"code": code, "message": message,
            "retryable": retryable, "details": snapshot(details)}}


class OperationFailure(Exception):
    """Typed failure from a real node operation, never detected by text parsing."""
    def __init__(self, code, message, *, retryable=False, details=None):
        super().__init__(message)
        self.outcome = failure(code, message, retryable=retryable, details=details)


def error_code(error):
    """The typed outcome code of a node failure, or None for an untyped exception."""
    outcome = getattr(error, "outcome", None)
    if isinstance(outcome, dict) and isinstance(outcome.get("error"), dict):
        code = outcome["error"].get("code")
        return code if isinstance(code, str) else None
    return None


def _protected_failure(error):
    """Execution authority failures are typed, never inferred from error text."""
    from magic_agents.coordination.budget import BudgetError
    from magic_agents.coordination.service import CoordinationError
    from magic_llm.agent.control import AgentControlError
    from magic_llm.agent.types import AgentBudgetExceeded
    from magic_llm.engine.attempt_control import ProviderAttemptControlError
    from magic_llm.exception.ChatException import RequestValidationError
    return isinstance(error, (asyncio.CancelledError, PermissionError, BudgetError,
                              CoordinationError, AgentControlError, ProviderAttemptControlError,
                              AgentBudgetExceeded, RequestValidationError)) or (
        isinstance(error, OperationFailure) and error_code(error) == "SKILLS_CONTEXT_LIMIT")


def _authority_outcome(error):
    if isinstance(error, asyncio.CancelledError):
        return {"status": "cancelled"}
    code = (getattr(error, "code", None) or getattr(error, "error_code", None)
            or error_code(error) or "COORDINATION_DENIED")
    return failure(str(code), "Invocation stopped by execution authority",
                   details={"exception_type": type(error).__name__})


class _InvocationAuthority:
    """One private, fail-closed lineage across all owned lifecycle children.

    A Hook may catch a child's exception. It cannot thereby undo the execution
    authority's terminal decision or start a sibling with fresh permissions.
    """
    def __init__(self, scope, node_id):
        self.scope, self.node_id, self.denial = scope, node_id, None
        self.owner_guard = scope.capture_guard(node_id)

    def protect(self, error):
        if self.denial is None:
            if isinstance(error, OperationFailure) and error_code(error) == "SKILLS_CONTEXT_LIMIT":
                from magic_llm.agent.control import AgentControlError
                error = AgentControlError("Skills request exceeds the admitted context budget", "SKILLS_CONTEXT_LIMIT")
            self.denial = error
        return self.denial

    def check(self, node_id):
        if self.denial is not None:
            raise self.denial
        try:
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError()
            # Retain the initiating owner's authority even for nonactor children.
            self.owner_guard()
            if node_id != self.node_id:
                self.scope.guard(node_id)
        except BaseException as error:
            if not isinstance(error, (Exception, asyncio.CancelledError)):
                raise
            if not _protected_failure(error):
                from magic_agents.coordination.service import CoordinationError
                error = CoordinationError("coordination_admission_failed", "Execution authority check failed")
            raise self.protect(error)


def _copy_run_log(chat_log):
    """Isolate ordinary Hook data without cloning live authority, locks or clients."""
    coordination = getattr(chat_log, "coordination", None)
    memo = {id(coordination): coordination} if coordination is not None else None
    return copy.deepcopy(chat_log, memo)


def _reject_sync_tool(chat_log):
    scope = getattr(chat_log, "coordination", None)
    if scope is not None and not getattr(scope.runtime, 'invocation_mode', False):
        from magic_agents.coordination.service import CoordinationError
        raise CoordinationError("unsupported_coordination_capability",
                                "Synchronous child tools require outstanding-work reconciliation")


def error_retryable(error):
    """``retryable`` of a typed node failure, or None when the failure is untyped.

    A step-mode Fetch failure maps to the same typed outcome as under Hook
    control (HTTP 429/5xx and network errors are retryable).
    """
    outcome = getattr(error, "outcome", None)
    if isinstance(outcome, dict) and isinstance(outcome.get("error"), dict):
        retryable = outcome["error"].get("retryable")
        return retryable if type(retryable) is bool else None
    from magic_agents.node_system.fetch_request import FetchError, operation_failure_for
    if isinstance(error, FetchError):
        mapped = operation_failure_for(error)
        if mapped is not error:
            return mapped.outcome["error"]["retryable"]
    return None


def _close_open_subgraphs(outputs, error):
    """SUBGRAPH_END frames for sub-flows an interrupted operation left open.

    An Inner Flow cancelled by its invocation budget (or interrupted by an
    unexpected exception) never reaches its own SUBGRAPH_END; the kept trace
    still pairs every SUBGRAPH_START with an end.
    """
    opened, closed = [], set()
    for item in outputs:
        content = item.get("content") if isinstance(item, dict) and item.get("type") == "debug" else None
        if not isinstance(content, dict):
            continue
        if content.get("event_type") == "SUBGRAPH_START":
            opened.append(item)
        elif content.get("event_type") == "SUBGRAPH_END":
            closed.add(content.get("child_run_id"))
    reason = "cancelled" if isinstance(error, asyncio.CancelledError) else "interrupted"
    frames = []
    # Innermost first, as a sub-flow ends before the one that contains it.
    for item in reversed(opened):
        start = item["content"]
        if start.get("child_run_id") in closed:
            continue
        frame = {"type": "debug", "content": {
            "event_type": "SUBGRAPH_END",
            "parent_execution_id": start.get("parent_execution_id"),
            "child_run_id": start.get("child_run_id"),
            "node_id": start.get("node_id"),
            "status": "error",
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        }}
        if item.get("source_node_path") is not None:
            frame["source_node_path"] = list(item["source_node_path"])
        frames.append(frame)
    return frames


class _Emit:
    def __init__(self, frame):
        self.frame, self.active = frame, True

    def _emit(self, kind, content):
        if self.active:
            self.frame["side_events"].append({"type": kind, "content": snapshot(content)})
        return None

    def user(self, content):
        return self._emit("user", content)

    def debug(self, content):
        return self._emit("debug", content)

    def feedback(self, content):
        return self._emit("feedback", content)


class ControlledTool:
    _disable_dedup = True

    def __init__(self, adapter, node_id, edge, function, chat_log):
        self.adapter, self.node_id, self.edge = adapter, node_id, edge
        self.function, self.chat_log = function, chat_log
        self.__name__ = getattr(function, "__name__", node_id)
        self._source_node_id = node_id
        self._events = []

    async def __call__(self, **arguments):
        from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL
        tool_call = CURRENT_TOOL_CALL.get()
        caller = {"node_id": self.edge.target, "edge_id": self.edge.id}
        if tool_call is not None:
            caller["tool_call_id"] = tool_call.id
        record = await self.adapter.invoke(self.node_id, arguments, self.chat_log,
                                           caller=caller, tool=self.function, admit_delivery=True)
        self._events.append({"type": "hook_result", "execution": snapshot(record)})
        outcome = record["outcome"]
        if outcome["status"] == "success":
            return outcome["content"]
        if outcome["status"] == "error":
            error = outcome["error"]
            raise OperationFailure(error["code"], error["message"],
                                   retryable=error["retryable"], details=error["details"])
        raise asyncio.CancelledError()

    def drain_events(self):
        events, self._events = self._events, []
        return events


class InvocationControl:
    """Per-graph adapter around actual node operations, with per-call state."""
    def __init__(self, nodes, edges, *, timeout=60, execution_id="", run_id=""):
        self.nodes, self.edges = nodes, list(edges)
        self.timeout = timeout
        self.execution_id, self.run_id = execution_id, run_id
        self.records, self.events = [], []
        self.bindings = [node for node in nodes.values()
                         if getattr(node, "lifecycle_event", None)]
        for hook in self.bindings:
            if any(target not in nodes or getattr(nodes[target], "node_type", None) == "hook" for target in hook.target_node_ids):
                raise ValueError("Lifecycle Hook target must reference an executable non-Hook node")
        self.connections = {edge.id: edge for edge in self.edges
                            if edge.source in nodes
                            and getattr(nodes[edge.source], "lifecycle_event", None)
                            and edge.sourceHandle == getattr(nodes[edge.source], "OUTPUT_HANDLE_CALL", None)}
        self.on_demand = {edge.target for edge in self.connections.values()}
        from magic_agents.hooks.messages_hook import bind_messages_hooks
        from magic_agents.node_system.NodeHook import NodeHook
        bind_messages_hooks(nodes)
        self.hook_ids = {hook.node_id for hook in self.bindings} | {
            node.node_id for node in nodes.values() if isinstance(node, NodeHook) and node.hook_mode == 'messages'}
        self._active_graph_calls = set()
        for node in nodes.values():
            node._invocation_control = self

    @staticmethod
    def exports_tool_definitions(node):
        data = getattr(node, "_data", None)
        return bool(getattr(node, "tool_mode", False) or getattr(data, "tool_mode", False)
                    or (getattr(node, "node_type", None) == "python_exec"
                        and callable(getattr(node, "_has_code", None)) and not node._has_code())
                    or getattr(node, "node_type", None) in ("mcp", "tool"))

    def tool_definition_edge(self, edge):
        target = self.nodes.get(edge.target)
        prefix = getattr(target, "INPUT_TOOL_PREFIX", "")
        return bool(getattr(target, "node_type", None) == "llm" and prefix
                    and (edge.targetHandle or "").startswith(prefix))

    def controlled(self, node_id, edge=None):
        if any(node_id in hook.target_node_ids for hook in self.bindings):
            return True
        candidates = [edge] if edge is not None else [
            item for item in self.edges if item.target == node_id
            and not self.tool_definition_edge(item)]
        return any(item.hooks and item.hooks.enabled and item.hooks.hook_node_id in self.hook_ids
                   for item in candidates)

    def select(self, node_id, caller, event):
        edge_ids = caller.get("edge_ids", [caller.get("edge_id")])
        node_hooks, edge_hooks = [], []
        for hook in self.bindings:
            if hook.lifecycle_event == event and node_id in hook.target_node_ids:
                node_hooks.append(hook)
        # Declared graph order makes fan-in deterministic. Each matching binding
        # gets its actual edge identity; no last-arriving input becomes caller.
        for edge in self.edges:
            if edge.id not in edge_ids or not edge.hooks or not edge.hooks.enabled:
                continue
            hook = self.nodes.get(edge.hooks.hook_node_id)
            if hook is None or not getattr(hook, "lifecycle_event", None) or hook.lifecycle_event != event or hook.target_node_ids:
                continue
            valid = (edge.source == node_id and edge.target == caller.get("node_id")
                     if self.tool_definition_edge(edge) else edge.target == node_id)
            if valid:
                selected = copy.copy(hook)
                selected._binding_edge_id = edge.id
                edge_hooks.append(selected)
        if event in ("onError", "onFinish", "onCancel"):
            return list(reversed(node_hooks)) + list(reversed(edge_hooks))
        return edge_hooks + node_hooks

    def wrap_tool(self, function, llm, handle, chat_log):
        origin = getattr(function, "_source_node_id", None)
        matches = [edge for edge in self.edges if edge.target == llm.node_id
                   and edge.targetHandle == handle and
                   (origin is None or origin == llm.node_id or edge.source == origin)]
        if len(matches) != 1:
            if any(self.controlled(edge.source, edge) for edge in matches):
                raise ValueError("Controlled tool needs unambiguous graph edge provenance")
            return function
        edge = matches[0]
        if self.controlled(edge.source, edge):
            return ControlledTool(self, edge.source, edge, function, chat_log)
        return function

    async def _operation(self, node_id, content, chat_log, *, tool=None, target_handle=None, raw_events=None):
        original = self.nodes[node_id]
        # A fresh configured instance protects concurrent calls from shared
        # node.inputs, outputs, response caches, and mutable processor fields.
        factory = getattr(original, "_invocation_factory", None)
        node = factory() if factory else copy.copy(original)
        node.inputs = dict(original.inputs)
        node.outputs, node._response = {}, None
        node._control_active = True
        node._invocation_control = self
        if tool is None and target_handle is None and isinstance(content, dict):
            # A replacement handle map removes omitted JSON data inputs, while
            # live clients/callables remain operational resources.
            for handle, value in list(node.inputs.items()):
                try:
                    snapshot(value)
                except (ValueError, TypeError):
                    continue
                del node.inputs[handle]
            for handle, value in content.items():
                original_value = node.inputs.get(handle)
                # A client/tool input has a JSON snapshot for the Hook, but its
                # unchanged operational object must still reach Node.process.
                if handle not in node.inputs or value != node._safe_value(original_value):
                    node.inputs[handle] = snapshot(value)
        if getattr(node, "node_type", None) == "fetch":
            if tool is None and not node.tool_mode:
                # A controlled step Fetch runs the same request/response core
                # as plain step mode. Its content is its input handle map (the
                # template context), also for a child call or redirect, so
                # another step Fetch's context.request["content"] replays as-is.
                from magic_agents.node_system.fetch_request import FetchError, operation_failure_for
                if not isinstance(content, dict):
                    raise OperationFailure("INVALID_INPUT", "Fetch operation arguments must be an object")
                if target_handle is not None:
                    node.inputs = snapshot(content)
                    # A child call or redirect is an explicit request: it is
                    # sent even with empty content (no "no inputs" skip).
                    node._explicit_invocation = True
                result = {}
                try:
                    async for item in node.process(chat_log):
                        if item.get("type") == node.OUTPUT_HANDLE:
                            value = item.get("content")
                            if isinstance(value, dict) and "node" in value and "content" in value:
                                value = value["content"]
                            result[node.OUTPUT_HANDLE] = node._safe_value(value)
                except FetchError as exc:
                    failure = operation_failure_for(exc)
                    if failure is exc:
                        raise  # TemplateError/UnexpectedError -> NODE_EXCEPTION
                    raise failure from exc
                return result
            # Tool mode keeps the LLM tool contract, with typed HTTP failures.
            function = node._build_tool_callable()
            function._strict_errors = True
            if getattr(chat_log, 'coordination', None) is not None:
                from magic_agents.coordination.dispatch import register_dispatch_callable
                function._dispatch_scope = chat_log.coordination
                function._source_node_id = node.node_id
                register_dispatch_callable(function)
            if not isinstance(content, dict):
                raise OperationFailure("INVALID_INPUT", "Fetch operation arguments must be an object")
            return await function(**content)
        if tool is not None:
            if not isinstance(content, dict):
                raise OperationFailure("INVALID_INPUT", "Tool arguments must be an object")
            if asyncio.iscoroutinefunction(tool) or asyncio.iscoroutinefunction(getattr(tool, "__call__", None)):
                return await tool(**content)
            _reject_sync_tool(chat_log)
            return await asyncio.to_thread(tool, **content)
        handle = target_handle or getattr(node, "INPUT_HANDLE_TEMPLATE_CONTEXT", None)
        if handle:
            node.inputs[handle] = snapshot(content)
        elif isinstance(content, dict):
            pass  # Handle-map input was applied above.
        else:
            # Named data inputs are authoritative; an operation with one data
            # input can accept its content directly.
            keys = list(node.inputs)
            if len(keys) == 1:
                node.inputs[keys[0]] = snapshot(content)
            else:
                node.inputs["content"] = snapshot(content)
        outputs = []
        node._hooks = getattr(original, "_hooks", None)
        node._tool_graph_factory = getattr(original, "_tool_graph_factory", None)
        try:
            async for item in node.process(chat_log):
                outputs.append(item)
        except BaseException as exc:
            # A failed (or timed-out) operation keeps the trace it produced,
            # e.g. an Inner Flow's SUBGRAPH_START/END and child error frames.
            if raw_events is not None:
                raw_events.extend(outputs)
                raw_events.extend(_close_open_subgraphs(outputs, exc))
            raise
        is_tool = self.exports_tool_definitions(node)
        if is_tool and target_handle is not None:
            # Resolve the existing real authored tool, then call that operation;
            # on-demand composition must not return its definition packet.
            functions = []
            for item in outputs:
                value = item.get("content")
                if isinstance(value, dict) and "node" in value and "content" in value:
                    value = value["content"]
                function = getattr(value, "tool_callable", None)
                if function is not None:
                    functions.append(function)
            if len(functions) != 1 or not isinstance(content, dict):
                raise OperationFailure("INVALID_INPUT", "Child tool operation needs one callable and object arguments")
            function = functions[0]
            if asyncio.iscoroutinefunction(function) or asyncio.iscoroutinefunction(getattr(function, "__call__", None)):
                return await function(**content)
            _reject_sync_tool(chat_log)
            return await asyncio.to_thread(function, **content)
        if raw_events is not None:
            raw_events.extend(outputs)
        from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes
        if any(item.get("type") == ConditionalSignalTypes.BYPASS_ALL for item in outputs):
            # BYPASS_ALL is a node's failure signal (configuration/input errors),
            # never a success handle.
            errors = [item for item in outputs if item.get("type") == "debug"
                      and isinstance(item.get("content"), dict) and item["content"].get("error_type")]
            message = str(errors[-1]["content"].get("error_message") or "") if errors else ""
            raise OperationFailure("NODE_ERROR", message or "Node signalled a failure without a result",
                                   details=node._safe_value(errors[-1]) if errors else None)
        data = [item for item in outputs if item.get("type") not in ("debug", "debug_summary")]
        if not data:
            errors = [item for item in outputs if item.get("type") == "debug"]
            if errors:
                raise OperationFailure("NODE_ERROR", "Node produced an error without a result",
                                       details=node._safe_value(errors[-1]))
            return {}
        result = {}
        for item in data:
            value = item.get("content")
            if isinstance(value, dict) and "node" in value and "content" in value:
                value = value["content"]
            result[item["type"]] = node._safe_value(value)
        return result

    async def invoke(self, node_id, content, chat_log, *, caller=None, parent=None,
                     state=None, stack=(), tool=None, event="onStart", target_handle=None, raw_events=None, delivery=False,
                     admit_delivery=False, record_id=None):
        caller, content = snapshot(caller or {}), snapshot(content)
        state = state or {"deadline": time.monotonic() + self.timeout, "admissions": [0]}
        scope = getattr(chat_log, "coordination", None)
        if scope is not None and "authority" not in state:
            owner_id = caller.get("node_id", node_id) if tool is not None else node_id
            state["authority"] = _InvocationAuthority(scope, owner_id)
        authority = state.get("authority")
        child_operation = parent is not None and caller.get("edge_id") in self.connections and not delivery
        ident = record_id or uuid.uuid4().hex
        frame = {"id": ident, "root_id": parent["root_id"] if parent else ident,
                 "parent_id": parent["id"] if parent else None, "kind": "node", "node_id": node_id,
                 "caller": caller, "input": snapshot(content), "request": {"content": content},
                 "outcome": None, "original_outcome": None, "executed": False,
                 "recovered": False, "child": []}
        if parent is not None:
            parent["child"].append(frame)
        self.records.append(frame)
        if delivery:
            frame["operation"] = "delivery"
        admission = self.select(node_id, caller, event)
        failed_hooks, finish_started = set(), set()
        phase_facts = {}
        outcome = None
        admitted = False

        async def phase(name, current, selected=None):
            fact = {"event": name, "event_id": uuid.uuid4().hex, "node_id": node_id,
                    "frame_id": ident, "executed": frame["executed"],
                    "request": snapshot(frame["request"]), "outcome": snapshot(current)}
            self.events.append(snapshot(fact))
            phase_facts[name] = fact
            selected = selected if selected is not None else self.select(node_id, caller, name)
            if delivery and name != "onDeliver":
                selected = [hook for hook in selected if not hook.target_node_ids]
            for hook in selected:
                if name == "onError" and hook.node_id in failed_hooks:
                    continue
                if name == "onFinish":
                    if hook.node_id in finish_started:
                        continue
                    finish_started.add(hook.node_id)
                current, terminal, failed = await self._hook(hook, frame, name, current,
                    fact, chat_log, state, (*stack, node_id))
                if failed:
                    failed_hooks.add(hook.node_id)
                if name in ("onStart", "onDeliver") and terminal or name == "onError" and current and current["status"] in ("success", "cancelled"):
                    break
            return current

        try:
            if authority is not None:
                if child_operation and node_id in authority.scope.roles:
                    authority.check(authority.node_id)
                    from magic_agents.coordination.service import CoordinationError
                    raise CoordinationError("unsupported_coordination_topology",
                                            "Participant children require scheduler-owned activation routing")
                authority.check(node_id)
            if node_id in stack or len(stack) >= 16 or state["admissions"][0] >= 100:
                if authority is not None:
                    from magic_agents.coordination.service import CoordinationError
                    raise CoordinationError("invocation_limit", "Connected invocation cycle/depth/count limit reached")
                outcome = failure("INVOCATION_LIMIT", "Connected invocation cycle/depth/count limit reached")
            elif time.monotonic() >= state["deadline"]:
                if authority is not None:
                    from magic_agents.coordination.service import CoordinationError
                    raise CoordinationError("deadline_exceeded", "Invocation deadline exhausted")
                outcome = failure("DEADLINE_EXCEEDED", "Invocation deadline exhausted")
            else:
                state["admissions"][0] += 1
                admitted = True
                if admit_delivery and self.select(node_id, caller, "onDeliver"):
                    delivery_caller = {key: value for key, value in caller.items() if key != "tool_call_id"}
                    delivery_frame = await self.invoke(node_id, frame["request"]["content"], chat_log,
                        caller=delivery_caller, parent=frame, state=state, stack=stack,
                        event="onDeliver", delivery=True)
                    if delivery_frame["executed"] and delivery_frame["outcome"]["status"] == "success":
                        frame["request"] = {"content": snapshot(delivery_frame["outcome"]["content"])}
                        outcome = await phase(event, None, admission)
                    else:
                        outcome = snapshot(delivery_frame["outcome"])
                else:
                    outcome = await phase(event, None, admission)
                if authority is not None:
                    authority.check(node_id)
                if outcome is None:
                    frame["executed"] = True
                    dispatch_task = None
                    try:
                        if delivery:
                            result = snapshot(frame["request"]["content"])
                        else:
                            async def operation():
                                if authority is not None:
                                    authority.check(node_id)
                                child_log = _copy_run_log(chat_log) if child_operation and authority is not None else chat_log
                                return await self._operation(node_id, frame["request"]["content"],
                                    child_log, tool=tool, target_handle=target_handle, raw_events=raw_events)
                            dispatch = (authority.scope.run_child_operation(node_id, ident,
                                frame["request"]["content"], operation)
                                if child_operation and authority is not None else operation())
                            dispatch_task = asyncio.ensure_future(dispatch)
                            result = await asyncio.wait_for(dispatch_task,
                                timeout=max(0, state["deadline"] - time.monotonic()))
                        if time.monotonic() >= state["deadline"]:
                            raise asyncio.TimeoutError()
                        if authority is not None:
                            authority.check(node_id)
                        outcome = success(result)
                    except OperationFailure as exc:
                        if authority is not None and _protected_failure(exc):
                            raise authority.protect(exc)
                        outcome = snapshot(exc.outcome)
                    except asyncio.TimeoutError as exc:
                        if authority is not None and _protected_failure(exc):
                            raise authority.protect(exc)
                        expired = (time.monotonic() >= state["deadline"] or
                                   dispatch_task is not None and dispatch_task.cancelled())
                        if authority is not None and expired:
                            from magic_agents.coordination.service import CoordinationError
                            raise CoordinationError("deadline_exceeded", "Node invocation deadline exhausted")
                        outcome = failure("NODE_TIMEOUT", "Node operation exceeded its invocation budget", retryable=True)
                    except Exception as exc:
                        if authority is not None and _protected_failure(exc):
                            raise authority.protect(exc)
                        from magic_agents.node_system.fetch_request import error_context
                        details = {"exception_type": type(exc).__name__}
                        context = error_context(exc)  # sanitized Fetch diagnostics
                        if context:
                            details["context"] = context
                        outcome = failure("NODE_EXCEPTION", str(exc) or type(exc).__name__,
                                          details=details)
            frame["original_outcome"] = snapshot(outcome)
            # Rejected work is a diagnostic record, not a lifecycle admission.
            # Keep its phase facts without redispatching callbacks that could
            # recursively request the same rejected work again.
            selected = None if admitted else []
            if outcome["status"] == "error":
                outcome = await phase("onError", outcome, selected)
            outcome = await phase("onFinish", outcome, selected)
            if authority is not None:
                authority.check(node_id)
            frame["outcome"] = snapshot(outcome)
        except asyncio.CancelledError:
            if authority is not None:
                authority.protect(asyncio.CancelledError())
            frame["original_outcome"] = frame["original_outcome"] or {"status": "cancelled"}
            frame["outcome"] = {"status": "cancelled"}
            outcome = {"status": "cancelled"}
            await phase("onCancel", outcome)
            # Started/interrupted finish bindings are never replayed.
            remaining = [hook for hook in self.select(node_id, caller, "onFinish")
                         if hook.node_id not in finish_started]
            outcome = await phase("onFinish", outcome, remaining) if not finish_started else await self._cancel_finish(
                remaining, frame, chat_log, state, (*stack, node_id), phase_facts.get("onFinish"))
            frame["outcome"] = {"status": "cancelled"}
            raise
        except Exception as error:
            if authority is not None and _protected_failure(error):
                error = authority.protect(error)
                frame["original_outcome"] = frame["original_outcome"] or _authority_outcome(error)
                frame["outcome"] = _authority_outcome(error)
                frame["protected"] = True
                raise error
            raise
        finally:
            if frame["outcome"] is not None:
                frame["recovered"] = (frame["original_outcome"]["status"] == "error"
                                      and frame["outcome"]["status"] == "success")
        return snapshot(frame)

    async def _cancel_finish(self, hooks, frame, chat_log, state, stack, fact=None):
        fact = fact or {"event": "onFinish", "event_id": uuid.uuid4().hex,
                       "node_id": frame["node_id"], "frame_id": frame["id"],
                       "executed": frame["executed"], "request": snapshot(frame["request"])}
        fact = {**fact, "outcome": {"status": "cancelled"}}
        for hook in hooks:
            await self._hook(hook, frame, "onFinish", {"status": "cancelled"}, fact, chat_log, state, stack)
        return {"status": "cancelled"}

    async def _hook(self, hook, subject, phase, outcome, fact, chat_log, state, stack):
        authority = state.get("authority")
        cancelling = phase == "onCancel" or outcome and outcome["status"] == "cancelled"
        frame = {"id": uuid.uuid4().hex, "root_id": subject["root_id"], "parent_id": subject["id"],
                 "kind": "hook", "node_id": hook.node_id, "caller": snapshot(subject["caller"]),
                 "event": phase, "input": snapshot(subject["request"]["content"]),
                 "child": [], "side_events": [], "outcome": None, "original_outcome": None,
                 "executed": False, "recovered": False}
        subject["child"].append(frame)
        self.records.append(frame)
        entry_outcome, entry_request = snapshot(outcome), snapshot(subject["request"])
        emit = _Emit(frame)
        edge = next((edge for edge in self.edges if edge.id == getattr(hook, "_binding_edge_id", subject["caller"].get("edge_id"))), None)
        cap = hook._timeout_seconds
        if getattr(hook, "_binding_edge_id", None) and edge and edge.hooks and edge.hooks.hook_node_id == hook.node_id and edge.hooks.timeout_override is not None:
            cap = edge.hooks.timeout_override
        hook_deadline = min(state["deadline"], time.monotonic() + cap)
        child_state = {**state, "deadline": hook_deadline}
        context = HookContextFactory.build_node_context(execution_id=self.execution_id,
            run_id=self.run_id, node_id=hook.node_id, node_type="hook", node_class="NodeHook",
            inputs=snapshot(subject["request"]))
        context.request = snapshot(subject["request"])
        context.outcome = snapshot(outcome)
        effective_caller = snapshot(subject["caller"])
        binding_edge_id = getattr(hook, "_binding_edge_id", None)
        if binding_edge_id is not None:
            binding_edge = next(edge for edge in self.edges if edge.id == binding_edge_id)
            effective_caller = {"node_id": binding_edge.target if self.tool_definition_edge(binding_edge) else binding_edge.source,
                                "edge_id": binding_edge_id}
            if "tool_call_id" in subject["caller"]:
                effective_caller["tool_call_id"] = subject["caller"]["tool_call_id"]
        frame["caller"] = effective_caller
        context.caller = MappingProxyType(effective_caller)
        context.event = {**snapshot(fact), "request": snapshot(subject["request"]), "outcome": snapshot(outcome)}
        context.child = snapshot(frame["child"])
        context.emit = emit

        async def call(connection, content):
            if phase == "onCancel" or outcome and outcome["status"] == "cancelled":
                raise ValueError("Cancellation controls cannot launch connected children")
            edge = self.connections.get(connection)
            if edge is None or edge.source != hook.node_id:
                raise ValueError("Child call requires this Hook's declared child-call edge ID")
            child_id = uuid.uuid4().hex
            try:
                return await self.invoke(edge.target, content, chat_log,
                    caller={"node_id": hook.node_id, "edge_id": edge.id},
                    parent=frame, state=child_state, stack=stack, target_handle=edge.targetHandle,
                    admit_delivery=True, record_id=child_id)
            except asyncio.CancelledError:
                if authority is not None or asyncio.current_task().cancelling():
                    raise
                return snapshot(next(child for child in frame["child"] if child["id"] == child_id))
        async def owned_call(connection, content):
            if not emit.active:
                raise ValueError("Hook invocation is closed")
            owner = asyncio.current_task()
            owned_tasks.add(owner)
            try:
                return await call(connection, content)
            finally:
                owned_tasks.discard(owner)
        owned_tasks = set()
        context.call = owned_call

        async def call_many(calls, mode="sequence"):
            if mode not in ("sequence", "parallel"):
                raise ValueError("call_many mode must be sequence or parallel")
            if mode == "sequence":
                return [await owned_call(item["connection"], item["content"]) for item in calls]
            tasks = [asyncio.create_task(owned_call(item["connection"], item["content"])) for item in calls]
            try:
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        context.call_many = call_many
        hook_task = None
        try:
            if authority is not None and not cancelling:
                authority.check(subject["node_id"])
            budget = hook_deadline - time.monotonic()
            if budget <= 0:
                raise asyncio.TimeoutError()
            frame["executed"] = True
            hook_task = asyncio.ensure_future(hook.invoke_control(context, _copy_run_log(chat_log)))
            reply = await asyncio.wait_for(hook_task, timeout=budget)
            if authority is not None and not cancelling:
                authority.check(subject["node_id"])
            if time.monotonic() >= hook_deadline:
                raise asyncio.TimeoutError()
            reply = {"action": "pass"} if reply is None else snapshot(reply)
            if not isinstance(reply, dict):
                raise ValueError("Hook must return an explicit action decision")
            action = reply.get("action")
            if phase == "onCancel" or outcome and outcome["status"] == "cancelled":
                if action != "pass":
                    raise ValueError("Cancelled lifecycle only accepts pass")
            terminal = False
            if action == "pass" and set(reply) == {"action"}:
                pass
            elif action == "input" and phase in ("onStart", "onDeliver") and set(reply) == {"action", "content"}:
                subject["request"] = {"content": snapshot(reply["content"])}
            elif action == "outcome" and set(reply) == {"action", "outcome"}:
                replacement = reply["outcome"]
                if replacement.get("status") == "success" and set(replacement) == {"status", "content"}:
                    outcome = success(replacement["content"])
                elif replacement.get("status") == "error" and set(replacement) == {"status", "error"}:
                    error = replacement["error"]
                    if set(error) != {"code", "message", "retryable", "details"} or not isinstance(error["code"], str) or not isinstance(error["message"], str) or type(error["retryable"]) is not bool:
                        raise ValueError("Malformed typed error outcome")
                    outcome = snapshot(replacement)
                else:
                    raise ValueError("Outcome decision requires success or typed error")
                terminal = True
            elif action == "redirect" and set(reply) == {"action", "connection", "content"}:
                child = await call(reply["connection"], reply["content"])
                outcome, terminal = child["outcome"], True
            else:
                raise ValueError("Invalid Hook action or fields")
            if time.monotonic() >= hook_deadline:
                raise asyncio.TimeoutError()
            frame["decision"] = reply
            frame["outcome"] = frame["original_outcome"] = success(reply)
            return outcome, terminal, False
        except asyncio.CancelledError:
            if authority is not None:
                authority.protect(asyncio.CancelledError())
            frame["outcome"] = frame["original_outcome"] = {"status": "cancelled"}
            raise
        except Exception as exc:
            subject["request"] = entry_request
            outcome = entry_outcome
            if authority is not None:
                if (isinstance(exc, asyncio.TimeoutError) and not _protected_failure(exc)
                        and (time.monotonic() >= hook_deadline or hook_task is not None and hook_task.cancelled())):
                    from magic_agents.coordination.service import CoordinationError
                    exc = CoordinationError("deadline_exceeded", "Lifecycle callback deadline exhausted")
                if _protected_failure(exc):
                    exc = authority.protect(exc)
                    frame["outcome"] = frame["original_outcome"] = _authority_outcome(exc)
                    frame["protected"] = True
                    raise exc
            code = "HOOK_TIMEOUT" if isinstance(exc, asyncio.TimeoutError) else "HOOK_FAILED"
            frame["outcome"] = frame["original_outcome"] = failure(code, str(exc) or code,
                details={"exception_type": type(exc).__name__})
            frame["diagnostic"] = snapshot(frame["outcome"]["error"])
            if hook.failure_policy == "fail" and phase != "onCancel" and not (outcome and outcome["status"] == "cancelled"):
                return failure("HOOK_FAILED", "Hook control failed", details=frame["diagnostic"]), True, True
            return outcome, False, True
        finally:
            emit.active = False
            current = asyncio.current_task()
            pending = [task for task in owned_tasks if task is not current and not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def prepare_delivery(self, edge, content):
        """Gate one isolated data delivery without executing its target.

        A terminal decision skips only this delivery. Its sibling edges and
        target invocation keep their independent lifecycle and data state.
        Tool-definition packets are controlled at actual tool call admission.
        """
        if self.tool_definition_edge(edge) or edge.id in self.connections:
            return True, content, None
        caller = {"node_id": edge.source, "edge_id": edge.id}
        if not self.select(edge.target, caller, "onDeliver"):
            return True, content, None
        target = self.nodes[edge.target]
        try:
            portable = snapshot(content)
        except (ValueError, TypeError):
            portable = target._safe_value(content)
        source = self.nodes.get(edge.source)
        chat_log = getattr(source, "_invocation_chat_log", None)
        frame = await self.invoke(edge.target, portable, chat_log, caller=caller,
                                  event="onDeliver", delivery=True)
        assign = frame["executed"] and frame["outcome"]["status"] == "success"
        effective = frame["outcome"].get("content")
        if effective == portable:
            effective = content  # Preserve unchanged opaque operational value.
        return assign, effective, frame

    async def process_node(self, node, chat_log):
        delivered = getattr(node, "_control_callers", {})
        active_edges = [edge for edge in self.edges if edge.id in delivered
                        and edge.target == node.node_id and not self.tool_definition_edge(edge)]
        caller = {"node_id": active_edges[0].source, "edge_id": active_edges[0].id} if active_edges else {}
        if len(active_edges) > 1:
            caller["edge_ids"] = [edge.id for edge in active_edges]
        inputs = {}
        for handle, value in node.inputs.items():
            try:
                inputs[handle] = snapshot(value)
            except (ValueError, TypeError):
                continue  # Clients/callable tools are operational resources.
        raw_events = []
        frame = await self.invoke(node.node_id, inputs, chat_log, caller=caller, raw_events=raw_events)
        node._last_invocation_record = frame
        outcome = frame["outcome"]
        yield {"type": "debug", "content": {"event_type": "HOOK_RESULT", "node_id": node.node_id,
                                           "data": {"execution": frame}}}
        if outcome["status"] == "success":
            value = outcome["content"]
            if not isinstance(value, dict):
                raise OperationFailure("INVALID_OUTPUT", "Normal node outcome must map output handles to content")
            originals = frame["original_outcome"].get("content", {})
            replayed = set()
            for item in raw_events:
                handle = item.get("type")
                if handle in ("debug", "debug_summary"):
                    yield item
                elif handle in value and handle in originals and value[handle] == originals[handle]:
                    # Unchanged opaque objects and every streaming chunk keep
                    # their exact operational identity after the finish gate.
                    yield item
                    replayed.add(handle)
            for handle, content in value.items():
                if handle not in replayed:
                    yield node.yield_static(content, content_type=handle)
        elif outcome["status"] == "error":
            # Keep the operation's own trace (child frames, SUBGRAPH_START/END)
            # before the node fails, exactly as on the success path.
            for item in raw_events:
                if item.get("type") in ("debug", "debug_summary"):
                    yield item
            error = outcome["error"]
            raise OperationFailure(error["code"], error["message"],
                                   retryable=error["retryable"], details=error["details"])
