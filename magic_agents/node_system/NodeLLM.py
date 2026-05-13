import asyncio
import json
import re
import uuid
import logging
from typing import Any, Optional, TYPE_CHECKING

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.primitive_coercion import coerce_primitive_by_type, input_has_value

if TYPE_CHECKING:
    from magic_llm.agent import TaskManifest, SubagentBundle
    from magic_agents.hooks.hook_registry import HookRegistry

logger = logging.getLogger(__name__)


class NodeLLM(Node):
    """
    LLM node - handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_CLIENT_PROVIDER = 'handle-client-provider'
    DEFAULT_INPUT_CHAT = 'handle-chat'
    DEFAULT_INPUT_SYSTEM_CONTEXT = 'handle-system-context'
    DEFAULT_INPUT_USER_MESSAGE = 'handle_user_message'
    DEFAULT_INPUT_TEMPERATURE = 'handle-llm-temperature'
    DEFAULT_INPUT_TOP_P = 'handle-llm-top_p'
    DEFAULT_INPUT_MAX_TOKENS = 'handle-llm-max_tokens'
    DEFAULT_INPUT_STREAM = 'handle-llm-stream'
    DEFAULT_INPUT_ITERATE = 'handle-llm-iterate'
    DEFAULT_INPUT_JSON_OUTPUT = 'handle-llm-json_output'
    # Output handles
    DEFAULT_OUTPUT_CONTENT = 'handle_streaming_content'
    DEFAULT_OUTPUT_GENERATED = 'handle_generated_content'
    DEFAULT_OUTPUT_TOOL_CALLS = 'handle-tool-calls'
    # Tool input handle prefix — dynamic collection from tool-prefixed handles
    DEFAULT_INPUT_TOOL_PREFIX = 'handle-tool-'
    # Engines known to NOT support tools via kwargs
    _UNSUPPORTED_ENGINES = {'google', 'cohere', 'cloudflare'}

    def __init__(self,
                 data: LlmNodeModel,
                 node_id: str,
                 debug: bool = False,
                 handles: Optional[dict] = None,
                 **kwargs):
        super().__init__(
            debug=debug,
            node_id=node_id,
            **kwargs)
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLER_CLIENT_PROVIDER = handles.get('client_provider', handles.get('client', self.DEFAULT_INPUT_CLIENT_PROVIDER))
        self.INPUT_HANDLER_CHAT = handles.get('chat', self.DEFAULT_INPUT_CHAT)
        self.INPUT_HANDLER_SYSTEM_CONTEXT = handles.get('system_context', handles.get('system', self.DEFAULT_INPUT_SYSTEM_CONTEXT))
        self.INPUT_HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_INPUT_USER_MESSAGE))
        self.INPUT_HANDLER_TEMPERATURE = handles.get('temperature', self.DEFAULT_INPUT_TEMPERATURE)
        self.INPUT_HANDLER_TOP_P = handles.get('top_p', self.DEFAULT_INPUT_TOP_P)
        self.INPUT_HANDLER_MAX_TOKENS = handles.get('max_tokens', self.DEFAULT_INPUT_MAX_TOKENS)
        self.INPUT_HANDLER_STREAM = handles.get('stream', self.DEFAULT_INPUT_STREAM)
        self.INPUT_HANDLER_ITERATE = handles.get('iterate', self.DEFAULT_INPUT_ITERATE)
        self.INPUT_HANDLER_JSON_OUTPUT = handles.get('json_output', handles.get('json_mode', self.DEFAULT_INPUT_JSON_OUTPUT))
        # Output handles
        self.OUTPUT_HANDLE_CONTENT = handles.get('output_content', handles.get('streaming', self.DEFAULT_OUTPUT_CONTENT))
        self.OUTPUT_HANDLE_GENERATED = handles.get('output_generated', handles.get('generated', self.DEFAULT_OUTPUT_GENERATED))
        self.OUTPUT_HANDLE_TOOL_CALLS = handles.get('output_tool_calls', self.DEFAULT_OUTPUT_TOOL_CALLS)
        # Tool input handle prefix (configurable via JSON data.handles.tool_prefix)
        self.INPUT_TOOL_PREFIX = handles.get('tool_prefix', self.DEFAULT_INPUT_TOOL_PREFIX)
        self._default_iterate = getattr(data, 'iterate', False)
        self._default_stream = data.stream
        self._default_json_output = data.json_output
        self._default_temperature = data.temperature
        self._default_top_p = data.top_p
        self._default_max_tokens = data.max_tokens
        self._base_extra_data = dict(data.extra_data or {})
        # allow re-execution inside Loop when requested
        self.iterate = self._default_iterate
        self.stream = self._default_stream
        self.json_output = self._default_json_output
        self.extra_data = dict(self._base_extra_data)
        self.generated = ''

    def _resolve_runtime_value(self, handle: str, default, value_type: str):
        if input_has_value(self.inputs, handle):
            return coerce_primitive_by_type(self.inputs[handle], value_type, field_name=handle)
        return default

    def _build_runtime_extra_data(self) -> dict:
        extra_data = dict(self._base_extra_data)
        has_temperature_input = input_has_value(self.inputs, self.INPUT_HANDLER_TEMPERATURE)
        has_top_p_input = input_has_value(self.inputs, self.INPUT_HANDLER_TOP_P)
        has_max_tokens_input = input_has_value(self.inputs, self.INPUT_HANDLER_MAX_TOKENS)
        runtime_temperature = self._resolve_runtime_value(self.INPUT_HANDLER_TEMPERATURE, self._default_temperature, 'float')
        runtime_top_p = self._resolve_runtime_value(self.INPUT_HANDLER_TOP_P, self._default_top_p, 'float')
        runtime_max_tokens = self._resolve_runtime_value(self.INPUT_HANDLER_MAX_TOKENS, self._default_max_tokens, 'int')

        if has_temperature_input:
            extra_data['temperature'] = runtime_temperature
        elif 'temperature' not in extra_data and runtime_temperature is not None:
            extra_data['temperature'] = runtime_temperature

        if has_top_p_input:
            extra_data['top_p'] = runtime_top_p
        elif 'top_p' not in extra_data and runtime_top_p is not None:
            extra_data['top_p'] = runtime_top_p

        if has_max_tokens_input:
            extra_data['max_tokens'] = runtime_max_tokens
        elif 'max_tokens' not in extra_data and runtime_max_tokens is not None:
            extra_data['max_tokens'] = runtime_max_tokens

        return extra_data

    async def _collect_tools(self) -> tuple[list, dict]:
        """Collect all tool definitions from tool-prefixed input handles.

        Scans self.inputs for keys starting with the tool prefix.
        Returns a tuple of (tools_schemas, tool_functions):
          - tools_schemas: list of OpenAI-compatible tool definition dicts
          - tool_functions: dict mapping tool name -> callable executor

        For objects conforming to the ToolProvider protocol, extracts
        both tool_schema and tool_callable. For plain callables, uses
        normalize_openai_tools for schema extraction. For plain dicts,
        uses them as-is (schema-only, no executor).

        Extended to support MCPToolBundle (multi-tool) inputs from MCP nodes:
          - MCPToolBundle.tool_schemas[] -> extend schema list
          - MCPToolBundle.tool_functions{} -> merge into functions dict
          - Collision detection raises MCPToolNameCollisionError
        
        Task subagents are loaded via MagicLLM.load_subagents() in process(),
        not collected here. magic-llm owns all subagent architecture.
        """
        from magic_agents.mcp.errors import MCPToolNameCollisionError
        
        tools_schemas = []
        tool_functions = {}

        for handle_name, value in sorted(self.inputs.items()):
            if not handle_name.startswith(self.INPUT_TOOL_PREFIX):
                continue
            if value is None:
                continue

            # Bundle path: MCPToolBundle with multiple tools
            if hasattr(value, 'tool_schemas') and hasattr(value, 'tool_functions'):
                # MCPToolBundle (or similar multi-tool container)
                bundle_node_id = getattr(value, 'node_id', 'unknown')
                bundle_prefix = getattr(value, 'prefix', '')
                bundle_count = len(value.tool_schemas)
                
                # Collision detection: check for duplicates before merging
                for tool_name in value.tool_functions.keys():
                    if tool_name in tool_functions:
                        # Collision detected - find source nodes
                        existing_source = getattr(tool_functions[tool_name], '_mcp_node_id', None) or 'existing_tool'
                        raise MCPToolNameCollisionError(
                            tool_name=tool_name,
                            source_nodes=[existing_source, bundle_node_id]
                        )
                
                # Extend schemas and merge functions
                tools_schemas.extend(value.tool_schemas)
                
                # Tag functions with their source node_id for collision tracking
                for tool_name, func in value.tool_functions.items():
                    # Add source metadata for collision detection
                    if not hasattr(func, '_mcp_node_id'):
                        func._mcp_node_id = bundle_node_id
                    tool_functions[tool_name] = func
                
                logger.debug(
                    "NodeLLM:%s flattened MCP bundle from node '%s': %d tools (prefix='%s')",
                    self.node_id,
                    bundle_node_id,
                    bundle_count,
                    bundle_prefix
                )
            
            # Single-tool path: existing ToolProvider (FetchToolCallable, PythonExecutor)
            elif hasattr(value, 'tool_schema') and hasattr(value, 'tool_callable'):
                tools_schemas.append(value.tool_schema)
                if value.tool_callable is not None:
                    name = getattr(value.tool_callable, '__name__', None)
                    if name:
                        # Collision detection for single tools too
                        if name in tool_functions:
                            raise MCPToolNameCollisionError(
                                tool_name=name,
                                source_nodes=['existing_tool', self.node_id]
                            )
                        tool_functions[name] = value.tool_callable
            
            # Plain callable: schema auto-extracted by magic-llm
            elif callable(value) and not isinstance(value, dict):
                tools_schemas.append(value)
                name = getattr(value, '__name__', None)
                if name:
                    if name in tool_functions:
                        raise MCPToolNameCollisionError(
                            tool_name=name,
                            source_nodes=['existing_tool', self.node_id]
                        )
                    tool_functions[name] = value
            
            # Schema-only dict (no executor)
            elif isinstance(value, dict):
                tools_schemas.append(value)

        # NOTE: Task subagents are loaded via MagicLLM.load_subagents() 
        # in process() where the client is available.
        # magic-llm owns ALL subagent architecture — no local registry.

        return tools_schemas, tool_functions

    def _warn_unsupported_engine(self, client) -> None:
        """Log a warning if the configured engine does not support tools."""
        engine = getattr(client.llm, 'engine_name', '') or getattr(client, 'engine', '')
        engine_lower = (engine or '').lower()
        for unsupported in self._UNSUPPORTED_ENGINES:
            if unsupported in engine_lower:
                logger.warning(
                    "NodeLLM:%s engine '%s' does not support tools — "
                    "tools will be passed but may be ignored by the provider. "
                    "This is a known limitation (WIP).",
                    self.node_id, engine
                )
                break
    
    async def _load_subagents_if_enabled(self, client: MagicLLM) -> "SubagentBundle":
        """Load task subagents via magic-llm unified API if feature enabled.
        
        ARCHITECTURE (Option 1 stricter boundary):
        - magic-llm owns ALL subagent architecture
        - magic-agents passes manifest_dir and code_registry
        - load_subagents() handles discovery → registration internally
        
        Args:
            client: MagicLLM instance with load_subagents() method
            
        Returns:
            SubagentBundle with tool schemas for agent loop injection
        """
        from magic_llm.agent.bundle import SubagentBundle
        from magic_agents.agt_flow import is_task_subagents_enabled, get_code_registry
        from pathlib import Path
        
        # Check application-level feature flag
        if not is_task_subagents_enabled():
            logger.debug("NodeLLM:%s task subagents feature disabled", self.node_id)
            return SubagentBundle()
        
        # Check if client has load_subagents() (requires magic-llm 0.1.28+)
        if not hasattr(client, 'load_subagents'):
            logger.warning(
                "NodeLLM:%s task subagents enabled but magic-llm lacks load_subagents() "
                "— upgrade magic-llm to version 0.1.28 or higher.",
                self.node_id
            )
            return SubagentBundle()
        
        # Default manifest directory (can be overridden in future)
        manifest_dir = Path("subagents")
        
        # Get application-level code registry
        code_registry = get_code_registry()
        
        # Reset depth counters for new execution (magic-llm handles internally)
        if hasattr(client, 'reset_depths'):
            client.reset_depths()
        
        # Call magic-llm unified API
        bundle = await client.load_subagents(manifest_dir, code_registry)
        
        if bundle.registered_count > 0:
            logger.info(
                "NodeLLM:%s loaded %d task subagents via MagicLLM.load_subagents()",
                self.node_id,
                bundle.registered_count
            )
        
        return bundle

    def _create_hook_relay(self, client: Any = None) -> Any:
        """Create a HookRelay for tool call/result collection.

        When hooks are registered (self._hooks set by Node.__call__), creates
        a HookRelay that bridges magic-agents hooks to magic-llm's AgentHooks
        Protocol and collects tool call/result data.

        When no hooks are registered, creates a standalone HookRelay collector
        that still captures on_tool_start/on_tool_complete data without
        invoking any flow hooks. This ensures the streaming tool path can
        emit TOOL_CALL/TOOL_RESULT debug events for persistence even for
        graphs without configured hooks.

        Args:
            client: Optional MagicLLM client. When provided, builds a best-effort
                llm_config dict with model, provider, streaming, and json_output
                fields for hook event propagation.

        Returns:
            HookRelay instance (always, never None).
        """
        from magic_agents.hooks.hook_relay import HookRelay

        # Build best-effort llm_config from available NodeLLM state
        llm_config: dict[str, Any] = {}
        if client is not None:
            llm_config["model"] = getattr(client.llm, 'model', '')
            llm_config["provider"] = (
                getattr(client.llm, 'engine_name', '')
                or getattr(client, 'engine', '')
            )
        llm_config["streaming"] = self.stream
        llm_config["json_output"] = self.json_output

        hooks = getattr(self, '_hooks', None)
        if hooks is not None and not hooks.is_empty():
            return HookRelay(
                registry=hooks,
                node_id=self.node_id or '',
                graph_id=getattr(hooks, 'execution_id', ''),
                run_id=getattr(hooks, 'run_id', ''),
                llm_config=llm_config,
                timeout=30.0,  # Increased from 5s default for persistence hooks
            )

        return HookRelay(
            node_id=self.node_id or '',
            llm_config=llm_config,
        )

    async def process(self, chat_log):
        self.stream = self._resolve_runtime_value(self.INPUT_HANDLER_STREAM, self._default_stream, 'bool')
        self.iterate = self._resolve_runtime_value(self.INPUT_HANDLER_ITERATE, self._default_iterate, 'bool')
        self.json_output = self._resolve_runtime_value(self.INPUT_HANDLER_JSON_OUTPUT, self._default_json_output, 'bool')
        self.extra_data = self._build_runtime_extra_data()
        params = self.inputs
        # Avoid logging full params to prevent leaking content; log keys only
        logger.debug("NodeLLM:%s inputs keys: %s", self.node_id, list(params.keys()))
        no_inputs = False
        if not params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT) and not params.get(self.INPUT_HANDLER_USER_MESSAGE):
            no_inputs = True

        def extract_message(msg):
            """Extract string message from various input types."""
            if isinstance(msg, str):
                return msg
            if isinstance(msg, list):
                return json.dumps(msg)
            if isinstance(msg, dict):
                # From conditional: try 'value' key first, then 'content', then serialize
                if 'value' in msg:
                    return extract_message(msg['value'])
                if 'content' in msg:
                    return extract_message(msg['content'])
                return json.dumps(msg)
            return str(msg) if msg is not None else ''

        client: MagicLLM = self.get_input(self.INPUT_HANDLER_CLIENT_PROVIDER, required=True)
        if c := params.get(self.INPUT_HANDLER_CHAT):
            chat = c
            if sys_prompt := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                chat.set_system(extract_message(sys_prompt))
            if user_prompt := self.get_input(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(extract_message(user_prompt))
        else:
            if no_inputs:
                logger.debug("NodeLLM:%s no inputs provided; yielding empty content", self.node_id)
                yield self.yield_static('', content_type=self.OUTPUT_HANDLE_GENERATED)
                return
            sys_context = params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT)
            chat = ModelChat(extract_message(sys_context) if sys_context else None)
            if k := params.get(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(extract_message(k))
            else:
                logger.error("NodeLLM:%s missing required input '%s'", self.node_id, self.INPUT_HANDLER_USER_MESSAGE)
                yield self.yield_debug_error(
                    error_type="InputError",
                    error_message=f"NodeLLM requires input '{self.INPUT_HANDLER_USER_MESSAGE}' with a user message.",
                    context={
                        "available_inputs": list(params.keys()),
                        "required_input": self.INPUT_HANDLER_USER_MESSAGE,
                        "node_config": {"stream": self.stream, "json_output": self.json_output}
                    }
                )
                return

        # Collect tools from tool-prefixed input handles
        tools_schemas, tool_functions = await self._collect_tools()
        
        # Load task subagents via magic-llm unified API (if feature enabled)
        # magic-llm handles discovery → registration → tool schema injection
        subagent_bundle = await self._load_subagents_if_enabled(client)
        
        # Extend schemas with subagent tool schemas if any registered
        if subagent_bundle.registered_count > 0:
            tools_schemas.extend(subagent_bundle.tool_schemas)
            # NOTE: tool_functions NOT merged — magic-llm routes via TaskExecutor
            # which wraps callables with safeguards (depth, timeout, semaphore)

        # Track tool calls for the handle-tool-calls output
        final_tool_calls: list = []

        if not self.stream:
            logger.info("NodeLLM:%s generating (non-stream) with model=%s", self.node_id, client.llm.model)

            if tool_functions:
                # Tool-enabled path: delegate to magic-llm's canonical agent loop
                self._warn_unsupported_engine(client)

                user_msg = extract_message(self.get_input(self.INPUT_HANDLER_USER_MESSAGE, ''))
                sys_msg = None
                if sys_ctx := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                    sys_msg = extract_message(sys_ctx)

                if not hasattr(client, 'run_agent_async'):
                    # Fallback: wrap sync run_agent via asyncio.to_thread
                    logger.warning(
                        "NodeLLM:%s async agent loop not available in installed magic-llm version — "
                        "falling back to sync run_agent() via asyncio.to_thread(). "
                        "Hook events will NOT be delivered via this path. "
                        "Upgrade magic-llm for native async hook support.",
                        self.node_id
                    )
                    if not hasattr(client, 'run_agent'):
                        raise RuntimeError(
                            f"NodeLLM:{self.node_id} has callable tools but magic-llm does not "
                            f"provide client.run_agent_async() or client.run_agent(). Upgrade magic-llm."
                        )
                    hook_relay = self._create_hook_relay(client=client)
                    intention = await asyncio.to_thread(
                        client.run_agent,
                        user_input=user_msg,
                        system_prompt=sys_msg,
                        tools=tools_schemas,
                        tool_functions=tool_functions,
                        hooks=hook_relay,
                        **self.extra_data
                    )
                    await hook_relay.flush_pending_hooks()
                else:
                    hook_relay = self._create_hook_relay(client=client)
                    intention = await client.run_agent_async(
                        user_input=user_msg,
                        system_prompt=sys_msg,
                        tools=tools_schemas,
                        tool_functions=tool_functions,
                        hooks=hook_relay,
                        **self.extra_data
                    )
                    await hook_relay.flush_pending_hooks()
                self.generated = intention.content
                # Use REAL collected tool calls, not intention.tool_calls (always empty for callable-tool path)
                if hook_relay is not None and hook_relay.collected_tool_calls:
                    final_tool_calls = list(hook_relay.collected_tool_calls)
                else:
                    final_tool_calls = getattr(intention, 'tool_calls', []) or []
                # P1-NEW: _emit_llm_generation removed for HookRelay path — data parity confirmed
                # (on_llm_end from HookRelay carries model, tokens, provider_request_id)

                # Emit tool_call/tool_result debug events for API persistence
                if hook_relay is not None:
                    for te in hook_relay.get_collected_tool_data_for_yield():
                        yield {
                            'type': 'debug',
                            'content': {
                                'event_type': 'TOOL_CALL' if te['type'] == 'tool_call' else 'TOOL_RESULT',
                                'node_id': self.node_id,
                                'data': te['data'],
                            }
                        }

            elif tools_schemas:
                # Schema-only tools: single generate call (LLM can reference tools but no executor)
                # === HOOK: on_llm_start (schema-only tools non-streaming path, Phase 0 R0.1) ===
                _llm_ctx = None
                if hasattr(self, '_hooks') and self._hooks is not None and not self._hooks.is_empty():
                    from magic_agents.hooks.context_factory import HookContextFactory
                    _llm_ctx = HookContextFactory.build_llm_context(
                        execution_id=getattr(self._hooks, 'execution_id', ''),
                        run_id=getattr(self._hooks, 'run_id', ''),
                        node_id=self.node_id,
                        node_type=self.node_type or '',
                        node_class=self.__class__.__name__,
                        model=client.llm.model,
                        streaming=False,
                        llm_config={
                            "model": client.llm.model,
                            "provider": getattr(client.llm, 'engine_name', ''),
                            "tools_schemas": tools_schemas,
                        },
                    )
                    await self._hooks.invoke("on_llm_start", _llm_ctx)

                self._warn_unsupported_engine(client)
                intention = await client.llm.async_generate(
                    chat, tools=tools_schemas, **self.extra_data
                )

                # === HOOK: on_llm_end (schema-only tools non-streaming path, Phase 0 R0.1) ===
                if _llm_ctx is not None:
                    usage = getattr(intention, 'usage', None)
                    finish_reason = None
                    if hasattr(intention, 'choices') and intention.choices:
                        finish_reason = intention.choices[0].finish_reason
                    _llm_ctx.outputs = {
                        "model": getattr(intention, 'model', ''),
                        "content": getattr(intention, 'content', ''),
                        "provider_request_id": getattr(intention, 'id', None),
                        "prompt_tokens": getattr(usage, 'prompt_tokens', None) if usage else None,
                        "completion_tokens": getattr(usage, 'completion_tokens', None) if usage else None,
                        "total_tokens": getattr(usage, 'total_tokens', None) if usage else None,
                        "finish_reason": finish_reason,
                    }
                    await self._hooks.invoke("on_llm_end", _llm_ctx)
                    # Single-call path — fire on_llm_loop_end with total_iterations: 1
                    _llm_ctx.outputs["total_iterations"] = 1
                    await self._hooks.invoke("on_llm_loop_end", _llm_ctx)

                self.generated = intention.content
                final_tool_calls = getattr(intention, 'tool_calls', []) or []
                # Phase 0: emit LLM_GENERATION for execution tree persistence
                # TODO: verify on_llm_end carries cached/reasoning/audio token fields
                # before removing _emit_llm_generation fallback (P1-NEW)
                yield self._emit_llm_generation(intention)

            else:
                # Existing path: no tools, single generation call
                # === HOOK: on_llm_start (non-tool path, Phase 9) ===
                _llm_ctx = None
                if hasattr(self, '_hooks') and self._hooks is not None and not self._hooks.is_empty():
                    from magic_agents.hooks.context_factory import HookContextFactory
                    _llm_ctx = HookContextFactory.build_llm_context(
                        execution_id=getattr(self._hooks, 'execution_id', ''),
                        run_id=getattr(self._hooks, 'run_id', ''),
                        node_id=self.node_id,
                        node_type=self.node_type or '',
                        node_class=self.__class__.__name__,
                        model=client.llm.model,
                        streaming=False,
                    )
                    await self._hooks.invoke("on_llm_start", _llm_ctx)

                intention = await client.llm.async_generate(chat, **self.extra_data)

                # === HOOK: on_llm_end (non-tool non-streaming path, Phase 0 R0.4) ===
                if _llm_ctx is not None:
                    usage = getattr(intention, 'usage', None)
                    finish_reason = None
                    if hasattr(intention, 'choices') and intention.choices:
                        finish_reason = intention.choices[0].finish_reason
                    _llm_ctx.outputs = {
                        "model": getattr(intention, 'model', ''),
                        "content": getattr(intention, 'content', ''),
                        "provider_request_id": getattr(intention, 'id', None),
                        "prompt_tokens": getattr(usage, 'prompt_tokens', None) if usage else None,
                        "completion_tokens": getattr(usage, 'completion_tokens', None) if usage else None,
                        "total_tokens": getattr(usage, 'total_tokens', None) if usage else None,
                        "finish_reason": finish_reason,
                    }
                    await self._hooks.invoke("on_llm_end", _llm_ctx)
                    # Single-call path — fire on_llm_loop_end with total_iterations: 1
                    _llm_ctx.outputs["total_iterations"] = 1
                    await self._hooks.invoke("on_llm_loop_end", _llm_ctx)

                self.generated = intention.content
                final_tool_calls = getattr(intention, 'tool_calls', []) or []
                # Phase 0: emit LLM_GENERATION for execution tree persistence
                # TODO: verify on_llm_end carries cached/reasoning/audio token fields
                # before removing _emit_llm_generation fallback (P1-NEW)
                yield self._emit_llm_generation(intention)

            yield self.yield_static(ChatCompletionModel(
                id=uuid.uuid4().hex,
                model=client.llm.model,
                choices=[ChoiceModel()],
                usage=intention.usage),
                content_type=self.OUTPUT_HANDLE_CONTENT)
        else:
            logger.info("NodeLLM:%s streaming generation with model=%s", self.node_id, client.llm.model)

            if tool_functions:
                # Streaming tool-enabled path
                self._warn_unsupported_engine(client)

                user_msg = extract_message(self.get_input(self.INPUT_HANDLER_USER_MESSAGE, ''))
                sys_msg = None
                if sys_ctx := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                    sys_msg = extract_message(sys_ctx)

                if not hasattr(client, 'run_agent_stream_async'):
                    # Fallback: wrap sync run_agent_stream via asyncio.to_thread
                    logger.warning(
                        "NodeLLM:%s async streaming agent loop not available in installed magic-llm version — "
                        "falling back to sync run_agent_stream() via asyncio.to_thread(). "
                        "Hook events will NOT be delivered via this path. "
                        "Upgrade magic-llm for native async hook support.",
                        self.node_id
                    )
                    if not hasattr(client, 'run_agent_stream'):
                        raise RuntimeError(
                            f"NodeLLM:{self.node_id} has callable tools with streaming but magic-llm does not "
                            f"provide client.run_agent_stream_async() or client.run_agent_stream(). Upgrade magic-llm."
                        )

                    hook_relay = self._create_hook_relay(client=client)
                    last_chunk = None
                    try:
                        for chunk in await asyncio.to_thread(
                            client.run_agent_stream,
                            user_input=user_msg,
                            system_prompt=sys_msg,
                            tools=tools_schemas,
                            tool_functions=tool_functions,
                            hooks=hook_relay,
                            **self.extra_data
                        ):
                            self.generated += chunk.choices[0].delta.content or ''
                            last_chunk = chunk
                            yield self.yield_static(chunk, content_type=self.OUTPUT_HANDLE_CONTENT)
                        if last_chunk:
                            if hook_relay is not None and hook_relay.collected_tool_calls:
                                final_tool_calls = list(hook_relay.collected_tool_calls)
                            else:
                                final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
                            # P1-NEW: _emit_llm_generation removed for HookRelay path — data parity confirmed
                            if hook_relay is not None:
                                for te in hook_relay.get_collected_tool_data_for_yield():
                                    yield {
                                        'type': 'debug',
                                        'content': {
                                            'event_type': 'TOOL_CALL' if te['type'] == 'tool_call' else 'TOOL_RESULT',
                                            'node_id': self.node_id,
                                            'data': te['data'],
                                        }
                                    }
                    finally:
                        await hook_relay.flush_pending_hooks()
                else:
                    hook_relay = self._create_hook_relay(client=client)
                    last_chunk = None
                    try:
                        async for chunk in client.run_agent_stream_async(
                            user_input=user_msg,
                            system_prompt=sys_msg,
                            tools=tools_schemas,
                            tool_functions=tool_functions,
                            hooks=hook_relay,
                            task_executor=getattr(subagent_bundle, 'task_executor', None),
                            **self.extra_data
                        ):
                            self.generated += chunk.choices[0].delta.content or ''
                            last_chunk = chunk
                            yield self.yield_static(chunk, content_type=self.OUTPUT_HANDLE_CONTENT)
                        # Capture tool_calls from the last chunk
                        if last_chunk:
                            if hook_relay is not None and hook_relay.collected_tool_calls:
                                final_tool_calls = list(hook_relay.collected_tool_calls)
                            else:
                                final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
                            # P1-NEW: _emit_llm_generation removed for HookRelay path — data parity confirmed
                            if hook_relay is not None:
                                for te in hook_relay.get_collected_tool_data_for_yield():
                                    yield {
                                        'type': 'debug',
                                        'content': {
                                            'event_type': 'TOOL_CALL' if te['type'] == 'tool_call' else 'TOOL_RESULT',
                                            'node_id': self.node_id,
                                            'data': te['data'],
                                        }
                                    }
                    finally:
                        await hook_relay.flush_pending_hooks()
            elif tools_schemas:
                # Schema-only tools with streaming
                # === HOOK: on_llm_start (schema-only tools streaming path, Phase 0 R0.2) ===
                _llm_ctx = None
                if hasattr(self, '_hooks') and self._hooks is not None and not self._hooks.is_empty():
                    from magic_agents.hooks.context_factory import HookContextFactory
                    _llm_ctx = HookContextFactory.build_llm_context(
                        execution_id=getattr(self._hooks, 'execution_id', ''),
                        run_id=getattr(self._hooks, 'run_id', ''),
                        node_id=self.node_id,
                        node_type=self.node_type or '',
                        node_class=self.__class__.__name__,
                        model=client.llm.model,
                        streaming=True,
                        llm_config={
                            "model": client.llm.model,
                            "provider": getattr(client.llm, 'engine_name', ''),
                            "tools_schemas": tools_schemas,
                        },
                    )
                    await self._hooks.invoke("on_llm_start", _llm_ctx)

                self._warn_unsupported_engine(client)
                last_chunk = None
                async for i in client.llm.async_stream_generate(chat, tools=tools_schemas, **self.extra_data):
                    self.generated += i.choices[0].delta.content or ''
                    last_chunk = i
                    yield self.yield_static(i, content_type=self.OUTPUT_HANDLE_CONTENT)
                if last_chunk:
                    # === HOOK: on_llm_end (schema-only tools streaming path, Phase 0 R0.2) ===
                    if _llm_ctx is not None:
                        usage = getattr(last_chunk, 'usage', None)
                        finish_reason = None
                        if hasattr(last_chunk, 'choices') and last_chunk.choices:
                            finish_reason = last_chunk.choices[0].finish_reason
                        _llm_ctx.outputs = {
                            "model": getattr(last_chunk, 'model', ''),
                            "content": self.generated,
                            "provider_request_id": getattr(last_chunk, 'id', None),
                            "prompt_tokens": getattr(usage, 'prompt_tokens', None) if usage else None,
                            "completion_tokens": getattr(usage, 'completion_tokens', None) if usage else None,
                            "total_tokens": getattr(usage, 'total_tokens', None) if usage else None,
                            "finish_reason": finish_reason,
                        }
                        await self._hooks.invoke("on_llm_end", _llm_ctx)
                        # Single-call path — fire on_llm_loop_end with total_iterations: 1
                        _llm_ctx.outputs["total_iterations"] = 1
                        await self._hooks.invoke("on_llm_loop_end", _llm_ctx)

                    final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
                    # Phase 0: emit LLM_GENERATION for execution tree persistence
                    # TODO: verify on_llm_end carries cached/reasoning/audio token fields
                    # before removing _emit_llm_generation fallback (P1-NEW)
                    yield self._emit_llm_generation(last_chunk)
            else:
                # Existing streaming path: no tools
                # === HOOK: on_llm_start (non-tool streaming, Phase 9) ===
                _llm_ctx = None
                if hasattr(self, '_hooks') and self._hooks is not None and not self._hooks.is_empty():
                    from magic_agents.hooks.context_factory import HookContextFactory
                    _llm_ctx = HookContextFactory.build_llm_context(
                        execution_id=getattr(self._hooks, 'execution_id', ''),
                        run_id=getattr(self._hooks, 'run_id', ''),
                        node_id=self.node_id,
                        node_type=self.node_type or '',
                        node_class=self.__class__.__name__,
                        model=client.llm.model,
                        streaming=True,
                    )
                    await self._hooks.invoke("on_llm_start", _llm_ctx)

                last_chunk = None
                async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                    self.generated += i.choices[0].delta.content or ''
                    last_chunk = i
                    yield self.yield_static(i, content_type=self.OUTPUT_HANDLE_CONTENT)
                if last_chunk:
                    final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
                    # Phase 0: emit LLM_GENERATION for execution tree persistence
                    # TODO: verify on_llm_end carries cached/reasoning/audio token fields
                    # before removing _emit_llm_generation fallback (P1-NEW)
                    yield self._emit_llm_generation(last_chunk)

                # === HOOK: on_llm_end (non-tool streaming path, Phase 0 R0.4) ===
                if _llm_ctx is not None:
                    usage = getattr(last_chunk, 'usage', None) if last_chunk else None
                    finish_reason = None
                    if last_chunk and hasattr(last_chunk, 'choices') and last_chunk.choices:
                        finish_reason = last_chunk.choices[0].finish_reason
                    _llm_ctx.outputs = {
                        "model": getattr(last_chunk, 'model', '') if last_chunk else '',
                        "content": self.generated,
                        "provider_request_id": getattr(last_chunk, 'id', None) if last_chunk else None,
                        "prompt_tokens": getattr(usage, 'prompt_tokens', None) if usage else None,
                        "completion_tokens": getattr(usage, 'completion_tokens', None) if usage else None,
                        "total_tokens": getattr(usage, 'total_tokens', None) if usage else None,
                        "finish_reason": finish_reason,
                    }
                    await self._hooks.invoke("on_llm_end", _llm_ctx)
                    # Single-call path — fire on_llm_loop_end with total_iterations: 1
                    _llm_ctx.outputs["total_iterations"] = 1
                    await self._hooks.invoke("on_llm_loop_end", _llm_ctx)
        # if self.json_output:
        #     print(self.generated)
        #     self.generated = json.loads(self.generated)

        if self.json_output:
            logger.debug("NodeLLM:%s parsing JSON output", self.node_id)

            # Extract JSON from markdown code blocks or plain text
            json_content = None

            # Try to find JSON in markdown code blocks (```json or ```)
            json_pattern = r'```(?:json)?\s*(.*?)\s*```'
            matches = re.findall(json_pattern, self.generated, re.DOTALL)

            if matches:
                # Use the first JSON block found
                json_content = matches[0].strip()
            else:
                # If no code blocks found, try to extract JSON from the entire string
                # Look for content that starts with { and ends with }
                brace_pattern = r'\{.*\}'
                match = re.search(brace_pattern, self.generated, re.DOTALL)
                if match:
                    json_content = match.group().strip()
                else:
                    json_content = self.generated.strip()
            if json_content:
                try:
                    self.generated = json.loads(json_content)
                    logger.debug("NodeLLM:%s JSON parsed successfully", self.node_id)
                except json.JSONDecodeError as e:
                    logger.error("NodeLLM:%s JSON parsing failed: %s", self.node_id, e)
                    yield self.yield_debug_error(
                        error_type="JSONParseError",
                        error_message=f"Failed to parse JSON content: {str(e)}",
                        context={
                            "json_content_preview": json_content[:200] if len(json_content) > 200 else json_content,
                            "full_generated_length": len(self.generated),
                            "error_position": getattr(e, 'pos', None)
                        }
                    )
                    return
            else:
                logger.error("NodeLLM:%s no JSON content found in generated output (%d chars)", self.node_id, len(self.generated))
                yield self.yield_debug_error(
                    error_type="JSONExtractionError",
                    error_message="No JSON content found in generated output. The LLM did not produce valid JSON.",
                    context={
                        "generated_preview": self.generated[:500] if len(self.generated) > 500 else self.generated,
                        "generated_length": len(self.generated),
                        "json_output_required": self.json_output,
                        "model": getattr(client.llm, 'model', 'unknown') if 'client' in locals() else 'unknown'
                    }
                )
                return
        # Yield tool calls on dedicated handle ONLY when tools are present (zero-regression for no-tool graphs)
        if tools_schemas or tool_functions:
            yield self.yield_static(final_tool_calls, content_type=self.OUTPUT_HANDLE_TOOL_CALLS)
        # Yield on the configured output handle
        yield self.yield_static(self.generated, content_type=self.OUTPUT_HANDLE_GENERATED)

    def _emit_llm_generation(self, intention, duration_ms: Optional[float] = None) -> dict:
        """Emit a structured LLM_GENERATION debug event for execution tree persistence.
        
        Phase 0 cross-repo instrumentation: called after every LLM provider response
        (both stream and non-stream). The event includes model, token counts,
        duration, and provider_request_id.
        
        Args:
            intention: The ChatCompletionModel or synthetic response.
            duration_ms: Optional measured call duration.
            
        Returns:
            Debug event dict suitable for yielding via SYSTEM_EVENT_DEBUG channel.
        """
        usage = getattr(intention, 'usage', None)
        event_payload = {
            'event_type': 'LLM_GENERATION',
            'node_id': self.node_id,
            'model': getattr(intention, 'model', 'unknown'),
            'provider_request_id': getattr(intention, 'id', None),
            'prompt_tokens': getattr(usage, 'prompt_tokens', 0) if usage else 0,
            'completion_tokens': getattr(usage, 'completion_tokens', 0) if usage else 0,
            'total_tokens': getattr(usage, 'total_tokens', 0) if usage else 0,
            'cached_tokens_read': getattr(usage, 'cached_read_tokens', 0) if usage else 0,
            'cached_tokens_write': getattr(usage, 'cached_write_tokens', 0) if usage else 0,
            'reasoning_tokens': getattr(usage, 'reasoning_tokens', 0) if usage else 0,
            'audio_tokens': getattr(usage, 'audio_tokens', 0) if usage else 0,
        }
        if duration_ms is not None:
            event_payload['duration_ms'] = duration_ms
        return {
            'type': 'debug',
            'content': event_payload,
        }

    def _capture_internal_state(self):
        """Capture LLM-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add LLM-specific variables
        state['stream'] = self.stream
        state['json_output'] = self.json_output
        state['iterate'] = self.iterate
        state['generated'] = self.generated[:500] if len(self.generated) > 500 else self.generated  # Truncate long outputs
        state['extra_data'] = self.extra_data
        
        return state
    
    async def __call__(self, chat_log, **kwargs):
        # if configured to iterate inside a Loop, always re-run instead of using cached response
        if getattr(self, 'iterate', False):
            self._response = None
        async for result in super().__call__(chat_log, **kwargs):
            yield result
