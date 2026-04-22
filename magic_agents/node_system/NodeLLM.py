import asyncio
import json
import re
import uuid
import logging
from typing import Optional, TYPE_CHECKING

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.Node import Node

if TYPE_CHECKING:
    from magic_llm.agent import TaskManifest

logger = logging.getLogger(__name__)


def _to_task_manifest(subagent_manifest) -> 'TaskManifest':
    """Convert SubagentManifest to TaskManifest for magic-llm registration.
    
    Excludes YAML-specific fields (apiVersion, kind, version, source_file, etc.)
    Keeps runtime policy fields: id, name, description, input_schema,
    timeout_seconds, max_concurrency, max_depth.
    
    Args:
        subagent_manifest: SubagentManifest from magic_agents
        
    Returns:
        TaskManifest for MagicLLM.register_task()
    """
    from magic_llm.agent import TaskManifest
    
    return TaskManifest(
        id=subagent_manifest.id,
        name=subagent_manifest.name,
        description=subagent_manifest.description,
        input_schema=subagent_manifest.input_schema,
        timeout_seconds=subagent_manifest.timeout_seconds,
        max_concurrency=subagent_manifest.max_concurrency,
        max_depth=subagent_manifest.max_depth,
    )


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
    # Output handles
    DEFAULT_OUTPUT_CONTENT = 'handle_streaming_content'
    DEFAULT_OUTPUT_GENERATED = 'handle_generated_content'
    DEFAULT_OUTPUT_TOOL_CALLS = 'handle-tool-calls'
    # Legacy output handle for backward compatibility with existing graphs
    LEGACY_OUTPUT_GENERATED = 'handle_generated_end'
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
        # Output handles
        self.OUTPUT_HANDLE_CONTENT = handles.get('output_content', handles.get('streaming', self.DEFAULT_OUTPUT_CONTENT))
        self.OUTPUT_HANDLE_GENERATED = handles.get('output_generated', handles.get('generated', self.DEFAULT_OUTPUT_GENERATED))
        self.OUTPUT_HANDLE_TOOL_CALLS = handles.get('output_tool_calls', self.DEFAULT_OUTPUT_TOOL_CALLS)
        # Tool input handle prefix (configurable via JSON data.handles.tool_prefix)
        self.INPUT_TOOL_PREFIX = handles.get('tool_prefix', self.DEFAULT_INPUT_TOOL_PREFIX)
        # allow re-execution inside Loop when requested
        self.iterate = getattr(data, 'iterate', False)
        self.stream = data.stream
        self.json_output = data.json_output
        self.extra_data = data.extra_data
        self.generated = ''
        if 'temperature' not in self.extra_data and data.temperature is not None:
            self.extra_data['temperature'] = data.temperature
        if 'top_p' not in self.extra_data and data.top_p is not None:
            self.extra_data['top_p'] = data.top_p
        if 'max_tokens' not in self.extra_data and data.max_tokens is not None:
            self.extra_data['max_tokens'] = data.max_tokens

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

        Extended to support TaskToolBundle from subagent registry:
          - TaskToolBundle.tool_schemas[] -> extend schema list
          - TaskToolBundle.tool_functions{} -> merge into functions dict
          - Feature flag: is_task_subagents_enabled() controls injection
        
        See `magic_agents.subagents.__init__` for subagent registration details.
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

        # NEW: Inject registered task subagents from registry (if feature enabled)
        # Feature flag: check if subagents module is initialized and enabled
        # Store task_bundle for later registration in process() where client is available
        self._pending_task_bundle = None
        try:
            from magic_agents.subagents import get_registry, TaskToolBundle
            from magic_agents.subagents.config import is_task_subagents_enabled
            
            if is_task_subagents_enabled():
                registry = get_registry()
                if registry.is_initialized() and registry.get_registered_ids():
                    # Build TaskToolBundle from registered subagents
                    task_bundle = await TaskToolBundle.from_registry(registry)
                    
                    if task_bundle.registered_count > 0:
                        # Collision detection: check for duplicates before merging
                        for tool_name in task_bundle.tool_functions.keys():
                            if tool_name in tool_functions:
                                raise MCPToolNameCollisionError(
                                    tool_name=tool_name,
                                    source_nodes=['existing_tool', 'subagent_registry']
                                )
                        
                        # Store bundle for registration in process()
                        self._pending_task_bundle = task_bundle
                        
                        # Extend schemas (schemas are needed for LLM tool selection)
                        tools_schemas.extend(task_bundle.tool_schemas)
                        
                        # NOTE: tool_functions NOT merged here - registration via
                        # MagicLLM.register_task() handles execution safeguards.
                        # The callables are passed through for backward compatibility,
                        # but TaskExecutor routes via _task_registry for task-specific
                        # handling (depth, timeout, semaphore).
                        tool_functions.update(task_bundle.tool_functions)
                        
                        logger.debug(
                            "NodeLLM:%s collected %d task subagents (registration pending)",
                            self.node_id,
                            task_bundle.registered_count
                        )
        except ImportError:
            # Subagents module not available - skip injection
            pass
        except RuntimeError:
            # Registry not initialized - skip injection
            pass

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
    
    def _register_task_subagents(self, client: MagicLLM) -> None:
        """Register task subagents with MagicLLM before agent loop execution.
        
        THIN WRAPPER ARCHITECTURE:
        - magic-agents collects manifests and callables
        - MagicLLM.register_task() wraps with safeguards (depth, timeout, semaphore)
        - TaskExecutor handles execution routing
        
        This method is called in process() after _collect_tools() collects
        the pending task bundle, and before run_agent_async() starts the loop.
        
        Args:
            client: MagicLLM instance with register_task() method
        """
        if not self._pending_task_bundle:
            return
        
        # Check if client has register_task() (requires updated magic-llm)
        if not hasattr(client, 'register_task'):
            logger.warning(
                "NodeLLM:%s task subagents collected but magic-llm lacks register_task() "
                "— falling back to legacy tool_functions path. "
                "Upgrade magic-llm to version with TaskExecutor support.",
                self.node_id
            )
            return
        
        bundle = self._pending_task_bundle
        
        for manifest in bundle.manifests:
            callable = bundle.tool_callables.get(manifest.id)
            if callable is None:
                logger.warning(
                    "NodeLLM:%s callable missing for manifest '%s' — skipping",
                    self.node_id,
                    manifest.id
                )
                continue
            
            # Convert SubagentManifest → TaskManifest
            task_manifest = _to_task_manifest(manifest)
            
            # Register with magic-llm (creates semaphore, wraps callable)
            client.register_task(task_manifest, callable)
            
            logger.debug(
                "NodeLLM:%s registered task '%s' with magic-llm TaskExecutor",
                self.node_id,
                manifest.id
            )
        
        logger.info(
            "NodeLLM:%s registered %d task subagents via MagicLLM.register_task()",
            self.node_id,
            len(bundle.manifests)
        )
        
        # Clear pending bundle after registration
        self._pending_task_bundle = None

    async def process(self, chat_log):
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
                # Also yield on legacy handle for backward compatibility
                if self.OUTPUT_HANDLE_GENERATED != self.LEGACY_OUTPUT_GENERATED:
                    yield self.yield_static('', content_type=self.LEGACY_OUTPUT_GENERATED)
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
        
        # Register task subagents with MagicLLM (if collected)
        # This happens BEFORE agent loop execution, ensuring TaskExecutor
        # wraps callables with depth/timeout/semaphore safeguards
        self._register_task_subagents(client)

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
                        "Upgrade magic-llm for native async support.",
                        self.node_id
                    )
                    if not hasattr(client, 'run_agent'):
                        raise RuntimeError(
                            f"NodeLLM:{self.node_id} has callable tools but magic-llm does not "
                            f"provide client.run_agent_async() or client.run_agent(). Upgrade magic-llm."
                        )
                    intention = await asyncio.to_thread(
                        client.run_agent,
                        user_input=user_msg,
                        system_prompt=sys_msg,
                        tools=tools_schemas,
                        tool_functions=tool_functions,
                        **self.extra_data
                    )
                else:
                    intention = await client.run_agent_async(
                        user_input=user_msg,
                        system_prompt=sys_msg,
                        tools=tools_schemas,
                        tool_functions=tool_functions,
                        **self.extra_data
                    )
                self.generated = intention.content
                final_tool_calls = getattr(intention, 'tool_calls', []) or []

            elif tools_schemas:
                # Schema-only tools: single generate call (LLM can reference tools but no executor)
                self._warn_unsupported_engine(client)
                intention = await client.llm.async_generate(
                    chat, tools=tools_schemas, **self.extra_data
                )
                self.generated = intention.content
                final_tool_calls = getattr(intention, 'tool_calls', []) or []

            else:
                # Existing path: no tools, single generation call
                intention = await client.llm.async_generate(chat, **self.extra_data)
                self.generated = intention.content
                final_tool_calls = getattr(intention, 'tool_calls', []) or []

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
                        "Upgrade magic-llm for native async support.",
                        self.node_id
                    )
                    if not hasattr(client, 'run_agent_stream'):
                        raise RuntimeError(
                            f"NodeLLM:{self.node_id} has callable tools with streaming but magic-llm does not "
                            f"provide client.run_agent_stream_async() or client.run_agent_stream(). Upgrade magic-llm."
                        )

                    last_chunk = None
                    for chunk in await asyncio.to_thread(
                        client.run_agent_stream,
                        user_input=user_msg,
                        system_prompt=sys_msg,
                        tools=tools_schemas,
                        tool_functions=tool_functions,
                        **self.extra_data
                    ):
                        self.generated += chunk.choices[0].delta.content or ''
                        last_chunk = chunk
                        yield self.yield_static(chunk, content_type=self.OUTPUT_HANDLE_CONTENT)
                    if last_chunk:
                        final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
                else:
                    last_chunk = None
                    async for chunk in client.run_agent_stream_async(
                        user_input=user_msg,
                        system_prompt=sys_msg,
                        tools=tools_schemas,
                        tool_functions=tool_functions,
                        **self.extra_data
                    ):
                        self.generated += chunk.choices[0].delta.content or ''
                        last_chunk = chunk
                        yield self.yield_static(chunk, content_type=self.OUTPUT_HANDLE_CONTENT)
                    # Capture tool_calls from the last chunk
                    if last_chunk:
                        final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
            elif tools_schemas:
                # Schema-only tools with streaming
                self._warn_unsupported_engine(client)
                last_chunk = None
                async for i in client.llm.async_stream_generate(chat, tools=tools_schemas, **self.extra_data):
                    self.generated += i.choices[0].delta.content or ''
                    last_chunk = i
                    yield self.yield_static(i, content_type=self.OUTPUT_HANDLE_CONTENT)
                if last_chunk:
                    final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
            else:
                # Existing streaming path: no tools
                last_chunk = None
                async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                    self.generated += i.choices[0].delta.content or ''
                    last_chunk = i
                    yield self.yield_static(i, content_type=self.OUTPUT_HANDLE_CONTENT)
                if last_chunk:
                    final_tool_calls = getattr(last_chunk.choices[0].delta, 'tool_calls', []) or []
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
        # Also yield on legacy handle for backward compatibility with existing graphs
        # This ensures graphs using 'handle_generated_end' still work
        if self.OUTPUT_HANDLE_GENERATED != self.LEGACY_OUTPUT_GENERATED:
            yield self.yield_static(self.generated, content_type=self.LEGACY_OUTPUT_GENERATED)

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
    
    async def __call__(self, chat_log):
        # if configured to iterate inside a Loop, always re-run instead of using cached response
        if getattr(self, 'iterate', False):
            self._response = None
        async for result in super().__call__(chat_log):
            yield result
