from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Dict, AsyncGenerator, Optional, List

import jinja2
from jinja2 import UndefinedError, TemplateSyntaxError, TemplateError

from magic_agents.node_system.Node import Node
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalNodeModel, ConditionalSignalTypes
from magic_agents.execution.condition_evaluator import ConditionEvaluator
from magic_agents.execution.condition_evaluator_jinja2 import Jinja2Evaluator, Jinja2StrictEvaluator
from magic_agents.execution.condition_evaluator_llm import build_judgment_chat, parse_judgments, validate_provider_answer
from magic_agents.execution.condition_evaluator_jev import evaluate_jev
from magic_agents.execution.condition_judgment_contract import validate_judgment_answers
from magic_agents.util.llm_usage import usage_detail_outputs

logger = logging.getLogger(__name__)


class NodeConditional(Node):
    """Route state deterministically, optionally after typed Jev or LLM judgments.
    
    Handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.

    The *condition* template must render to the **name of the output handle** that
    should continue execution. All other outgoing handles are considered bypassed.

    Supports multiple inputs that are merged into a single context for evaluation.

    Inputs (configurable via data.handles)
    ------
    handle_input (default): Any (required)
        Primary JSON string or Python object used as the context when evaluating
        the Jinja2 template.
    Additional inputs will be merged according to merge_strategy.

    Data fields (kwargs)
    --------------------
    condition : str (required)
        Jinja2 template string that will be rendered with the parsed input as
        its context. The rendered string decides which handle is emitted.
    merge_strategy : str (optional, default='flat')
        How to merge multiple inputs:
        - 'flat': Merge all inputs into single flat dict (later overrides earlier)
        - 'namespaced': Keep inputs separate under their handle names
    handles : dict (optional)
        Override default handle names. Example: {"input": "my_custom_input"}
    output_handles : list (optional)
        Declared output handle names for build-time validation.
    default_handle : str (optional)
        Fallback handle if condition evaluates to empty/invalid string.
    evaluation_mode : str (optional, default='jinja')
        'llm' and 'jev' evaluate typed questions against state before routing. The
        condition can read `answers` and `state`; branch payloads retain the
        original merged state. 'jinja' preserves existing exact-rule behavior.
    questions : dict (required in llm and jev modes)
        Independent named choice, score, or noul questions in Jev's grammar.
    evaluation_timeout : float (optional, default=30)
        Maximum seconds for judgment evaluation, including Jev retries.
    jev : dict (optional)
        Jev base_url (default https://api.typesafe.ai/v1) and optional api_key.
        Missing keys use server JEV_API_KEY; Jev never needs an LLM provider.
    evaluation_error_policy : str (optional, default='fail')
        'default' routes original state to the declared default_handle when
        judgment evaluation fails. Condition and graph routing errors still fail.
    """
    # Default handle name - can be overridden by JSON data.handles
    DEFAULT_INPUT_HANDLE_CTX = "handle_input"
    DEFAULT_INPUT_HANDLE_CLIENT = "handle-client-provider"

    def __init__(
        self,
        *,
        condition: str = None,
        merge_strategy: str = "flat",
        handles: Optional[dict] = None,
        output_handles: Optional[List[str]] = None,
        default_handle: Optional[str] = None,
        evaluator: Optional[ConditionEvaluator] = None,
        evaluation_mode: str = "jinja",
        questions: Optional[Dict[str, Any]] = None,
        evaluation_timeout: float = 30.0,
        evaluation_error_policy: str = 'fail',
        jev: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.condition_template = condition
        self.merge_strategy = merge_strategy
        self.output_handles = output_handles
        self.default_handle = default_handle
        self.init_error = None
        self.evaluation_mode = evaluation_mode
        self.questions = questions
        self.evaluation_timeout = evaluation_timeout
        self.evaluation_error_policy = evaluation_error_policy
        self.used_default = False
        self.judgment_error = None
        self.jev = jev
        self.answers: Dict[str, Any] = {}
        self.judgment_diagnostics: Dict[str, Any] = {}
        self._evaluation_sequence = 0
        
        # Set up the condition evaluator (default: Jinja2Evaluator)
        self._evaluator: ConditionEvaluator = evaluator or (
            Jinja2StrictEvaluator() if evaluation_mode in {'llm', 'jev'} else Jinja2Evaluator()
        )
        
        # Track key sources for collision detection
        self._key_sources: Dict[str, str] = {}
        self._merge_collisions: List[Dict[str, Any]] = []
        
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLE_CTX = handles.get('input', handles.get('context', self.DEFAULT_INPUT_HANDLE_CTX))
        self.INPUT_HANDLE_CLIENT = handles.get('client_provider', handles.get('client', self.DEFAULT_INPUT_HANDLE_CLIENT))
        
        if not condition:
            self.init_error = "NodeConditional requires a non-empty 'condition' template"
        elif merge_strategy not in ("flat", "namespaced"):
            self.init_error = f"Invalid merge_strategy '{merge_strategy}'. Must be 'flat' or 'namespaced'"
        elif evaluation_mode != 'jinja' or questions is not None or evaluation_error_policy != 'fail':
            validated = ConditionalNodeModel(
                condition=condition, merge_strategy=merge_strategy, handles=handles,
                output_handles=output_handles, default_handle=default_handle,
                evaluation_mode=evaluation_mode, questions=questions,
                evaluation_timeout=evaluation_timeout,
                evaluation_error_policy=evaluation_error_policy,
                jev=jev,
            )
            # Preserve explicit null criterion descriptions in both providers.
            self.questions = {key: value.model_dump(exclude_none=True) | (
                {'criteria': value.criteria} if getattr(value, 'criteria', None) is not None else {}
            ) for key, value in (validated.questions or {}).items()}
            self.jev = validated.jev.model_dump(exclude_none=True) if validated.jev is not None else None
        
        super().__init__(**kwargs)

        # Pre-compile template for efficiency (only if condition is valid)
        # This is kept for backward compatibility — the evaluator handles actual evaluation.
        if self.condition_template:
            env = jinja2.Environment()
            self._template = env.from_string(self.condition_template)
        else:
            self._template = None

    def _parse_input_data(self, raw_data: Any) -> Any:
        """Parse input data, attempting JSON decode for strings."""
        if isinstance(raw_data, str):
            try:
                return json.loads(raw_data)
            except json.JSONDecodeError:
                # Treat as plain string value
                return raw_data
        return raw_data

    def _is_client_input(self, handle: str) -> bool:
        if self.evaluation_mode in {'llm', 'jev'}:
            return handle in {self.INPUT_HANDLE_CLIENT, self.DEFAULT_INPUT_HANDLE_CLIENT}
        # Before LLM mode existed, exact-rule graphs could name their primary
        # state port anything, including the new client's default name.
        return handle == self.INPUT_HANDLE_CLIENT and (
            handle != self.INPUT_HANDLE_CTX
        )

    def has_state_inputs(self) -> bool:
        """Client readiness alone never makes a conditional ready for loop state."""
        return any(value is not None for handle, value in self.inputs.items() if not self._is_client_input(handle))

    async def _generate_judgments(self, render_ctx):
        client = self.get_input(self.INPUT_HANDLE_CLIENT, required=True) if self.evaluation_mode == 'llm' else None
        chat = build_judgment_chat(render_ctx, self.questions) if client is not None else None
        hooks = self._hooks
        self._evaluation_sequence += 1
        ctx = None
        model = getattr(client.llm, 'model', 'unknown') if client is not None else 'jev-latest'
        provider = (getattr(client.llm, 'engine_name', '') or getattr(client, 'engine', 'unknown')) if client is not None else 'typesafe'
        if hooks is not None and not hooks.is_empty():
            from magic_agents.hooks.context_factory import HookContextFactory
            ctx = HookContextFactory.build_llm_context(
                execution_id=getattr(hooks, 'execution_id', ''),
                run_id=getattr(hooks, 'run_id', ''), node_id=self.node_id,
                node_type=self.node_type or 'conditional', node_class=type(self).__name__,
                model=model, provider=provider, streaming=False,
                sequence_number=self._evaluation_sequence,
                evaluation_mode=self.evaluation_mode, question_ids=list(self.questions),
            )
            await hooks.invoke('on_llm_start', ctx)
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(
                client.llm.async_generate(chat, json_output=True) if client is not None else
                    evaluate_jev(render_ctx, self.questions, self.jev, timeout=self.evaluation_timeout),
                timeout=self.evaluation_timeout,
            )
        except BaseException as exc:
            if ctx is not None:
                public_error = exc
                if self.evaluation_error_policy == 'default' and isinstance(exc, Exception):
                    public_error = asyncio.TimeoutError('Conditional judgment evaluation timed out') if isinstance(exc, asyncio.TimeoutError) else RuntimeError(
                        'Conditional judgment provider request failed')
                ctx.error = public_error
                ctx.outputs = {'model': model, 'provider': provider, 'finish_reason': 'error', 'error': str(public_error)}
                await hooks.invoke('on_llm_end', ctx)
            raise
        usage = usage_detail_outputs(getattr(response, 'usage', None), response_id=getattr(response, 'id', None))
        duration_ms = (time.monotonic() - started) * 1000
        output = {
            'model': getattr(response, 'model', None) or model,
            'provider': provider, 'content': getattr(response, 'content', None),
            'duration_ms': duration_ms, **usage,
        }
        choices = getattr(response, 'choices', None)
        if choices:
            output['finish_reason'] = getattr(choices[0], 'finish_reason', None)
        if ctx is not None:
            ctx.outputs = output
            await hooks.invoke('on_llm_end', ctx)
            ctx.outputs = {**output, 'total_iterations': 1}
            await hooks.invoke('on_llm_loop_end', ctx)
        return response, {'type': 'debug', 'content': {
            'event_type': 'LLM_GENERATION', 'node_id': self.node_id,
            'node_type': 'conditional',
            # Internal judgment calls have no user-facing content chunk. Give
            # usage consumers a stable per-call identity, including adapters
            # that omit their provider request ID.
            'id': usage.get('provider_request_id') or getattr(response, 'id', None) or f'conditional-{uuid.uuid4().hex}',
            'model': output['model'], 'provider': provider,
            'duration_ms': duration_ms, **usage,
        }}

    def _merge_inputs(self) -> Dict[str, Any]:
        """
        Merge all input data into a single context dictionary.
        
        For flat merge strategy, tracks key collisions for debugging.
        
        Returns
        -------
        dict
            Merged context dictionary for template evaluation.
        
        Raises
        ------
        ValueError
            If no inputs are available or if merge fails.
        """
        merged_context = {}
        self._key_sources.clear()
        self._merge_collisions.clear()
        
        # Collect all input handles that have data
        available_inputs = [
            (handle_name, self.inputs.get(handle_name))
            for handle_name in self.inputs.keys()
            if self.inputs.get(handle_name) is not None and not self._is_client_input(handle_name)
        ]
        
        if not available_inputs:
            return None  # Will be handled in process method
        
        if self.debug:
            logger.debug(
                "NodeConditional (%s): Merging %d inputs with strategy '%s'",
                self.node_id,
                len(available_inputs),
                self.merge_strategy
            )
        
        # Process each input according to merge strategy
        for handle_name, raw_data in available_inputs:
            parsed_data = self._parse_input_data(raw_data)
            
            if self.merge_strategy == "namespaced":
                # Store under handle name to prevent collisions
                merged_context[handle_name] = parsed_data
                # Convenience alias: expose the primary input as `value`
                # This matches existing docs/examples that reference `value` directly.
                if handle_name == self.INPUT_HANDLE_CTX:
                    merged_context.setdefault("value", parsed_data)
                if self.debug:
                    logger.debug(
                        "NodeConditional (%s): Added input '%s' under namespace",
                        self.node_id,
                        handle_name
                    )
            else:
                # Flat merge with collision detection
                if isinstance(parsed_data, dict):
                    for key in parsed_data.keys():
                        if key in merged_context:
                            # Track collision
                            self._merge_collisions.append({
                                "key": key,
                                "previous_handle": self._key_sources.get(key),
                                "new_handle": handle_name,
                                "previous_value": str(merged_context[key])[:50],
                                "new_value": str(parsed_data[key])[:50]
                            })
                        self._key_sources[key] = handle_name
                    merged_context.update(parsed_data)
                    if self.debug:
                        logger.debug(
                            "NodeConditional (%s): Merged dict from '%s' (keys: %s)",
                            self.node_id,
                            handle_name,
                            list(parsed_data.keys())
                        )
                else:
                    # Non-dict inputs stored by handle name
                    if handle_name in merged_context:
                        self._merge_collisions.append({
                            "key": handle_name,
                            "type": "handle_collision"
                        })
                    merged_context[handle_name] = parsed_data
                    self._key_sources[handle_name] = handle_name
                    if self.debug:
                        logger.debug(
                            "NodeConditional (%s): Added non-dict input '%s' by handle name",
                            self.node_id,
                            handle_name
                        )
                
                # Convenience alias: expose the primary input as `value` (without overriding a real `value`)
                if handle_name == self.INPUT_HANDLE_CTX:
                    merged_context.setdefault("value", parsed_data)
        
        # Log collision warnings
        if self._merge_collisions and self.debug:
            logger.warning(
                "NodeConditional (%s): %d key collision(s) in flat merge: %s",
                self.node_id, len(self._merge_collisions),
                [c['key'] for c in self._merge_collisions]
            )
        
        return merged_context

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:  # noqa: D401
        """Evaluate condition, emit chosen handle, update bypass metadata."""
        self.selected_handle = None
        self._response = None
        self.answers = {}
        self.judgment_diagnostics = {}
        self.used_default = False
        self.judgment_error = None
        self.outputs.clear()
        
        # Check for initialization errors
        if self.init_error:
            yield self.yield_debug_error(
                error_type="ConfigurationError",
                error_message=self.init_error,
                context={
                    "condition": self.condition_template,
                    "merge_strategy": self.merge_strategy
                }
            )
            yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
            return
        
        # Merge all available inputs into a dict context
        render_ctx = self._merge_inputs()
        
        # Check if merge failed (no inputs available)
        if render_ctx is None:
            yield self.yield_debug_error(
                error_type="InputError",
                error_message=f"NodeConditional '{self.node_id}' requires at least one input. No data received on any input handle.",
                context={
                    "available_handles": list(self.inputs.keys()),
                    "condition": self.condition_template,
                    "merge_strategy": self.merge_strategy
                }
            )
            # Signal bypass to all downstream
            yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
            return

        # Providers judge state; deterministic application code selects a route.
        route_ctx = render_ctx
        selected_handle = None
        if self.evaluation_mode in {'llm', 'jev'}:
            provider_returned = False
            try:
                response, generation_event = await self._generate_judgments(render_ctx)
                provider_returned = True
                # Usage is emitted even if the provider returned malformed JSON.
                yield generation_event
                if self.evaluation_mode == 'jev':
                    self.answers = validate_judgment_answers(response.answers, self.questions)
                else:
                    validate_provider_answer(response)
                    self.answers = parse_judgments(response.content, self.questions, diagnostics=self.judgment_diagnostics)
                route_ctx = {**render_ctx, 'state': render_ctx, 'answers': self.answers}
            except Exception as exc:
                timed_out = isinstance(exc, asyncio.TimeoutError)
                self.judgment_error = {
                    'code': 'EVALUATION_TIMEOUT' if timed_out else 'JUDGMENT_ERROR',
                    'message': 'Conditional judgment evaluation timed out' if timed_out else (
                        'Conditional judgment answer validation failed' if provider_returned else
                        'Conditional judgment provider request failed'),
                }
                if self.evaluation_error_policy == 'default':
                    self.answers = {}
                    self.judgment_diagnostics = {}
                    self.used_default = True
                    selected_handle = self.default_handle
                    yield {'type': 'debug', 'content': {
                        'event_type': 'CONDITIONAL_FALLBACK', 'node_id': self.node_id,
                        'node_type': self.node_type or 'conditional',
                        'evaluation_mode': self.evaluation_mode, 'default_handle': self.default_handle,
                        'used_default': True, 'judgment_error': self.judgment_error,
                    }}
                else:
                    yield self.yield_debug_error(
                        error_type='EvaluationTimeout' if timed_out else 'JudgmentError',
                        error_message=str(exc) or self.judgment_error['message'],
                        context={'evaluation_mode': self.evaluation_mode, 'question_ids': list(self.questions), 'evaluation_timeout': self.evaluation_timeout},
                    )
                    yield {'type': ConditionalSignalTypes.BYPASS_ALL, 'content': None}
                    return

        if not self.used_default:
            # Evaluate the condition template using the configured evaluator
            try:
                selected_handle = self._evaluator.evaluate(self.condition_template, route_ctx)
            except UndefinedError as e:
                logger.error(
                    "NodeConditional (%s): Undefined variable in template: %s",
                    self.node_id,
                    e,
                )
                available_keys = list(render_ctx.keys()) if isinstance(render_ctx, dict) else []
                yield self.yield_debug_error(
                    error_type="TemplateError",
                    error_message=f"Template references undefined variable: {str(e)}",
                    context={
                        "condition": self.condition_template,
                        "available_context_keys": available_keys,
                        "context_preview": {k: str(v)[:100] for k, v in (render_ctx.items() if isinstance(render_ctx, dict) else [])},
                        "merge_strategy": self.merge_strategy
                    }
                )
                yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
                return
            except TemplateSyntaxError as e:
                logger.error(
                    "NodeConditional (%s): Invalid Jinja2 syntax: %s",
                    self.node_id,
                    e,
                )
                yield self.yield_debug_error(
                    error_type="TemplateSyntaxError",
                    error_message=f"Invalid Jinja2 syntax in condition: {str(e)}",
                    context={
                        "condition": self.condition_template,
                        "error_line": getattr(e, 'lineno', None),
                        "merge_strategy": self.merge_strategy
                    }
                )
                yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
                return
            except TemplateError as e:
                logger.error(
                    "NodeConditional (%s): Template evaluation failed: %s",
                    self.node_id,
                    e,
                )
                yield self.yield_debug_error(
                    error_type="TemplateEvaluationError",
                    error_message=f"Failed to evaluate condition template: {str(e)}",
                    context={
                        "condition": self.condition_template,
                        "available_context_keys": list(render_ctx.keys()) if isinstance(render_ctx, dict) else [],
                        "merge_strategy": self.merge_strategy
                    }
                )
                yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
                return
            except Exception as e:
                logger.error(
                    "NodeConditional (%s): Unexpected error during evaluation: %s",
                    self.node_id,
                    e,
                )
                yield self.yield_debug_error(
                    error_type="UnexpectedError",
                    error_message=f"Unexpected error during condition evaluation: {str(e)}",
                    context={
                        "condition": self.condition_template,
                        "exception_type": type(e).__name__,
                        "merge_strategy": self.merge_strategy
                    }
                )
                yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
                return

        if self.debug:
            logger.debug(
                "NodeConditional (%s): evaluated template to '%s' with ctx=%s",
                self.node_id,
                selected_handle,
                render_ctx,
            )

        # Handle empty result with default_handle fallback
        if not selected_handle:
            if self.default_handle:
                selected_handle = self.default_handle
                self.used_default = True
                logger.warning(
                    "NodeConditional (%s): Condition evaluated to empty, using default_handle: %s",
                    self.node_id, self.default_handle
                )
            else:
                yield self.yield_debug_error(
                    error_type="EmptyHandleError",
                    error_message="Condition evaluated to empty string with no default_handle configured.",
                    context={
                        "condition": self.condition_template,
                        "rendered_result": repr(selected_handle),
                        "context_keys": list(render_ctx.keys()) if isinstance(render_ctx, dict) else [],
                        "merge_strategy": self.merge_strategy,
                        "suggestion": "Add 'default_handle' to conditional config for fallback routing."
                    }
                )
                # Emit BYPASS_ALL signal so executor can bypass all downstream
                yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
                return

        if self.evaluation_mode in {'llm', 'jev'} and selected_handle not in self.output_handles:
            yield self.yield_debug_error(
                error_type='GraphRoutingError',
                error_message=f"Condition selected undeclared output handle '{selected_handle}'",
                context={'output_handles': self.output_handles},
            )
            yield {'type': ConditionalSignalTypes.BYPASS_ALL, 'content': None}
            return

        # Persist selection for executors that need deterministic bypass routing
        self.selected_handle = selected_handle

        # Record selected handle in outputs for downstream nodes.
        # Pass the merged context to the selected output
        self.outputs[selected_handle] = self.prep(render_ctx)

        # Emit event; downstream executor will route by content_type.
        yield {
            "type": selected_handle,
            "content": self.prep(render_ctx),
        }

        # Yield 'end' for bookkeeping so executors know node completed.
        yield self.yield_static({
            "selected": selected_handle,
            "merge_strategy": self.merge_strategy,
            "input_count": sum(value is not None for handle, value in self.inputs.items() if not self._is_client_input(handle)),
            "merge_collisions": self._merge_collisions if self._merge_collisions else None,
            "output_handles": self.output_handles,
            "default_handle": self.default_handle,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_error_policy": self.evaluation_error_policy,
            "used_default": self.used_default,
            "judgment_error": self.judgment_error,
            **({
                'answers': self.answers,
                'confidence_source': 'jev_calibrated' if self.evaluation_mode == 'jev' else 'llm_estimated_not_calibrated',
                'judgment_diagnostics': self.judgment_diagnostics,
            } if self.evaluation_mode in {'llm', 'jev'} else {}),
        })

    def _capture_internal_state(self):
        """Capture Conditional-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Conditional-specific variables as documented
        state['condition'] = self.condition_template
        state['merge_strategy'] = self.merge_strategy
        state['output_handles'] = self.output_handles
        state['default_handle'] = self.default_handle
        state['evaluation_mode'] = self.evaluation_mode
        state['evaluation_error_policy'] = self.evaluation_error_policy
        state['used_default'] = self.used_default
        state['judgment_error'] = self.judgment_error
        if self.evaluation_mode in {'llm', 'jev'}:
            state['answers'] = self.answers
            state['questions'] = self.questions
            state['judgment_diagnostics'] = self.judgment_diagnostics
            state['confidence_source'] = 'jev_calibrated' if self.evaluation_mode == 'jev' else 'llm_estimated_not_calibrated'
        
        # Capture the selected handle if available
        if hasattr(self, 'selected_handle'):
            state['selected_handle'] = self.selected_handle
        
        # Include merge collision warnings if any
        if self._merge_collisions:
            state['merge_collisions'] = self._merge_collisions
        
        # Capture context data used for evaluation (truncated for large contexts)
        try:
            context_data = self._merge_inputs()
            if context_data:
                state['context_data'] = self._safe_copy_dict(context_data)
        except Exception:
            pass  # Context may not be available yet
        
        return state

    def get_possible_outputs(self) -> List[str]:
        """
        Return declared or inferred output handles.
        
        Used for graph validation and documentation.
        
        Returns:
            List of possible output handle names
        """
        if self.output_handles:
            return list(self.output_handles)
        
        # Try to infer from condition template (basic heuristics)
        # This is limited but catches common patterns
        inferred = []
        condition = self.condition_template or ""
        
        # Pattern: {{ 'handle_a' if ... else 'handle_b' }}
        string_matches = re.findall(r"'([^']+)'", condition)
        inferred.extend(string_matches)
        
        # Also match double-quoted strings
        double_matches = re.findall(r'"([^"]+)"', condition)
        inferred.extend(double_matches)
        
        return list(set(inferred)) if inferred else []

    def validate_against_edges(self, edges: List) -> Dict[str, Any]:
        """
        Validate this conditional against provided edges.
        
        Args:
            edges: List of edge objects with source, target, sourceHandle attributes
        
        Returns:
            Validation result dict with 'valid', 'warnings', 'errors' keys
        """
        outgoing = [e for e in edges if e.source == self.node_id]
        edge_handles = {e.sourceHandle for e in outgoing}
        possible_outputs = self.get_possible_outputs()
        
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "edge_handles": list(edge_handles),
            "declared_outputs": possible_outputs
        }
        
        if possible_outputs:
            missing = set(possible_outputs) - edge_handles
            extra = edge_handles - set(possible_outputs)
            
            if missing:
                result["valid"] = False
                result["errors"].append({
                    "type": "missing_edges",
                    "handles": list(missing),
                    "message": f"No edges for declared outputs: {list(missing)}"
                })
            
            if extra:
                result["warnings"].append({
                    "type": "extra_edges",
                    "handles": list(extra),
                    "message": f"Edges exist for handles not in declared outputs: {list(extra)}"
                })
        
        # Check default_handle has an edge
        if self.default_handle and self.default_handle not in edge_handles:
            result["valid"] = False
            result["errors"].append({
                "type": "missing_default_edge",
                "handle": self.default_handle,
                "message": f"No edge for default_handle: {self.default_handle}"
            })
        
        return result
