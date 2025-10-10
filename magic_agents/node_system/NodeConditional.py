from __future__ import annotations

import json
import logging
from typing import Any, Dict, AsyncGenerator

import jinja2
from jinja2 import UndefinedError, TemplateSyntaxError, TemplateError

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeConditional(Node):
    """Branching node that routes execution based on a Jinja2-evaluated condition.

    The *condition* template must render to the **name of the output handle** that
    should continue execution. All other outgoing handles are considered bypassed.

    Supports multiple inputs that are merged into a single context for evaluation.

    Inputs
    ------
    handle_input : Any (required)
        Primary JSON string or Python object used as the context when evaluating
        the Jinja2 template.
    handle_input_1, handle_input_2, ... : Any (optional)
        Additional inputs that will be merged with the primary input according
        to the merge_strategy.

    Data fields (kwargs)
    --------------------
    condition : str (required)
        Jinja2 template string that will be rendered with the parsed input as
        its context. The rendered string decides which handle is emitted.
    merge_strategy : str (optional, default='flat')
        How to merge multiple inputs:
        - 'flat': Merge all inputs into single flat dict (later overrides earlier)
        - 'namespaced': Keep inputs separate under their handle names

    Examples
    --------
    IF Pattern:
        condition: "{{ 'adult' if age >= 18 else 'minor' }}"
        
    SWITCH Pattern:
        condition: "{{ status }}"
        
    Complex Multi-Input:
        condition: "{{ 'approved' if user.age >= 18 and account.balance > 1000 else 'denied' }}"
        merge_strategy: 'namespaced'
    """

    INPUT_HANDLE_CTX = "handle_input"

    def __init__(self, *, condition: str, merge_strategy: str = "flat", **kwargs):
        if not condition:
            raise ValueError("NodeConditional requires a non-empty 'condition' template")
        self.condition_template = condition
        self.merge_strategy = merge_strategy
        
        if merge_strategy not in ("flat", "namespaced"):
            raise ValueError(f"Invalid merge_strategy '{merge_strategy}'. Must be 'flat' or 'namespaced'")
        
        super().__init__(**kwargs)

        # Pre-compile template for efficiency
        env = jinja2.Environment()
        self._template = env.from_string(self.condition_template)

    def _parse_input_data(self, raw_data: Any) -> Any:
        """Parse input data, attempting JSON decode for strings."""
        if isinstance(raw_data, str):
            try:
                return json.loads(raw_data)
            except json.JSONDecodeError:
                # Treat as plain string value
                return raw_data
        return raw_data

    def _merge_inputs(self) -> Dict[str, Any]:
        """
        Merge all input data into a single context dictionary.
        
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
        
        # Collect all input handles that have data
        available_inputs = [
            (handle_name, self.inputs.get(handle_name))
            for handle_name in self.inputs.keys()
            if self.inputs.get(handle_name) is not None
        ]
        
        if not available_inputs:
            raise ValueError(
                f"NodeConditional '{self.node_id}' requires at least one input. "
                f"No data received on any input handle."
            )
        
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
                if self.debug:
                    logger.debug(
                        "NodeConditional (%s): Added input '%s' under namespace",
                        self.node_id,
                        handle_name
                    )
            else:
                # Flat merge: combine all dicts, later inputs override earlier
                if isinstance(parsed_data, dict):
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
                    merged_context[handle_name] = parsed_data
                    if self.debug:
                        logger.debug(
                            "NodeConditional (%s): Added non-dict input '%s' by handle name",
                            self.node_id,
                            handle_name
                        )
        
        return merged_context

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:  # noqa: D401
        """Evaluate condition, emit chosen handle, update bypass metadata."""
        # Merge all available inputs into a dict context
        render_ctx = self._merge_inputs()

        # Evaluate the condition template
        try:
            selected_handle = str(self._template.render(**render_ctx)).strip()
        except jinja2.UndefinedError as e:
            logger.error(
                "NodeConditional (%s): Undefined variable in template: %s",
                self.node_id,
                e,
            )
            available_keys = list(render_ctx.keys()) if isinstance(render_ctx, dict) else []
            raise ValueError(
                f"NodeConditional '{self.node_id}' template references undefined variable. "
                f"Condition: {self.condition_template}. "
                f"Available context keys: {available_keys}. "
                f"Error: {e}"
            ) from e
        except jinja2.TemplateSyntaxError as e:
            logger.error(
                "NodeConditional (%s): Invalid Jinja2 syntax: %s",
                self.node_id,
                e,
            )
            raise ValueError(
                f"NodeConditional '{self.node_id}' has invalid Jinja2 syntax in condition. "
                f"Condition: {self.condition_template}. "
                f"Error: {e}"
            ) from e
        except jinja2.TemplateError as e:
            logger.error(
                "NodeConditional (%s): Template evaluation failed: %s",
                self.node_id,
                e,
            )
            raise ValueError(
                f"NodeConditional '{self.node_id}' failed to evaluate condition. "
                f"Condition: {self.condition_template}. "
                f"Error: {e}"
            ) from e
        except Exception as e:
            logger.error(
                "NodeConditional (%s): Unexpected error during evaluation: %s",
                self.node_id,
                e,
            )
            raise ValueError(
                f"NodeConditional '{self.node_id}' encountered unexpected error. "
                f"Condition: {self.condition_template}. "
                f"Error: {e}"
            ) from e

        if self.debug:
            logger.debug(
                "NodeConditional (%s): evaluated template to '%s' with ctx=%s",
                self.node_id,
                selected_handle,
                render_ctx,
            )

        if not selected_handle:
            raise ValueError(
                f"NodeConditional '{self.node_id}' rendered empty handle. "
                f"Condition: {self.condition_template}. "
                f"The condition must return a non-empty string matching an output handle name."
            )

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
            "input_count": len([k for k in self.inputs.keys() if self.inputs.get(k) is not None])
        })
