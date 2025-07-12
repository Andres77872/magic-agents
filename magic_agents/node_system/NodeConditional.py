from __future__ import annotations

import json
import logging
from typing import Any, Dict, AsyncGenerator

import jinja2

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeConditional(Node):
    """Branching node that routes execution based on a Jinja2-evaluated condition.

    The *condition* template must render to the **name of the output handle** that
    should continue execution. All other outgoing handles are considered bypassed.

    Inputs
    ------
    handle_input : Any (required)
        JSON string or Python object used as the context when evaluating the
        Jinja2 template.

    Data fields (kwargs)
    --------------------
    condition : str (required)
        Jinja2 template string that will be rendered with the parsed input as
        its context. The rendered string decides which handle is emitted.
    """

    INPUT_HANDLE_CTX = "handle_input"

    def __init__(self, *, condition: str, **kwargs):
        if not condition:
            raise ValueError("NodeConditional requires a non-empty 'condition' template")
        self.condition_template = condition
        super().__init__(**kwargs)

        # Pre-compile template for efficiency
        env = jinja2.Environment()
        self._template = env.from_string(self.condition_template)

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:  # noqa: D401
        """Evaluate condition, emit chosen handle, update bypass metadata."""
        raw_ctx = self.get_input(self.INPUT_HANDLE_CTX, required=True)

        # Attempt to JSON-decode strings
        if isinstance(raw_ctx, str):
            try:
                ctx: Any = json.loads(raw_ctx)
            except json.JSONDecodeError:
                # treat as plain string value
                ctx = raw_ctx
        else:
            ctx = raw_ctx

        # Jinja2 requires dict-like context. If primitive, expose as 'value'.
        if not isinstance(ctx, dict):
            render_ctx = {"value": ctx}
        else:
            render_ctx = ctx

        selected_handle = str(self._template.render(**render_ctx)).strip()
        if self.debug:
            logger.debug(
                "NodeConditional (%s): evaluated template to '%s' with ctx=%s",
                self.node_id,
                selected_handle,
                render_ctx,
            )

        if not selected_handle:
            raise ValueError(
                f"NodeConditional '{self.node_id}' rendered empty handle from condition: {self.condition_template}"
            )

        # Record selected handle in outputs for downstream nodes.
        self.outputs[selected_handle] = self.prep(ctx)

        # Emit event; downstream executor will route by content_type.
        yield {
            "type": selected_handle,
            "content": self.prep(ctx),
        }

        # Yield 'end' for bookkeeping so executors know node completed.
        yield self.yield_static({"selected": selected_handle})
