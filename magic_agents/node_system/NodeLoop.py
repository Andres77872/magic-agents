import json
import logging
from typing import Optional

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeLoop(Node):
    """
    Loop node: iterates over a list of items, then aggregates results.
    
    Handle names are configurable via JSON data.handles - JSON is the source of truth.
    Default handle names are used only if not specified in JSON.

    Inputs (configurable via data.handles):
      - 'handle_list' (default): JSON string or Python list of items to iterate.
      - 'handle_loop' (default): Optional per-iteration result to aggregate.

    Outputs (configurable via data.handles):
      - 'handle_item' (default): Each item during iteration
      - 'handle_end' (default): Aggregated results after iteration
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_HANDLE_LIST = 'handle_list'
    DEFAULT_INPUT_HANDLE_LOOP = 'handle_loop'
    DEFAULT_OUTPUT_HANDLE_ITEM = 'handle_item'
    DEFAULT_OUTPUT_HANDLE_END = 'handle_end'

    def __init__(self, handles: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLE_LIST = handles.get('input_list', handles.get('list', self.DEFAULT_INPUT_HANDLE_LIST))
        self.INPUT_HANDLE_LOOP = handles.get('input_loop', handles.get('loop', self.DEFAULT_INPUT_HANDLE_LOOP))
        self.OUTPUT_HANDLE_ITEM = handles.get('output_item', handles.get('item', self.DEFAULT_OUTPUT_HANDLE_ITEM))
        self.OUTPUT_HANDLE_END = handles.get('output_end', handles.get('end', self.DEFAULT_OUTPUT_HANDLE_END))

    async def process(self, chat_log):
        raw = self.get_input(self.INPUT_HANDLE_LIST, required=False)
        
        # Validate input exists
        if raw is None:
            yield self.yield_debug_error(
                error_type="InputError",
                error_message=f"NodeLoop requires input '{self.INPUT_HANDLE_LIST}' with a list of items to iterate.",
                context={
                    "available_inputs": list(self.inputs.keys()),
                    "required_input": self.INPUT_HANDLE_LIST
                }
            )
            return
        
        # parse JSON string or accept list directly
        if isinstance(raw, str):
            try:
                items = json.loads(raw)
            except json.JSONDecodeError as e:
                yield self.yield_debug_error(
                    error_type="JSONParseError",
                    error_message=f"Invalid JSON list: {str(e)}",
                    context={
                        "input_value_preview": raw[:200] if len(raw) > 200 else raw,
                        "input_length": len(raw),
                        "error_position": getattr(e, 'pos', None)
                    }
                )
                return
        else:
            items = raw
        
        if not isinstance(items, list):
            yield self.yield_debug_error(
                error_type="ValidationError",
                error_message=f"NodeLoop expects a list, got {type(items).__name__}",
                context={
                    "received_type": type(items).__name__,
                    "value_preview": str(items)[:200] if len(str(items)) > 200 else str(items)
                }
            )
            return
        
        logger.info("NodeLoop:%s iterating over %d items", self.node_id, len(items))
        # yield each item for downstream processing - use configured output handle
        for idx, item in enumerate(items):
            if self.debug and idx < 5:
                logger.debug("NodeLoop:%s yielding item index=%d", self.node_id, idx)
            yield self.yield_static(item, content_type=self.OUTPUT_HANDLE_ITEM)
        # after iteration, aggregate any loop inputs collected
        agg = self.inputs.get(self.INPUT_HANDLE_LOOP, [])
        if self.debug:
            try:
                agg_len = len(agg)
            except Exception:
                agg_len = 1 if agg else 0
            logger.debug("NodeLoop:%s yielding aggregation length=%d", self.node_id, agg_len)
        yield self.yield_static(agg, content_type=self.OUTPUT_HANDLE_END)

    def _capture_internal_state(self):
        """Capture Loop-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Loop-specific variables as documented
        state['iterate'] = True  # Loop nodes always iterate
        
        # Capture handle configuration
        state['input_handle_list'] = self.INPUT_HANDLE_LIST
        state['input_handle_loop'] = self.INPUT_HANDLE_LOOP
        state['output_handle_item'] = self.OUTPUT_HANDLE_ITEM
        state['output_handle_end'] = self.OUTPUT_HANDLE_END
        
        return state