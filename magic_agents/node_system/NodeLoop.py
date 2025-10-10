import json
import logging

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeLoop(Node):
    """
    Loop node: iterates over a list of items, then aggregates results.

    Inputs:
      - 'handle_list': JSON string or Python list of items to iterate.
      - 'handle_loop': Optional per-iteration result to aggregate.

    Outputs (via content_type, not handle names):
      - Items are emitted with content_type='content' (not 'handle_item')
      - Aggregation is emitted with content_type='end' (not 'handle_end')
    
    Edge Configuration:
      - For iteration: sourceHandle="content" (receives each item)
      - For aggregation: sourceHandle="default" or "end" (receives aggregated results)
    
    Note: OUTPUT_HANDLE_* constants are for reference/documentation only.
    The implementation uses generic content_type values for backward compatibility.
    """
    INPUT_HANDLE_LIST = 'handle_list'
    INPUT_HANDLE_LOOP = 'handle_loop'
    # NOTE: These output constants are for reference only - not used as actual content_type values
    OUTPUT_HANDLE_ITEM = 'handle_item'  # Reference: actual content_type='content'
    OUTPUT_HANDLE_END = 'handle_end'    # Reference: actual content_type='end' (default)

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
        # yield each item for downstream processing
        for idx, item in enumerate(items):
            if self.debug and idx < 5:
                logger.debug("NodeLoop:%s yielding item index=%d", self.node_id, idx)
            yield self.yield_static(item, content_type='content')
        # after iteration, aggregate any loop inputs collected
        agg = self.inputs.get(self.INPUT_HANDLE_LOOP, [])
        if self.debug:
            try:
                agg_len = len(agg)
            except Exception:
                agg_len = 1 if agg else 0
            logger.debug("NodeLoop:%s yielding aggregation length=%d", self.node_id, agg_len)
        yield self.yield_static(agg)