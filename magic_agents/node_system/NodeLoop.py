import json
import logging

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeLoop(Node):
    """
    Loop node: iterates over a list of items, then aggregates results.

    Inputs:
      - 'list': JSON string or Python list of items to iterate.
      - 'loop': Optional per-iteration result to aggregate.

    Outputs:
      - 'item': each element from the list, as a 'content' event.
      - 'end': final aggregation of all 'loop' inputs, as an 'end' event.
    """
    INPUT_HANDLE_LIST = 'handle_list'
    INPUT_HANDLE_LOOP = 'handle_loop'
    OUTPUT_HANDLE_ITEM = 'handle_item'
    OUTPUT_HANDLE_END = 'handle_end'

    async def process(self, chat_log):
        raw = self.get_input(self.INPUT_HANDLE_LIST, required=True)
        # parse JSON string or accept list directly
        if isinstance(raw, str):
            try:
                items = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"NodeLoop '{self.node_id}': invalid JSON list: {e}")
        else:
            items = raw
        if not isinstance(items, list):
            raise ValueError(f"NodeLoop '{self.node_id}' expects a list, got {type(items)}")
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