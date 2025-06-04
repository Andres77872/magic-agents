import json

from magic_agents.node_system.Node import Node


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
    INPUT_HANDLE_LIST = 'list'
    INPUT_HANDLE_LOOP = 'loop'
    OUTPUT_HANDLE_ITEM = 'item'
    OUTPUT_HANDLE_END = 'end'

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
        # yield each item for downstream processing
        for item in items:
            yield self.yield_static(item, content_type='content')
        # after iteration, aggregate any loop inputs collected
        agg = self.inputs.get(self.INPUT_HANDLE_LOOP, [])
        yield self.yield_static(agg)