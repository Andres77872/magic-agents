import json
import logging
from typing import Optional

from magic_agents.models.factory.Nodes import ClientNodeModel
from magic_agents.node_system.Node import Node
from magic_llm import MagicLLM

logger = logging.getLogger(__name__)


class NodeClientLLM(Node):
    """
    ClientLLM node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default output handle name - can be overridden by JSON data.handles
    # Uses hyphen to match JSON graph convention (handle-client-provider)
    DEFAULT_OUTPUT_HANDLE = 'handle-client-provider'

    def __init__(self,
                 data: ClientNodeModel,
                 node_id: str,
                 debug: bool = False,
                 handles: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(
            node_id=node_id,
            debug=debug,
            **kwargs
        )
        self.client = None
        self.init_error = None
        self.init_error_type = None
        # Allow JSON to override handle names
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', handles.get('client', self.DEFAULT_OUTPUT_HANDLE))
        self._client_args_preview = {
            "engine": getattr(data, "engine", None),
            "model": getattr(data, "model", None),
        }
        self._raw_api_info_type = type(getattr(data, "api_info", None)).__name__

        try:
            raw_api_info = data.api_info
            if raw_api_info is None or raw_api_info == "":
                api_info = {}
            elif isinstance(raw_api_info, dict):
                api_info = raw_api_info
            else:
                api_info = json.loads(raw_api_info)

            args = {
                'engine': data.engine,
                'model': data.model,
                **api_info,
                **data.extra_data
            }

            # MagicLLM uses `private_key` but many configs provide `api_key`
            if 'api_key' in args and 'private_key' not in args:
                args['private_key'] = args['api_key']

            if self.debug:
                logger.debug(
                    "NodeClientLLM:%s initializing client engine=%s model=%s",
                    self.node_id,
                    data.engine,
                    data.model
                )
            self.client = MagicLLM(**args)
            logger.info("NodeClientLLM:%s client initialized", self.node_id)
        except Exception as e:
            self.init_error = str(e)
            self.init_error_type = type(e).__name__
            logger.error("NodeClientLLM:%s failed to initialize client: %s", self.node_id, self.init_error)

    async def process(self, chat_log):
        if self.init_error:
            yield self.yield_debug_error(
                error_type="ConfigurationError",
                error_message=f"NodeClientLLM failed to initialize MagicLLM client: {self.init_error}",
                context={
                    "exception_type": self.init_error_type,
                    "client_config_preview": self._client_args_preview,
                    "raw_api_info_type": self._raw_api_info_type,
                }
            )
            return
        if self.debug:
            logger.debug("NodeClientLLM:%s yielding MagicLLM client", self.node_id)
        yield self.yield_static(self.client, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture ClientLLM-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add ClientLLM-specific variables (without sensitive data)
        state['engine'] = self._client_args_preview.get('engine')
        state['model'] = self._client_args_preview.get('model')
        state['client_initialized'] = self.client is not None
        
        if self.init_error:
            state['init_error'] = self.init_error
            state['init_error_type'] = self.init_error_type
        
        return state
