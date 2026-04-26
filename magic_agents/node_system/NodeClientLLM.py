import json
import logging
from typing import Optional

from magic_agents.models.factory.Nodes import ClientNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.env_resolver import resolve_env_placeholders, resolve_env_string
from magic_agents.util.primitive_coercion import coerce_primitive_by_type, input_has_value
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
    DEFAULT_INPUT_MODEL = 'handle-client-model'
    DEFAULT_INPUT_ENGINE = 'handle-client-engine'

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
        self.INPUT_HANDLE_MODEL = handles.get('model', self.DEFAULT_INPUT_MODEL)
        self.INPUT_HANDLE_ENGINE = handles.get('engine', self.DEFAULT_INPUT_ENGINE)
        self.OUTPUT_HANDLE = handles.get('output', handles.get('client', self.DEFAULT_OUTPUT_HANDLE))
        self._default_engine = getattr(data, 'engine', None)
        self._default_model = getattr(data, 'model', None)
        self._default_api_info = getattr(data, 'api_info', None)
        self._base_extra_data = dict(getattr(data, 'extra_data', {}) or {})
        self._client_args_preview = {
            "engine": self._default_engine,
            "model": self._default_model,
        }
        self._raw_api_info_type = type(self._default_api_info).__name__
        self._current_engine = self._default_engine
        self._current_model = self._default_model

        self._initialize_client(self._default_engine, self._default_model)

    def _build_magic_llm_args(self, engine: Optional[str], model: Optional[str]) -> dict:
        raw_api_info = self._default_api_info
        if raw_api_info is None or raw_api_info == "":
            api_info = {}
        elif isinstance(raw_api_info, dict):
            api_info = resolve_env_placeholders(raw_api_info)
        else:
            api_info = json.loads(resolve_env_string(raw_api_info))

        extra_data = resolve_env_placeholders(self._base_extra_data)

        args = {
            'engine': engine,
            'model': model,
            **api_info,
            **extra_data
        }

        if 'api_key' in args and 'private_key' not in args:
            args['private_key'] = args['api_key']

        return args

    def _initialize_client(self, engine: Optional[str], model: Optional[str]) -> None:
        self.client = None
        self.init_error = None
        self.init_error_type = None
        self._client_args_preview = {
            'engine': engine,
            'model': model,
        }

        try:
            args = self._build_magic_llm_args(engine, model)

            if self.debug:
                logger.debug(
                    "NodeClientLLM:%s initializing client engine=%s model=%s",
                    self.node_id,
                    engine,
                    model
                )
            self.client = MagicLLM(**args)
            self._current_engine = engine
            self._current_model = model
            logger.info("NodeClientLLM:%s client initialized", self.node_id)
        except Exception as e:
            self.init_error = str(e)
            self.init_error_type = type(e).__name__
            logger.error("NodeClientLLM:%s failed to initialize client: %s", self.node_id, self.init_error)

    def _resolve_runtime_engine_model(self) -> tuple[Optional[str], Optional[str]]:
        engine = self._default_engine
        model = self._default_model

        if input_has_value(self.inputs, self.INPUT_HANDLE_ENGINE):
            engine = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_ENGINE], 'str', field_name=self.INPUT_HANDLE_ENGINE)
        if input_has_value(self.inputs, self.INPUT_HANDLE_MODEL):
            model = coerce_primitive_by_type(self.inputs[self.INPUT_HANDLE_MODEL], 'str', field_name=self.INPUT_HANDLE_MODEL)

        return engine, model

    async def process(self, chat_log):
        runtime_engine, runtime_model = self._resolve_runtime_engine_model()
        if (
            self.client is None
            or runtime_engine != self._current_engine
            or runtime_model != self._current_model
        ):
            self._initialize_client(runtime_engine, runtime_model)

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
