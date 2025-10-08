import functools
import inspect
import time
import logging

logger = logging.getLogger(__name__)

SENSITIVE_KEYS = {"api_key", "private_key", "authorization", "password", "token", "bearer", "secret"}

def _redact(value):
    """Recursively redact sensitive keys in nested structures."""
    try:
        if isinstance(value, dict):
            return {k: ("***" if str(k).lower() in SENSITIVE_KEYS else _redact(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_redact(v) for v in value)
    except Exception:
        # If anything goes wrong during redaction, fallback to original value
        return value
    return value

from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel


def magic_telemetry(func):
    qualname = func.__qualname__.split('.')[0]
    if not inspect.isasyncgenfunction(func):
        raise TypeError(f"Function {qualname} is not an async generator. "
                        f"magic_telemetry can only be applied to async generator functions.")

    @functools.wraps(func)
    async def wrapper(self, chat_log, *args, **kwargs):
        debug = self.get_debug()
        start_time = time.monotonic()
        logger.info(f"Executing {qualname}:{self.node_id}...")
        if debug:
            logger.debug("Node %s:%s inputs: %s", qualname, self.node_id, _redact(getattr(self, 'inputs', {})))
        e_intput = {
            'meta': {
                'node_id': self.node_id,
                'node_type': self.extra_params['node_type'],
                'node_class': qualname,
                'start_time': start_time,
                'end_time': 0,
                'execution_time': 0,
            }
        }
        yield {
            'type': 'content',
            'content': {
                "node": qualname,
                "content": ChatCompletionModel(id='',
                                               model='',
                                               choices=[ChoiceModel()],
                                               extras=e_intput),
            }
        }

        async for i in func(self, chat_log, *args, **kwargs):
            yield i
        end_time = time.monotonic()
        execution_time = end_time - start_time
        logger.info(f"{qualname}:{self.node_id} execution time: {execution_time:.4f} seconds")
        if debug:
            logger.debug("Node %s:%s outputs: %s", qualname, self.node_id, _redact(getattr(self, 'response', None)))

        e_intput = {
            'meta': {
                'node_id': self.node_id,
                'node_type': self.extra_params['node_type'],
                'node_class': qualname,
                'start_time': start_time,
                'end_time': end_time,
                'execution_time': execution_time,
            }
        }
        yield {
            'type': 'content',
            'content': {
                "node": qualname,
                "content": ChatCompletionModel(id='',
                                               model='',
                                               choices=[ChoiceModel()],
                                               extras=e_intput),
            }
        }

    return wrapper
