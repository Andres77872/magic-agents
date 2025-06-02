import functools
import inspect
import time

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
        if debug:
            print(f"Executing {qualname}:{self.node_id}...")
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
        if debug:
            print(f"{qualname}:{self.node_id} execution time: {execution_time:.4f} seconds")

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
