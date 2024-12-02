import functools
import inspect
import time


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
            print(f"Executing {qualname}...")
        async for i in func(self, chat_log, *args, **kwargs):
            yield i
        end_time = time.monotonic()
        execution_time = end_time - start_time
        if debug:
            print(f"{qualname} execution time: {execution_time:.4f} seconds")

    return wrapper
