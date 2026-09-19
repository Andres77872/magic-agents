"""
Python Code Runner for run(handler) contract execution.

Provides CodeRunner, which compiles the reserved ``run(handler)`` entrypoint.
``subprocess`` mode executes it in a separate, terminable process.
``restricted_builtins`` and ``in_process`` are explicit local-execution modes
and use a worker thread so the event loop is not blocked.

SECURITY WARNING: This is NOT a security boundary. The exec() namespace is
restricted but trivially escapable via the Python introspection chain.
Do NOT use with untrusted/third-party code.
"""

import asyncio
import builtins
import logging
import multiprocessing
import time
from multiprocessing.connection import Connection
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Restricted builtins for Phase 1: safe subset excluding dangerous functions.
# This follows the same pattern as NodeHook._compile_hook_function().
# Blocked: open, exec, eval, compile, __import__, input, breakpoint
_RESTRICTED_BUILTINS: dict[str, Any] = {
    'abs': abs,
    'all': all,
    'any': any,
    'ascii': ascii,
    'bin': bin,
    'bool': bool,
    'bytearray': bytearray,
    'bytes': bytes,
    'callable': callable,
    'chr': chr,
    'complex': complex,
    'dict': dict,
    'dir': dir,
    'divmod': divmod,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'getattr': getattr,
    'hasattr': hasattr,
    'hash': hash,
    'hex': hex,
    'id': id,
    'int': int,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'iter': iter,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'next': next,
    'object': object,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'print': print,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'slice': slice,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'super': super,
    'tuple': tuple,
    'type': type,
    'vars': vars,
    'zip': zip,
}

# Builtins that are blocked for security reasons:
#   open        — filesystem access
#   exec, eval  — code injection
#   compile     — code compilation from within user code
#   __import__  — dynamic imports
#   input       — blocking I/O that could hang
#   breakpoint  — debugger access (pdb)


def _compile_run_function(code: str, namespace_builtins: Any) -> Callable:
    """Compile *code* and return its required ``run(handler)`` callable."""

    if not code or not code.strip():
        raise ValueError("User code must define run(handler) function")

    namespace: dict[str, Any] = {"__builtins__": namespace_builtins}
    try:
        exec(code, namespace)
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in user code: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Compilation failed: {exc}") from exc

    run_func = namespace.get("run")
    if run_func is None:
        raise ValueError("User code must define run(handler) function")
    if not callable(run_func):
        raise ValueError("'run' must be a callable function")
    return run_func


def _subprocess_entrypoint(connection: Connection, code: str, handler: dict) -> None:
    """Execute a handler in a spawned child and return one tagged payload."""

    try:
        run_func = _compile_run_function(code, builtins.__dict__)
        connection.send(("result", run_func(handler)))
    except BaseException as exc:  # The child must report failures, then exit.
        try:
            connection.send(("error", str(exc)))
        except BaseException:
            pass
    finally:
        connection.close()


class CodeRunner:
    """Execute the ``run(handler)`` contract using the configured safety mode.

    ``subprocess`` provides process separation and a hard, terminating timeout,
    but it is NOT a security boundary: the child inherits the environment and
    normal filesystem/network permissions. ``restricted_builtins`` is
    introspection-escapable, and ``in_process`` intentionally exposes full
    Python builtins. Do NOT use with untrusted/third-party code without an
    external sandbox.
    """

    def __init__(
        self,
        safety_mode: str = "subprocess",
        timeout: float = 30.0,
        max_output_chars: int = 8000,
    ):
        """Initialize CodeRunner.

        Args:
            safety_mode: Execution mode ('subprocess', 'in_process', or 'restricted_builtins').
            timeout: Maximum execution time in seconds.
            max_output_chars: Maximum output length before truncation.

        Warning:
            NOT a security boundary. Do not use with untrusted code without an
            external sandbox.
            See class-level docstring for details.
        """
        if safety_mode not in {"subprocess", "restricted_builtins", "in_process"}:
            raise ValueError(f"Unsupported Python execution safety mode: {safety_mode!r}")
        if timeout <= 0:
            raise ValueError("Python execution timeout must be greater than zero")
        if max_output_chars < 0:
            raise ValueError("max_output_chars must be non-negative")

        self.safety_mode = safety_mode
        self._timeout = timeout
        self._max_output_chars = max_output_chars

    def compile(self, code: str) -> Callable:
        """Compile user code and extract the run() function.

        Uses exec() with a constrained namespace matching the NodeHook pattern.
        The namespace contains only restricted builtins (no open, exec, eval,
        compile, __import__, input, breakpoint).

        Args:
            code: Python source code string containing a `def run(handler):` function.

        Returns:
            The compiled run() callable function.

        Raises:
            ValueError: If code is empty, missing run function, run is not callable,
                       or compilation fails (syntax error).
        """
        return _compile_run_function(code, _RESTRICTED_BUILTINS)

    def _truncate_result(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self._max_output_chars:
            omitted = len(value) - self._max_output_chars
            return value[: self._max_output_chars] + f"\n... [truncated {omitted} chars]"
        return value

    @staticmethod
    def _stop_process(process: multiprocessing.Process) -> None:
        if not process.is_alive():
            process.join(timeout=0.1)
            return
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join(timeout=0.5)

    def _execute_subprocess_sync(self, code: str, handler: dict) -> dict:
        """Run code in a spawned child, receiving or timing out synchronously."""

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_subprocess_entrypoint,
            args=(child_connection, code, handler),
            name="magic-agents-python-exec",
        )
        try:
            process.start()
            child_connection.close()
            deadline = time.monotonic() + self._timeout

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_process(process)
                    return {"error": f"execution timed out after {self._timeout} seconds"}

                if parent_connection.poll(min(remaining, 0.05)):
                    try:
                        kind, payload = parent_connection.recv()
                    except EOFError:
                        kind, payload = "error", "Python worker exited without a result"
                    process.join(timeout=0.5)
                    if process.is_alive():
                        self._stop_process(process)
                    if kind == "result":
                        return {"result": self._truncate_result(payload)}
                    return {"error": str(payload)}

                if not process.is_alive():
                    process.join(timeout=0.1)
                    return {"error": "Python worker exited without a result"}
        except Exception as exc:
            self._stop_process(process)
            return {"error": str(exc)}
        finally:
            parent_connection.close()
            child_connection.close()

    async def _execute_local(self, code: str, handler: dict) -> dict:
        namespace_builtins = (
            _RESTRICTED_BUILTINS
            if self.safety_mode == "restricted_builtins"
            else builtins.__dict__
        )
        run_func = _compile_run_function(code, namespace_builtins)
        result = await asyncio.wait_for(
            asyncio.to_thread(run_func, handler),
            timeout=self._timeout,
        )
        return {"result": self._truncate_result(result)}

    async def execute(self, code: str, handler: dict) -> dict:
        """Compile code and execute run(handler) with timeout enforcement.

        Args:
            code: Python source code string with a run(handler) function.
            handler: Dict of input values to pass to run().

        Returns:
            On success: {"result": <return value>}
            On error:   {"error": <error message string>}
            Never raises — all exceptions are caught and returned as error dicts.
        """
        try:
            if self.safety_mode == "subprocess":
                return await asyncio.to_thread(self._execute_subprocess_sync, code, handler)
            return await self._execute_local(code, handler)

        except asyncio.TimeoutError:
            return {"error": f"execution timed out after {self._timeout} seconds"}
        except ValueError as e:
            # Compilation errors from self.compile()
            return {"error": str(e)}
        except Exception as e:
            # Runtime errors from run(handler) execution
            return {"error": str(e)}
