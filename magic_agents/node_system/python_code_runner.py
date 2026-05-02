"""
Python Code Runner for run(handler) contract execution.

Provides CodeRunner, a wrapper class that handles exec()-based compilation
of user Python code, extracts the reserved run(handler) entrypoint, bridges
handler dicts, and wraps PythonExecutor for subprocess execution.

Phase 1: constrained exec() with restricted builtins.
Phase 2+: subprocess-only mode with import restrictions.

SECURITY WARNING: This is NOT a security boundary. The exec() namespace is
restricted but trivially escapable via the Python introspection chain.
Do NOT use with untrusted/third-party code.
"""

import asyncio
import logging
from typing import Any, Callable, Optional

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


class CodeRunner:
    """Wraps PythonExecutor for run(handler) execution.

    Phase 1 uses constrained exec() — this is NOT a security boundary.
    The exec() namespace is restricted but trivially escapable via the
    `().__class__.__bases__[0].__subclasses__()` introspection chain.

    Do NOT use with untrusted/third-party code.

    For untrusted code, configure safety_mode='subprocess' and understand
    that even subprocess mode does NOT prevent all escape vectors.
    Strong subprocess isolation (import restrictions, filesystem boundary,
    network containment) is deferred to Phase 2+.
    """

    def __init__(
        self,
        safety_mode: str = "subprocess",
        timeout: float = 30.0,
        max_output_chars: int = 8000,
    ):
        """Initialize CodeRunner.

        Internally creates a PythonExecutor for subprocess execution bridging.

        Args:
            safety_mode: Execution mode ('subprocess', 'in_process', or 'restricted_builtins').
            timeout: Maximum execution time in seconds.
            max_output_chars: Maximum output length before truncation.

        Warning:
            NOT a security boundary. Do not use with untrusted code.
            See class-level docstring for details.
        """
        from magic_llm.util.python_executor import PythonExecutor

        self._executor = PythonExecutor(
            safety_mode=safety_mode,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )
        self._timeout = timeout

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
        if not code or not code.strip():
            raise ValueError("User code must define run(handler) function")

        # Create execution namespace with restricted builtins
        namespace: dict[str, Any] = {
            '__builtins__': _RESTRICTED_BUILTINS,
        }

        try:
            exec(code, namespace)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in user code: {e}") from e
        except Exception as e:
            raise ValueError(f"Compilation failed: {e}") from e

        # Extract the run function from the namespace
        run_func = namespace.get('run')

        if run_func is None:
            raise ValueError("User code must define run(handler) function")

        if not callable(run_func):
            raise ValueError("'run' must be a callable function")

        return run_func

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
            run_func = self.compile(code)

            # Execute via asyncio.to_thread to avoid blocking event loop,
            # wrapped in asyncio.wait_for for timeout enforcement.
            result = await asyncio.wait_for(
                asyncio.to_thread(run_func, handler),
                timeout=self._timeout,
            )

            return {"result": result}

        except asyncio.TimeoutError:
            return {"error": f"execution timed out after {self._timeout} seconds"}
        except ValueError as e:
            # Compilation errors from self.compile()
            return {"error": str(e)}
        except Exception as e:
            # Runtime errors from run(handler) execution
            return {"error": str(e)}
