"""Asyncio helpers with the same semantics on every supported Python.

``asyncio.wait_for`` changed in Python 3.12 (gh-96764): it now awaits in the
calling task under ``asyncio.timeout``. On 3.11 it wraps the awaitable in a new
Task, which has two consequences this package cannot tolerate:

* a cancellation that arrives in the same loop step the inner awaitable
  completes is swallowed and the result returned (gh-86296), so cancelled
  graphs, hooks and coordination waits keep running;
* code that enters and exits AnyIO cancel scopes (the MCP SDK session) runs in
  a different task from its caller and fails on exit.

``wait_for`` here is ``asyncio.wait_for`` on 3.12+ and a backport of the 3.12
implementation on 3.11.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Awaitable, Optional, TypeVar

T = TypeVar("T")

__all__ = ["wait_for"]


if sys.version_info >= (3, 12):
    wait_for = asyncio.wait_for
else:  # pragma: no cover - exercised by the Python 3.11 CI job
    async def wait_for(aw: Awaitable[T], timeout: Optional[float]) -> T:
        if timeout is not None and timeout <= 0:
            fut = asyncio.ensure_future(aw)
            if fut.done():
                return fut.result()
            fut.cancel()
            await asyncio.wait((fut,))
            try:
                return fut.result()
            except asyncio.CancelledError as exc:
                raise TimeoutError() from exc
        async with asyncio.timeout(timeout):
            return await aw
