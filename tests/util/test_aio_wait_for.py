"""magic_agents.util.aio.wait_for keeps Python 3.12 semantics on every version."""
import asyncio

import pytest

from magic_agents.util.aio import wait_for


async def test_awaits_in_the_calling_task():
    async def current():
        return asyncio.current_task()
    assert await wait_for(current(), 1) is asyncio.current_task()


async def test_timeout_raises_and_cancels_the_awaitable():
    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    with pytest.raises(TimeoutError):
        await wait_for(slow(), 0.01)
    assert cancelled.is_set()


async def test_nonpositive_timeout_returns_done_results_and_times_out_pending_work():
    done = asyncio.get_running_loop().create_future()
    done.set_result('ready')
    assert await wait_for(done, 0) == 'ready'
    with pytest.raises(TimeoutError):
        await wait_for(asyncio.Event().wait(), 0)


async def test_outer_cancellation_propagates():
    started = asyncio.Event()

    async def waiter():
        started.set()
        await wait_for(asyncio.Event().wait(), 5)
    task = asyncio.create_task(waiter())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
