"""Tests for sub-agent base infrastructure."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.subagents.base import SubAgentManager, SubAgentTask


@pytest.fixture
def manager():
    return SubAgentManager(max_concurrent=3, max_depth=2)


@pytest.mark.asyncio
async def test_submit_task(manager):
    async def dummy_work(ctx):
        return "done"

    task = await manager.submit(
        name="test-task",
        work_fn=dummy_work,
        callback=AsyncMock(),
    )
    assert isinstance(task, SubAgentTask)
    assert task.name == "test-task"


@pytest.mark.asyncio
async def test_concurrency_cap(manager):
    started = asyncio.Event()
    gate = asyncio.Event()

    async def blocking_work(ctx):
        started.set()
        await gate.wait()
        return "done"

    callbacks = [AsyncMock() for _ in range(5)]

    # Submit 3 tasks (at the cap)
    for i in range(3):
        await manager.submit(name=f"task-{i}", work_fn=blocking_work, callback=callbacks[i])

    # Wait a tick for tasks to start
    await asyncio.sleep(0.05)

    assert manager.active_count == 3

    # Fourth task should be queued, not running
    task4 = await manager.submit(name="task-3", work_fn=blocking_work, callback=callbacks[3])
    await asyncio.sleep(0.05)
    assert manager.active_count == 3  # Still 3, not 4

    # Release all
    gate.set()
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_depth_limit(manager):
    async def nested_work(ctx):
        if ctx.get("depth", 0) >= 2:
            return "too deep"
        return "ok"

    task = await manager.submit(
        name="deep-task",
        work_fn=nested_work,
        callback=AsyncMock(),
        depth=3,
    )
    # Task at depth 3 should be rejected (max_depth=2)
    assert task is None or task.rejected


@pytest.mark.asyncio
async def test_callback_receives_result(manager):
    async def simple_work(ctx):
        return "result-value"

    callback = AsyncMock()
    await manager.submit(name="cb-test", work_fn=simple_work, callback=callback)
    await asyncio.sleep(0.1)
    callback.assert_called_once_with("result-value")


@pytest.mark.asyncio
async def test_callback_receives_error(manager):
    async def failing_work(ctx):
        raise ValueError("boom")

    callback = AsyncMock()
    await manager.submit(name="fail-test", work_fn=failing_work, callback=callback)
    await asyncio.sleep(0.1)
    # Callback should receive the error info
    callback.assert_called_once()
    arg = callback.call_args[0][0]
    assert "error" in str(arg).lower() or "boom" in str(arg).lower()
