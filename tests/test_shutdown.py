"""Tests for graceful shutdown."""

import pytest

from src.shutdown import GracefulShutdown


@pytest.mark.asyncio
async def test_shutdown_runs_callbacks():
    shutdown = GracefulShutdown()
    called = []

    async def cb1():
        called.append("cb1")

    async def cb2():
        called.append("cb2")

    shutdown.register(cb1)
    shutdown.register(cb2)
    await shutdown.shutdown()

    assert "cb1" in called
    assert "cb2" in called


@pytest.mark.asyncio
async def test_shutdown_handles_callback_errors():
    shutdown = GracefulShutdown()

    async def failing_cb():
        raise ValueError("boom")

    async def good_cb():
        pass

    shutdown.register(failing_cb)
    shutdown.register(good_cb)
    # Should not raise
    await shutdown.shutdown()


@pytest.mark.asyncio
async def test_shutdown_only_runs_once():
    shutdown = GracefulShutdown()
    count = [0]

    async def cb():
        count[0] += 1

    shutdown.register(cb)
    await shutdown.shutdown()
    await shutdown.shutdown()  # Second call should be no-op
    assert count[0] == 1
