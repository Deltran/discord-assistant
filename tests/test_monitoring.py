"""Tests for monitoring channel."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.monitoring import MonitoringChannel


@pytest.fixture
def monitoring():
    bot = MagicMock()
    channel = MagicMock()
    channel.send = AsyncMock()
    bot.get_channel = MagicMock(return_value=channel)
    mon = MonitoringChannel(bot=bot, channel_id=12345)
    return mon, channel


@pytest.mark.asyncio
async def test_initialize(monitoring):
    mon, channel = monitoring
    await mon.initialize()
    assert mon._channel is not None


@pytest.mark.asyncio
async def test_post_message(monitoring):
    mon, channel = monitoring
    await mon.initialize()
    await mon.post("Test message")
    channel.send.assert_called_once_with("Test message")


@pytest.mark.asyncio
async def test_post_startup(monitoring):
    mon, channel = monitoring
    await mon.initialize()
    await mon.post_startup()
    call_args = channel.send.call_args[0][0]
    assert "started" in call_args.lower()


@pytest.mark.asyncio
async def test_post_without_channel():
    bot = MagicMock()
    bot.get_channel = MagicMock(return_value=None)
    mon = MonitoringChannel(bot=bot, channel_id=0)
    await mon.initialize()
    # Should not raise
    await mon.post("test")
