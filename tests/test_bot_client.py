"""Tests for the Discord bot client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.client import AssistantBot


@pytest.fixture
def bot():
    with patch.dict("os.environ", {
        "DISCORD_TOKEN": "test",
        "MINIMAX_API_KEY": "test",
        "MONITORING_CHANNEL_ID": "1",
    }):
        from importlib import reload
        import src.settings
        reload(src.settings)
        settings = src.settings.Settings()
        return AssistantBot(settings=settings)


def test_bot_initializes(bot):
    assert bot is not None
    assert bot.settings.discord_token.get_secret_value() == "test"


@pytest.mark.asyncio
async def test_on_message_ignores_self(bot):
    bot._user = MagicMock()
    bot._user.id = 999

    msg = MagicMock()
    msg.author.id = 999  # Same as bot
    msg.author.bot = True

    with patch.object(bot, "_handle_message", new_callable=AsyncMock) as mock_handle:
        await bot.on_message(msg)
        mock_handle.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_responds_to_dm(bot):
    bot._user = MagicMock()
    bot._user.id = 999

    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.author.display_name = "Alice"
    msg.content = "Hello"
    msg.channel.type = MagicMock()
    msg.channel.type.name = "private"
    msg.channel.name = None
    msg.channel.send = AsyncMock()
    msg.mentions = []
    msg.guild = None

    callback = AsyncMock(return_value="Hi Alice!")
    bot._agent_callback = callback

    await bot.on_message(msg)
    callback.assert_called_once()


@pytest.mark.asyncio
async def test_on_message_ignores_ignored_channel(bot):
    bot._user = MagicMock()
    bot._user.id = 999

    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.channel.type = MagicMock()
    msg.channel.type.name = "text"
    msg.channel.name = "general"
    msg.mentions = []
    msg.guild = MagicMock()

    callback = AsyncMock()
    bot._agent_callback = callback

    await bot.on_message(msg)
    callback.assert_not_called()
