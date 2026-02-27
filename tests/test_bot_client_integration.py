"""Integration tests for bot client with message store."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.client import AssistantBot
from src.memory.store import MessageStore


@pytest.fixture
async def store(tmp_path):
    s = MessageStore(tmp_path / "test.sqlite")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def make_bot(store):
    with patch.dict("os.environ", {
        "DISCORD_TOKEN": "t", "MINIMAX_API_KEY": "k", "MONITORING_CHANNEL_ID": "1",
    }):
        from importlib import reload
        import src.settings
        reload(src.settings)
        settings = src.settings.Settings()

        callback = AsyncMock(return_value="Bot response")
        bot = AssistantBot(settings=settings, agent_callback=callback, message_store=store)
        bot._user = MagicMock()
        bot._user.id = 999
        return bot


@pytest.mark.asyncio
async def test_bot_stores_incoming_message(make_bot, store):
    bot = make_bot
    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.author.display_name = "Alice"
    msg.content = "Hello bot"
    msg.channel.type.name = "text"
    msg.channel.name = "dev"
    msg.channel.id = 555
    msg.channel.send = AsyncMock()
    msg.mentions = []
    msg.guild = MagicMock()

    await bot.on_message(msg)

    messages = await store.get_messages(channel_id="555", limit=10)
    user_msgs = [m for m in messages if not m["is_bot"]]
    assert len(user_msgs) >= 1
    assert user_msgs[0]["content"] == "Hello bot"
    assert user_msgs[0]["user_name"] == "Alice"


@pytest.mark.asyncio
async def test_bot_stores_response(make_bot, store):
    bot = make_bot
    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.author.display_name = "Alice"
    msg.content = "Hello"
    msg.channel.type.name = "text"
    msg.channel.name = "dev"
    msg.channel.id = 555
    msg.channel.send = AsyncMock()
    msg.mentions = []
    msg.guild = MagicMock()

    await bot.on_message(msg)

    messages = await store.get_messages(channel_id="555", limit=10)
    bot_msgs = [m for m in messages if m["is_bot"]]
    assert len(bot_msgs) >= 1
    assert bot_msgs[0]["content"] == "Bot response"


@pytest.mark.asyncio
async def test_bot_stores_read_only_message(make_bot, store):
    """READ_ONLY messages (from other bots) should also be stored."""
    bot = make_bot
    msg = MagicMock()
    msg.author.id = 456
    msg.author.bot = True
    msg.author.display_name = "OtherBot"
    msg.content = "Scheduled reminder"
    msg.channel.type.name = "text"
    msg.channel.name = "dev"
    msg.channel.id = 555
    msg.channel.send = AsyncMock()
    msg.mentions = []
    msg.guild = MagicMock()

    await bot.on_message(msg)

    messages = await store.get_messages(channel_id="555", limit=10)
    assert len(messages) == 1
    assert messages[0]["content"] == "Scheduled reminder"
    assert messages[0]["is_bot"] is True


@pytest.mark.asyncio
async def test_bot_does_not_store_ignored_messages(make_bot, store):
    """IGNORE messages should not be stored."""
    bot = make_bot
    # Mention someone other than our bot -> IGNORE
    msg = MagicMock()
    msg.author.id = 123
    msg.author.bot = False
    msg.author.display_name = "Alice"
    msg.content = "@someone hey"
    msg.channel.type.name = "text"
    msg.channel.name = "dev"
    msg.channel.id = 555
    msg.channel.send = AsyncMock()
    other_user = MagicMock()
    other_user.id = 888
    msg.mentions = [other_user]
    msg.guild = MagicMock()

    await bot.on_message(msg)

    messages = await store.get_messages(channel_id="555", limit=10)
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_bot_works_without_message_store():
    """Bot should work fine when no message store is provided."""
    with patch.dict("os.environ", {
        "DISCORD_TOKEN": "t", "MINIMAX_API_KEY": "k", "MONITORING_CHANNEL_ID": "1",
    }):
        from importlib import reload
        import src.settings
        reload(src.settings)
        settings = src.settings.Settings()

        callback = AsyncMock(return_value="response")
        bot = AssistantBot(settings=settings, agent_callback=callback)
        bot._user = MagicMock()
        bot._user.id = 999

        msg = MagicMock()
        msg.author.id = 123
        msg.author.bot = False
        msg.author.display_name = "Alice"
        msg.content = "Hello"
        msg.channel.type.name = "text"
        msg.channel.name = "dev"
        msg.channel.id = 555
        msg.channel.send = AsyncMock()
        msg.mentions = []
        msg.guild = MagicMock()

        # Should not raise
        await bot.on_message(msg)
        msg.channel.send.assert_called_once_with("response")
