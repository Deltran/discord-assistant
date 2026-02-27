"""Tests for Discord message filtering rules."""

from unittest.mock import MagicMock

from src.bot.filters import MessageAction, evaluate_message


def _make_message(
    *,
    author_bot: bool = False,
    mentions: list | None = None,
    channel_type: str = "text",
    channel_name: str = "test-channel",
    bot_id: int = 999,
) -> MagicMock:
    msg = MagicMock()
    msg.author.bot = author_bot
    msg.author.id = 123 if not author_bot else 456
    msg.mentions = mentions or []
    msg.channel.name = channel_name

    if channel_type == "dm":
        msg.channel.type = MagicMock()
        msg.channel.type.name = "private"
        msg.guild = None
    else:
        msg.channel.type = MagicMock()
        msg.channel.type.name = "text"
        msg.guild = MagicMock()

    return msg


def test_bot_message_is_read_only():
    msg = _make_message(author_bot=True)
    action = evaluate_message(msg, bot_user_id=999, ignored_channels={"general"})
    assert action == MessageAction.READ_ONLY


def test_mention_other_user_is_ignore():
    other_user = MagicMock()
    other_user.id = 777
    msg = _make_message(mentions=[other_user])
    action = evaluate_message(msg, bot_user_id=999, ignored_channels={"general"})
    assert action == MessageAction.IGNORE


def test_mention_bot_is_respond():
    bot_user = MagicMock()
    bot_user.id = 999
    msg = _make_message(mentions=[bot_user])
    action = evaluate_message(msg, bot_user_id=999, ignored_channels={"general"})
    assert action == MessageAction.RESPOND


def test_ignored_channel_is_ignore():
    msg = _make_message(channel_name="general")
    action = evaluate_message(msg, bot_user_id=999, ignored_channels={"general"})
    assert action == MessageAction.IGNORE


def test_dm_is_respond():
    msg = _make_message(channel_type="dm")
    action = evaluate_message(msg, bot_user_id=999, ignored_channels={"general"})
    assert action == MessageAction.RESPOND


def test_other_channel_message_is_respond():
    msg = _make_message(channel_name="dev-chat")
    action = evaluate_message(msg, bot_user_id=999, ignored_channels={"general"})
    assert action == MessageAction.RESPOND
