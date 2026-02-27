"""Tests for session routing."""

from unittest.mock import MagicMock

from src.agent.router import get_session_id


def test_dm_session_id():
    msg = MagicMock()
    msg.channel.type.name = "private"
    msg.author.id = 12345
    msg.guild = None
    result = get_session_id(msg)
    assert result == "dm-12345"


def test_channel_session_id():
    msg = MagicMock()
    msg.channel.type.name = "text"
    msg.channel.id = 67890
    msg.guild = MagicMock()
    result = get_session_id(msg)
    assert result == "channel-67890"
