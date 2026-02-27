"""Session routing logic.

- DMs: One persistent session per user (dm-{user_id})
- Channels: One session per channel (channel-{channel_id})
"""

from typing import Any


def get_session_id(message: Any) -> str:
    """Determine the session ID for a Discord message."""
    if message.channel.type.name == "private":
        return f"dm-{message.author.id}"
    return f"channel-{message.channel.id}"
