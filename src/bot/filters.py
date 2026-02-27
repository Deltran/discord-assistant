"""Message filtering rules evaluated in priority order.

Priority:
1. Bot message -> READ_ONLY (store in context, don't respond)
2. @mentions someone other than our bot -> IGNORE
3. @mentions our bot -> RESPOND
4. Ignored channel -> IGNORE
5. DM -> RESPOND
6. All other messages -> RESPOND
"""

from enum import Enum
from typing import Any


class MessageAction(Enum):
    RESPOND = "respond"
    READ_ONLY = "read_only"
    IGNORE = "ignore"


def evaluate_message(
    message: Any,
    *,
    bot_user_id: int,
    ignored_channels: set[str],
) -> MessageAction:
    """Evaluate a Discord message and return the appropriate action."""
    if message.author.bot:
        return MessageAction.READ_ONLY

    if message.mentions:
        bot_mentioned = any(m.id == bot_user_id for m in message.mentions)
        if bot_mentioned:
            return MessageAction.RESPOND
        return MessageAction.IGNORE

    channel_name = getattr(message.channel, "name", None)
    if channel_name and channel_name in ignored_channels:
        return MessageAction.IGNORE

    if message.channel.type.name == "private":
        return MessageAction.RESPOND

    return MessageAction.RESPOND
