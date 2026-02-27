"""Monitoring channel -- operational messages posted to a dedicated Discord channel."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MonitoringChannel:
    def __init__(self, *, bot, channel_id: int):
        self._bot = bot
        self._channel_id = channel_id
        self._channel = None

    async def initialize(self):
        """Resolve the channel from the bot's cache."""
        if self._channel_id:
            self._channel = self._bot.get_channel(self._channel_id)
            if not self._channel:
                logger.warning(f"Monitoring channel {self._channel_id} not found")

    async def post(self, message: str):
        """Post a message to the monitoring channel."""
        if not self._channel:
            logger.warning(f"Monitoring channel not available, logging instead: {message}")
            return
        try:
            from src.bot.formatters import split_message
            for chunk in split_message(message):
                await self._channel.send(chunk)
        except Exception as e:
            logger.error(f"Failed to post to monitoring channel: {e}")

    async def post_startup(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await self.post(f"\U0001f7e2 **Bot started** at {ts}")

    async def post_shutdown(self):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        await self.post(f"\U0001f534 **Bot shutting down** at {ts}")

    async def post_heartbeat(self):
        ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
        await self.post(f"\U0001f493 Heartbeat at {ts}")

    async def post_error(self, error: str):
        await self.post(f"\u26a0\ufe0f **Error:** {error}")

    async def post_safety_rule(self, rule: str):
        await self.post(f"\U0001f6e1\ufe0f **New safety rule:** {rule}")

    async def post_compaction(self, session_id: str):
        await self.post(f"\U0001f4e6 **Compaction:** Session `{session_id}` was compacted")

    async def post_subagent_complete(self, name: str, summary: str):
        await self.post(f"\u2705 **Sub-agent `{name}` complete:** {summary[:200]}")

    async def post_soul_proposal(self, diff: str, reason: str):
        await self.post(f"\u270f\ufe0f **SOUL.md change proposed**\nReason: {reason}\n```diff\n{diff}\n```\nReply with `approve` or `reject`.")
