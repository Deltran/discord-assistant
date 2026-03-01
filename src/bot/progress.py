"""Progress reporter — sends execution updates to a Discord channel."""

import logging

import discord

from src.bot.formatters import split_message

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Reports plan execution progress to a Discord channel."""

    def __init__(self, channel: discord.abc.Messageable):
        self.channel = channel

    async def report(self, message: str):
        """Send a progress update."""
        for chunk in split_message(message):
            await self.channel.send(chunk)

    async def complete(self, summary: str):
        """Send the final completion summary."""
        for chunk in split_message(f"---\n{summary}"):
            await self.channel.send(chunk)
