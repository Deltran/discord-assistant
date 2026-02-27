"""Discord bot client — the gateway between Discord and the agent."""

import logging
from pathlib import Path
from typing import Callable, Awaitable

import discord
import yaml

from src.bot.filters import MessageAction, evaluate_message
from src.bot.formatters import split_message
from src.settings import Settings

logger = logging.getLogger(__name__)


class AssistantBot(discord.Client):
    def __init__(self, *, settings: Settings, agent_callback: Callable | None = None):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)

        self.settings = settings
        self._agent_callback = agent_callback
        self._ignored_channels = self._load_ignored_channels()

    def _load_ignored_channels(self) -> set[str]:
        config_path = Path("config/channels.yaml")
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            return set(data.get("ignored_channels", []))
        return {"general"}

    async def on_ready(self):
        logger.info(f"Bot connected as {self.user} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message):
        if self.user and message.author.id == self.user.id:
            return

        action = evaluate_message(
            message,
            bot_user_id=self.user.id if self.user else 0,
            ignored_channels=self._ignored_channels,
        )

        if action == MessageAction.IGNORE:
            return

        if action == MessageAction.READ_ONLY:
            logger.debug(f"Read-only message from {message.author}: {message.content[:50]}")
            return

        await self._handle_message(message)

    async def _handle_message(self, message: discord.Message):
        if self._agent_callback is None:
            await message.channel.send("I received your message. Agent not yet connected.")
            return

        response = await self._agent_callback(message)
        if response:
            for chunk in split_message(response):
                await message.channel.send(chunk)
