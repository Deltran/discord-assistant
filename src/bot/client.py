"""Discord bot client — the gateway between Discord and the agent."""

import logging
from pathlib import Path
from typing import Callable, Awaitable

import discord
import yaml

from src.agent.core import LLMProviderError
from src.bot.filters import MessageAction, evaluate_message
from src.bot.formatters import split_message
from src.memory.store import MessageStore
from src.settings import Settings

logger = logging.getLogger(__name__)


class AssistantBot(discord.Client):
    def __init__(
        self,
        *,
        settings: Settings,
        agent_callback: Callable | None = None,
        message_store: MessageStore | None = None,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)

        self.settings = settings
        self._agent_callback = agent_callback
        self._message_store = message_store
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
        if self._message_store:
            await self._message_store.initialize()
            logger.info("Message store initialized")

    async def close(self):
        if self._message_store:
            await self._message_store.close()
            logger.info("Message store closed")
        await super().close()

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

        await self._save_incoming(message)

        if action == MessageAction.READ_ONLY:
            logger.debug(f"Read-only message from {message.author}: {message.content[:50]}")
            return

        await self._handle_message(message)

    async def _handle_message(self, message: discord.Message):
        if self._agent_callback is None:
            await message.channel.send("I received your message. Agent not yet connected.")
            return

        try:
            async with message.channel.typing():
                response = await self._agent_callback(message)
        except LLMProviderError as e:
            logger.error("LLM provider error: %s (recoverable=%s)", e, e.recoverable)
            if e.recoverable:
                await message.channel.send(
                    f"Sorry, I'm having trouble thinking right now — {e}. Try again shortly."
                )
            else:
                await message.channel.send(
                    f"I've hit a problem I can't recover from — {e}. My operator will need to look into this."
                )
            return
        except Exception:
            logger.exception("Unexpected error handling message")
            await message.channel.send(
                "Something went wrong on my end. Please try again later."
            )
            return

        if response:
            for chunk in split_message(response):
                await message.channel.send(chunk)
            await self._save_bot_response(message.channel.id, response)

    async def _save_incoming(self, message: discord.Message):
        if self._message_store is None:
            return
        try:
            await self._message_store.save_message(
                channel_id=str(message.channel.id),
                user_id=str(message.author.id),
                user_name=message.author.display_name,
                content=message.content,
                is_bot=message.author.bot,
                bot_name=message.author.display_name if message.author.bot else None,
            )
        except Exception:
            logger.exception("Failed to save incoming message")

    async def _save_bot_response(self, channel_id: int, content: str):
        if self._message_store is None:
            return
        try:
            bot_name = self.user.name if self.user else "assistant"
            await self._message_store.save_message(
                channel_id=str(channel_id),
                user_id=str(self.user.id) if self.user else "0",
                user_name=bot_name,
                content=content,
                is_bot=True,
                bot_name=bot_name,
            )
        except Exception:
            logger.exception("Failed to save bot response")
