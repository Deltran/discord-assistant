"""Entry point for the Discord AI assistant daemon."""

import logging

from src.bot.client import AssistantBot
from src.agent.core import invoke_agent
from src.providers.minimax import create_llm
from src.settings import Settings
from src.soul import load_soul

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> AssistantBot:
    """Create and configure the bot with all dependencies."""
    settings = Settings()

    system_prompt = load_soul(settings.soul_path)
    logger.info(f"Loaded SOUL.md from {settings.soul_path}")

    llm = create_llm(settings)

    async def agent_callback(message) -> str:
        return await invoke_agent(
            llm=llm,
            system_prompt=system_prompt,
            user_message=message.content,
            user_name=message.author.display_name,
        )

    return AssistantBot(settings=settings, agent_callback=agent_callback)


def main():
    bot = create_app()
    logger.info("Starting Discord assistant...")
    bot.run(bot.settings.discord_token.get_secret_value(), log_handler=None)


if __name__ == "__main__":
    main()
