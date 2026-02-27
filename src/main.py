"""Entry point for the Discord AI assistant daemon."""

import logging

from src.bot.client import AssistantBot
from src.agent.core import CoreAgent
from src.agent.router import get_session_id
from src.memory.operational import OperationalMemory
from src.memory.store import MessageStore
from src.memory.vector import VectorMemory
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

    vector_memory = VectorMemory(persist_dir=settings.data_dir / "vectors")
    logger.info("Vector memory initialized at %s", settings.data_dir / "vectors")

    operational_memory = OperationalMemory(memory_dir=settings.memory_dir)
    operational_memory.initialize()
    logger.info("Operational memory initialized at %s", settings.memory_dir)

    agent = CoreAgent(
        llm=llm,
        system_prompt=system_prompt,
        vector_memory=vector_memory,
        operational_memory=operational_memory,
    )

    async def agent_callback(message) -> str:
        return await agent.invoke(
            session_id=get_session_id(message),
            user_message=message.content,
            user_name=message.author.display_name,
        )

    message_store = MessageStore(settings.data_dir / "messages.sqlite")

    return AssistantBot(
        settings=settings,
        agent_callback=agent_callback,
        message_store=message_store,
    )


def main():
    bot = create_app()
    logger.info("Starting Discord assistant...")
    bot.run(bot.settings.discord_token.get_secret_value(), log_handler=None)


if __name__ == "__main__":
    main()
