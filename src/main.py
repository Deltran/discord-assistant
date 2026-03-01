"""Entry point for the Discord AI assistant daemon."""

import logging
from pathlib import Path

from src.agent.core import CoreAgent, LLMProviderError
from src.agent.router import get_session_id
from src.bot.client import AssistantBot
from src.memory.operational import OperationalMemory
from src.memory.store import MessageStore
from src.memory.vector import VectorMemory
from src.monitoring import MonitoringChannel
from src.providers.minimax import create_llm
from src.scheduler.heartbeat import HeartbeatRunner
from src.scheduler.jobs import SchedulerManager
from src.settings import Settings
from src.skills.loader import load_manifests
from src.skills.registry import SkillRegistry
from src.soul import load_soul
from src.tools.files import file_read, file_write
from src.tools.shell import shell_exec
from src.tools.skill_author import create_skill_author_tool
from src.tools.skill_dispatch import create_dispatch_skill_tool
from src.tools.web import http_request, scrape_url, web_search

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

    # Skill registry — load builtins and user skills
    registry = SkillRegistry()
    builtin_dir = Path(__file__).parent / "skills" / "builtin"
    for manifest in load_manifests(builtin_dir):
        registry.register(manifest)
        logger.info("Registered builtin skill: %s", manifest.name)
    for manifest in load_manifests(settings.skills_dir):
        registry.register(manifest)
        logger.info("Registered user skill: %s", manifest.name)

    # Core tools available to the agent
    base_tools = [web_search, scrape_url, http_request, shell_exec, file_read, file_write]

    # Skill dispatch meta-tool
    dispatch_tool = create_dispatch_skill_tool(
        registry=registry, llm=llm, available_tools=base_tools,
    )

    # Skill authoring tool — lets the agent create new skills
    author_tool = create_skill_author_tool(
        skills_dir=settings.skills_dir, registry=registry,
    )

    tools = base_tools + [dispatch_tool, author_tool]

    agent = CoreAgent(
        llm=llm,
        system_prompt=system_prompt,
        vector_memory=vector_memory,
        operational_memory=operational_memory,
        tools=tools,
        skill_registry=registry,
    )

    message_store = MessageStore(settings.data_dir / "messages.sqlite")

    bot = AssistantBot(
        settings=settings,
        agent_callback=None,  # set below after we have monitoring
        message_store=message_store,
    )

    # Monitoring channel
    monitoring = MonitoringChannel(
        bot=bot,
        channel_id=settings.monitoring_channel_id,
    )

    # Heartbeat runner
    heartbeat = HeartbeatRunner(
        llm=llm,
        monitoring=monitoring,
        assistant_home=settings.assistant_home,
    )

    # Agent callback — records errors to heartbeat
    async def agent_callback(message) -> str:
        try:
            return await agent.invoke(
                session_id=get_session_id(message),
                user_message=message.content,
                user_name=message.author.display_name,
            )
        except LLMProviderError as e:
            heartbeat.record_error(str(e))
            raise

    bot._agent_callback = agent_callback

    # Scheduler
    scheduler = SchedulerManager()
    scheduler.setup_default_jobs(
        heartbeat_fn=heartbeat.run,
    )

    # Hook into bot lifecycle
    original_on_ready = bot.on_ready

    async def on_ready_with_infra():
        await original_on_ready()
        await monitoring.initialize()
        scheduler.start()
        await monitoring.post_startup()
        logger.info("Monitoring, heartbeat, and scheduler started")

    bot.on_ready = on_ready_with_infra

    original_close = bot.close

    async def close_with_infra():
        scheduler.stop()
        await monitoring.post_shutdown()
        logger.info("Scheduler stopped")
        await original_close()

    bot.close = close_with_infra

    return bot


def main():
    bot = create_app()
    logger.info("Starting Discord assistant...")
    bot.run(bot.settings.discord_token.get_secret_value(), log_handler=None)


if __name__ == "__main__":
    main()
