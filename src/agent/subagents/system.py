"""System sub-agent — shell commands and file operations."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a system administration assistant. Execute commands safely.
NEVER execute destructive commands without explicit confirmation.
Report results clearly."""


async def run_system_task(*, llm: ChatOpenAI, task: str) -> str:
    """Run a system task (shell command, file operation)."""
    messages = normalize_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Task: {task}\n\n"
                "Analyze this task and determine the appropriate commands to run."
            )
        ),
    ])

    response = await llm.ainvoke(messages)
    return response.content
