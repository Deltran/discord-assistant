"""Builder sub-agent — plan-driven step-by-step implementation."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)

BUILDER_PROMPT = """You are a builder assistant. Execute implementation plans step by step.
For each step: announce what you're doing, execute it, verify the result.
Stop and report on failure rather than pushing through."""


async def run_builder(*, llm: ChatOpenAI, plan: str) -> str:
    """Execute a build plan step by step."""
    messages = normalize_messages([
        SystemMessage(content=BUILDER_PROMPT),
        HumanMessage(
            content=f"Execute this plan:\n\n{plan}\n\nReport progress for each step."
        ),
    ])

    response = await llm.ainvoke(messages)
    return response.content
