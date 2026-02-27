"""Periodic self-review of operational memory."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.providers.minimax import normalize_messages
from src.memory.operational import OperationalMemory

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """Review the following operational memory files. Your job is to:
1. Identify redundant entries that can be consolidated
2. Flag stale items that may no longer be relevant
3. Suggest improvements to organization

Be conservative — only suggest removing items you're confident are stale.
Output a summary of what you found and any recommendations."""


async def run_self_review(*, llm: ChatOpenAI, operational_memory: OperationalMemory) -> str:
    """Review operational memory and return recommendations."""
    all_memory = operational_memory.read_all()

    content = ""
    for key, text in all_memory.items():
        content += f"\n## {key}\n{text}\n"

    messages = normalize_messages([
        SystemMessage(content=REVIEW_PROMPT),
        HumanMessage(content=content),
    ])

    response = await llm.ainvoke(messages)
    return response.content
