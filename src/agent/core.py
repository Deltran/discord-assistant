"""Core agent — Phase 1: simple LLM call with system prompt.

In later phases this becomes a full LangGraph state machine with routing,
sub-agent delegation, and checkpointing.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.providers.minimax import normalize_messages


async def invoke_agent(
    *,
    llm: ChatOpenAI,
    system_prompt: str,
    user_message: str,
    user_name: str,
) -> str:
    """Invoke the agent with a single user message and return the response."""
    messages = normalize_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"[{user_name}]: {user_message}"),
    ])

    response = await llm.ainvoke(messages)
    return response.content
