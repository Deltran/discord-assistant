"""Context compaction — summarize old messages when approaching limits."""

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def should_compact(messages: list[BaseMessage], *, max_messages: int = 50) -> bool:
    """Check if the message list needs compaction."""
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    return len(non_system) > max_messages


async def compact_messages(
    messages: list[BaseMessage],
    *,
    llm: ChatOpenAI,
    keep_recent: int = 10,
) -> list[BaseMessage]:
    """Compact old messages into a summary, keeping recent ones intact."""
    if len(messages) <= keep_recent:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]

    conversation_text = "\n".join(
        f"{type(m).__name__}: {m.content}" for m in old_messages
    )

    summary_prompt = [
        SystemMessage(
            content="Summarize the following conversation concisely, preserving key facts, decisions, and context that would be needed to continue the conversation."
        ),
        HumanMessage(content=conversation_text),
    ]

    response = await llm.ainvoke(summary_prompt)

    summary_msg = HumanMessage(
        content=f"[Conversation summary]: {response.content}"
    )

    return [summary_msg] + recent_messages
