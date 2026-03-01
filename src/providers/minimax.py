"""MiniMax m2.5 provider configuration and message normalization."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.settings import Settings


def normalize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure strict user/assistant alternation as required by MiniMax.

    Rules:
    - System messages stay at the front
    - After system messages, first non-system must be HumanMessage
    - Consecutive same-role *conversational* messages are merged
    - AIMessages with tool_calls are never merged (they must pair with ToolMessages)
    - ToolMessages are never merged (they must pair with the preceding AIMessage)
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system:
        return system_msgs

    # Ensure first non-system message is HumanMessage
    if not isinstance(non_system[0], HumanMessage):
        non_system.insert(0, HumanMessage(content="[conversation start]"))

    # Merge consecutive same-role conversational messages, but preserve tool-call sequences
    merged: list[BaseMessage] = [non_system[0]]
    for msg in non_system[1:]:
        # Never merge ToolMessages
        if isinstance(msg, ToolMessage):
            merged.append(msg)
            continue

        # Never merge AIMessages that have tool_calls
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            merged.append(msg)
            continue

        # Never merge into an AIMessage that has tool_calls
        prev = merged[-1]
        if isinstance(prev, AIMessage) and getattr(prev, "tool_calls", None):
            merged.append(msg)
            continue

        # Don't merge into or from ToolMessages
        if isinstance(prev, ToolMessage):
            merged.append(msg)
            continue

        # Merge consecutive same-type conversational messages
        if type(msg) is type(prev):
            merged[-1] = type(msg)(content=prev.content + "\n" + msg.content)
        else:
            merged.append(msg)

    return system_msgs + merged


def create_llm(settings: Settings) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for MiniMax."""
    return ChatOpenAI(
        model=settings.minimax_model,
        api_key=settings.minimax_api_key.get_secret_value(),
        base_url=settings.minimax_base_url,
        temperature=0.7,
    )
