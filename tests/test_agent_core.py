"""Tests for the core agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.core import invoke_agent


@pytest.mark.asyncio
async def test_invoke_agent_returns_string():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Hello!"))

    result = await invoke_agent(
        llm=mock_llm,
        system_prompt="You are helpful.",
        user_message="Hi",
        user_name="TestUser",
    )
    assert result == "Hello!"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_invoke_agent_includes_user_attribution():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))

    await invoke_agent(
        llm=mock_llm,
        system_prompt="System",
        user_message="Hello",
        user_name="Alice",
    )

    call_args = mock_llm.ainvoke.call_args[0][0]
    human_msgs = [m for m in call_args if hasattr(m, "content") and "Alice" in str(m.content)]
    assert len(human_msgs) > 0


@pytest.mark.asyncio
async def test_invoke_agent_has_system_message():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="OK"))

    await invoke_agent(
        llm=mock_llm,
        system_prompt="Be helpful",
        user_message="test",
        user_name="Bob",
    )

    from langchain_core.messages import SystemMessage
    call_args = mock_llm.ainvoke.call_args[0][0]
    system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
    assert len(system_msgs) == 1
    assert "Be helpful" in system_msgs[0].content
