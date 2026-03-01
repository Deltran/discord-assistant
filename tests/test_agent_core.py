"""Tests for the LangGraph-based core agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError
from httpx import Response, Request

from src.agent.core import CoreAgent, LLMProviderError


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Hello!"))
    return llm


@pytest.fixture
def agent(mock_llm, tmp_path):
    return CoreAgent(
        llm=mock_llm,
        system_prompt="You are helpful.",
    )


@pytest.mark.asyncio
async def test_agent_responds(agent):
    result = await agent.invoke(
        session_id="dm-123",
        user_message="Hi",
        user_name="Alice",
    )
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_agent_maintains_session(agent, mock_llm):
    await agent.invoke(session_id="dm-123", user_message="Hi", user_name="Alice")
    await agent.invoke(session_id="dm-123", user_message="How are you?", user_name="Alice")

    # Second call should have conversation history
    second_call_messages = mock_llm.ainvoke.call_args_list[1][0][0]
    assert len(second_call_messages) > 2  # system + first exchange + new message


@pytest.mark.asyncio
async def test_agent_separate_sessions(agent, mock_llm):
    await agent.invoke(session_id="dm-123", user_message="Hi from Alice", user_name="Alice")
    await agent.invoke(session_id="dm-456", user_message="Hi from Bob", user_name="Bob")

    # Bob's session should not contain Alice's messages
    bob_messages = mock_llm.ainvoke.call_args_list[1][0][0]
    message_contents = " ".join(m.content for m in bob_messages)
    assert "Alice" not in message_contents


@pytest.mark.asyncio
async def test_agent_compacts_long_sessions():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))

    agent = CoreAgent(
        llm=mock_llm,
        system_prompt="System",
        max_session_messages=10,
    )

    for i in range(12):
        await agent.invoke(session_id="dm-1", user_message=f"Msg {i}", user_name="Alice")

    session = agent._get_session("dm-1")
    # 12 human + 12 AI = 24 without compaction, should be compacted
    assert len(session) < 24


@pytest.mark.asyncio
async def test_auth_error_raises_unrecoverable():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=AuthenticationError(
            message="Invalid API key",
            response=Response(status_code=401, request=Request("POST", "https://api.minimax.io/v1")),
            body=None,
        )
    )
    agent = CoreAgent(llm=mock_llm, system_prompt="System")

    with pytest.raises(LLMProviderError) as exc_info:
        await agent.invoke(session_id="dm-1", user_message="Hi", user_name="Alice")

    assert not exc_info.value.recoverable
    # Session should not contain the failed user message
    assert len(agent._get_session("dm-1")) == 0


@pytest.mark.asyncio
async def test_rate_limit_raises_recoverable():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=RateLimitError(
            message="Rate limit exceeded",
            response=Response(status_code=429, request=Request("POST", "https://api.minimax.io/v1")),
            body=None,
        )
    )
    agent = CoreAgent(llm=mock_llm, system_prompt="System")

    with pytest.raises(LLMProviderError) as exc_info:
        await agent.invoke(session_id="dm-1", user_message="Hi", user_name="Alice")

    assert exc_info.value.recoverable
    assert len(agent._get_session("dm-1")) == 0


@pytest.mark.asyncio
async def test_connection_error_raises_recoverable():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=APIConnectionError(request=Request("POST", "https://api.minimax.io/v1"))
    )
    agent = CoreAgent(llm=mock_llm, system_prompt="System")

    with pytest.raises(LLMProviderError) as exc_info:
        await agent.invoke(session_id="dm-1", user_message="Hi", user_name="Alice")

    assert exc_info.value.recoverable


@pytest.mark.asyncio
async def test_server_error_raises_recoverable():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=APIStatusError(
            message="Internal server error",
            response=Response(status_code=500, request=Request("POST", "https://api.minimax.io/v1")),
            body=None,
        )
    )
    agent = CoreAgent(llm=mock_llm, system_prompt="System")

    with pytest.raises(LLMProviderError) as exc_info:
        await agent.invoke(session_id="dm-1", user_message="Hi", user_name="Alice")

    assert exc_info.value.recoverable
