"""Tests for context compaction."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from src.memory.compaction import should_compact, compact_messages


def test_should_compact_under_limit():
    messages = [HumanMessage(content="short")] * 5
    assert should_compact(messages, max_messages=50) is False


def test_should_compact_over_limit():
    messages = [HumanMessage(content="msg")] * 60
    assert should_compact(messages, max_messages=50) is True


@pytest.mark.asyncio
async def test_compact_messages():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="Summary of the conversation so far.")
    )

    messages = [
        HumanMessage(content=f"[User]: Message {i}")
        for i in range(30)
    ]

    result = await compact_messages(messages, llm=mock_llm, keep_recent=10)

    assert len(result) == 11  # 1 summary + 10 recent
    assert "Summary" in result[0].content


@pytest.mark.asyncio
async def test_compact_preserves_recent():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Summary"))

    messages = [HumanMessage(content=f"msg-{i}") for i in range(20)]

    result = await compact_messages(messages, llm=mock_llm, keep_recent=5)

    recent_contents = [m.content for m in result[1:]]  # skip summary
    assert recent_contents == [f"msg-{i}" for i in range(15, 20)]


@pytest.mark.asyncio
async def test_compact_short_list_no_op():
    mock_llm = MagicMock()
    messages = [HumanMessage(content="hello")]
    result = await compact_messages(messages, llm=mock_llm, keep_recent=10)
    assert result == messages
    mock_llm.ainvoke.assert_not_called()
