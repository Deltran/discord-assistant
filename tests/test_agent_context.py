"""Tests for cross-session context injection."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.core import CoreAgent
from src.memory.vector import VectorMemory
from src.memory.operational import OperationalMemory


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))
    return llm


@pytest.fixture
def vector_memory(tmp_path):
    return VectorMemory(persist_dir=tmp_path / "vectors")


@pytest.fixture
def opmem(tmp_path):
    om = OperationalMemory(memory_dir=tmp_path / "memory")
    om.initialize()
    return om


@pytest.mark.asyncio
async def test_agent_searches_vector_memory(mock_llm, vector_memory):
    vector_memory.add(
        text="Alice said she loves pizza",
        metadata={"channel_id": "ch-1", "user_name": "Alice"},
    )

    agent = CoreAgent(
        llm=mock_llm,
        system_prompt="Be helpful",
        vector_memory=vector_memory,
    )

    await agent.invoke(session_id="dm-1", user_message="What food does Alice like?", user_name="Bob")

    # The LLM should have been called with context that includes pizza
    call_args = mock_llm.ainvoke.call_args[0][0]
    all_content = " ".join(m.content for m in call_args)
    assert "pizza" in all_content


@pytest.mark.asyncio
async def test_agent_injects_operational_memory(mock_llm, opmem):
    opmem.append_safety_rule("Never visit badsite.com")

    agent = CoreAgent(
        llm=mock_llm,
        system_prompt="Be helpful",
        operational_memory=opmem,
    )

    await agent.invoke(session_id="dm-1", user_message="Hello", user_name="Alice")

    call_args = mock_llm.ainvoke.call_args[0][0]
    all_content = " ".join(m.content for m in call_args)
    assert "badsite.com" in all_content


@pytest.mark.asyncio
async def test_agent_indexes_messages(mock_llm, vector_memory):
    agent = CoreAgent(
        llm=mock_llm,
        system_prompt="Be helpful",
        vector_memory=vector_memory,
    )

    await agent.invoke(session_id="dm-1", user_message="I love sushi", user_name="Alice")

    # The message should now be searchable
    results = vector_memory.search("sushi", k=1)
    assert len(results) == 1
    assert "sushi" in results[0]["text"]


@pytest.mark.asyncio
async def test_agent_works_without_memory(mock_llm):
    agent = CoreAgent(llm=mock_llm, system_prompt="Be helpful")
    result = await agent.invoke(session_id="dm-1", user_message="Hi", user_name="Alice")
    assert result == "Response"
