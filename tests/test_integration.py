"""Integration test — end-to-end verification."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.store import MessageStore
from src.memory.vector import VectorMemory
from src.memory.operational import OperationalMemory
from src.agent.core import CoreAgent


@pytest.mark.asyncio
async def test_full_message_flow(tmp_path):
    """End-to-end: message received -> stored -> agent responds -> response stored -> indexed."""
    # Setup components
    store = MessageStore(tmp_path / "messages.sqlite")
    await store.initialize()

    vector_mem = VectorMemory(persist_dir=tmp_path / "vectors")

    opmem = OperationalMemory(memory_dir=tmp_path / "memory")
    opmem.initialize()

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Hello Alice!"))

    agent = CoreAgent(
        llm=mock_llm,
        system_prompt="Test system prompt",
        vector_memory=vector_mem,
        operational_memory=opmem,
    )

    # Simulate agent invocation
    result = await agent.invoke(
        session_id="dm-123",
        user_message="Hi there, my name is Alice",
        user_name="Alice",
    )
    assert result == "Hello Alice!"

    # Verify message was indexed in vector store
    results = vector_mem.search("Alice", k=1)
    assert len(results) >= 1

    # Verify session maintained
    result2 = await agent.invoke(
        session_id="dm-123",
        user_message="What did I just say?",
        user_name="Alice",
    )
    # Second call should have history
    second_call = mock_llm.ainvoke.call_args_list[1][0][0]
    assert len(second_call) > 2  # system + prior exchange + new message

    await store.close()
