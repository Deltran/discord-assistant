"""Tests for self-review cycle."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.self_review import run_self_review
from src.memory.operational import OperationalMemory


@pytest.mark.asyncio
async def test_self_review(tmp_path):
    opmem = OperationalMemory(memory_dir=tmp_path)
    opmem.initialize()
    opmem.append_safety_rule("Rule 1")
    opmem.add_operational_note("Note 1")

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Review: all looks good"))

    result = await run_self_review(llm=mock_llm, operational_memory=opmem)
    assert "all looks good" in result
    mock_llm.ainvoke.assert_called_once()
