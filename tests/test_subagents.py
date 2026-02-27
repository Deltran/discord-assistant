"""Tests for all sub-agents."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.subagents.research import run_research
from src.agent.subagents.system import run_system_task
from src.agent.subagents.briefing import run_briefing
from src.agent.subagents.builder import run_builder


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="LLM response"))
    return llm


@pytest.mark.asyncio
async def test_research_agent(mock_llm):
    with patch("src.tools.web.web_search") as mock_search:
        mock_search.ainvoke = AsyncMock(return_value="Search result about AI")
        result = await run_research(llm=mock_llm, query="latest AI news")
        assert result == "LLM response"
        mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_research_agent_deep(mock_llm):
    with patch("src.tools.web.web_search") as mock_search:
        mock_search.ainvoke = AsyncMock(return_value="Deep results")
        result = await run_research(llm=mock_llm, query="quantum computing", depth="deep")
        assert result == "LLM response"


@pytest.mark.asyncio
async def test_system_agent(mock_llm):
    result = await run_system_task(llm=mock_llm, task="check disk space")
    assert result == "LLM response"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_briefing_agent(mock_llm):
    with patch("src.tools.web.web_search") as mock_search:
        mock_search.ainvoke = AsyncMock(return_value="News results")
        result = await run_briefing(llm=mock_llm)
        assert result == "LLM response"
        # Should search for each default topic
        assert mock_search.ainvoke.call_count == 4


@pytest.mark.asyncio
async def test_briefing_custom_topics(mock_llm):
    with patch("src.tools.web.web_search") as mock_search:
        mock_search.ainvoke = AsyncMock(return_value="Results")
        await run_briefing(llm=mock_llm, topics=["weather", "sports"])
        assert mock_search.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_builder_agent(mock_llm):
    result = await run_builder(llm=mock_llm, plan="Step 1: Create file\nStep 2: Write code")
    assert result == "LLM response"
    mock_llm.ainvoke.assert_called_once()
