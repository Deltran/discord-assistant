"""Tests for the heartbeat runner."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.scheduler.heartbeat import HeartbeatRunner


@pytest.fixture
def monitoring():
    m = AsyncMock()
    m.post = AsyncMock()
    m.post_error = AsyncMock()
    return m


@pytest.fixture
def heartbeat_dir(tmp_path):
    md = tmp_path / "HEARTBEAT.md"
    md.write_text("# Heartbeat\nCheck everything.\n")
    return tmp_path


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content="HEARTBEAT_OK"))
    return llm


@pytest.mark.asyncio
async def test_heartbeat_ok_stays_silent(mock_llm, monitoring, heartbeat_dir):
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=heartbeat_dir
    )
    await runner.run()
    monitoring.post.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_surfaces_issues(mock_llm, monitoring, heartbeat_dir):
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(content="Vector store has 3 errors in the last hour.")
    )
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=heartbeat_dir
    )
    await runner.run()
    monitoring.post.assert_called_once()
    assert "Vector store" in monitoring.post.call_args[0][0]


@pytest.mark.asyncio
async def test_heartbeat_includes_recorded_errors(mock_llm, monitoring, heartbeat_dir):
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=heartbeat_dir
    )
    runner.record_error("Rate limit hit at 14:30")
    runner.record_error("Connection timeout at 14:35")

    await runner.run()

    # The prompt sent to the LLM should include the errors
    call_args = mock_llm.ainvoke.call_args[0][0]
    prompt_text = call_args[1].content  # HumanMessage
    assert "Rate limit hit" in prompt_text
    assert "Connection timeout" in prompt_text


@pytest.mark.asyncio
async def test_heartbeat_clears_errors_after_review(mock_llm, monitoring, heartbeat_dir):
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=heartbeat_dir
    )
    runner.record_error("Some error")
    await runner.run()
    assert len(runner._error_log) == 0


@pytest.mark.asyncio
async def test_heartbeat_skips_without_md(mock_llm, monitoring, tmp_path):
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=tmp_path
    )
    await runner.run()
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_handles_llm_failure(mock_llm, monitoring, heartbeat_dir):
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM down"))
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=heartbeat_dir
    )
    await runner.run()
    monitoring.post_error.assert_called_once()
    assert "could not reach LLM" in monitoring.post_error.call_args[0][0]


@pytest.mark.asyncio
async def test_error_log_caps_at_20(mock_llm, monitoring, heartbeat_dir):
    runner = HeartbeatRunner(
        llm=mock_llm, monitoring=monitoring, assistant_home=heartbeat_dir
    )
    for i in range(25):
        runner.record_error(f"Error {i}")
    assert len(runner._error_log) == 20
    assert runner._error_log[0] == "Error 5"
