"""Tests for scheduler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.scheduler.jobs import SchedulerManager


def test_scheduler_loads_config():
    manager = SchedulerManager(config_path="config/schedule.yaml")
    assert manager._config.get("briefing", {}).get("time") == "07:00"


def test_scheduler_setup_default_jobs():
    manager = SchedulerManager(config_path="config/schedule.yaml")
    manager.setup_default_jobs(
        briefing_fn=AsyncMock(),
        compaction_fn=AsyncMock(),
        review_fn=AsyncMock(),
        heartbeat_fn=AsyncMock(),
    )
    jobs = manager.list_jobs()
    names = [j["name"] for j in jobs]
    assert "daily_briefing" in names
    assert "health_check" in names
    assert "memory_compaction" in names
    assert "memory_review" in names


def test_scheduler_no_config():
    manager = SchedulerManager(config_path="/nonexistent/path.yaml")
    assert manager._config == {}
