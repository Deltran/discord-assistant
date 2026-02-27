"""Scheduled task definitions using APScheduler."""

import logging
from pathlib import Path
from typing import Callable, Awaitable

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class SchedulerManager:
    def __init__(self, config_path: str = "config/schedule.yaml"):
        self._scheduler = AsyncIOScheduler()
        self._config = self._load_config(config_path)
        self._jobs: dict[str, dict] = {}

    @staticmethod
    def _load_config(config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def register_job(self, name: str, func: Callable, trigger, **kwargs):
        """Register a scheduled job."""
        job = self._scheduler.add_job(func, trigger, name=name, **kwargs)
        self._jobs[name] = {"job": job, "name": name}

    def setup_default_jobs(
        self,
        *,
        briefing_fn: Callable | None = None,
        compaction_fn: Callable | None = None,
        review_fn: Callable | None = None,
        heartbeat_fn: Callable | None = None,
    ):
        """Register all default scheduled jobs from config."""
        briefing_cfg = self._config.get("briefing", {})
        briefing_time = briefing_cfg.get("time", "07:00")
        tz = briefing_cfg.get("timezone", "America/Chicago")
        hour, minute = briefing_time.split(":")

        if briefing_fn:
            self.register_job(
                "daily_briefing",
                briefing_fn,
                CronTrigger(hour=int(hour), minute=int(minute), timezone=tz),
            )

        compaction_hours = self._config.get("compaction_check_hours", 6)
        if compaction_fn:
            self.register_job(
                "memory_compaction",
                compaction_fn,
                IntervalTrigger(hours=compaction_hours),
            )

        review_day = self._config.get("memory_review_day", "monday")
        # APScheduler expects abbreviated day names (mon, tue, etc.)
        review_day = review_day[:3].lower()
        if review_fn:
            self.register_job(
                "memory_review",
                review_fn,
                CronTrigger(day_of_week=review_day, hour=3, minute=0),
            )

        heartbeat_minutes = self._config.get("heartbeat_minutes", 5)
        if heartbeat_fn:
            self.register_job(
                "health_check",
                heartbeat_fn,
                IntervalTrigger(minutes=heartbeat_minutes),
            )

    def start(self):
        self._scheduler.start()
        logger.info(f"Scheduler started with {len(self._jobs)} jobs")

    def stop(self):
        self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    def list_jobs(self) -> list[dict]:
        return [{"name": name} for name in self._jobs]
