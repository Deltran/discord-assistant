"""Sub-agent base infrastructure -- spawning, concurrency, depth control."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class SubAgentTask:
    name: str
    rejected: bool = False
    _task: asyncio.Task | None = field(default=None, repr=False)


class SubAgentManager:
    def __init__(self, *, max_concurrent: int = 5, max_depth: int = 2):
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active: list[SubAgentTask] = []
        self._active_count = 0

    @property
    def active_count(self) -> int:
        return self._active_count

    async def submit(
        self,
        *,
        name: str,
        work_fn: Callable[[dict], Awaitable[Any]],
        callback: Callable[[Any], Awaitable[None]],
        depth: int = 0,
    ) -> SubAgentTask | None:
        """Submit a sub-agent task for background execution."""
        if depth > self.max_depth:
            logger.warning(
                "Sub-agent '%s' rejected: depth %d exceeds max %d",
                name, depth, self.max_depth,
            )
            return SubAgentTask(name=name, rejected=True)

        agent_task = SubAgentTask(name=name)

        async def _run():
            async with self._semaphore:
                self._active_count += 1
                try:
                    result = await work_fn({"depth": depth})
                    await callback(result)
                except Exception as e:
                    logger.error("Sub-agent '%s' failed: %s", name, e)
                    await callback(f"Error in sub-agent '{name}': {e}")
                finally:
                    self._active_count -= 1

        agent_task._task = asyncio.create_task(_run())
        self._active.append(agent_task)
        return agent_task
