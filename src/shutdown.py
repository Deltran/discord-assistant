"""Graceful shutdown with state checkpointing."""

import asyncio
import logging
import signal
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class GracefulShutdown:
    def __init__(self):
        self._callbacks: list[Callable[[], Awaitable[None]]] = []
        self._shutting_down = False

    def register(self, callback: Callable[[], Awaitable[None]]):
        """Register a cleanup callback to run on shutdown."""
        self._callbacks.append(callback)

    def setup_signals(self, loop: asyncio.AbstractEventLoop):
        """Register signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.shutdown(s)))

    async def shutdown(self, sig=None):
        """Run all cleanup callbacks."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if sig:
            logger.info(f"Received signal {sig.name}, shutting down...")
        else:
            logger.info("Shutdown initiated...")

        for callback in reversed(self._callbacks):
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
