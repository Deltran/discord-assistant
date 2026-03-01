"""Heartbeat runner — periodic self-check that surfaces issues proactively."""

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.monitoring import MonitoringChannel

logger = logging.getLogger(__name__)

HEARTBEAT_OK = "HEARTBEAT_OK"


def _load_heartbeat_md(assistant_home: Path) -> str:
    """Load HEARTBEAT.md from the assistant home directory."""
    path = assistant_home / "HEARTBEAT.md"
    if path.exists():
        return path.read_text()
    return ""


class HeartbeatRunner:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        monitoring: MonitoringChannel,
        assistant_home: Path,
    ):
        self._llm = llm
        self._monitoring = monitoring
        self._assistant_home = assistant_home
        self._error_log: list[str] = []

    def record_error(self, error: str) -> None:
        """Record an error for the next heartbeat to review."""
        self._error_log.append(error)
        # Keep only the last 20 errors to avoid unbounded growth
        if len(self._error_log) > 20:
            self._error_log = self._error_log[-20:]

    async def run(self) -> None:
        """Execute one heartbeat cycle."""
        checklist = _load_heartbeat_md(self._assistant_home)
        if not checklist:
            logger.debug("No HEARTBEAT.md found, skipping")
            return

        # Build context with any recent errors
        context_parts = [checklist]
        if self._error_log:
            error_summary = "\n".join(f"- {e}" for e in self._error_log)
            context_parts.append(
                f"\n## Recent Errors (since last heartbeat)\n{error_summary}"
            )

        prompt = "\n".join(context_parts)

        try:
            response = await self._llm.ainvoke([
                SystemMessage(content="You are an operational monitor for a Discord AI assistant. Be concise."),
                HumanMessage(content=prompt),
            ])
            reply = response.content.strip()
        except Exception:
            logger.exception("Heartbeat LLM call failed")
            await self._monitoring.post_error("Heartbeat failed — could not reach LLM")
            return

        # Clear error log after successful review
        self._error_log.clear()

        if HEARTBEAT_OK in reply:
            logger.info("Heartbeat: all clear")
            return

        # The LLM had something to report — post it
        logger.info("Heartbeat surfaced: %s", reply[:100])
        await self._monitoring.post(f"\U0001f4ac **Heartbeat check-in:**\n{reply}")
