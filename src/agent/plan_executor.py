"""Plan executor — runs structured plans step-by-step with progress reporting."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.plan_parser import ExecutionPlan, PlanPhase, PlanStep
from src.agent.tool_loop import run_tool_loop
from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)

STEP_PROMPT = """You are executing a step in an implementation plan.
Complete the task described below using the available tools.
Be thorough but concise. Report what you did and the outcome."""


class PlanExecutor:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        tools: list,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        max_step_iterations: int = 15,
    ):
        self.llm = llm
        self.tools = tools
        self.on_progress = on_progress
        self.max_step_iterations = max_step_iterations

    async def _report(self, message: str):
        """Send a progress report."""
        if self.on_progress:
            try:
                await self.on_progress(message)
            except Exception:
                logger.exception("Progress report failed")

    async def execute(self, plan: ExecutionPlan) -> str:
        """Execute a full plan, returning a summary of results."""
        await self._report(
            f"**Starting plan:** {plan.title} "
            f"({plan.total_steps} steps across {len(plan.phases)} phases)"
        )

        results = []

        for i, phase in enumerate(plan.phases, 1):
            phase.status = "running"
            await self._report(f"**Phase {i}/{len(plan.phases)}:** {phase.name}")

            if phase.parallel:
                phase_result = await self._execute_parallel_phase(phase)
            else:
                phase_result = await self._execute_sequential_phase(phase)

            if phase_result["failed"]:
                phase.status = "failed"
                results.append(f"**{phase.name}**: FAILED — {phase_result['summary']}")
                await self._report(f"Phase '{phase.name}' failed. Stopping execution.")
                break
            else:
                phase.status = "completed"
                results.append(f"**{phase.name}**: {phase_result['summary']}")

        completed = plan.completed_steps
        total = plan.total_steps
        summary = (
            f"**Plan complete:** {plan.title}\n"
            f"**Progress:** {completed}/{total} steps completed\n\n"
            + "\n".join(results)
        )

        await self._report(summary)
        return summary

    async def _execute_sequential_phase(self, phase: PlanPhase) -> dict:
        """Execute phase steps one at a time."""
        summaries = []
        failed = False

        for j, step in enumerate(phase.steps, 1):
            step.status = "running"
            await self._report(f"  Step {j}/{len(phase.steps)}: {step.description[:100]}")

            result = await self._execute_step(step)
            step.result = result

            if step.status == "failed":
                summaries.append(f"Step {j} FAILED: {result[:200]}")
                failed = True
                break
            else:
                summaries.append(f"Step {j} done")

        return {"failed": failed, "summary": "; ".join(summaries)}

    async def _execute_parallel_phase(self, phase: PlanPhase) -> dict:
        """Execute phase steps concurrently."""
        tasks = [self._execute_step(step) for step in phase.steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries = []
        failed = False
        for j, (step, result) in enumerate(zip(phase.steps, results, strict=True), 1):
            if isinstance(result, Exception):
                step.status = "failed"
                step.result = str(result)
                summaries.append(f"Step {j} FAILED: {result}")
                failed = True
            else:
                step.result = result
                summaries.append(f"Step {j} done")

        return {"failed": failed, "summary": "; ".join(summaries)}

    async def _execute_step(self, step: PlanStep) -> str:
        """Execute a single plan step using the tool loop."""
        step.status = "running"

        prompt = step.description
        if step.validation:
            prompt += f"\n\nAfter completing the task, verify: {step.validation}"

        messages = normalize_messages([
            SystemMessage(content=STEP_PROMPT),
            HumanMessage(content=prompt),
        ])

        try:
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await run_tool_loop(
                llm=llm_with_tools,
                messages=messages,
                tools=self.tools,
                max_iterations=self.max_step_iterations,
            )
            step.status = "completed"
            return response.content
        except Exception as e:
            step.status = "failed"
            logger.exception("Step execution failed: %s", step.description[:100])
            return f"Error: {e}"
