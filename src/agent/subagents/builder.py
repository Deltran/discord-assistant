"""Builder sub-agent — plan-driven step-by-step implementation with plan executor."""

import logging
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.plan_executor import PlanExecutor
from src.agent.plan_parser import parse_plan
from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)

BUILDER_PROMPT = """You are a builder assistant. Execute implementation plans step by step.
For each step: announce what you're doing, execute it using available tools, verify the result.
Stop and report on failure rather than pushing through."""


async def run_builder(
    *,
    llm: ChatOpenAI,
    plan: str,
    tools: list | None = None,
    on_progress: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """Execute a build plan.

    If tools are provided, uses PlanExecutor for autonomous multi-step execution.
    Falls back to single-shot LLM call without tools.
    """
    if tools:
        # Parse the plan into structured form
        execution_plan = await parse_plan(llm=llm, plan_text=plan)

        # Execute with the plan executor
        executor = PlanExecutor(
            llm=llm,
            tools=tools,
            on_progress=on_progress,
        )
        return await executor.execute(execution_plan)
    else:
        messages = normalize_messages([
            SystemMessage(content=BUILDER_PROMPT),
            HumanMessage(
                content=f"Execute this plan:\n\n{plan}\n\nReport progress for each step."
            ),
        ])
        response = await llm.ainvoke(messages)
        return response.content
