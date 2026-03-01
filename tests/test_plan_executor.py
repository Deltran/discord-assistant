"""Tests for plan parsing and execution."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.agent.plan_executor import PlanExecutor
from src.agent.plan_parser import ExecutionPlan, PlanPhase, PlanStep, parse_plan


@tool
async def mock_tool(action: str) -> str:
    """A mock tool for testing.

    Args:
        action: What action to take.
    """
    return f"done: {action}"


# --- Plan Parser Tests ---


@pytest.mark.asyncio
async def test_parse_plan_valid_json():
    """LLM returns valid structured JSON."""
    plan_json = json.dumps({
        "title": "Test Plan",
        "phases": [
            {
                "name": "Setup",
                "parallel": False,
                "steps": [
                    {
                        "description": "Create directory",
                        "validation": "dir exists",
                        "tools_needed": ["shell_exec"],
                    },
                    {
                        "description": "Write config",
                        "validation": None,
                        "tools_needed": ["file_write"],
                    },
                ],
            },
            {
                "name": "Build",
                "parallel": True,
                "steps": [
                    {
                        "description": "Compile module A",
                        "validation": "no errors",
                        "tools_needed": ["shell_exec"],
                    },
                ],
            },
        ],
    })

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=plan_json))

    plan = await parse_plan(llm=llm, plan_text="some markdown plan")

    assert plan.title == "Test Plan"
    assert len(plan.phases) == 2
    assert plan.phases[0].name == "Setup"
    assert len(plan.phases[0].steps) == 2
    assert plan.phases[0].steps[0].description == "Create directory"
    assert plan.phases[0].steps[0].validation == "dir exists"
    assert plan.phases[1].parallel is True
    assert plan.total_steps == 3
    assert plan.completed_steps == 0


@pytest.mark.asyncio
async def test_parse_plan_with_code_fences():
    """LLM wraps JSON in markdown code fences."""
    plan_json = json.dumps({
        "title": "Fenced Plan",
        "phases": [{"name": "Phase 1", "steps": [{"description": "Do it"}]}],
    })

    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=f"```json\n{plan_json}\n```"))

    plan = await parse_plan(llm=llm, plan_text="plan")
    assert plan.title == "Fenced Plan"


@pytest.mark.asyncio
async def test_parse_plan_invalid_json_fallback():
    """Invalid JSON falls back to single-step plan."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="This is not JSON at all"))

    plan = await parse_plan(llm=llm, plan_text="some plan text")
    assert plan.title == "Unparsed Plan"
    assert len(plan.phases) == 1
    assert len(plan.phases[0].steps) == 1


# --- Plan Executor Tests ---


@pytest.mark.asyncio
async def test_executor_sequential():
    """Sequential phase executes steps in order."""
    plan = ExecutionPlan(
        title="Test",
        phases=[PlanPhase(
            name="Phase 1",
            steps=[
                PlanStep(description="Step 1"),
                PlanStep(description="Step 2"),
            ],
        )],
    )

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Step completed"))

    executor = PlanExecutor(llm=llm, tools=[mock_tool])
    result = await executor.execute(plan)

    assert "Plan complete" in result
    assert plan.completed_steps == 2


@pytest.mark.asyncio
async def test_executor_failure_stops():
    """Execution stops when a step fails."""
    plan = ExecutionPlan(
        title="Failing Plan",
        phases=[PlanPhase(
            name="Phase 1",
            steps=[
                PlanStep(description="Step 1"),
                PlanStep(description="Step 2 will fail"),
                PlanStep(description="Step 3 should not run"),
            ],
        )],
    )

    call_count = 0

    async def failing_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Something broke")
        return AIMessage(content="OK")

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(side_effect=failing_ainvoke)

    executor = PlanExecutor(llm=llm, tools=[mock_tool])
    result = await executor.execute(plan)

    assert "FAILED" in result
    assert plan.phases[0].steps[2].status == "pending"  # never reached


@pytest.mark.asyncio
async def test_executor_parallel_phase():
    """Parallel phase executes all steps concurrently."""
    plan = ExecutionPlan(
        title="Parallel Test",
        phases=[PlanPhase(
            name="Parallel Phase",
            parallel=True,
            steps=[
                PlanStep(description="Task A"),
                PlanStep(description="Task B"),
            ],
        )],
    )

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Done"))

    executor = PlanExecutor(llm=llm, tools=[mock_tool])
    result = await executor.execute(plan)

    assert plan.completed_steps == 2
    assert "Plan complete" in result


@pytest.mark.asyncio
async def test_executor_progress_callback():
    """Progress callback fires during execution."""
    progress_messages = []

    async def on_progress(msg):
        progress_messages.append(msg)

    plan = ExecutionPlan(
        title="Progress Test",
        phases=[PlanPhase(
            name="Phase",
            steps=[PlanStep(description="Do thing")],
        )],
    )

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Done"))

    executor = PlanExecutor(llm=llm, tools=[mock_tool], on_progress=on_progress)
    await executor.execute(plan)

    assert len(progress_messages) >= 3  # start, phase, step, summary
    assert any("Starting plan" in m for m in progress_messages)
    assert any("Plan complete" in m for m in progress_messages)


@pytest.mark.asyncio
async def test_executor_with_validation():
    """Steps with validation include it in the prompt."""
    plan = ExecutionPlan(
        title="Validation Test",
        phases=[PlanPhase(
            name="Phase",
            steps=[PlanStep(
                description="Create file",
                validation="File exists at /tmp/test.txt",
            )],
        )],
    )

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Created and verified"))

    executor = PlanExecutor(llm=llm, tools=[mock_tool])
    await executor.execute(plan)

    # Check the prompt included validation
    call_args = llm.ainvoke.call_args_list[0][0][0]
    prompt_text = " ".join(m.content for m in call_args)
    assert "verify" in prompt_text.lower()
