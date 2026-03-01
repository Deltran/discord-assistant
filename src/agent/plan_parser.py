"""Plan parser — decompose markdown plans into structured execution plans."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)

PARSER_PROMPT = """\
You are a plan parser. Given a markdown plan, extract it into structured JSON.

Output ONLY valid JSON with this schema:
{
  "title": "Plan title",
  "phases": [
    {
      "name": "Phase name",
      "parallel": false,
      "steps": [
        {
          "description": "What to do",
          "validation": "How to verify success (optional, null if none)",
          "tools_needed": ["shell_exec", "file_write"]
        }
      ]
    }
  ]
}

Rules:
- Each phase is a logical group of related steps
- Set parallel=true only if the phase's steps can run independently
- Phases themselves always run sequentially
- tools_needed is your best guess at which tools each step needs
- Keep step descriptions actionable and specific"""


@dataclass
class PlanStep:
    description: str
    validation: str | None = None
    tools_needed: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: str | None = None


@dataclass
class PlanPhase:
    name: str
    steps: list[PlanStep] = field(default_factory=list)
    parallel: bool = False
    status: str = "pending"


@dataclass
class ExecutionPlan:
    title: str
    phases: list[PlanPhase] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return sum(len(p.steps) for p in self.phases)

    @property
    def completed_steps(self) -> int:
        return sum(
            1 for p in self.phases for s in p.steps if s.status == "completed"
        )


async def parse_plan(*, llm: ChatOpenAI, plan_text: str) -> ExecutionPlan:
    """Parse a markdown plan into a structured ExecutionPlan using the LLM."""
    messages = normalize_messages([
        SystemMessage(content=PARSER_PROMPT),
        HumanMessage(content=f"Parse this plan:\n\n{plan_text}"),
    ])

    response = await llm.ainvoke(messages)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]  # remove opening ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse plan JSON: %s\nContent: %s", e, content[:500])
        # Fallback: single phase with one step
        return ExecutionPlan(
            title="Unparsed Plan",
            phases=[PlanPhase(
                name="Execution",
                steps=[PlanStep(description=plan_text[:500])],
            )],
        )

    phases = []
    for phase_data in data.get("phases", []):
        steps = []
        for step_data in phase_data.get("steps", []):
            steps.append(PlanStep(
                description=step_data.get("description", ""),
                validation=step_data.get("validation"),
                tools_needed=step_data.get("tools_needed", []),
            ))
        phases.append(PlanPhase(
            name=phase_data.get("name", "Unnamed"),
            steps=steps,
            parallel=phase_data.get("parallel", False),
        ))

    return ExecutionPlan(
        title=data.get("title", "Plan"),
        phases=phases,
    )
