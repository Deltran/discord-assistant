"""System sub-agent — shell commands and file operations with tool loop."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.tool_loop import run_tool_loop
from src.providers.minimax import normalize_messages

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a system administration assistant. Execute commands safely.
Use the available tools to complete the task, then report results clearly."""


async def run_system_task(*, llm: ChatOpenAI, task: str, tools: list | None = None) -> str:
    """Run a system task (shell command, file operation)."""
    messages = normalize_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Task: {task}\n\n"
                "Use the available tools to complete this task."
            )
        ),
    ])

    if tools:
        llm_with_tools = llm.bind_tools(tools)
        response = await run_tool_loop(llm=llm_with_tools, messages=messages, tools=tools)
        return response.content
    else:
        response = await llm.ainvoke(messages)
        return response.content
