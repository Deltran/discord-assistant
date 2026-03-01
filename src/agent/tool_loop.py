"""Generic async tool loop for agentic LLM execution."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


def _sanitize_minimax_response(content: str) -> tuple[str, bool]:
    """Strip MiniMax XML artifacts from response content.

    Returns:
        (cleaned_content, had_tool_calls) where had_tool_calls is True
        if <minimax:tool_call> blocks were found.
    """
    had_tool_calls = bool(
        re.search(r"<minimax:tool_call>.*?</minimax:tool_call>", content, re.DOTALL)
    )
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    cleaned = re.sub(
        r"<minimax:tool_call>.*?</minimax:tool_call>", "", cleaned, flags=re.DOTALL
    )
    return cleaned.strip(), had_tool_calls


async def run_tool_loop(
    *,
    llm: Any,
    messages: list[BaseMessage],
    tools: list,
    max_iterations: int = 10,
    on_tool_call: Callable[[str, dict], Awaitable[None]] | None = None,
) -> AIMessage:
    """Run an LLM-tool execution loop until the model produces a final text response.

    Args:
        llm: A LangChain chat model (already bound with tools via bind_tools).
        messages: The conversation history to send.
        tools: List of LangChain @tool functions for execution lookup.
        max_iterations: Safety cap on loop iterations.
        on_tool_call: Optional async callback(tool_name, tool_args) for progress reporting.

    Returns:
        The final AIMessage (with text content, no tool_calls).
    """
    tool_map = {t.name: t for t in tools}
    working_messages = list(messages)

    for iteration in range(max_iterations):
        response: AIMessage = await llm.ainvoke(working_messages)
        working_messages.append(response)

        if not response.tool_calls:
            content = response.content or ""
            cleaned, had_tool_calls = _sanitize_minimax_response(content)

            if had_tool_calls:
                # Replace the AI message with stripped version, ask LLM to retry
                working_messages[-1] = AIMessage(content=cleaned)
                tool_names = ", ".join(tool_map.keys())
                working_messages.append(HumanMessage(content=(
                    "You attempted to call a tool using XML syntax, but that"
                    " doesn't work. Use the function calling interface instead."
                    f" Available tools: {tool_names}"
                )))
                continue

            if cleaned != content:
                # Only think blocks were stripped — return cleaned content
                return AIMessage(content=cleaned)

            return response

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call.get("id", f"call_{iteration}_{tool_name}")

            if on_tool_call:
                try:
                    await on_tool_call(tool_name, tool_args)
                except Exception:
                    logger.exception("on_tool_call callback failed")

            if tool_name not in tool_map:
                available = ", ".join(tool_map.keys())
                result = f"Error: Unknown tool '{tool_name}'. Available: {available}"
                logger.warning("LLM requested unknown tool: %s", tool_name)
            else:
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
                    logger.exception("Tool %s failed", tool_name)

            working_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call_id)
            )

    logger.warning("Tool loop hit max iterations (%d), returning partial", max_iterations)
    final = await llm.ainvoke(working_messages)
    return final
