"""Tests for the agentic tool loop."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.agent.tool_loop import run_tool_loop


@tool
async def add_numbers(a: int, b: int) -> str:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return str(a + b)


@tool
async def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: The person's name.
    """
    return f"Hello, {name}!"


@pytest.fixture
def tools():
    return [add_numbers, greet]


def _make_ai_with_tool_calls(tool_calls):
    """Create an AIMessage with tool_calls."""
    msg = AIMessage(content="", tool_calls=tool_calls)
    return msg


def _make_final_ai(content):
    """Create a final AIMessage with no tool_calls."""
    return AIMessage(content=content)


@pytest.mark.asyncio
async def test_no_tool_calls_returns_immediately(tools):
    """LLM returns text with no tool_calls — loop returns immediately."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=_make_final_ai("Just text"))

    messages = [SystemMessage(content="sys"), HumanMessage(content="hi")]
    result = await run_tool_loop(llm=llm, messages=messages, tools=tools)

    assert result.content == "Just text"
    assert llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_tool_call_executes_and_loops(tools):
    """LLM requests a tool call, gets result, then returns final text."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_ai_with_tool_calls([{
            "name": "add_numbers",
            "args": {"a": 2, "b": 3},
            "id": "call_1",
        }]),
        _make_final_ai("The sum is 5"),
    ])

    messages = [HumanMessage(content="add 2 and 3")]
    result = await run_tool_loop(llm=llm, messages=messages, tools=tools)

    assert result.content == "The sum is 5"
    assert llm.ainvoke.call_count == 2

    # Verify the tool result was passed back
    second_call_messages = llm.ainvoke.call_args_list[1][0][0]
    tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "5"


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_response(tools):
    """LLM requests multiple tool calls in a single response."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_ai_with_tool_calls([
            {"name": "add_numbers", "args": {"a": 1, "b": 2}, "id": "call_1"},
            {"name": "greet", "args": {"name": "Alice"}, "id": "call_2"},
        ]),
        _make_final_ai("Done: 3 and Hello, Alice!"),
    ])

    messages = [HumanMessage(content="do both")]
    result = await run_tool_loop(llm=llm, messages=messages, tools=tools)

    assert result.content == "Done: 3 and Hello, Alice!"
    second_call_messages = llm.ainvoke.call_args_list[1][0][0]
    tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2


@pytest.mark.asyncio
async def test_unknown_tool_returns_error(tools):
    """LLM requests a tool that doesn't exist — error message returned."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_ai_with_tool_calls([{
            "name": "nonexistent_tool",
            "args": {},
            "id": "call_1",
        }]),
        _make_final_ai("Sorry, that tool doesn't exist"),
    ])

    messages = [HumanMessage(content="use fake tool")]
    await run_tool_loop(llm=llm, messages=messages, tools=tools)

    second_call_messages = llm.ainvoke.call_args_list[1][0][0]
    tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "Unknown tool" in tool_msgs[0].content


@pytest.mark.asyncio
async def test_max_iterations_cap(tools):
    """Loop terminates at max_iterations and returns partial."""
    llm = MagicMock()
    # Always return tool calls — never a final text
    llm.ainvoke = AsyncMock(return_value=_make_ai_with_tool_calls([{
        "name": "add_numbers",
        "args": {"a": 1, "b": 1},
        "id": "call_loop",
    }]))

    messages = [HumanMessage(content="loop forever")]
    await run_tool_loop(
        llm=llm, messages=messages, tools=tools, max_iterations=3
    )

    # 3 iterations + 1 final call
    assert llm.ainvoke.call_count == 4


@pytest.mark.asyncio
async def test_on_tool_call_callback(tools):
    """on_tool_call callback fires for each tool call."""
    calls_received = []

    async def callback(name, args):
        calls_received.append((name, args))

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_ai_with_tool_calls([{
            "name": "greet",
            "args": {"name": "Bob"},
            "id": "call_1",
        }]),
        _make_final_ai("Done"),
    ])

    messages = [HumanMessage(content="greet bob")]
    await run_tool_loop(
        llm=llm, messages=messages, tools=tools, on_tool_call=callback
    )

    assert len(calls_received) == 1
    assert calls_received[0] == ("greet", {"name": "Bob"})


@pytest.mark.asyncio
async def test_tool_execution_error_caught(tools):
    """Tool that raises an exception returns error text, doesn't crash."""
    @tool
    async def failing_tool(x: str) -> str:
        """A tool that always fails.

        Args:
            x: Input string.
        """
        raise RuntimeError("kaboom")

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_ai_with_tool_calls([{
            "name": "failing_tool",
            "args": {"x": "test"},
            "id": "call_1",
        }]),
        _make_final_ai("Tool failed, sorry"),
    ])

    messages = [HumanMessage(content="fail")]
    result = await run_tool_loop(
        llm=llm, messages=messages, tools=[failing_tool]
    )

    assert result.content == "Tool failed, sorry"
    second_call_messages = llm.ainvoke.call_args_list[1][0][0]
    tool_msgs = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert "Error executing failing_tool" in tool_msgs[0].content


@pytest.mark.asyncio
async def test_minimax_xml_tool_call_triggers_retry(tools):
    """LLM returns XML tool call text — loop retries with correction message."""
    xml_response = (
        "<think>Let me fetch that</think>"
        '<minimax:tool_call><invoke name="fetch_fetch">'
        '<parameter name="url">https://example.com</parameter>'
        "</invoke></minimax:tool_call>"
    )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_final_ai(xml_response),
        _make_final_ai("Here is the actual answer"),
    ])

    messages = [HumanMessage(content="fetch example.com")]
    result = await run_tool_loop(llm=llm, messages=messages, tools=tools)

    assert result.content == "Here is the actual answer"
    assert llm.ainvoke.call_count == 2

    # Verify the retry message mentions function calling
    second_call_messages = llm.ainvoke.call_args_list[1][0][0]
    human_msgs = [m for m in second_call_messages if isinstance(m, HumanMessage)]
    retry_msg = human_msgs[-1].content
    assert "function calling interface" in retry_msg


@pytest.mark.asyncio
async def test_minimax_think_blocks_stripped_from_final(tools):
    """LLM returns think blocks with real content — think blocks stripped."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=_make_final_ai(
        "<think>Let me think about this</think>Here is the actual answer"
    ))

    messages = [HumanMessage(content="answer me")]
    result = await run_tool_loop(llm=llm, messages=messages, tools=tools)

    assert result.content == "Here is the actual answer"
    assert llm.ainvoke.call_count == 1


@pytest.mark.asyncio
async def test_minimax_xml_tool_call_with_real_tool_available(tools):
    """LLM returns XML tool call, retries with proper function calling, then final."""
    xml_response = (
        '<minimax:tool_call><invoke name="add_numbers">'
        '<parameter name="a">2</parameter>'
        '<parameter name="b">3</parameter>'
        "</invoke></minimax:tool_call>"
    )

    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=[
        _make_final_ai(xml_response),
        _make_ai_with_tool_calls([{
            "name": "add_numbers",
            "args": {"a": 2, "b": 3},
            "id": "call_1",
        }]),
        _make_final_ai("The sum is 5"),
    ])

    messages = [HumanMessage(content="add 2 and 3")]
    result = await run_tool_loop(llm=llm, messages=messages, tools=tools)

    assert result.content == "The sum is 5"
    assert llm.ainvoke.call_count == 3
