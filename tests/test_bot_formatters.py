"""Tests for Discord output formatting."""

from src.bot.formatters import split_message, format_code_block


def test_split_message_short():
    result = split_message("Hello world")
    assert result == ["Hello world"]


def test_split_message_long():
    long_text = "a" * 2500
    result = split_message(long_text, max_length=2000)
    assert len(result) == 2
    assert len(result[0]) <= 2000
    assert len(result[1]) <= 2000
    assert "".join(result) == long_text


def test_split_message_respects_newlines():
    text = ("line\n" * 399) + ("x" * 500)  # ~2495 chars
    result = split_message(text, max_length=2000)
    assert all(len(chunk) <= 2000 for chunk in result)


def test_format_code_block():
    result = format_code_block("print('hello')", language="python")
    assert result == "```python\nprint('hello')\n```"


def test_format_code_block_no_language():
    result = format_code_block("some output")
    assert result == "```\nsome output\n```"
