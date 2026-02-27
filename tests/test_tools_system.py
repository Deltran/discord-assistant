"""Tests for system tools."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.shell import shell_exec, is_destructive_command
from src.tools.files import file_read, file_write


def test_is_destructive_rm():
    assert is_destructive_command("rm -rf /tmp/test") is True


def test_is_destructive_safe():
    assert is_destructive_command("ls -la") is False


def test_is_destructive_dd():
    assert is_destructive_command("dd if=/dev/zero of=/dev/sda") is True


def test_is_destructive_chmod():
    assert is_destructive_command("chmod 777 /etc/passwd") is True


@pytest.mark.asyncio
async def test_file_read(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World")
    result = await file_read.ainvoke({"path": str(test_file)})
    assert "Hello World" in result


@pytest.mark.asyncio
async def test_file_read_missing(tmp_path):
    result = await file_read.ainvoke({"path": str(tmp_path / "missing.txt")})
    assert "not found" in result.lower() or "error" in result.lower()


@pytest.mark.asyncio
async def test_file_write(tmp_path):
    test_file = tmp_path / "output.txt"
    result = await file_write.ainvoke({"path": str(test_file), "content": "Test content"})
    assert test_file.read_text() == "Test content"
    assert "success" in result.lower() or "wrote" in result.lower()


@pytest.mark.asyncio
async def test_shell_exec_safe_command():
    result = await shell_exec.ainvoke({"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_shell_exec_destructive_blocked():
    result = await shell_exec.ainvoke({"command": "rm -rf /tmp/test"})
    assert "destructive" in result.lower() or "confirmation" in result.lower()
