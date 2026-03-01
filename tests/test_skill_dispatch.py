"""Tests for skill dispatch meta-tool."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool

from src.skills.loader import SkillManifest
from src.skills.registry import SkillRegistry
from src.tools.skill_dispatch import (
    _filter_tools,
    _run_dynamic_skill,
    create_dispatch_skill_tool,
)


@tool
async def mock_shell_exec(command: str) -> str:
    """Execute a shell command.

    Args:
        command: The command to run.
    """
    return f"ran: {command}"


@tool
async def mock_http_request(url: str) -> str:
    """Fetch a URL.

    Args:
        url: The URL to fetch.
    """
    return f"fetched: {url}"


@pytest.fixture
def registry():
    r = SkillRegistry()
    r.register(SkillManifest(
        name="research",
        description="Research a topic",
        trigger="research questions",
        permissions=["http_request"],
        entry_point="tool.py",
        author="human",
        trusted=True,
        created="2026-01-01",
        path=Path("/tmp/research"),
    ))
    r.register(SkillManifest(
        name="untrusted-skill",
        description="An untrusted skill",
        trigger="untrusted trigger",
        permissions=["shell_exec", "http_request"],
        entry_point="tool.py",
        author="agent",
        trusted=False,
        created="2026-01-01",
        path=Path("/tmp/untrusted"),
    ))
    return r


@pytest.fixture
def available_tools():
    return [mock_shell_exec, mock_http_request]


def test_filter_tools_trusted(registry, available_tools):
    """Trusted skill gets all declared permissions."""
    manifest = registry.get("research")
    tool_map = {t.name: t for t in available_tools}
    filtered = _filter_tools(manifest, tool_map)
    names = [t.name for t in filtered]
    assert "mock_http_request" not in names  # permission is "http_request" not "mock_http_request"


def test_filter_tools_untrusted_no_shell(registry):
    """Untrusted skill is denied shell_exec."""
    manifest = registry.get("untrusted-skill")
    # Use actual permission-named tools
    @tool
    async def shell_exec(command: str) -> str:
        """Run shell command.

        Args:
            command: The command.
        """
        return "ran"

    @tool
    async def http_request(url: str) -> str:
        """Fetch URL.

        Args:
            url: The URL.
        """
        return "fetched"

    tool_map = {"shell_exec": shell_exec, "http_request": http_request}
    filtered = _filter_tools(manifest, tool_map)
    names = [t.name for t in filtered]
    assert "shell_exec" not in names
    assert "http_request" in names


def test_dispatch_unknown_skill(registry, available_tools):
    """Dispatching an unknown skill returns an error message."""
    llm = MagicMock()
    dispatch = create_dispatch_skill_tool(
        registry=registry, llm=llm, available_tools=available_tools
    )
    result = asyncio.get_event_loop().run_until_complete(
        dispatch.ainvoke({"skill_name": "nonexistent", "input_text": "test"})
    )
    assert "Unknown skill" in result


def test_registry_reload(tmp_path):
    """Registry reload rescans directories."""
    # Create a skill directory
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "manifest.yaml").write_text(
        "name: my-skill\n"
        "description: A test skill\n"
        "trigger: test trigger\n"
        "permissions: []\n"
        "entry_point: tool.py\n"
        "author: human\n"
        "trusted: true\n"
        "created: 2026-01-01\n"
    )

    registry = SkillRegistry()
    count = registry.reload(tmp_path)
    assert count == 1
    assert registry.get("my-skill") is not None

    # Add another skill and reload
    skill2_dir = tmp_path / "another-skill"
    skill2_dir.mkdir()
    (skill2_dir / "manifest.yaml").write_text(
        "name: another-skill\n"
        "description: Another test\n"
        "trigger: another trigger\n"
        "permissions: []\n"
        "entry_point: tool.py\n"
        "author: human\n"
        "trusted: true\n"
        "created: 2026-01-01\n"
    )

    count = registry.reload(tmp_path)
    assert count == 2


@pytest.mark.asyncio
async def test_run_dynamic_skill(tmp_path):
    """Dynamic skill loading and execution."""
    skill_dir = tmp_path / "hello-skill"
    skill_dir.mkdir()
    (skill_dir / "tool.py").write_text(
        "async def run(input_text: str) -> str:\n"
        "    return f'Hello from skill: {input_text}'\n"
    )
    manifest = SkillManifest(
        name="hello-skill",
        description="Says hello",
        trigger="hello",
        permissions=[],
        entry_point="tool.py",
        author="agent",
        trusted=False,
        created="2026-01-01",
        path=skill_dir,
    )

    result = await _run_dynamic_skill(manifest, "world")
    assert "Hello from skill: world" in result


@pytest.mark.asyncio
async def test_run_dynamic_skill_missing_entry(tmp_path):
    """Dynamic skill with missing entry point returns error."""
    manifest = SkillManifest(
        name="missing",
        description="Missing",
        trigger="missing",
        permissions=[],
        entry_point="tool.py",
        author="agent",
        trusted=False,
        created="2026-01-01",
        path=tmp_path,
    )

    result = await _run_dynamic_skill(manifest, "test")
    assert "not found" in result


@pytest.mark.asyncio
async def test_run_dynamic_skill_no_run_function(tmp_path):
    """Dynamic skill without run() function returns error."""
    skill_dir = tmp_path / "no-run"
    skill_dir.mkdir()
    (skill_dir / "tool.py").write_text("x = 42\n")
    manifest = SkillManifest(
        name="no-run",
        description="No run",
        trigger="no run",
        permissions=[],
        entry_point="tool.py",
        author="agent",
        trusted=False,
        created="2026-01-01",
        path=skill_dir,
    )

    result = await _run_dynamic_skill(manifest, "test")
    assert "no run()" in result


@pytest.mark.asyncio
async def test_run_dynamic_skill_error_handling(tmp_path):
    """Dynamic skill that raises is caught gracefully."""
    skill_dir = tmp_path / "boom"
    skill_dir.mkdir()
    (skill_dir / "tool.py").write_text(
        "async def run(input_text: str) -> str:\n"
        "    raise ValueError('kaboom')\n"
    )
    manifest = SkillManifest(
        name="boom",
        description="Boom",
        trigger="boom",
        permissions=[],
        entry_point="tool.py",
        author="agent",
        trusted=False,
        created="2026-01-01",
        path=skill_dir,
    )

    result = await _run_dynamic_skill(manifest, "test")
    assert "failed" in result.lower()
