"""Tests for skill authoring tool."""


import pytest
import yaml

from src.skills.registry import SkillRegistry
from src.tools.skill_author import create_skill_author_tool


@pytest.fixture
def skills_dir(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    return d


@pytest.fixture
def registry():
    return SkillRegistry()


@pytest.fixture
def create_skill(skills_dir, registry):
    return create_skill_author_tool(skills_dir=skills_dir, registry=registry)


@pytest.mark.asyncio
async def test_create_skill_writes_files(create_skill, skills_dir):
    """create_skill writes manifest.yaml and tool.py."""
    result = await create_skill.ainvoke({
        "name": "weather-check",
        "description": "Check the weather",
        "trigger": "weather questions",
        "code": (
            'async def run(input_text: str) -> str:\n'
            '    return f"Weather for {input_text}: sunny"\n'
        ),
        "permissions": "http_request",
    })

    assert "created" in result
    skill_dir = skills_dir / "weather-check"
    assert skill_dir.exists()
    assert (skill_dir / "manifest.yaml").exists()
    assert (skill_dir / "tool.py").exists()

    manifest = yaml.safe_load((skill_dir / "manifest.yaml").read_text())
    assert manifest["name"] == "weather-check"
    assert manifest["trusted"] is False
    assert manifest["author"] == "agent"
    assert "http_request" in manifest["permissions"]


@pytest.mark.asyncio
async def test_create_skill_reloads_registry(create_skill, registry):
    """Created skill is immediately available in the registry."""
    await create_skill.ainvoke({
        "name": "hello-skill",
        "description": "Says hello",
        "trigger": "hello",
        "code": 'async def run(input_text: str) -> str:\n    return f"Hello {input_text}"\n',
        "permissions": "",
    })

    assert registry.get("hello-skill") is not None


@pytest.mark.asyncio
async def test_create_skill_rejects_spaces_in_name(create_skill):
    """Skill names with spaces are rejected."""
    result = await create_skill.ainvoke({
        "name": "bad name",
        "description": "Bad",
        "trigger": "bad",
        "code": 'async def run(input_text: str) -> str:\n    return "bad"\n',
        "permissions": "",
    })

    assert "Error" in result


@pytest.mark.asyncio
async def test_create_skill_rejects_no_run_function(create_skill):
    """Skill code without run() function is rejected."""
    result = await create_skill.ainvoke({
        "name": "no-run",
        "description": "No run",
        "trigger": "no run",
        "code": "x = 42\n",
        "permissions": "",
    })

    assert "Error" in result


@pytest.mark.asyncio
async def test_create_skill_rejects_duplicate(create_skill, skills_dir):
    """Creating a skill with an existing name is rejected."""
    # First creation succeeds
    await create_skill.ainvoke({
        "name": "dupe",
        "description": "First",
        "trigger": "first",
        "code": 'async def run(input_text: str) -> str:\n    return "first"\n',
        "permissions": "",
    })

    # Second creation fails
    result = await create_skill.ainvoke({
        "name": "dupe",
        "description": "Second",
        "trigger": "second",
        "code": 'async def run(input_text: str) -> str:\n    return "second"\n',
        "permissions": "",
    })

    assert "already exists" in result


@pytest.mark.asyncio
async def test_create_skill_empty_permissions(create_skill, skills_dir):
    """Skill with empty permissions creates valid manifest."""
    await create_skill.ainvoke({
        "name": "simple",
        "description": "Simple",
        "trigger": "simple",
        "code": 'async def run(input_text: str) -> str:\n    return "simple"\n',
        "permissions": "",
    })

    manifest = yaml.safe_load((skills_dir / "simple" / "manifest.yaml").read_text())
    assert manifest["permissions"] == []
