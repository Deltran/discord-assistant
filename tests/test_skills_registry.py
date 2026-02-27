"""Tests for skill registry."""

from pathlib import Path

import pytest

from src.skills.loader import SkillManifest
from src.skills.registry import SkillRegistry


@pytest.fixture
def registry():
    r = SkillRegistry()
    r.register(SkillManifest(
        name="weather", description="Fetch weather", trigger="weather questions",
        permissions=["http_request"], entry_point="tool.py",
        author="human", trusted=True, created="2026-01-01", path=Path("/tmp/weather"),
    ))
    r.register(SkillManifest(
        name="stocks", description="Check stocks", trigger="stock questions",
        permissions=["http_request"], entry_point="tool.py",
        author="agent", trusted=False, created="2026-01-01", path=Path("/tmp/stocks"),
    ))
    return r


def test_get_skill(registry):
    assert registry.get("weather") is not None
    assert registry.get("weather").name == "weather"


def test_get_missing_skill(registry):
    assert registry.get("nonexistent") is None


def test_get_skill_index(registry):
    index = registry.get_skill_index()
    assert "weather" in index
    assert "stocks" in index
    assert "trusted" in index
    assert "untrusted" in index


def test_all_skills(registry):
    assert len(registry.all_skills()) == 2


def test_empty_registry():
    r = SkillRegistry()
    assert r.get_skill_index() == "No skills available."
    assert r.all_skills() == []
