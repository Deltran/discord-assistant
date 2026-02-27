"""Tests for skills manifest discovery and loading."""

import pytest
import yaml

from src.skills.loader import SkillManifest, load_manifests


@pytest.fixture
def skills_dir(tmp_path):
    weather = tmp_path / "weather"
    weather.mkdir()
    manifest = {
        "name": "weather",
        "description": "Fetch weather forecasts",
        "trigger": "When the user asks about weather",
        "permissions": ["http_request"],
        "entry_point": "tool.py",
        "author": "human",
        "trusted": True,
        "created": "2026-01-01",
    }
    (weather / "manifest.yaml").write_text(yaml.dump(manifest))
    (weather / "tool.py").write_text("def run(): pass")
    return tmp_path


def test_load_manifests(skills_dir):
    manifests = load_manifests(skills_dir)
    assert len(manifests) == 1
    assert manifests[0].name == "weather"
    assert manifests[0].trusted is True
    assert manifests[0].path == skills_dir / "weather"


def test_load_manifests_skips_invalid(skills_dir):
    bad = skills_dir / "broken"
    bad.mkdir()
    (bad / "tool.py").write_text("def run(): pass")
    manifests = load_manifests(skills_dir)
    assert len(manifests) == 1


def test_load_manifests_empty_dir(tmp_path):
    manifests = load_manifests(tmp_path)
    assert manifests == []


def test_load_manifests_nonexistent_dir(tmp_path):
    manifests = load_manifests(tmp_path / "nope")
    assert manifests == []
