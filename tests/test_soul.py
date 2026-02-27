"""Tests for SOUL.md loading."""

from pathlib import Path

from src.soul import load_soul, SOUL_SEED


def test_load_soul_creates_seed_if_missing(tmp_path: Path):
    soul_path = tmp_path / "SOUL.md"
    content = load_soul(soul_path)
    assert soul_path.exists()
    assert content == SOUL_SEED
    assert "AI assistant" in content


def test_load_soul_reads_existing(tmp_path: Path):
    soul_path = tmp_path / "SOUL.md"
    soul_path.write_text("Custom soul content")
    content = load_soul(soul_path)
    assert content == "Custom soul content"


def test_soul_seed_contains_hostility_policy():
    assert "untrusted" in SOUL_SEED.lower() or "hostile" in SOUL_SEED.lower()
    assert "prompt injection" in SOUL_SEED.lower()
