"""Tests for operational memory."""

import pytest

from src.memory.operational import OperationalMemory


@pytest.fixture
def opmem(tmp_path):
    return OperationalMemory(memory_dir=tmp_path)


def test_initializes_files(opmem):
    opmem.initialize()
    assert opmem.safety_rules_path.exists()
    assert opmem.preferences_path.exists()
    assert opmem.operational_notes_path.exists()


def test_append_safety_rule(opmem):
    opmem.initialize()
    opmem.append_safety_rule("Never visit example-malware.com — detected prompt injection")
    content = opmem.safety_rules_path.read_text()
    assert "example-malware.com" in content


def test_safety_rules_append_only(opmem):
    opmem.initialize()
    opmem.append_safety_rule("Rule 1")
    opmem.append_safety_rule("Rule 2")
    content = opmem.safety_rules_path.read_text()
    assert "Rule 1" in content
    assert "Rule 2" in content


def test_update_preference(opmem):
    opmem.initialize()
    opmem.update_preference("formatting", "User prefers bullet points")
    content = opmem.preferences_path.read_text()
    assert "bullet points" in content


def test_add_operational_note(opmem):
    opmem.initialize()
    opmem.add_operational_note("SerpAPI works better than Tavily for news queries")
    content = opmem.operational_notes_path.read_text()
    assert "SerpAPI" in content


def test_read_all(opmem):
    opmem.initialize()
    opmem.append_safety_rule("Test rule")
    opmem.update_preference("test", "Test pref")
    opmem.add_operational_note("Test note")
    all_mem = opmem.read_all()
    assert "safety_rules" in all_mem
    assert "preferences" in all_mem
    assert "operational_notes" in all_mem
    assert "Test rule" in all_mem["safety_rules"]
