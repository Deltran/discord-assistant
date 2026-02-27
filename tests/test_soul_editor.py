"""Tests for SOUL.md propose-and-approve editing."""

from pathlib import Path

import pytest

from src.soul_editor import SoulEditor, SoulProposal


@pytest.fixture
def editor(tmp_path):
    soul_path = tmp_path / "SOUL.md"
    soul_path.write_text("# Identity\n\nOriginal content.\n")
    return SoulEditor(soul_path)


def test_create_proposal(editor):
    proposal = editor.create_proposal(
        new_content="# Identity\n\nUpdated content.\n",
        reason="User demonstrated preference for concise responses",
    )
    assert isinstance(proposal, SoulProposal)
    assert "Updated content" in proposal.new_content
    assert proposal.reason == "User demonstrated preference for concise responses"
    assert proposal.diff  # Should have a diff


def test_apply_proposal(editor):
    proposal = editor.create_proposal(
        new_content="# Identity\n\nNew content.\n",
        reason="Test",
    )
    editor.apply_proposal(proposal)
    assert editor.soul_path.read_text() == "# Identity\n\nNew content.\n"


def test_reject_proposal_preserves_original(editor):
    original = editor.soul_path.read_text()
    proposal = editor.create_proposal(new_content="Changed", reason="Test")
    # Don't apply it
    assert editor.soul_path.read_text() == original


def test_proposal_diff_is_readable(editor):
    proposal = editor.create_proposal(
        new_content="# Identity\n\nCompletely different.\n",
        reason="Test",
    )
    assert "Original content" in proposal.diff or "---" in proposal.diff
    assert "Completely different" in proposal.diff or "+++" in proposal.diff
