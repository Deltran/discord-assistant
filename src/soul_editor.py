"""SOUL.md propose-and-approve editing.

Agent never writes directly. It creates proposals with diffs,
which are posted to the monitoring channel for user approval.
"""

import difflib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SoulProposal:
    old_content: str
    new_content: str
    reason: str
    diff: str


class SoulEditor:
    def __init__(self, soul_path: Path):
        self.soul_path = soul_path

    def create_proposal(self, *, new_content: str, reason: str) -> SoulProposal:
        old_content = self.soul_path.read_text() if self.soul_path.exists() else ""
        diff = "\n".join(
            difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile="SOUL.md (current)",
                tofile="SOUL.md (proposed)",
                lineterm="",
            )
        )
        return SoulProposal(
            old_content=old_content,
            new_content=new_content,
            reason=reason,
            diff=diff,
        )

    def apply_proposal(self, proposal: SoulProposal):
        self.soul_path.write_text(proposal.new_content)
