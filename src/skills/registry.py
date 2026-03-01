"""Skill registry for routing — builds condensed index for system prompt."""

import logging
from pathlib import Path

from src.skills.loader import SkillManifest, load_manifests

logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, SkillManifest] = {}

    def register(self, manifest: SkillManifest):
        self._skills[manifest.name] = manifest

    def get(self, name: str) -> SkillManifest | None:
        return self._skills.get(name)

    def get_skill_index(self) -> str:
        """Build condensed skill index for injection into system prompt."""
        if not self._skills:
            return "No skills available."
        lines = []
        for skill in self._skills.values():
            trust = "trusted" if skill.trusted else "untrusted"
            lines.append(f"- **{skill.name}** ({trust}): {skill.trigger}")
        return "\n".join(lines)

    def all_skills(self) -> list[SkillManifest]:
        return list(self._skills.values())

    def reload(self, *dirs: Path) -> int:
        """Re-scan directories and reload all skill manifests.

        Returns the number of skills loaded.
        """
        self._skills.clear()
        for d in dirs:
            if d and d.exists():
                for manifest in load_manifests(d):
                    self.register(manifest)
                    logger.info("Loaded skill: %s", manifest.name)
        return len(self._skills)
