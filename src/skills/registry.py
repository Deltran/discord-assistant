"""Skill registry for routing — builds condensed index for system prompt."""

from src.skills.loader import SkillManifest


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
