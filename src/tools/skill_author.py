"""Skill authoring tool — lets the agent create new skills at runtime."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


def create_skill_author_tool(*, skills_dir: Path, registry: SkillRegistry):
    """Factory: create a create_skill @tool closure bound to a skills directory and registry."""

    @tool
    async def create_skill(
        name: str,
        description: str,
        trigger: str,
        code: str,
        permissions: str = "",
    ) -> str:
        """Create a new skill that can be invoked later.

        The skill code must define an async function: async def run(input_text: str) -> str

        Args:
            name: Unique skill name (lowercase, hyphens allowed, no spaces).
            description: What the skill does.
            trigger: When to use this skill (natural language).
            code: Python source code. Must define
                async def run(input_text: str) -> str.
            permissions: Comma-separated tool permissions
                (e.g. "http_request,file_read"). Empty for none.
        """
        # Validate name
        if not name or " " in name:
            return "Error: Skill name must be non-empty with no spaces. Use hyphens."

        # Validate code has run function
        if "async def run(" not in code:
            return "Error: Skill code must define 'async def run(input_text: str) -> str'"

        skill_dir = skills_dir / name
        if skill_dir.exists():
            return f"Error: Skill '{name}' already exists at {skill_dir}"

        # Parse permissions
        perm_list = [p.strip() for p in permissions.split(",") if p.strip()] if permissions else []

        # Create skill directory and files
        try:
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Write manifest
            manifest_data = {
                "name": name,
                "description": description,
                "trigger": trigger,
                "permissions": perm_list,
                "entry_point": "tool.py",
                "author": "agent",
                "trusted": False,
                "created": str(date.today()),
            }
            manifest_yaml = yaml.dump(manifest_data, default_flow_style=False)
            (skill_dir / "manifest.yaml").write_text(manifest_yaml)

            # Write code
            (skill_dir / "tool.py").write_text(code)

            # Reload registry to pick up the new skill
            builtin_dir = Path(__file__).parent.parent / "skills" / "builtin"
            registry.reload(builtin_dir, skills_dir)

            logger.info("Created new skill: %s", name)
            return (
                f"Skill '{name}' created at {skill_dir}. "
                "It is now available for use."
            )
        except Exception as e:
            logger.exception("Failed to create skill '%s'", name)
            # Clean up on failure
            if skill_dir.exists():
                import shutil
                shutil.rmtree(skill_dir, ignore_errors=True)
            return f"Error creating skill: {e}"

    return create_skill
