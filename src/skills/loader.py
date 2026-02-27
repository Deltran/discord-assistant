"""Skill manifest discovery, loading, and hot-reload."""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SkillManifest:
    name: str
    description: str
    trigger: str
    permissions: list[str]
    entry_point: str
    author: str
    trusted: bool
    created: str
    path: Path


def load_manifests(skills_dir: Path) -> list[SkillManifest]:
    """Discover and load all skill manifests from a directory."""
    manifests = []
    if not skills_dir.exists():
        return manifests

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        manifest_path = skill_dir / "manifest.yaml"
        if not manifest_path.exists():
            logger.warning(f"Skill {skill_dir.name}: no manifest.yaml, skipping")
            continue
        try:
            with open(manifest_path) as f:
                data = yaml.safe_load(f)
            manifests.append(SkillManifest(
                name=data["name"],
                description=data["description"],
                trigger=data["trigger"],
                permissions=data.get("permissions", []),
                entry_point=data.get("entry_point", "tool.py"),
                author=data.get("author", "unknown"),
                trusted=data.get("trusted", False),
                created=data.get("created", "unknown"),
                path=skill_dir,
            ))
        except Exception as e:
            logger.error(f"Failed to load {skill_dir.name}: {e}")

    return manifests
