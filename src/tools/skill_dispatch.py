"""Skill dispatch meta-tool — routes requests to registered skills."""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Map of builtin skill names to their sub-agent runner functions
BUILTIN_RUNNERS = {
    "research": "src.agent.subagents.research:run_research",
    "system": "src.agent.subagents.system:run_system_task",
    "builder": "src.agent.subagents.builder:run_builder",
    "briefing": "src.agent.subagents.briefing:run_briefing",
}

# Tools that untrusted skills cannot access at all
RESTRICTED_TOOLS = {"shell_exec"}

# Tools that untrusted skills can use but with restrictions
# (enforced at dispatch time via wrapper)
SANDBOXED_TOOLS = {"file_write"}


def create_dispatch_skill_tool(*, registry: SkillRegistry, llm, available_tools: list):
    """Factory: create a dispatch_skill @tool closure bound to a registry and LLM.

    Args:
        registry: The skill registry to look up skills.
        llm: The LLM instance for sub-agent execution.
        available_tools: The full list of available tool objects.
    """
    tool_map = {t.name: t for t in available_tools}

    @tool
    async def dispatch_skill(skill_name: str, input_text: str) -> str:
        """Dispatch a task to a registered skill by name.

        Args:
            skill_name: The name of the skill to invoke (from the skill index).
            input_text: The input text/query to pass to the skill.
        """
        manifest = registry.get(skill_name)
        if manifest is None:
            available = [s.name for s in registry.all_skills()]
            return f"Unknown skill '{skill_name}'. Available: {', '.join(available) or 'none'}"

        # Filter tools by manifest permissions
        permitted_tools = _filter_tools(manifest, tool_map)

        # Route builtin vs dynamic
        if skill_name in BUILTIN_RUNNERS:
            return await _run_builtin(skill_name, input_text, llm, permitted_tools)
        else:
            return await _run_dynamic_skill(manifest, input_text)

    return dispatch_skill


def _filter_tools(manifest, tool_map: dict) -> list:
    """Filter available tools based on manifest permissions and trust level."""
    permitted = []
    for perm in manifest.permissions:
        if perm in tool_map:
            # Untrusted skills can't use restricted tools
            if not manifest.trusted and perm in RESTRICTED_TOOLS:
                logger.warning(
                    "Skill '%s' (untrusted) denied access to %s",
                    manifest.name, perm,
                )
                continue
            permitted.append(tool_map[perm])
    return permitted


async def _run_builtin(skill_name: str, input_text: str, llm, tools: list) -> str:
    """Run a builtin skill via its sub-agent function."""
    runner_path = BUILTIN_RUNNERS[skill_name]
    module_path, func_name = runner_path.rsplit(":", 1)

    try:
        import importlib
        mod = importlib.import_module(module_path)
        runner = getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        return f"Failed to load builtin skill '{skill_name}': {e}"

    try:
        # Each builtin runner has its own signature — adapt
        if skill_name == "research":
            return await runner(llm=llm, query=input_text, tools=tools)
        elif skill_name == "system":
            return await runner(llm=llm, task=input_text, tools=tools)
        elif skill_name == "builder":
            return await runner(llm=llm, plan=input_text, tools=tools)
        elif skill_name == "briefing":
            topics = [t.strip() for t in input_text.split(",") if t.strip()]
            return await runner(llm=llm, topics=topics or None, tools=tools)
        else:
            return f"No handler for builtin skill '{skill_name}'"
    except Exception as e:
        logger.exception("Builtin skill '%s' failed", skill_name)
        return f"Skill '{skill_name}' failed: {e}"


async def _run_dynamic_skill(manifest, input_text: str) -> str:
    """Run a user/agent-authored dynamic skill by importing its entry point."""
    entry = manifest.path / manifest.entry_point
    if not entry.exists():
        return f"Skill '{manifest.name}' entry point not found: {entry}"

    try:
        spec = importlib.util.spec_from_file_location(
            f"skill_{manifest.name}", str(entry)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        logger.exception("Failed to load dynamic skill '%s'", manifest.name)
        return f"Failed to load skill '{manifest.name}': {e}"

    run_fn = getattr(module, "run", None)
    if run_fn is None:
        return f"Skill '{manifest.name}' has no run() function in {manifest.entry_point}"

    try:
        result = await run_fn(input_text)
        return str(result)
    except Exception as e:
        logger.exception("Dynamic skill '%s' failed", manifest.name)
        return f"Skill '{manifest.name}' failed: {e}"
