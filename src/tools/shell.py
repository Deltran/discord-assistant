"""Shell command execution with safety checks."""

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DESTRUCTIVE_PATTERNS = [
    "rm ", "rm\t", "rmdir",
    "dd ", "mkfs",
    "chmod", "chown",
    "> /dev/", ">/dev/",
    "kill -9", "killall",
    "format ",
    "fdisk",
]

TIMEOUT_SECONDS = 30


def is_destructive_command(command: str) -> bool:
    """Check if a command is potentially destructive."""
    cmd_lower = command.lower().strip()
    return any(pattern in cmd_lower for pattern in DESTRUCTIVE_PATTERNS)


@tool
async def shell_exec(command: str) -> str:
    """Execute a shell command and return the output.

    Destructive commands (rm, dd, chmod, etc.) are blocked and require
    explicit user confirmation before execution.

    Args:
        command: The shell command to execute.
    """
    if is_destructive_command(command):
        return (
            f"⚠️ Destructive command detected: `{command}`\n"
            "This requires explicit user confirmation before execution. "
            "Please confirm this action."
        )

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
        output = stdout.decode().strip()
        errors = stderr.decode().strip()

        result = ""
        if output:
            result += output
        if errors:
            result += f"\nSTDERR: {errors}"
        if proc.returncode != 0:
            result += f"\nExit code: {proc.returncode}"

        return result or "(no output)"
    except asyncio.TimeoutError:
        return f"Command timed out after {TIMEOUT_SECONDS}s"
    except Exception as e:
        return f"Error executing command: {e}"
