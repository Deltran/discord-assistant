"""Shell command execution with safety checks."""

import asyncio
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

CATASTROPHIC_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "dd if=/dev/zero of=/dev/",
    "dd if=/dev/random of=/dev/",
    "mkfs.",
    "> /dev/sda",
    ">/dev/sda",
    "fdisk /dev/",
]

TIMEOUT_SECONDS = 30


def is_catastrophic_command(command: str) -> bool:
    """Check if a command would be catastrophically destructive."""
    cmd_lower = command.lower().strip()
    return any(pattern in cmd_lower for pattern in CATASTROPHIC_PATTERNS)


@tool
async def shell_exec(command: str) -> str:
    """Execute a shell command and return the output.

    Catastrophic commands (rm -rf /, dd to disk, mkfs, etc.) are permanently blocked.

    Args:
        command: The shell command to execute.
    """
    if is_catastrophic_command(command):
        return (
            f"🚫 Blocked: `{command}` — this command could cause catastrophic damage "
            "and will never be executed."
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
