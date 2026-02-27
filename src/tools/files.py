"""File system tools with safety checks."""

import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

MAX_READ_SIZE = 500_000  # 500KB


@tool
async def file_read(path: str) -> str:
    """Read the contents of a file.

    Args:
        path: Absolute or relative path to the file to read.
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        content = p.read_text()
        if len(content) > MAX_READ_SIZE:
            content = content[:MAX_READ_SIZE] + "\n[Truncated]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
async def file_write(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed.

    Args:
        path: Path to the file to write.
        content: The content to write to the file.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"
