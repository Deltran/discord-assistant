"""Web content hostility detection and sanitization."""

import logging
import re

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+(instructions|prompts)",
    r"you\s+are\s+now\s+",
    r"system\s+override",
    r"new\s+instructions?\s*:",
    r"forget\s+(all|everything|your)\s+",
    r"disregard\s+(all|previous|your)\s+",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"act\s+as\s+(if\s+you\s+are|a)\s+",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> list[str]:
    """Scan text for prompt injection attempts. Returns list of matched patterns."""
    matches = []
    for pattern in _compiled_patterns:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_web_content(content: str, *, source_url: str = "") -> tuple[str, list[str]]:
    """Sanitize web content and detect injection attempts.

    Returns (sanitized_content, list_of_detected_injections).
    """
    injections = detect_prompt_injection(content)
    if injections:
        logger.warning(f"Prompt injection detected from {source_url}: {injections}")
    # Content is returned as-is (it's data, not instructions)
    # The injections list is returned so the caller can log/report
    return content, injections
