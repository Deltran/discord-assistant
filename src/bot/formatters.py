"""Output formatting for Discord messages."""

DISCORD_MAX_LENGTH = 2000


def split_message(text: str, *, max_length: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split a long message into chunks that fit within Discord's limit.

    Tries to split on newlines when possible, otherwise splits at max_length.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1 or split_at == 0:
            split_at = max_length

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


def format_code_block(code: str, language: str = "") -> str:
    """Wrap text in a Discord code block."""
    return f"```{language}\n{code}\n```"
