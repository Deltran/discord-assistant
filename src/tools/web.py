"""Web research tools — search, fetch, and browse via Firecrawl."""

import logging
from urllib.parse import urlparse

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

BLOCKED_PROTOCOLS = {"file", "ftp", "data", "javascript"}
MAX_RESPONSE_SIZE = 100_000  # 100KB text limit


def sanitize_url(url: str) -> str:
    """Validate and sanitize a URL for safe fetching."""
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")
    parsed = urlparse(url)
    if parsed.scheme in BLOCKED_PROTOCOLS:
        raise ValueError(f"Blocked protocol: {parsed.scheme}")
    if not parsed.scheme:
        url = "https://" + url
    return url


def _get_firecrawl_client():
    """Get an async Firecrawl client, or None if not configured."""
    try:
        from firecrawl import AsyncFirecrawl
        from src.settings import Settings
        settings = Settings()
        if settings.firecrawl_api_key is None:
            logger.warning("FIRECRAWL_API_KEY not set")
            return None
        return AsyncFirecrawl(api_key=settings.firecrawl_api_key.get_secret_value())
    except Exception:
        logger.exception("Failed to create Firecrawl client")
        return None


@tool
async def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query string.
    """
    client = _get_firecrawl_client()
    if client is None:
        return "Web search is not configured. Set FIRECRAWL_API_KEY in .env."

    try:
        results = await client.search(query, limit=5)
        # Firecrawl v4 returns SearchData (Pydantic model), not a dict
        # Access .data attribute to get the list of results
        items = results.data if hasattr(results, 'data') else []
        if not items:
            return "No search results found."
        lines = []
        for r in items:
            title = r.get("title", "Untitled") if isinstance(r, dict) else getattr(r, "title", "Untitled")
            url = r.get("url", "") if isinstance(r, dict) else getattr(r, "url", "")
            snippet = r.get("description", "") if isinstance(r, dict) else getattr(r, "description", "")
            lines.append(f"**{title}** ({url})\n{snippet}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Firecrawl search failed")
        return f"Search failed: {e}"


@tool
async def scrape_url(url: str) -> str:
    """Scrape a URL and return its content as clean markdown.

    Args:
        url: The URL to scrape. Must be http or https.
    """
    url = sanitize_url(url)
    client = _get_firecrawl_client()
    if client is None:
        return "Web scraping is not configured. Set FIRECRAWL_API_KEY in .env."

    try:
        result = await client.scrape(url, formats=["markdown"])
        # Handle both dict and Pydantic model responses
        markdown = result.get("markdown", "") if isinstance(result, dict) else getattr(result, "markdown", "")
        if not markdown:
            return "No content could be extracted from this page."
        if len(markdown) > MAX_RESPONSE_SIZE:
            markdown = markdown[:MAX_RESPONSE_SIZE] + "\n[Truncated]"
        return markdown
    except Exception as e:
        logger.exception("Firecrawl scrape failed for %s", url)
        return f"Scrape failed: {e}"


@tool
async def http_request(url: str) -> str:
    """Fetch raw content from a URL (no rendering or markdown conversion).

    Use scrape_url for clean content. This is for raw HTTP when needed.

    Args:
        url: The URL to fetch. Must be http or https.
    """
    import aiohttp

    url = sanitize_url(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            max_redirects=5,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            text = await response.text()
            if len(text) > MAX_RESPONSE_SIZE:
                text = text[:MAX_RESPONSE_SIZE] + "\n[Truncated]"
            return text
