"""Web research tools — search, fetch, and browse."""

import logging
from urllib.parse import urlparse

import aiohttp
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

BLOCKED_PROTOCOLS = {"file", "ftp", "data", "javascript"}
MAX_RESPONSE_SIZE = 100_000  # 100KB text limit
MAX_REDIRECTS = 5


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


async def _do_web_search(query: str) -> list[dict]:
    """Perform a web search. Placeholder for search API integration."""
    # TODO: Integrate SerpAPI, Tavily, or similar
    logger.warning("Web search not yet configured — returning empty results")
    return []


@tool
async def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Args:
        query: The search query string.
    """
    results = await _do_web_search(query)
    if not results:
        return "No search results found."
    lines = []
    for r in results:
        lines.append(f"**{r['title']}** ({r['url']})\n{r.get('snippet', '')}")
    return "\n\n".join(lines)


@tool
async def http_request(url: str) -> str:
    """Fetch content from a URL.

    Args:
        url: The URL to fetch. Must be http or https.
    """
    url = sanitize_url(url)
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            max_redirects=MAX_REDIRECTS,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            text = await response.text()
            if len(text) > MAX_RESPONSE_SIZE:
                text = text[:MAX_RESPONSE_SIZE] + "\n[Truncated]"
            return text


@tool
async def headless_browser(url: str) -> str:
    """Browse a URL and return the accessibility tree for page understanding.

    Uses Playwright to render JavaScript-heavy pages and extract the
    accessibility tree instead of screenshots.

    Args:
        url: The URL to browse.
    """
    url = sanitize_url(url)
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return "Error: Playwright not available. Install with: playwright install chromium"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=30000)
        tree = await page.accessibility.snapshot()
        await browser.close()

    if not tree:
        return "Could not extract accessibility tree from page."

    return _format_accessibility_tree(tree)


def _format_accessibility_tree(node: dict, indent: int = 0) -> str:
    """Format an accessibility tree node into readable text."""
    prefix = "  " * indent
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [f"{prefix}{role}"]
    if name:
        parts.append(f'"{name}"')
    if value:
        parts.append(f"value={value}")

    line = " ".join(parts)
    lines = [line]

    for child in node.get("children", []):
        lines.append(_format_accessibility_tree(child, indent + 1))

    return "\n".join(lines)
