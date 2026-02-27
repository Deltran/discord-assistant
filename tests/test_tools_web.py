"""Tests for web research tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.tools.web import web_search, http_request, sanitize_url


def test_sanitize_url_valid():
    assert sanitize_url("https://example.com") == "https://example.com"


def test_sanitize_url_rejects_file_protocol():
    with pytest.raises(ValueError, match="protocol"):
        sanitize_url("file:///etc/passwd")


def test_sanitize_url_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_url("")


@pytest.mark.asyncio
async def test_http_request_fetches():
    with patch("src.tools.web.aiohttp") as mock_aiohttp:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="<html>Hello</html>")
        mock_response.headers = {"content-type": "text/html"}

        # Build a context-manager mock for session.get(...)
        mock_get_cm = AsyncMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.ClientTimeout = aiohttp.ClientTimeout  # pass through

        result = await http_request.ainvoke({"url": "https://example.com"})
        assert "Hello" in result


@pytest.mark.asyncio
async def test_web_search_returns_results():
    with patch("src.tools.web._do_web_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            {"title": "Result 1", "url": "https://example.com", "snippet": "First result"},
        ]
        result = await web_search.ainvoke({"query": "test query"})
        assert "Result 1" in result
