"""Tests for web research tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.tools.web import web_search, scrape_url, http_request, sanitize_url


def test_sanitize_url_valid():
    assert sanitize_url("https://example.com") == "https://example.com"


def test_sanitize_url_rejects_file_protocol():
    with pytest.raises(ValueError, match="protocol"):
        sanitize_url("file:///etc/passwd")


def test_sanitize_url_rejects_empty():
    with pytest.raises(ValueError):
        sanitize_url("")


def test_sanitize_url_adds_https():
    assert sanitize_url("example.com") == "https://example.com"


@pytest.mark.asyncio
async def test_http_request_fetches():
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="<html>Hello</html>")
    mock_response.headers = {"content-type": "text/html"}

    mock_get_cm = AsyncMock()
    mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_get_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock(return_value=mock_get_cm)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await http_request.ainvoke({"url": "https://example.com"})
        assert "Hello" in result


@pytest.mark.asyncio
async def test_web_search_returns_results():
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={
        "data": [
            {"title": "Result 1", "url": "https://example.com", "description": "First result"},
        ]
    })

    with patch("src.tools.web._get_firecrawl_client", return_value=mock_client):
        result = await web_search.ainvoke({"query": "test query"})
        assert "Result 1" in result
        mock_client.search.assert_called_once_with("test query", limit=5)


@pytest.mark.asyncio
async def test_web_search_no_api_key():
    with patch("src.tools.web._get_firecrawl_client", return_value=None):
        result = await web_search.ainvoke({"query": "test"})
        assert "not configured" in result


@pytest.mark.asyncio
async def test_scrape_url_returns_markdown():
    mock_client = AsyncMock()
    mock_client.scrape = AsyncMock(return_value={
        "markdown": "# Hello World\n\nThis is the page content."
    })

    with patch("src.tools.web._get_firecrawl_client", return_value=mock_client):
        result = await scrape_url.ainvoke({"url": "https://example.com"})
        assert "Hello World" in result
        mock_client.scrape.assert_called_once_with("https://example.com", formats=["markdown"])


@pytest.mark.asyncio
async def test_scrape_url_no_api_key():
    with patch("src.tools.web._get_firecrawl_client", return_value=None):
        result = await scrape_url.ainvoke({"url": "https://example.com"})
        assert "not configured" in result


@pytest.mark.asyncio
async def test_scrape_url_truncates_large_content():
    mock_client = AsyncMock()
    mock_client.scrape = AsyncMock(return_value={
        "markdown": "x" * 200_000
    })

    with patch("src.tools.web._get_firecrawl_client", return_value=mock_client):
        result = await scrape_url.ainvoke({"url": "https://example.com"})
        assert result.endswith("[Truncated]")
        assert len(result) < 200_000
