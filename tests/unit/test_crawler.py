import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services import crawler
from crawl4ai import CrawlResult

@pytest.mark.asyncio
@patch("app.services.crawler.AsyncWebCrawler")

# Tests that fetch_web_page returns a successful fetch correctly
@pytest.mark.asyncio
async def test_fetch_web_page_success(mock_crawler_cls):

    mock_result = MagicMock(spec=CrawlResult)
    mock_result.url = "https://example.com"
    mock_result.success = True
    mock_result.status_code = 200
    mock_result.markdown = "<html>Testi</html>"

    with patch("app.services.crawler.AsyncWebCrawler") as mock_crawler_cls:
        mock_crawler = AsyncMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)

        mock_crawler_cls.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler_cls.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await crawler.fetch_web_page("https://example.com")

        mock_crawler.arun.assert_awaited_once_with("https://example.com")
        assert result.success is True
        assert result.markdown == "<html>Testi</html>"

@pytest.mark.asyncio
@patch("app.services.crawler.AsyncWebCrawler")

# Tests that fetch_web_page handles a failed fetch correctly
async def test_fetch_web_page_failure(mock_crawler_cls):
    mock_crawler = AsyncMock()
    mock_crawler.arun.return_value = AsyncMock(
        success=False,
        markdown="Not Found"
    )
    mock_crawler_cls.return_value.__aenter__.return_value = mock_crawler

    result = await crawler.fetch_web_page("https://example.com")

    mock_crawler.arun.assert_awaited_once_with("https://example.com")
    assert result.success is False
    assert result.markdown == "Not Found"