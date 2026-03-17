import pytest
from unittest.mock import AsyncMock, patch
from app.services import crawler

@pytest.mark.asyncio
@patch("app.services.crawler.AsyncWebCrawler")

# Tests that fetch_web_page returns a successful fetch correctly
async def test_fetch_web_page_success(mock_crawler_cls):
    mock_crawler = AsyncMock()
    mock_crawler.arun.return_value = AsyncMock(
        success=True,
        markdown="<html>Testi</html>"
    )
    
    mock_crawler_cls.return_value.__aenter__.return_value = mock_crawler
    result = await crawler.fetch_web_page("https://example.com")

    mock_crawler.arun.assert_awaited_once_with("https://example.com")
    assert result.success is True
    assert "<html>Testi</html>" == result.markdown

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