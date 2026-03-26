from crawl4ai import AsyncWebCrawler, CrawlResult


async def fetch_web_page(url: str) -> CrawlResult:
    """Fetches web page content asynchronously using Crawl4AI library."""
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url)
            if not result.success:
                return result
            if result.status_code >= 400:
                result.success = False
                result.error_message = f"HTTP Error {result.status_code}: {result.url}"
        return result
    except Exception as e:
        return CrawlResult(
            url=url,
            success=False,
            error_message=f"Connection failed: {str(e)}",
            status_code=400,
            markdown="",
            html="",
            metadata={}
        )
