from crawl4ai import AsyncWebCrawler, CrawlResult


async def fetch_web_page(url: str) -> CrawlResult:
    """Fetches web page content asynchronously using Crawl4AI library."""
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(url)
