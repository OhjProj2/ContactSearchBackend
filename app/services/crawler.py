from crawl4ai import AsyncWebCrawler, CrawlResult


async def fetch_web_page(url: str) -> CrawlResult:
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(url)
